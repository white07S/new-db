import json
import uuid
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dbgpt.core import Chunk
from dbgpt.storage.vector_store.filters import (
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

from .retriever import InMemoryBM25Retriever
from . import logger


TABLE_DOC = "table"
COLUMN_DOC = "column"
COMMENT_FIELD = "comment"
BUSINESS_FIELD = "business"


@dataclass
class SchemaSearchResult:
    context: Dict[str, Dict[str, Any]]
    ordered_tables: List[str]
    new_tables: List[str]
    expanded_tables: List[str]
    columns_added: Dict[str, List[str]]

    def is_empty(self) -> bool:
        return not self.context


class SchemaSearchEngine:
    """Search engine over JSON schema definitions."""

    def __init__(
        self,
        schema_path: Path,
        vector_store,
        max_table_candidates: int = 3,
        max_columns_per_table: int = 5,
        bm25_top_k: int = 8,
    ) -> None:
        self.schema_path = schema_path
        self.vector_store = vector_store
        self.max_table_candidates = max_table_candidates
        self.max_columns_per_table = max_columns_per_table
        self.bm25_top_k = bm25_top_k

        self.schema_data = self._load_schema()
        self.table_comment_chunks: List[Chunk] = []
        self.table_business_chunks: List[Chunk] = []
        self.column_comment_chunks: List[Chunk] = []
        self.column_business_chunks: List[Chunk] = []

        self._index_schema()
        self._init_bm25()

    def _load_schema(self) -> Dict[str, Any]:
        with self.schema_path.open("r") as f:
            schema = json.load(f)
        tables = schema.get("tables", [])
        return {table["table_name"]: table for table in tables}

    def _index_schema(self) -> None:
        """Convert schema json into vector-store chunks."""
        all_chunks: List[Chunk] = []
        for table_name, table_info in self.schema_data.items():
            table_comment = table_info.get("table_name_comment", "")
            if table_comment:
                chunk = self._build_chunk(
                    content=table_comment,
                    table_name=table_name,
                    column_name=None,
                    doc_type=TABLE_DOC,
                    field_type=COMMENT_FIELD,
                )
                self.table_comment_chunks.append(chunk)
                all_chunks.append(chunk)

            table_business = table_info.get("table_name_business_interpretations", "")
            if table_business:
                chunk = self._build_chunk(
                    content=table_business,
                    table_name=table_name,
                    column_name=None,
                    doc_type=TABLE_DOC,
                    field_type=BUSINESS_FIELD,
                )
                self.table_business_chunks.append(chunk)
                all_chunks.append(chunk)

            for column in table_info.get("columns", []):
                column_name = column.get("column_name")
                if not column_name:
                    continue
                column_comment = column.get("column_comment", "")
                if column_comment:
                    chunk = self._build_chunk(
                        content=column_comment,
                        table_name=table_name,
                        column_name=column_name,
                        doc_type=COLUMN_DOC,
                        field_type=COMMENT_FIELD,
                    )
                    self.column_comment_chunks.append(chunk)
                    all_chunks.append(chunk)

                column_business = column.get("column_business_interpretation", "")
                if column_business:
                    chunk = self._build_chunk(
                        content=column_business,
                        table_name=table_name,
                        column_name=column_name,
                        doc_type=COLUMN_DOC,
                        field_type=BUSINESS_FIELD,
                    )
                    self.column_business_chunks.append(chunk)
                    all_chunks.append(chunk)

        if all_chunks:
            self.vector_store.load_document(all_chunks)
            logger.info(
                "Indexed schema into vector store",
                extra={
                    "props": {
                        "chunks": len(all_chunks),
                        "tables": len(self.schema_data),
                    }
                },
            )

    def _init_bm25(self) -> None:
        """Create BM25 retrievers for each chunk family."""
        self.table_comment_bm25 = InMemoryBM25Retriever(
            self.table_comment_chunks, top_k=self.bm25_top_k
        )
        self.table_business_bm25 = InMemoryBM25Retriever(
            self.table_business_chunks, top_k=self.bm25_top_k
        )
        self.column_comment_bm25 = InMemoryBM25Retriever(
            self.column_comment_chunks, top_k=self.bm25_top_k
        )
        self.column_business_bm25 = InMemoryBM25Retriever(
            self.column_business_chunks, top_k=self.bm25_top_k
        )

    def _build_chunk(
        self,
        *,
        content: str,
        table_name: str,
        column_name: Optional[str],
        doc_type: str,
        field_type: str,
    ) -> Chunk:
        metadata = {
            "table_name": table_name,
            "doc_type": doc_type,
            "field": field_type,
        }
        if column_name:
            metadata["column_name"] = column_name
        return Chunk(
            content=content,
            metadata=metadata,
            chunk_id=f"{doc_type}:{field_type}:{table_name}:{column_name or 'na'}::"
            f"{uuid.uuid4()}",
        )

    def _vector_search(
        self, text: str, doc_type: str, field_type: str, topk: int
    ) -> List[Chunk]:
        filters = MetadataFilters(
            condition=FilterCondition.AND,
            filters=[
                MetadataFilter(key="doc_type", value=doc_type),
                MetadataFilter(key="field", value=field_type),
            ],
        )
        try:
            return self.vector_store.similar_search(text, topk, filters=filters)
        except Exception as exc:  # pragma: no cover - safety net
            logger.warning(
                "Vector search failed",
                extra={"props": {"error": str(exc), "doc_type": doc_type}},
            )
            return []

    @staticmethod
    def _rrf(results: Iterable[List[Chunk]], exclude_key: Optional[str] = None):
        """Reciprocal rank fusion with optional exclusion key."""
        scores: Dict[str, float] = {}
        meta: Dict[str, Chunk] = {}
        for result in results:
            for rank, chunk in enumerate(result):
                key = chunk.metadata.get(exclude_key) if exclude_key else chunk.chunk_id
                if not key:
                    continue
                if key not in scores:
                    scores[key] = 0.0
                    meta[key] = chunk
                scores[key] += 1.0 / (rank + 60)
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(key, score, meta[key]) for key, score in ordered]

    def _aggregate_tables(
        self, query: str, exclude_tables: Set[str]
    ) -> List[str]:
        table_rankings = self._rrf(
            [
                self._vector_search(
                    query,
                    doc_type=TABLE_DOC,
                    field_type=COMMENT_FIELD,
                    topk=self.max_table_candidates * 2,
                ),
                self._vector_search(
                    query,
                    doc_type=TABLE_DOC,
                    field_type=BUSINESS_FIELD,
                    topk=self.max_table_candidates * 2,
                ),
                self.table_comment_bm25._retrieve(query),
                self.table_business_bm25._retrieve(query),
            ],
            exclude_key="table_name",
        )
        ordered_tables: List[str] = []
        for table_name, _score, _chunk in table_rankings:
            if table_name in exclude_tables:
                continue
            if table_name not in self.schema_data:
                continue
            if table_name not in ordered_tables:
                ordered_tables.append(table_name)
            if len(ordered_tables) >= self.max_table_candidates:
                break
        return ordered_tables

    def _aggregate_columns(
        self,
        query: str,
        candidate_tables: List[str],
        existing_columns: Dict[str, Set[str]],
    ) -> Dict[str, List[str]]:
        column_rankings = self._rrf(
            [
                self._vector_search(
                    query,
                    doc_type=COLUMN_DOC,
                    field_type=COMMENT_FIELD,
                    topk=self.max_columns_per_table * len(candidate_tables) * 2 or 10,
                ),
                self._vector_search(
                    query,
                    doc_type=COLUMN_DOC,
                    field_type=BUSINESS_FIELD,
                    topk=self.max_columns_per_table * len(candidate_tables) * 2 or 10,
                ),
                self.column_comment_bm25._retrieve(query),
                self.column_business_bm25._retrieve(query),
            ],
            exclude_key=None,
        )

        table_columns: Dict[str, List[str]] = defaultdict(list)
        existing_tables = set(existing_columns.keys())
        allowed_tables = set(candidate_tables) | existing_tables

        for _key, _score, chunk in column_rankings:
            table_name = chunk.metadata.get("table_name")
            column_name = chunk.metadata.get("column_name")
            if not table_name or not column_name:
                continue
            if table_name not in allowed_tables:
                continue
            if column_name in existing_columns.get(table_name, set()):
                continue
            if column_name in table_columns[table_name]:
                continue
            if len(table_columns[table_name]) >= self.max_columns_per_table:
                continue
            table_columns[table_name].append(column_name)
        return table_columns

    def _build_table_context(
        self, table_name: str, selected_columns: List[str]
    ) -> Dict[str, Any]:
        table = self.schema_data[table_name]
        columns_info = []
        column_lookup = {
            column["column_name"]: column for column in table.get("columns", [])
        }
        for column_name in selected_columns:
            data = column_lookup.get(column_name)
            if not data:
                continue
            columns_info.append(
                {
                    "column_name": column_name,
                    "column_dtype": data.get("column_dtype", ""),
                    "column_comment": data.get("column_comment", ""),
                    "column_business_interpretation": data.get(
                        "column_business_interpretation", ""
                    ),
                }
            )

        context = {
            "table_name_comment": table.get("table_name_comment", ""),
            "table_name_business_interpretations": table.get(
                "table_name_business_interpretations", ""
            ),
            "primary_key": {
                "name": table.get("primary_key"),
                "comment": table.get("primary_key_comment", ""),
                "business_interpretation": table.get(
                    "primary_key_business_interpretations", ""
                ),
            },
            "foreign_keys": {
                "comment": table.get("foreign_key_comment", ""),
                "business_interpretations": table.get(
                    "foreign_key_business_interpretations", ""
                ),
                "relationships": table.get("foreign_keys", []),
            },
            "columns": columns_info,
        }
        return context

    async def search(
        self,
        query: str,
        existing_tables: Set[str],
        existing_columns: Dict[str, Set[str]],
    ) -> SchemaSearchResult:
        """Search schema for tables/columns relevant to query."""
        table_candidates = self._aggregate_tables(query, exclude_tables=existing_tables)
        column_candidates = self._aggregate_columns(
            query, candidate_tables=table_candidates, existing_columns=existing_columns
        )

        # Include tables that received new columns even if they were not in table search
        tables_to_return = list(OrderedDict.fromkeys(table_candidates))
        for table_name in column_candidates:
            if table_name not in tables_to_return:
                tables_to_return.append(table_name)

        context: Dict[str, Dict[str, Any]] = OrderedDict()
        columns_added: Dict[str, List[str]] = {}
        for table_name in tables_to_return:
            selected_columns = column_candidates.get(table_name, [])
            if not selected_columns and table_name not in table_candidates:
                continue
            context[table_name] = self._build_table_context(
                table_name, selected_columns
            )
            columns_added[table_name] = selected_columns

        new_tables = [t for t in tables_to_return if t not in existing_tables]
        expanded_tables = [t for t in tables_to_return if t in existing_tables]
        ordered_tables = tables_to_return

        return SchemaSearchResult(
            context=context,
            ordered_tables=ordered_tables,
            new_tables=new_tables,
            expanded_tables=expanded_tables,
            columns_added=columns_added,
        )

