"""
SQL Retrieval module using Qdrant for schema search.

This module provides SQL query generation and execution capabilities
with schema-aware retrieval using Qdrant vector store.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import copy

from providers.qdrant import QdrantProvider
from providers.embeddings import EmbeddingProvider
from providers.database import DatabaseProvider
from providers.llm import LLMProvider

from retrieval.sql_rag.prompts.database_manager import (
    get_sql_generation_prompt,
    get_error_recovery_prompt
)

from common.logger import get_logger

logger = get_logger(__name__)


class SQLRetrieval:
    """
    SQL Retrieval system using Qdrant for schema search.
    """

    def __init__(
        self,
        db_provider: DatabaseProvider,
        qdrant_provider: QdrantProvider,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        collection_name: str = "sql_schema",
        schema_path: str = "retrieval/context_provider/test_data_schema/output_schema.json",
        vega_schema_path: str = "retrieval/context_provider/chart_schema/vega-lite-schema-v5.json",
        chart_output_dir: str = "output/charts",
        recreate_collection: bool = False,
        filter_importance_label: bool = True
    ):
        """
        Initialize SQL Retrieval system.

        Args:
            db_provider: Database provider
            qdrant_provider: Qdrant vector store provider
            embedding_provider: Embedding provider
            llm_provider: LLM provider
            collection_name: Qdrant collection name
            schema_path: Path to schema JSON file
            vega_schema_path: Path to Vega-Lite schema
            chart_output_dir: Directory for chart outputs
            recreate_collection: Whether to recreate collection
            filter_importance_label: Whether to filter importance labels
        """
        self.db_provider = db_provider
        self.qdrant_provider = qdrant_provider
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.collection_name = collection_name
        self.schema_path = Path(schema_path)
        self.vega_schema_path = Path(vega_schema_path)
        self.chart_output_dir = Path(chart_output_dir)
        self.filter_importance_label = filter_importance_label

        # Create output directory
        self.chart_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize collection
        self._initialize_collection(recreate_collection)

        # Schema data (to be loaded)
        self.schema_data = None

        logger.info(f"SQL Retrieval initialized with collection: {collection_name}")

    def _initialize_collection(self, recreate: bool = False):
        """Initialize Qdrant collection for schema vectors."""
        try:
            embedding_dimensions = self.embedding_provider.dimensions

            self.qdrant_provider.create_collection(
                collection_name=self.collection_name,
                vector_size=embedding_dimensions,
                distance_metric="cosine",
                recreate_if_exists=recreate
            )

            logger.info(f"Collection '{self.collection_name}' ready")

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    def _filter_schema(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove importance_label from schema if configured.

        Args:
            schema_data: Original schema data

        Returns:
            Filtered schema data
        """
        if not self.filter_importance_label:
            return schema_data

        # Deep copy to avoid modifying original
        filtered_schema = copy.deepcopy(schema_data)

        # Remove importance_label from all columns
        if "tables" in filtered_schema:
            for table in filtered_schema["tables"]:
                if "columns" in table:
                    for column in table["columns"]:
                        column.pop("importance_label", None)

        logger.debug("Filtered importance_label from schema")
        return filtered_schema

    async def build_index(self):
        """Build vector index from schema."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        # Load schema
        with open(self.schema_path, 'r') as f:
            raw_schema = json.load(f)

        # Filter schema
        self.schema_data = self._filter_schema(raw_schema)

        # Create chunks
        chunks = self._create_schema_chunks()

        if not chunks:
            logger.warning("No chunks created from schema")
            return

        # Generate embeddings
        texts = [chunk["content"] for chunk in chunks]
        embeddings = await self.embedding_provider.aembed_texts(texts)

        # Prepare documents for Qdrant
        documents = []
        for chunk in chunks:
            documents.append({
                "content": chunk["content"],
                "doc_type": chunk.get("doc_type", ""),
                "table_name": chunk.get("table_name", ""),
                "column_name": chunk.get("column_name", ""),
                "field_type": chunk.get("field_type", ""),
            })

        # Upsert to Qdrant
        self.qdrant_provider.upsert_documents(
            collection_name=self.collection_name,
            documents=documents,
            embeddings=embeddings,
            batch_size=100
        )

        logger.info(f"Indexed {len(chunks)} schema chunks")

    def _create_schema_chunks(self) -> List[Dict[str, Any]]:
        """Create searchable chunks from schema."""
        chunks = []

        if "tables" not in self.schema_data:
            return chunks

        for table in self.schema_data["tables"]:
            # Table-level chunk
            table_content = self._format_table_chunk(table)
            chunks.append({
                "content": table_content,
                "doc_type": "table",
                "table_name": table["table_name"],
                "field_type": "table"
            })

            # Column-level chunks
            for column in table.get("columns", []):
                column_content = self._format_column_chunk(table["table_name"], column)
                chunks.append({
                    "content": column_content,
                    "doc_type": "column",
                    "table_name": table["table_name"],
                    "column_name": column["column_name"],
                    "field_type": "column"
                })

            # Foreign key chunks
            for fk in table.get("foreign_keys", []):
                fk_content = f"Foreign Key: {table['table_name']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
                chunks.append({
                    "content": fk_content,
                    "doc_type": "foreign_key",
                    "table_name": table["table_name"],
                    "field_type": "relationship"
                })

        return chunks

    def _format_table_chunk(self, table: Dict[str, Any]) -> str:
        """Format table information for embedding."""
        content_parts = [
            f"Table: {table['table_name']}"
        ]

        if "table_name_comment" in table:
            content_parts.append(f"Description: {table['table_name_comment']}")

        if "table_name_business_interpretations" in table:
            content_parts.append(f"Business: {table['table_name_business_interpretations']}")

        # Add column names
        columns = [col["column_name"] for col in table.get("columns", [])]
        if columns:
            content_parts.append(f"Columns: {', '.join(columns)}")

        # Add primary key
        if "primary_key" in table:
            content_parts.append(f"Primary Key: {table['primary_key']}")

        return "\n".join(content_parts)

    def _format_column_chunk(self, table_name: str, column: Dict[str, Any]) -> str:
        """Format column information for embedding."""
        content_parts = [
            f"Table: {table_name}",
            f"Column: {column['column_name']}",
            f"Type: {column['column_dtype']}"
        ]

        if "column_comment" in column:
            content_parts.append(f"Technical: {column['column_comment']}")

        if "column_business_interpretation" in column:
            content_parts.append(f"Business: {column['column_business_interpretation']}")

        return "\n".join(content_parts)

    async def search_schema(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant schema elements.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Relevant schema chunks
        """
        # Generate query embedding
        query_embedding = await self.embedding_provider.aembed_text(query)

        # Search in Qdrant
        results = self.qdrant_provider.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        return results

    async def generate_sql(
        self,
        query: str,
        schema_context: str
    ) -> Dict[str, Any]:
        """
        Generate SQL from natural language query.

        Args:
            query: Natural language query
            schema_context: Relevant schema information

        Returns:
            Generated SQL and metadata
        """
        # Build prompt
        system_prompt = get_sql_generation_prompt(include_examples=False)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Schema:\n{schema_context}\n\nQuery: {query}"}
        ]

        # Generate SQL using LLM
        response = await self.llm_provider.chat(messages)

        # Parse response
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'choices'):
            response_text = response.choices[0].message.content
        else:
            response_text = str(response)

        # Try to parse as JSON
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text

            result = json.loads(json_str)
            return result

        except json.JSONDecodeError:
            # Fallback: extract SQL from response
            sql = self._extract_sql_from_text(response_text)
            return {
                "sql": sql,
                "reasoning": "Extracted from response",
                "explanation": response_text[:200]
            }

    def _extract_sql_from_text(self, text: str) -> str:
        """Extract SQL from text response."""
        # Look for SQL in code blocks
        if "```sql" in text:
            return text.split("```sql")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()

        # Look for SELECT statement
        lines = text.split('\n')
        sql_lines = []
        in_sql = False

        for line in lines:
            if 'SELECT' in line.upper():
                in_sql = True
            if in_sql:
                sql_lines.append(line)
                if ';' in line:
                    break

        return '\n'.join(sql_lines).strip()

    async def execute_sql(
        self,
        sql: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Execute SQL with error recovery.

        Args:
            sql: SQL query
            max_retries: Maximum retry attempts

        Returns:
            Execution results
        """
        for attempt in range(max_retries):
            try:
                # Execute SQL
                df_result = self.db_provider.execute_query(sql)

                logger.info(f"SQL executed successfully", extra={"props": {
                    "rows": len(df_result),
                    "columns": len(df_result.columns),
                    "attempt": attempt + 1
                }})

                return {
                    "success": True,
                    "data": df_result,
                    "row_count": len(df_result),
                    "column_count": len(df_result.columns),
                    "sql": sql,
                    "attempts": attempt + 1
                }

            except Exception as e:
                logger.warning(f"SQL execution failed on attempt {attempt + 1}: {e}")

                if attempt < max_retries - 1:
                    # Try to recover
                    recovery_prompt = get_error_recovery_prompt(
                        error_message=str(e),
                        previous_sql=sql
                    )

                    messages = [
                        {"role": "system", "content": "You are a SQL expert. Fix the SQL query."},
                        {"role": "user", "content": recovery_prompt}
                    ]

                    try:
                        response = await self.llm_provider.chat(messages)

                        if hasattr(response, 'text'):
                            response_text = response.text
                        elif hasattr(response, 'choices'):
                            response_text = response.choices[0].message.content
                        else:
                            response_text = str(response)

                        sql = self._extract_sql_from_text(response_text)
                        logger.info(f"SQL corrected for retry {attempt + 2}")

                    except Exception as llm_error:
                        logger.error(f"LLM error recovery failed: {llm_error}")

        return {
            "success": False,
            "error": f"Failed after {max_retries} attempts",
            "sql": sql,
            "attempts": max_retries
        }

    async def query(self, query: str) -> Dict[str, Any]:
        """
        Execute complete SQL RAG pipeline.

        Args:
            query: Natural language query

        Returns:
            Query results
        """
        logger.info(f"Processing SQL query: {query}")

        # Step 1: Search for relevant schema
        schema_results = await self.search_schema(query, top_k=15)

        if not schema_results:
            return {
                "success": False,
                "error": "No relevant schema found",
                "query": query
            }

        # Step 2: Format schema context
        schema_context = self._format_schema_context(schema_results)

        # Step 3: Generate SQL
        sql_result = await self.generate_sql(query, schema_context)

        if not sql_result.get("sql"):
            return {
                "success": False,
                "error": "Failed to generate SQL",
                "query": query,
                "reasoning": sql_result.get("reasoning")
            }

        # Step 4: Execute SQL
        execution_result = await self.execute_sql(sql_result["sql"])

        # Step 5: Format response
        result = {
            "query": query,
            "sql": sql_result["sql"],
            "reasoning": sql_result.get("reasoning", ""),
            "explanation": sql_result.get("explanation", ""),
            **execution_result
        }

        # Step 6: Generate chart if data available and successful
        if execution_result.get("success") and execution_result.get("data") is not None:
            chart_result = await self._generate_chart(query, execution_result["data"])
            if chart_result.get("success"):
                result["chart"] = chart_result

        return result

    def _format_schema_context(self, schema_results: List[Dict[str, Any]]) -> str:
        """Format schema search results into context."""
        # Group by table
        tables = {}
        for result in schema_results:
            metadata = result.get("metadata", {})
            table_name = metadata.get("table_name")

            if table_name:
                if table_name not in tables:
                    tables[table_name] = {
                        "description": "",
                        "columns": [],
                        "relationships": []
                    }

                if metadata.get("doc_type") == "table":
                    tables[table_name]["description"] = metadata.get("content", "")
                elif metadata.get("doc_type") == "column":
                    tables[table_name]["columns"].append(metadata.get("content", ""))
                elif metadata.get("doc_type") == "foreign_key":
                    tables[table_name]["relationships"].append(metadata.get("content", ""))

        # Format as text
        context_parts = []
        for table_name, info in tables.items():
            context_parts.append(f"Table: {table_name}")
            if info["description"]:
                context_parts.append(info["description"])
            if info["columns"]:
                context_parts.append("Columns:")
                context_parts.extend(info["columns"])
            if info["relationships"]:
                context_parts.append("Relationships:")
                context_parts.extend(info["relationships"])
            context_parts.append("")

        return "\n".join(context_parts)

    async def _generate_chart(
        self,
        query: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate chart visualization for query results.

        Args:
            query: Original query
            df: Result dataframe

        Returns:
            Chart generation results
        """
        try:
            # Simple chart generation logic
            # This could be enhanced with more sophisticated chart selection

            # Convert dataframe to dict for JSON serialization
            data_dict = df.to_dict(orient='records')

            # Determine chart type based on data
            chart_type = self._determine_chart_type(df)

            # Generate simple Vega-Lite spec
            vega_spec = self._create_vega_spec(data_dict, chart_type, df.columns.tolist())

            # Save chart
            import json
            chart_file = self.chart_output_dir / f"chart_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(chart_file, 'w') as f:
                json.dump(vega_spec, f, indent=2)

            logger.info(f"Chart saved to {chart_file}")

            return {
                "success": True,
                "chart_type": chart_type,
                "chart_file": str(chart_file),
                "spec": vega_spec
            }

        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _determine_chart_type(self, df: pd.DataFrame) -> str:
        """Determine appropriate chart type based on data."""
        if len(df) <= 10 and len(df.columns) == 2:
            # Check if one column is numeric
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) == 1:
                return "bar"

        if len(df.columns) >= 2:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) >= 2:
                return "scatter"

        return "table"

    def _create_vega_spec(
        self,
        data: List[Dict],
        chart_type: str,
        columns: List[str]
    ) -> Dict[str, Any]:
        """Create basic Vega-Lite specification."""
        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": data},
            "mark": chart_type if chart_type != "table" else "text"
        }

        if chart_type == "bar" and len(columns) >= 2:
            spec["encoding"] = {
                "x": {"field": columns[0], "type": "nominal"},
                "y": {"field": columns[1], "type": "quantitative"}
            }
        elif chart_type == "scatter" and len(columns) >= 2:
            numeric_cols = [col for col in columns if any(isinstance(row.get(col), (int, float)) for row in data)]
            if len(numeric_cols) >= 2:
                spec["encoding"] = {
                    "x": {"field": numeric_cols[0], "type": "quantitative"},
                    "y": {"field": numeric_cols[1], "type": "quantitative"}
                }

        spec["width"] = 600
        spec["height"] = 400

        return spec