"""
Document Retrieval module using Qdrant for vector storage.

This module provides document indexing and retrieval capabilities
using Qdrant vector store (no ChromaDB dependencies).
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

from providers.qdrant import QdrantProvider
from providers.embeddings import EmbeddingProvider
from providers.llm import LLMProvider

from common.logger import get_logger

logger = get_logger(__name__)


class DocumentRetrieval:
    """
    Document Retrieval system using Qdrant for vector storage.
    """

    def __init__(
        self,
        qdrant_provider: QdrantProvider,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        collection_name: str = "documents",
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        use_hybrid_search: bool = True,
        recreate_collection: bool = False
    ):
        """
        Initialize Document Retrieval system.

        Args:
            qdrant_provider: Qdrant vector store provider
            embedding_provider: Embedding provider
            llm_provider: LLM provider
            collection_name: Qdrant collection name
            chunk_size: Size of document chunks in characters
            chunk_overlap: Overlap between chunks
            use_hybrid_search: Whether to use hybrid search
            recreate_collection: Whether to recreate collection
        """
        self.qdrant_provider = qdrant_provider
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_hybrid_search = use_hybrid_search

        # Initialize collection
        self._initialize_collection(recreate_collection)

        # BM25 components for hybrid search
        self.bm25_index = None
        self.document_cache = []

        logger.info(f"Document Retrieval initialized with collection: {collection_name}")

    def _initialize_collection(self, recreate: bool = False):
        """Initialize Qdrant collection for document vectors."""
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

    async def index_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Index a document into Qdrant.

        Args:
            file_path: Path to the document
            metadata: Optional metadata

        Returns:
            Indexing results
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Document not found: {file_path}")

            # Load document based on type
            if file_path.suffix.lower() == '.pdf':
                chunks = await self._process_pdf(file_path, metadata)
            elif file_path.suffix.lower() in ['.txt', '.md']:
                chunks = await self._process_text(file_path, metadata)
            elif file_path.suffix.lower() == '.json':
                chunks = await self._process_json(file_path, metadata)
            else:
                # Default text processing
                chunks = await self._process_text(file_path, metadata)

            if not chunks:
                return {"success": False, "error": "No chunks created from document"}

            # Generate embeddings
            texts = [chunk["content"] for chunk in chunks]
            embeddings = await self.embedding_provider.aembed_texts(texts)

            # Prepare documents for Qdrant
            documents = []
            ids = []

            for i, chunk in enumerate(chunks):
                chunk_id = self._generate_chunk_id(file_path, i)
                ids.append(chunk_id)

                doc_metadata = {
                    "content": chunk["content"],
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk["content"]),
                }

                if metadata:
                    doc_metadata.update(metadata)

                if "metadata" in chunk:
                    doc_metadata.update(chunk["metadata"])

                documents.append(doc_metadata)

            # Upsert to Qdrant
            self.qdrant_provider.upsert_documents(
                collection_name=self.collection_name,
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                batch_size=100
            )

            # Invalidate BM25 cache
            self.bm25_index = None

            logger.info(f"Indexed {len(chunks)} chunks from {file_path}")

            return {
                "success": True,
                "file": str(file_path),
                "chunks": len(chunks),
                "collection": self.collection_name
            }

        except Exception as e:
            logger.error(f"Failed to index document {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file": str(file_path)
            }

    async def index_documents(
        self,
        file_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Index multiple documents.

        Args:
            file_paths: List of document paths
            metadata: Optional metadata for all documents

        Returns:
            List of indexing results
        """
        results = []
        for file_path in file_paths:
            result = await self.index_document(file_path, metadata)
            results.append(result)
        return results

    async def _process_pdf(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process PDF document."""
        try:
            # Try using Azure Document Intelligence if available
            from retrieval.doc_rag.pdf_loader import MetadataAwarePDFLoader

            loader = MetadataAwarePDFLoader(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            chunks = await loader.load_and_chunk(str(file_path))
            return chunks

        except ImportError:
            # Fallback to basic PDF processing
            try:
                import PyPDF2

                chunks = []
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)

                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()

                        # Create chunks from page text
                        page_chunks = self._create_text_chunks(
                            text,
                            {"page": page_num + 1, "source": str(file_path)}
                        )
                        chunks.extend(page_chunks)

                return chunks

            except ImportError:
                logger.warning("PyPDF2 not installed, treating PDF as text")
                return await self._process_text(file_path, metadata)

    async def _process_text(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process text document."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks = self._create_text_chunks(
            content,
            {"source": str(file_path)}
        )

        return chunks

    async def _process_json(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process JSON document."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert JSON to readable text
        content = json.dumps(data, indent=2)

        chunks = self._create_text_chunks(
            content,
            {"source": str(file_path), "type": "json"}
        )

        return chunks

    def _create_text_chunks(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from text with overlap.

        Args:
            text: Text to chunk
            metadata: Optional metadata

        Returns:
            List of chunks
        """
        chunks = []
        text = text.strip()

        if not text:
            return chunks

        # Simple character-based chunking
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]

            chunk = {
                "content": chunk_text,
                "start_index": i,
                "end_index": min(i + self.chunk_size, len(text))
            }

            if metadata:
                chunk["metadata"] = metadata

            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, file_path: Path, chunk_index: int) -> str:
        """Generate unique ID for a chunk."""
        id_string = f"{file_path.absolute()}_{chunk_index}"
        return hashlib.md5(id_string.encode()).hexdigest()

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            top_k: Number of results
            filters: Optional metadata filters

        Returns:
            Search results
        """
        # Generate query embedding
        query_embedding = await self.embedding_provider.aembed_text(query)

        # Vector search
        vector_results = self.qdrant_provider.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k * 2 if self.use_hybrid_search else top_k,
            filter_dict=filters
        )

        if not self.use_hybrid_search:
            return vector_results[:top_k]

        # BM25 search for hybrid
        bm25_results = await self._bm25_search(query, top_k * 2)

        # Combine using RRF
        combined = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            top_k
        )

        return combined

    async def _bm25_search(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        # Initialize BM25 if needed
        if self.bm25_index is None:
            await self._build_bm25_index()

        if not self.bm25_index:
            return []

        try:
            from rank_bm25 import BM25Okapi

            # Tokenize query
            query_tokens = query.lower().split()

            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)

            # Get top-k indices
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            # Return documents with scores
            results = []
            for idx in top_indices:
                if idx < len(self.document_cache):
                    doc = self.document_cache[idx].copy()
                    doc["score"] = float(scores[idx])
                    results.append(doc)

            return results

        except ImportError:
            logger.warning("rank_bm25 not installed, skipping BM25 search")
            return []

    async def _build_bm25_index(self):
        """Build BM25 index from all documents."""
        try:
            from rank_bm25 import BM25Okapi

            # Fetch all documents
            all_docs = []
            offset = None

            while True:
                docs, offset = self.qdrant_provider.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset
                )

                all_docs.extend(docs)

                if offset is None:
                    break

            if not all_docs:
                logger.warning("No documents for BM25 indexing")
                return

            # Build corpus
            corpus = []
            self.document_cache = []

            for doc in all_docs:
                content = doc.get("metadata", {}).get("content", "")
                corpus.append(content.lower().split())
                self.document_cache.append(doc)

            # Create BM25 index
            self.bm25_index = BM25Okapi(corpus)

            logger.info(f"Built BM25 index with {len(corpus)} documents")

        except ImportError:
            logger.warning("rank_bm25 not installed")
            self.use_hybrid_search = False

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.use_hybrid_search = False

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Combine results using RRF."""
        rrf_scores = {}

        # Add vector search scores
        for rank, result in enumerate(vector_results):
            doc_id = result.get("id", str(result))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        # Add BM25 scores
        for rank, result in enumerate(bm25_results):
            doc_id = result.get("id", str(result))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        # Create result map
        result_map = {}
        for result in vector_results + bm25_results:
            doc_id = result.get("id", str(result))
            if doc_id not in result_map:
                result_map[doc_id] = result

        # Sort by RRF score
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )

        # Return top-k
        combined_results = []
        for doc_id in sorted_ids[:top_k]:
            if doc_id in result_map:
                result = result_map[doc_id].copy()
                result["rrf_score"] = rrf_scores[doc_id]
                combined_results.append(result)

        return combined_results

    async def query(self, query: str) -> Dict[str, Any]:
        """
        Execute document retrieval and answer generation.

        Args:
            query: Natural language query

        Returns:
            Query results with answer
        """
        logger.info(f"Processing document query: {query}")

        # Step 1: Search for relevant documents
        search_results = await self.search(query, top_k=10)

        if not search_results:
            return {
                "success": False,
                "answer": "No relevant documents found.",
                "query": query,
                "sources": []
            }

        # Step 2: Format context
        context = self._format_context(search_results)

        # Step 3: Generate answer
        system_prompt = """You are a helpful assistant that answers questions based on provided documents.
        Use the document context to provide accurate and detailed answers.
        If the answer cannot be found in the documents, say so clearly.
        Always cite the source documents when providing information."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the documents."}
        ]

        response = await self.llm_provider.chat(messages)

        # Extract answer
        if hasattr(response, 'text'):
            answer = response.text
        elif hasattr(response, 'choices'):
            answer = response.choices[0].message.content
        else:
            answer = str(response)

        # Step 4: Extract sources
        sources = self._extract_sources(search_results)

        return {
            "success": True,
            "answer": answer,
            "query": query,
            "sources": sources,
            "documents_retrieved": len(search_results)
        }

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as context."""
        context_parts = []

        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            content = metadata.get("content", "")
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "")

            context_parts.append(f"[Document {i}]")
            context_parts.append(f"Source: {source}")
            if page:
                context_parts.append(f"Page: {page}")
            context_parts.append(f"Content: {content}")
            context_parts.append("")

        return "\n".join(context_parts)

    def _extract_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique sources from results."""
        sources = {}

        for result in results:
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown")
            file_name = metadata.get("file_name", Path(source).name)

            if source not in sources:
                sources[source] = {
                    "source": source,
                    "file_name": file_name,
                    "chunks": 0,
                    "pages": set()
                }

            sources[source]["chunks"] += 1

            if "page" in metadata:
                sources[source]["pages"].add(metadata["page"])

        # Convert to list and format pages
        source_list = []
        for source_info in sources.values():
            info = {
                "source": source_info["source"],
                "file_name": source_info["file_name"],
                "chunks": source_info["chunks"]
            }
            if source_info["pages"]:
                info["pages"] = sorted(list(source_info["pages"]))
            source_list.append(info)

        return source_list