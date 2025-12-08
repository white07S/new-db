"""Hybrid retriever combining vector search and BM25 with reranking.

Implements the hybrid search pattern from sql_rag with RRF fusion.
"""

from typing import List, Optional, Dict, Any
from rank_bm25 import BM25Okapi

from dbgpt.core import Chunk
from dbgpt.rag.retriever.base import BaseRetriever
from dbgpt.storage.vector_store.filters import MetadataFilters

from . import logger


class InMemoryBM25Retriever(BaseRetriever):
    """In-memory BM25 retriever for document chunks."""

    def __init__(self, chunks: List[Chunk], top_k: int = 5):
        """Initialize BM25 retriever with chunks.
        
        Args:
            chunks: List of Chunk objects to index
            top_k: Number of results to return
        """
        self._chunks = chunks
        self._top_k = top_k
        self._corpus = [chunk.content for chunk in chunks]
        self._tokenized_corpus = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        
        logger.info(f"Initialized BM25 retriever", extra={"props": {"corpus_size": len(chunks)}})

    async def _aretrieve(
        self, 
        query: str, 
        filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        return self._retrieve(query, filters)

    def _retrieve(
        self, 
        query: str, 
        filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        return self._retrieve_with_score(query, 0.0, filters)

    def _retrieve_with_score(
        self, 
        query: str, 
        score_threshold: float, 
        filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self._top_k]
        
        results = []
        for i in top_n:
            if scores[i] < score_threshold:
                continue
            chunk = self._chunks[i].model_copy()
            chunk.score = float(scores[i])
            results.append(chunk)
        
        return results

    async def _aretrieve_with_score(
        self, 
        query: str, 
        score_threshold: float, 
        filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        return self._retrieve_with_score(query, score_threshold, filters)


def rrf_fusion(results_list: List[List[Chunk]], k: int = 60) -> List[Chunk]:
    """Reciprocal Rank Fusion for combining multiple result sets.
    
    RRF formula: score = sum(1 / (k + rank)) for each result set
    
    Args:
        results_list: List of result lists from different retrievers
        k: Constant for RRF (typically 60)
        
    Returns:
        Fused and reranked list of chunks
    """
    fused_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}
    
    for results in results_list:
        for rank, chunk in enumerate(results):
            # Use content as key (could also use hash for efficiency)
            key = chunk.content[:200]  # Use first 200 chars as key
            
            if key not in fused_scores:
                fused_scores[key] = 0.0
                chunk_map[key] = chunk
            
            fused_scores[key] += 1.0 / (rank + k)
    
    # Sort by fused score and return
    reranked = []
    for key in sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True):
        chunk = chunk_map[key]
        chunk.score = fused_scores[key]
        reranked.append(chunk)
    
    return reranked


class HybridDocRetriever:
    """Hybrid retriever combining vector search and BM25.
    
    Features:
    - Vector search for semantic similarity
    - BM25 for keyword matching
    - RRF fusion for combining results
    - Preserves metadata through retrieval
    """
    
    def __init__(
        self,
        vector_store,
        chunks: List[Chunk],
        bm25_top_k: int = 10,
        vector_top_k: int = 10,
        final_top_k: int = 5
    ):
        """Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store for semantic search (ChromaStore)
            chunks: List of chunks for BM25 indexing
            bm25_top_k: Number of BM25 results
            vector_top_k: Number of vector search results
            final_top_k: Final number of results after fusion
        """
        self.vector_store = vector_store
        self.bm25_retriever = InMemoryBM25Retriever(chunks, top_k=bm25_top_k)
        self.vector_top_k = vector_top_k
        self.final_top_k = final_top_k
        
        logger.info(f"Initialized HybridDocRetriever", extra={"props": {
            "bm25_top_k": bm25_top_k,
            "vector_top_k": vector_top_k,
            "final_top_k": final_top_k
        }})
    
    async def retrieve(self, query: str) -> List[Chunk]:
        """Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            
        Returns:
            List of chunks sorted by relevance with metadata preserved
        """
        logger.info(f"Hybrid retrieval", extra={"props": {"query": query[:100]}})
        
        # Vector search
        try:
            vector_results = await self._vector_search(query)
            logger.info(f"Vector search results", extra={"props": {"count": len(vector_results)}})
        except Exception as e:
            logger.warning(f"Vector search failed", extra={"props": {"error": str(e)}})
            vector_results = []
        
        # BM25 search
        try:
            bm25_results = await self.bm25_retriever._aretrieve(query)
            logger.info(f"BM25 search results", extra={"props": {"count": len(bm25_results)}})
        except Exception as e:
            logger.warning(f"BM25 search failed", extra={"props": {"error": str(e)}})
            bm25_results = []
        
        # RRF Fusion
        if vector_results and bm25_results:
            fused_results = rrf_fusion([vector_results, bm25_results])
        elif vector_results:
            fused_results = vector_results
        else:
            fused_results = bm25_results
        
        final_results = fused_results[:self.final_top_k]
        
        logger.info(f"Hybrid retrieval complete", extra={"props": {
            "total_fused": len(fused_results),
            "returned": len(final_results)
        }})
        
        return final_results
    
    async def _vector_search(self, query: str) -> List[Chunk]:
        """Perform vector similarity search.
        
        Args:
            query: Search query
            
        Returns:
            List of chunks from vector search
        """
        # ChromaStore uses similar_search or query method
        if hasattr(self.vector_store, 'similar_search'):
            results = await self.vector_store.similar_search(query, topk=self.vector_top_k)
        elif hasattr(self.vector_store, 'query'):
            results = self.vector_store.query(query, top_k=self.vector_top_k)
        else:
            # Fallback: try aquery
            results = await self.vector_store.aquery(query, top_k=self.vector_top_k)
        
        return results
    
    def get_chunks_with_metadata(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Convert chunks to dictionaries with full metadata.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of dictionaries with content and metadata
        """
        return [
            {
                "content": chunk.content,
                "score": chunk.score if hasattr(chunk, 'score') else None,
                "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
            }
            for chunk in chunks
        ]
