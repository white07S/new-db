from typing import Optional, List
from rank_bm25 import BM25Okapi
from dbgpt.rag.retriever.base import BaseRetriever
from dbgpt.core import Chunk
from dbgpt.storage.vector_store.filters import MetadataFilters

class InMemoryBM25Retriever(BaseRetriever):
    """In-memory BM25 retriever."""

    def __init__(self, chunks: List[Chunk], top_k: int = 4):
        self._chunks = chunks
        self._top_k = top_k
        self._corpus = [chunk.content for chunk in chunks]
        self._tokenized_corpus = [doc.split(" ") for doc in self._corpus]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    async def _aretrieve(self, query: str, filters: Optional[MetadataFilters] = None) -> List[Chunk]:
        return self._retrieve(query, filters)

    def _retrieve(self, query: str, filters: Optional[MetadataFilters] = None) -> List[Chunk]:
        return self._retrieve_with_score(query, 0.0, filters)

    def _retrieve_with_score(self, query: str, score_threshold: float, filters: Optional[MetadataFilters] = None) -> List[Chunk]:
        tokenized_query = query.split(" ")
        # Get top_k scores
        scores = self._bm25.get_scores(tokenized_query)
        # Get indices of top_k scores
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self._top_k]
        
        results = []
        for i in top_n:
            chunk = self._chunks[i]
            if scores[i] < score_threshold:
                continue
            # Copy chunk to avoid modifying original
            new_chunk = chunk.model_copy()
            new_chunk.score = scores[i]
            results.append(new_chunk)
        return results

    async def _aretrieve_with_score(self, query: str, score_threshold: float, filters: Optional[MetadataFilters] = None) -> List[Chunk]:
        return self._retrieve_with_score(query, score_threshold, filters)

def rrf_fusion(results_list: List[List[Chunk]], k: int = 60) -> List[Chunk]:
    """Reciprocal Rank Fusion."""
    fused_scores = {}
    
    for results in results_list:
        for rank, chunk in enumerate(results):
            if chunk.content not in fused_scores:
                fused_scores[chunk.content] = 0
                # store chunk for reconstruction
                fused_scores[chunk.content + "_chunk_obj"] = chunk
            
            fused_scores[chunk.content] += 1 / (rank + k)
    
    # Sort by fused score
    reranked_results = []
    for content, score in fused_scores.items():
        if content.endswith("_chunk_obj"):
            continue
        chunk = fused_scores[content + "_chunk_obj"]
        chunk.score = score
        reranked_results.append(chunk)
        
    reranked_results.sort(key=lambda x: x.score, reverse=True)
    return reranked_results
