"""Reranking utilities for document retrieval.

Provides RRF fusion and optional LLM-based reranking.
"""

from typing import List, Optional, Callable
from dbgpt.core import Chunk

from . import logger


# Type alias for rank functions
RANK_FUNC = Callable[[List[Chunk]], List[Chunk]]


class Reranker:
    """Base reranker class."""
    
    def __init__(self, top_k: int = 5, rank_fn: Optional[RANK_FUNC] = None):
        """Initialize reranker.
        
        Args:
            top_k: Number of results to return
            rank_fn: Optional custom ranking function
        """
        self.top_k = top_k
        self.rank_fn = rank_fn
    
    def rank(self, candidates: List[Chunk], query: Optional[str] = None) -> List[Chunk]:
        """Rank candidates and return top_k.
        
        Args:
            candidates: List of candidate chunks
            query: Optional query for context-aware ranking
            
        Returns:
            Ranked list of chunks
        """
        if self.rank_fn:
            return self.rank_fn(candidates)[:self.top_k]
        
        # Default: sort by score
        return sorted(candidates, key=lambda x: getattr(x, 'score', 0), reverse=True)[:self.top_k]


class DefaultReranker(Reranker):
    """Default reranker using score-based sorting."""
    
    def rank(self, candidates: List[Chunk], query: Optional[str] = None) -> List[Chunk]:
        """Rank by existing scores.
        
        Args:
            candidates: List of candidate chunks
            query: Optional query (not used in default ranking)
            
        Returns:
            Ranked list of chunks
        """
        ranked = sorted(candidates, key=lambda x: getattr(x, 'score', 0), reverse=True)
        return ranked[:self.top_k]


class RRFReranker(Reranker):
    """Reranker using Reciprocal Rank Fusion.
    
    Used when combining results from multiple retrieval methods.
    """
    
    def __init__(self, top_k: int = 5, rrf_k: int = 60):
        """Initialize RRF reranker.
        
        Args:
            top_k: Number of results to return
            rrf_k: RRF constant (typically 60)
        """
        super().__init__(top_k)
        self.rrf_k = rrf_k
    
    def rank_multiple(
        self, 
        result_sets: List[List[Chunk]], 
        query: Optional[str] = None
    ) -> List[Chunk]:
        """Rank using RRF across multiple result sets.
        
        Args:
            result_sets: List of result lists from different retrievers
            query: Optional query (not used in RRF)
            
        Returns:
            Fused and ranked list of chunks
        """
        from .retriever import rrf_fusion
        
        fused = rrf_fusion(result_sets, k=self.rrf_k)
        return fused[:self.top_k]
    
    def rank(self, candidates: List[Chunk], query: Optional[str] = None) -> List[Chunk]:
        """Rank single result set (falls back to score sorting).
        
        Args:
            candidates: List of candidate chunks
            query: Optional query
            
        Returns:
            Ranked list of chunks
        """
        return sorted(candidates, key=lambda x: getattr(x, 'score', 0), reverse=True)[:self.top_k]


class LLMReranker(Reranker):
    """LLM-based reranker for high-quality relevance scoring.
    
    Uses an LLM to score each chunk's relevance to the query.
    """
    
    def __init__(
        self, 
        llm_client, 
        model_name: str,
        top_k: int = 5,
        batch_size: int = 5
    ):
        """Initialize LLM reranker.
        
        Args:
            llm_client: LLM client for scoring
            model_name: Model to use for scoring
            top_k: Number of results to return
            batch_size: Number of chunks to score in parallel
        """
        super().__init__(top_k)
        self.llm_client = llm_client
        self.model_name = model_name
        self.batch_size = batch_size
    
    async def arank(
        self, 
        candidates: List[Chunk], 
        query: str
    ) -> List[Chunk]:
        """Async rank using LLM scoring.
        
        Args:
            candidates: List of candidate chunks
            query: Query for relevance scoring
            
        Returns:
            Reranked list of chunks
        """
        if not candidates:
            return []
        
        logger.info(f"LLM reranking", extra={"props": {
            "candidates": len(candidates),
            "query": query[:50]
        }})
        
        # Score each chunk
        scored_chunks = []
        for chunk in candidates:
            score = await self._score_chunk(chunk, query)
            chunk.score = score
            scored_chunks.append(chunk)
        
        # Sort by score
        ranked = sorted(scored_chunks, key=lambda x: x.score, reverse=True)
        
        logger.info(f"LLM reranking complete", extra={"props": {"returned": len(ranked[:self.top_k])}})
        
        return ranked[:self.top_k]
    
    async def _score_chunk(self, chunk: Chunk, query: str) -> float:
        """Score a single chunk's relevance to query.
        
        Args:
            chunk: Chunk to score
            query: Query for relevance
            
        Returns:
            Relevance score (0-1)
        """
        from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType
        from dbgpt.core.interface.llm import ModelRequest
        
        prompt = f"""Rate the relevance of the following passage to the query on a scale of 0 to 10.
Only respond with a number.

Query: {query}

Passage: {chunk.content[:500]}

Relevance score (0-10):"""
        
        try:
            messages = [ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)]
            request = ModelRequest(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_new_tokens=10
            )
            
            response = await self.llm_client.generate(request)
            
            if response.success:
                # Parse score from response
                score_text = response.text.strip()
                score = float(score_text.split()[0]) / 10.0  # Normalize to 0-1
                return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"LLM scoring failed", extra={"props": {"error": str(e)}})
        
        return 0.5  # Default score on failure
    
    def rank(self, candidates: List[Chunk], query: Optional[str] = None) -> List[Chunk]:
        """Sync rank (uses async internally via asyncio.run).
        
        Args:
            candidates: List of candidate chunks
            query: Query for relevance scoring
            
        Returns:
            Reranked list of chunks
        """
        if not query:
            return sorted(candidates, key=lambda x: getattr(x, 'score', 0), reverse=True)[:self.top_k]
        
        import asyncio
        return asyncio.run(self.arank(candidates, query))
