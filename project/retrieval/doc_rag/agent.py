"""Document retrieval agent for iterative retrieval with sufficiency checking.

Similar to ScratchpadSchemaAgent but for document retrieval.
"""

import json
from typing import List, Dict, Any, Optional

from dbgpt.core import Chunk
from dbgpt.core.interface.llm import LLMClient, ModelRequest
from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType
from dbgpt.core import ModelOutput

from .retriever import HybridDocRetriever
from . import logger


class DocumentRetrievalAgent:
    """Agent for iterative document retrieval with sufficiency checking.
    
    Similar to ScratchpadSchemaAgent, this agent:
    1. Retrieves initial chunks
    2. Checks if they're sufficient to answer the query
    3. Iteratively retrieves more if needed
    4. Preserves metadata throughout the process
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str,
        retriever: HybridDocRetriever,
        max_iterations: int = 3,
        min_chunks: int = 3,
        max_chunks: int = 10
    ):
        """Initialize the document retrieval agent.
        
        Args:
            llm_client: LLM client for sufficiency checking
            model_name: Model name for LLM calls
            retriever: Hybrid retriever for document search
            max_iterations: Maximum retrieval iterations
            min_chunks: Minimum chunks to retrieve
            max_chunks: Maximum chunks to collect
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks
    
    async def _call_llm(self, prompt: str) -> str:
        """Helper to call LLM.
        
        Args:
            prompt: Prompt to send to LLM
            
        Returns:
            LLM response text
        """
        messages = [ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)]
        request = ModelRequest(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_new_tokens=512
        )
        response: ModelOutput = await self.llm_client.generate(request)
        if not response.success:
            raise Exception(f"LLM call failed: {response.text}")
        return response.text
    
    def _format_chunks_context(self, chunks: List[Chunk]) -> str:
        """Format chunks into a summary for LLM.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Formatted context string
        """
        summary = []
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata if hasattr(chunk, 'metadata') else {}
            page = meta.get('page', 'N/A')
            section = meta.get('section', 'N/A')
            
            # Truncate content for summary
            content_preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            summary.append(f"[{i+1}] Page {page}, {section}: {content_preview}")
        
        return "\n\n".join(summary)
    
    async def check_sufficiency(self, query: str, chunks: List[Chunk]) -> bool:
        """Check if current chunks are sufficient to answer the query.
        
        Args:
            query: User query
            chunks: Current retrieved chunks
            
        Returns:
            True if sufficient, False otherwise
        """
        context = self._format_chunks_context(chunks)
        
        prompt = f"""User Query: "{query}"

Retrieved Context:
{context}

Task: Determine if the retrieved context contains enough information to fully answer the user's query.

Reasoning Steps:
1. Identify what information is needed to answer the query
2. Check if all required information is present in the context
3. If any critical information is missing, answer NO

Response Format (JSON):
{{
    "reasoning": "Brief explanation of what's present/missing...",
    "sufficient": "YES" or "NO"
}}
"""
        
        try:
            response_text = await self._call_llm(prompt)
            
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(response_text[start:end])
                
                logger.info(f"Sufficiency check", extra={"props": {
                    "reasoning": result.get("reasoning", "")[:100],
                    "sufficient": result.get("sufficient", "")
                }})
                
                return result.get("sufficient", "").upper() == "YES"
        
        except Exception as e:
            logger.error(f"Sufficiency check error: {e}")
        
        # Default to sufficient if we have enough chunks
        return len(chunks) >= self.min_chunks
    
    async def generate_followup_query(self, original_query: str, chunks: List[Chunk]) -> Optional[str]:
        """Generate a followup query to find missing information.
        
        Args:
            original_query: Original user query
            chunks: Current retrieved chunks
            
        Returns:
            Followup query or None if no more needed
        """
        context = self._format_chunks_context(chunks)
        
        prompt = f"""Original Query: "{original_query}"

Already Retrieved:
{context}

Task: If the retrieved content is insufficient, generate a followup search query to find the missing information.

Response Format (JSON):
{{
    "reasoning": "What's missing and why...",
    "followup_query": "search query to find missing info" or null
}}
"""
        
        try:
            response_text = await self._call_llm(prompt)
            
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(response_text[start:end])
                
                followup = result.get("followup_query")
                if followup and followup.lower() != "null":
                    logger.info(f"Generated followup query", extra={"props": {
                        "followup": followup
                    }})
                    return followup
        
        except Exception as e:
            logger.error(f"Followup generation error: {e}")
        
        return None
    
    async def run(self, query: str) -> List[Chunk]:
        """Run the iterative retrieval process.
        
        Args:
            query: User query
            
        Returns:
            List of retrieved chunks with metadata
        """
        logger.info(f"Starting document retrieval", extra={"props": {"query": query[:100]}})
        
        # Collect unique chunks (by content hash)
        collected: Dict[str, Chunk] = {}
        
        # Initial retrieval
        initial_chunks = await self.retriever.retrieve(query)
        for chunk in initial_chunks:
            key = chunk.content[:100]  # Use content prefix as key
            if key not in collected:
                collected[key] = chunk
        
        logger.info(f"Initial retrieval", extra={"props": {"count": len(collected)}})
        
        # Check sufficiency
        current_chunks = list(collected.values())
        if await self.check_sufficiency(query, current_chunks):
            logger.info("Sufficient after initial retrieval")
            return current_chunks
        
        # Iterative retrieval
        for iteration in range(self.max_iterations):
            if len(collected) >= self.max_chunks:
                logger.info(f"Max chunks reached: {len(collected)}")
                break
            
            # Generate followup query
            followup = await self.generate_followup_query(query, list(collected.values()))
            if not followup:
                logger.info("No followup query needed")
                break
            
            # Retrieve more
            new_chunks = await self.retriever.retrieve(followup)
            added = 0
            for chunk in new_chunks:
                key = chunk.content[:100]
                if key not in collected:
                    collected[key] = chunk
                    added += 1
            
            logger.info(f"Iteration {iteration + 1}", extra={"props": {
                "new_chunks": added,
                "total": len(collected)
            }})
            
            if added == 0:
                logger.info("No new unique chunks found")
                break
            
            # Check sufficiency again
            if await self.check_sufficiency(query, list(collected.values())):
                logger.info(f"Sufficient after iteration {iteration + 1}")
                break
        
        final_chunks = list(collected.values())
        logger.info(f"Retrieval complete", extra={"props": {"total_chunks": len(final_chunks)}})
        
        return final_chunks
