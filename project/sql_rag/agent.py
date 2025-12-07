import json
import asyncio
from typing import List, Dict, Any, Optional
from dbgpt.core import Chunk
from dbgpt.model.proxy import OpenAILLMClient
from dbgpt.core.interface.llm import LLMClient, ModelRequest
from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType
from dbgpt.core import ModelOutput
from . import logger

class ScratchpadSchemaAgent:
    def __init__(
        self, 
        llm_client: LLMClient, 
        model_name: str,
        vector_retriever,
        bm25_retriever,
        stop_threshold: int = 3,
        max_iterations: int = 5
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.stop_threshold = stop_threshold
        self.max_iterations = max_iterations

    async def _call_llm(self, prompt: str) -> str:
        """Helper to call LLM."""
        messages = [ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)]
        request = ModelRequest(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_new_tokens=512
        )
        # Assuming async call matches the client interface
        response: ModelOutput = await self.llm_client.generate(request)
        if not response.success:
            raise Exception(f"LLM call failed: {response.error_code} - {response.text}")
        return response.text



    def _format_schema_context(self, chunks: List[Chunk]) -> str:
        """Format chunks into a clean schema summary."""
        summary = []
        for chunk in chunks:
            table_name = self._extract_table_name(chunk.content)
            if table_name:
                # Extract columns if possible (simple heuristic)
                columns = []
                import re
                col_matches = re.finditer(r'["`]?([\w_]+)["`]?\s+[A-Z]+', chunk.content)
                for m in col_matches:
                    columns.append(m.group(1))
                # simplistic column extraction, defaults to content if fails
                if len(columns) > 2:
                    summary.append(f"Table: {table_name}\nColumns: {', '.join(columns[:10])}...")
                else:
                    summary.append(f"Table: {table_name}\nContent Preview: {chunk.content[:150]}...")
        return "\n\n".join(summary)

    async def check_sufficiency(self, query: str, chunks: List[Chunk]) -> bool:
        """Check if current chunks are sufficient to answer the query."""
        schema_summary = self._format_schema_context(chunks)

        prompt = f"""
User Query: "{query}"

Current Retrieved Schema:
{schema_summary}

Task: Determine if the valid SQL query can be constructed with the above schema.

Reasoning Steps:
1. Identify the entities and attributes in the User Query (e.g. "orders", "customer city").
2. Check if the Current Retrieved Schema contains tables/columns for ALL identified attributes.
3. If any table or column is missing, the answer is NO.
4. If all are present, the answer is YES.

Response Format (JSON):
{{
    "reasoning": "I found table X for attribute Y, but missing Z...",
    "sufficient": "YES" or "NO"
}}
"""

        try:
            response_text = await self._call_llm(prompt)

            # Parse JSON response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                result = json.loads(json_str)

                logger.info(f"[Scratchpad] Sufficiency Check", extra={"props": {
                    "reasoning": result.get("reasoning", ""),
                    "sufficient": result.get("sufficient", "")
                }})

                return result.get("sufficient", "").upper() == "YES"
            else:
                # Fallback to text parsing
                return "YES" in response_text.upper() and ('"sufficient": "YES"' in response_text or '"sufficient": "yes"' in response_text)

        except Exception as e:
            logger.error(f"Error parsing sufficiency check: {e}")
            # Fallback to basic text parsing
            return "YES" in response_text.upper() and ('"sufficient": "YES"' in response_text or '"sufficient": "yes"' in response_text)

    async def plan_next_step(self, query: str, existing_tables: List[str]) -> Dict[str, Any]:
        """Plan next retrieval step based on what we already have."""

        prompt = f"""
User Query: "{query}"

Already Found Tables: {existing_tables}

Task: Identify MISSING information.

Reasoning Steps:
1. Compare User Query requirements vs Already Found Tables.
2. If a required entity (e.g. "product category") is missing, generate a keyword for it.
3. Do NOT generate keywords for tables we already have.

Response Format (JSON):
{{
    "reasoning": "Missing product info...",
    "keywords": ["missing_entity_keyword"],
    "sub_questions": ["Where is X stored?"]
}}
"""

        try:
            response_text = await self._call_llm(prompt)

            # Parse JSON response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                result = json.loads(json_str)

                logger.info(f"[Scratchpad] Next Step Plan", extra={"props": {
                    "reasoning": result.get("reasoning", ""),
                    "keywords": result.get("keywords", [])
                }})

                return result
            else:
                return {"keywords": [], "sub_questions": [], "reasoning": ""}

        except Exception as e:
            logger.error(f"Error parsing plan JSON: {e}")
            # Fallback
            return {"keywords": [], "sub_questions": [], "reasoning": ""}

    async def retrieve_table_first(self, keywords: List[str]) -> List[Chunk]:
        """Retrieve by table names using vector search."""
        if not keywords: return []
        search_query = " ".join(keywords)
        try:
            return await self.vector_retriever.aretrieve(search_query)
        except Exception:
            return []

    async def retrieve_column_first(self, keywords: List[str]) -> List[Chunk]:
        """Retrieve by column/content using BM25."""
        if not keywords: return []
        search_query = " ".join(keywords)
        try:
            return await self.bm25_retriever.aretrieve(search_query)
        except Exception:
            return []

    def _extract_table_name(self, content: str) -> Optional[str]:
        import re
        match = re.search(r"table_name:\s*([^\s]+)", content)
        if match: return match.group(1).strip("`\"'")
        match = re.search(r"CREATE TABLE\s+[`\"']?([^\s`\"'(]+)", content, re.IGNORECASE)
        if match: return match.group(1).strip("`\"'")
        return None

    async def run(self, query: str) -> List[Chunk]:
        """Main scratchpad loop."""
        scratchpad: Dict[str, Chunk] = {} # table_name -> Chunk
        
        logger.info(f"Scratchpad Query", extra={"props": {"query": query}})
        
        # --- Phase 1: Fast Path (Initial Scan) ---
        logger.info("Phase 1: Fast Path (Hybrid Retrieval)")
        # Use simple keywords from query
        initial_keywords = query.split() # Simple heuristic for first pass
        
        # Limit initial check to top 3-4 most relevant to avoid noise
        table_chunks = await self.retrieve_table_first(initial_keywords)
        col_chunks = await self.retrieve_column_first(initial_keywords)
        initial_chunks = table_chunks + col_chunks
        
        for chunk in initial_chunks:
            table_name = self._extract_table_name(chunk.content)
            if table_name and table_name not in scratchpad:
                scratchpad[table_name] = chunk
        
        logger.info(f"Phase 1 found tables", extra={"props": {"count": len(scratchpad)}})
        
        current_chunks = list(scratchpad.values())
        if await self.check_sufficiency(query, current_chunks):
             logger.info("Sufficiency Check: YES. Exiting early.")
             return current_chunks
        
        logger.info("Sufficiency Check: NO. Entering Iterative Loop.")
        
        # --- Phase 2: Iterative Loop ---
        iteration = 0
        while iteration < self.max_iterations:
            logger.info(f"Iteration", extra={"props": {"iteration": iteration + 1}})
            
            existing_tables = list(scratchpad.keys())
            plan = await self.plan_next_step(query, existing_tables)
            keywords = plan.get("keywords", [])
            logger.info(f"Reasoning", extra={"props": {"text": plan.get('reasoning', '')}})
            logger.info(f"New Keywords", extra={"props": {"keywords": keywords}})
            
            if not keywords:
                logger.info("No new keywords generated. Stopping.")
                break
            
            # Retrieve
            table_chunks = await self.retrieve_table_first(keywords)
            col_chunks = await self.retrieve_column_first(keywords)
            new_chunks = table_chunks + col_chunks
            
            added_count = 0
            for chunk in new_chunks:
                table_name = self._extract_table_name(chunk.content)
                if table_name and table_name not in scratchpad:
                    scratchpad[table_name] = chunk
                    added_count += 1
            
            logger.info(f"Added new chunks", extra={"props": {"count": added_count}})
            
            if added_count == 0:
                logger.info("No new unique info found. Stopping.")
                break
            
            current_chunks = list(scratchpad.values())
            if await self.check_sufficiency(query, current_chunks):
                 logger.info("Sufficiency Check: YES. Stopping.")
                 break
                
            iteration += 1
            
        logger.info(f"Finished", extra={"props": {"total_chunks": len(scratchpad)}})
        return list(scratchpad.values())
