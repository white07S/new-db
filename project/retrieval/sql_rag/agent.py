import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from dbgpt.core.interface.llm import LLMClient, ModelRequest
from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType
from dbgpt.core.interface.output_parser import BaseOutputParser
from dbgpt.core import ModelOutput

from .schema_search import SchemaSearchEngine, SchemaSearchResult
from . import logger


class _JsonResponseParser(BaseOutputParser):
    def get_format_instructions(self) -> str:
        return "Return only valid JSON."


@dataclass
class SufficiencyResult:
    sufficient: bool
    reasoning: str


@dataclass
class PlanResult:
    reasoning: str
    entities: List[str]
    sub_questions: List[str]


class ScratchpadSchemaAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str,
        schema_search_engine: SchemaSearchEngine,
        max_iterations: int = 3,
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.schema_search_engine = schema_search_engine
        self.max_iterations = max_iterations
        self._json_parser = _JsonResponseParser(is_stream_out=False)

    async def _call_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """Helper to call LLM."""
        messages = [ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)]
        request = ModelRequest(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_new_tokens=max_tokens,
        )
        response: ModelOutput = await self.llm_client.generate(request)
        if not response.success:
            raise RuntimeError(
                f"LLM call failed: {response.error_code} - {response.text}"
            )
        return response.text

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        cleaned = self._json_parser.parse_prompt_response(response_text)
        return json.loads(cleaned)

    async def check_sufficiency(self, query: str, schema_snapshot: str) -> SufficiencyResult:
        """Ask LLM if collected schema is sufficient."""
        prompt = f"""
User Question: "{query}"

Current Schema Context (JSON):
{schema_snapshot}

Determine if the schema contains every table, column, and key detail needed to answer the question.
Respond strictly in JSON:
{{
  "reasoning": "...explain gaps or confirmation...",
  "sufficient": "YES" or "NO"
}}
"""

        response_text = await self._call_llm(prompt)
        try:
            parsed = self._parse_json_response(response_text)
        except Exception as exc:
            logger.error(
                "Failed to parse sufficiency response",
                extra={"props": {"error": str(exc), "raw": response_text}},
            )
            raise

        reasoning = parsed.get("reasoning", "").strip()
        sufficient = parsed.get("sufficient", "").strip().upper() == "YES"
        logger.info(
            "[Scratchpad] Sufficiency",
            extra={"props": {"sufficient": sufficient, "reasoning": reasoning}},
        )
        return SufficiencyResult(sufficient=sufficient, reasoning=reasoning)

    async def plan_next_step(
        self, query: str, schema_snapshot: str, previous_reasoning: str
    ) -> PlanResult:
        """Break the task into sub-questions when schema is insufficient."""
        prompt = f"""
User Question: "{query}"

Current Schema Context:
{schema_snapshot}

Last Gap Analysis:
{previous_reasoning}

When the schema is insufficient, list missing entities and derive focused sub-questions.
Follow the style:
│  │ → Entities: [time_range=last_quarter, aggregation=top]
│  │ → Sub-questions:
│  │   1. Which table has customer data?
│  │   2. Which table has revenue/payment data?

Respond strictly in JSON:
{{
  "reasoning": "...",
  "entities": ["entity_keyword_1", "entity_keyword_2"],
  "sub_questions": ["Sub question 1", "Sub question 2"]
}}
"""

        response_text = await self._call_llm(prompt, max_tokens=768)
        try:
            parsed = self._parse_json_response(response_text)
        except Exception as exc:
            logger.error(
                "Failed to parse plan response",
                extra={"props": {"error": str(exc), "raw": response_text}},
            )
            return PlanResult(reasoning="", entities=[], sub_questions=[])

        reasoning = parsed.get("reasoning", "").strip()
        entities = [item.strip() for item in parsed.get("entities", []) if item]
        sub_questions = [
            item.strip() for item in parsed.get("sub_questions", []) if item
        ]

        logger.info(
            "[Scratchpad] Plan Next Step",
            extra={"props": {"entities": entities, "sub_questions": sub_questions}},
        )
        return PlanResult(reasoning=reasoning, entities=entities, sub_questions=sub_questions)

    def _merge_context(
        self,
        scratchpad: "OrderedDict[str, Dict[str, Any]]",
        new_context: Dict[str, Dict[str, Any]],
    ) -> None:
        for table_name, context in new_context.items():
            if table_name not in scratchpad:
                scratchpad[table_name] = context
            else:
                existing_columns = {
                    col["column_name"] for col in scratchpad[table_name]["columns"]
                }
                for column in context.get("columns", []):
                    if column["column_name"] not in existing_columns:
                        scratchpad[table_name]["columns"].append(column)

    def _serialize_context(self, scratchpad: "OrderedDict[str, Dict[str, Any]]") -> str:
        if not scratchpad:
            return "{}"
        return json.dumps(scratchpad, indent=2, ensure_ascii=False)

    def _sync_existing_columns(
        self,
        scratchpad: "OrderedDict[str, Dict[str, Any]]",
        existing_columns: Dict[str, Set[str]],
    ) -> None:
        for table_name, context in scratchpad.items():
            cols = {col["column_name"] for col in context.get("columns", [])}
            if table_name not in existing_columns:
                existing_columns[table_name] = set()
            existing_columns[table_name].update(cols)

    async def run(self, query: str) -> str:
        """Retrieve structured schema context ready for prompting."""
        scratchpad: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        existing_columns: Dict[str, Set[str]] = defaultdict(set)
        pending_queries: List[str] = [query]
        iteration = 0
        last_reason = ""
        seen_signatures: Set[str] = set()

        logger.info("Scratchpad schema search", extra={"props": {"query": query}})

        while iteration < self.max_iterations and pending_queries:
            signature = "|".join(sorted({term.strip() for term in pending_queries if term.strip()}))
            if signature in seen_signatures:
                logger.info(
                    "Search signature already processed; stopping to avoid redundant loops.",
                    extra={"props": {"search_terms": pending_queries}},
                )
                break
            seen_signatures.add(signature)

            logger.info(
                "Retrieval iteration",
                extra={
                    "props": {
                        "iteration": iteration + 1,
                        "search_terms": pending_queries,
                    }
                },
            )

            search_payload = " ".join(pending_queries).strip()
            if not search_payload:
                break

            result: SchemaSearchResult = await self.schema_search_engine.search(
                search_payload,
                existing_tables=set(scratchpad.keys()),
                existing_columns=existing_columns,
            )
            new_schema_found = bool(result.context)

            if result.context:
                self._merge_context(scratchpad, result.context)
                self._sync_existing_columns(scratchpad, existing_columns)

            for table in result.expanded_tables:
                columns = result.columns_added.get(table, [])
                if columns:
                    logger.info(
                        "Additional columns discovered",
                        extra={"props": {"table": table, "columns": columns}},
                    )

            if not scratchpad:
                logger.warning("No schema information found for current query terms.")
                last_reason = "No schema match found for the provided question."
                plan = PlanResult(reasoning="", entities=[], sub_questions=[])
            else:
                schema_snapshot = self._serialize_context(scratchpad)
                sufficiency = await self.check_sufficiency(query, schema_snapshot)
                last_reason = sufficiency.reasoning
                if sufficiency.sufficient:
                    logger.info("Schema sufficiency achieved.")
                    return schema_snapshot
                plan = await self.plan_next_step(query, schema_snapshot, sufficiency.reasoning)

            next_terms = plan.entities + plan.sub_questions
            if not next_terms:
                if not new_schema_found:
                    break
                # If we found new schema but still insufficient without new hints, reuse reasoning
                next_terms = [plan.reasoning] if plan.reasoning else []

            pending_queries = [term for term in next_terms if term]
            iteration += 1

        error_message = (
            "Unable to assemble sufficient schema context after "
            f"{self.max_iterations} iterations. Reason: {last_reason or 'Unknown'}"
        )
        logger.error(error_message)
        raise RuntimeError(error_message)
