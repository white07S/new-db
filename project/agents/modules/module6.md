Module 6: Guardrails Implementation
Why this approach
The OpenAI Agents SDK provides input and output guardrails that run in parallel with agent execution. We extend this with domain-specific guardrails for SQL agents.
Code Citation
File: openai-agents-python/src/agents/guardrail.py
python@dataclass
class GuardrailFunctionOutput:
    output_info: Any
    tripwire_triggered: bool
Implementation
python# agents/guardrails/input_guardrails.py
from pydantic import BaseModel
from typing import Callable, Awaitable

class GuardrailResult(BaseModel):
    passed: bool
    reason: str | None = None
    suggested_response: str | None = None
    metadata: dict = {}

class InputGuardrail:
    """Base class for input validation"""
    name: str
    run_in_parallel: bool = True
    
    async def check(self, user_input: str, context: dict) -> GuardrailResult:
        raise NotImplementedError

class OutOfContextGuardrail(InputGuardrail):
    """Detect queries outside the database domain"""
    name = "out_of_context"
    
    async def check(self, user_input: str, context: dict) -> GuardrailResult:
        # Use LLM to classify
        schema_summary = context.get("schema_summary", "")
        
        result = await self.llm.create(
            model="gpt-4o-mini",  # Fast model for guardrails
            messages=[
                {"role": "system", "content": f"""Determine if this query relates to the database schema.
Schema: {schema_summary}
Respond with JSON: {{"is_relevant": bool, "reason": str}}"""},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"}
        )
        
        parsed = json.loads(result.choices[0].message.content)
        if not parsed["is_relevant"]:
            return GuardrailResult(
                passed=False,
                reason=parsed["reason"],
                suggested_response="I can only help with questions about your data. Could you rephrase your question to relate to the available tables and columns?"
            )
        return GuardrailResult(passed=True)

class AmbiguousReferenceGuardrail(InputGuardrail):
    """Detect ambiguous pronouns/references without context"""
    name = "ambiguous_reference"
    
    AMBIGUOUS_PATTERNS = ["this", "that", "these", "those", "it", "they", "the previous", "of which", "from before"]
    
    async def check(self, user_input: str, context: dict) -> GuardrailResult:
        has_history = bool(context.get("session_history"))
        input_lower = user_input.lower()
        
        for pattern in self.AMBIGUOUS_PATTERNS:
            if pattern in input_lower:
                if not has_history:
                    return GuardrailResult(
                        passed=False,
                        reason=f"Ambiguous reference '{pattern}' without conversation context",
                        suggested_response=f"I'm not sure what '{pattern}' refers to. Could you be more specific?"
                    )
        return GuardrailResult(passed=True)

class MisleadingQueryGuardrail(InputGuardrail):
    """Detect potentially misleading or impossible queries"""
    name = "misleading_query"
    
    async def check(self, user_input: str, context: dict) -> GuardrailResult:
        # Pattern from WrenAI's misleading_assistance pipeline
        result = await self.llm.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Analyze if this query makes invalid assumptions about the data.
Examples of misleading queries:
- "Show me future sales" (asking for non-existent future data)
- "Compare apples to oranges table" (comparing unrelated entities)
- "Find negative customer IDs" (illogical constraint)

Respond: {"is_misleading": bool, "reason": str, "clarification_needed": str}"""},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"}
        )
        
        parsed = json.loads(result.choices[0].message.content)
        if parsed["is_misleading"]:
            return GuardrailResult(
                passed=False,
                reason=parsed["reason"],
                suggested_response=parsed["clarification_needed"]
            )
        return GuardrailResult(passed=True)

# Output Guardrails
class OutputGuardrail:
    """Base class for output validation"""
    name: str
    
    async def check(self, output: str, context: dict) -> GuardrailResult:
        raise NotImplementedError

class NoFabricatedDataGuardrail(OutputGuardrail):
    """Ensure LLM doesn't generate data that should come from SQL"""
    name = "no_fabricated_data"
    
    async def check(self, output: str, context: dict) -> GuardrailResult:
        has_sql_result = "sql_result" in context
        
        # Check for numeric patterns that weren't in SQL results
        result = await self.llm.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""Check if this response contains specific data values (numbers, names, dates) 
that should have come from a database query but weren't provided.

SQL was executed: {has_sql_result}
SQL Result snippet: {str(context.get('sql_result', 'None'))[:500]}

Respond: {{"has_fabricated_data": bool, "problematic_values": list[str]}}"""},
                {"role": "user", "content": output}
            ],
            response_format={"type": "json_object"}
        )
        
        parsed = json.loads(result.choices[0].message.content)
        if parsed["has_fabricated_data"]:
            return GuardrailResult(
                passed=False,
                reason=f"Response contains fabricated data: {parsed['problematic_values']}",
                metadata={"fabricated": parsed["problematic_values"]}
            )
        return GuardrailResult(passed=True)

# Guardrail Runner
class GuardrailRunner:
    """Execute guardrails in parallel or sequence"""
    
    def __init__(self, input_guardrails: list[InputGuardrail], output_guardrails: list[OutputGuardrail]):
        self.input_guardrails = input_guardrails
        self.output_guardrails = output_guardrails
    
    async def check_input(self, user_input: str, context: dict) -> list[GuardrailResult]:
        """Run all input guardrails"""
        parallel = [g for g in self.input_guardrails if g.run_in_parallel]
        sequential = [g for g in self.input_guardrails if not g.run_in_parallel]
        
        results = []
        
        # Run parallel guardrails
        if parallel:
            parallel_results = await asyncio.gather(*[
                g.check(user_input, context) for g in parallel
            ])
            results.extend(parallel_results)
        
        # Run sequential guardrails
        for g in sequential:
            result = await g.check(user_input, context)
            results.append(result)
            if not result.passed:
                break  # Stop on first failure
        
        return results