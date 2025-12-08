Module 1: Main Agent System Prompt (agents.md)
Why this approach
Following the OpenAI Agents SDK pattern, system prompts should be dynamic functions rather than static strings. OpenAIOpenAI This enables runtime injection of schema context, user state, and available tools—a pattern proven in production by both Codex and WrenAI.

Code Citation
File: openai-agents-python/src/agents/agent.py
python@dataclass
class Agent[TContext]:
    instructions: str | Callable[[RunContextWrapper[TContext], Agent[TContext]], str]
Implementation
python# agents/prompts/system_prompt.py
from typing import Callable
from pydantic import BaseModel

class AgentContext(BaseModel):
    """Runtime context injected into system prompt"""
    schema_context: str          # DDL + column descriptions
    available_tools: list[str]   # Currently registered tools
    session_summary: str | None  # Compressed conversation history
    user_preferences: dict       # User-specific settings

def build_system_prompt(ctx: AgentContext) -> str:
    return f"""You are a data analysis assistant specialized in SQL query generation and document retrieval.

## PERSONA
- Expert in SQL query construction and data interpretation
- Precise, factual, and transparent about limitations
- Never fabricate data—all data must come from SQL execution or document retrieval

## CAPABILITIES
You have access to these tools: {', '.join(ctx.available_tools)}

## DATABASE SCHEMA
{ctx.schema_context}

## GUARDRAILS
1. NEVER generate synthetic data that should come from SQL execution
2. NEVER answer questions outside your knowledge domain without retrieval
3. For ambiguous references ("this", "that", "the previous"), ALWAYS clarify or use session context
4. Only generate SELECT statements—no INSERT, UPDATE, DELETE, or DDL

## ORCHESTRATION RULES
1. For data questions: ALWAYS use sql_rag tool first
2. For document questions: Use doc_rag tool
3. For chart requests: Execute SQL first, then generate Vega-Lite spec
4. For follow-ups: Check session_summary for context before asking for clarification

## SESSION CONTEXT
{ctx.session_summary or "No prior conversation context."}
"""
Pydantic Interface
python# agents/models/agent_config.py
from pydantic import BaseModel, Field
from typing import Literal

class AgentConfig(BaseModel):
    name: str
    model: str = "gpt-4o"
    instructions: str | Callable = Field(...)
    tools: list[str] = Field(default_factory=list)
    handoffs: list[str] = Field(default_factory=list)
    input_guardrails: list[str] = Field(default_factory=list)
    output_guardrails: list[str] = Field(default_factory=list)
    output_type: type | None = None
    tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"] = "run_llm_aga
    