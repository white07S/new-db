Module 3: Scratchpad Module (Context Assembler)
Why this approach
The OpenAI Agents SDK handoff pattern shows that context must flow explicitly between agents. The Scratchpad tracks available context, identifies gaps, and assembles final context before LLM calls.
Code Citation
File: openai-agents-python/src/agents/handoffs.py
python@dataclass
class HandoffInputData:
    input_history: list[TResponseInputItem]  # Original conversation history
    pre_handoff_items: tuple[RunItem, ...]   # Items before handoff
    new_items: tuple[RunItem, ...]           # New items including handoff
Implementation
python# agents/scratchpad/context_tracker.py
from pydantic import BaseModel, Field
from typing import Any
from enum import Enum

class ContextStatus(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    STALE = "stale"
    PENDING = "pending"

class ContextItem(BaseModel):
    """Single piece of context"""
    key: str
    value: Any
    source: str                              # "sql_rag", "doc_rag", "user_input"
    status: ContextStatus
    timestamp: float
    token_count: int = 0
    
class ContextGap(BaseModel):
    """Identified missing context"""
    key: str
    reason: str
    suggested_action: str                    # Tool to fill gap

class Scratchpad(BaseModel):
    """Tracks context assembly state"""
    session_id: str
    items: dict[str, ContextItem] = Field(default_factory=dict)
    gaps: list[ContextGap] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 100000                 # Context budget
    
    def add_context(self, key: str, value: Any, source: str, tokens: int = 0):
        """Add context item to scratchpad"""
        self.items[key] = ContextItem(
            key=key,
            value=value,
            source=source,
            status=ContextStatus.AVAILABLE,
            timestamp=time.time(),
            token_count=tokens
        )
        self.total_tokens += tokens
        self.reasoning_trace.append(f"Added {key} from {source} ({tokens} tokens)")
    
    def identify_gaps(self, required_context: list[str]) -> list[ContextGap]:
        """Check for missing required context"""
        self.gaps = []
        for key in required_context:
            if key not in self.items or self.items[key].status != ContextStatus.AVAILABLE:
                self.gaps.append(ContextGap(
                    key=key,
                    reason=f"Required context '{key}' not available",
                    suggested_action=self._suggest_action(key)
                ))
        return self.gaps
    
    def is_complete(self, required: list[str]) -> bool:
        """Check if all required context is available"""
        return all(
            key in self.items and self.items[key].status == ContextStatus.AVAILABLE
            for key in required
        )
    
    def assemble_context(self) -> str:
        """Assemble all context for LLM consumption"""
        sections = []
        for key, item in self.items.items():
            if item.status == ContextStatus.AVAILABLE:
                sections.append(f"## {key.upper()}\n{item.value}")
        return "\n\n".join(sections)
    
    def _suggest_action(self, key: str) -> str:
        """Map context key to retrieval action"""
        mapping = {
            "schema_context": "sql_rag.get_schema",
            "query_results": "sql_rag.execute",
            "document_context": "doc_rag.retrieve",
            "prior_conversation": "session.get_history"
        }
        return mapping.get(key, "unknown")