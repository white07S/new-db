Module 8: Real-time Progress Tracking
Why this approach
Codex's JSONL event protocol enables real-time UI updates. Each step emits events with brief descriptions, allowing streaming progress to clients.
Code Citation
File: codex-rs/protocol/src/protocol.rs
jsonl{"type":"item.started","item":{"id":"item_1","type":"command_execution","status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_1","status":"completed"}}
Implementation
python# agents/events/protocol.py
from pydantic import BaseModel
from typing import Literal, Any
from datetime import datetime
import asyncio

class EventType(str, Enum):
    # Turn lifecycle
    TURN_STARTED = "turn.started"
    TURN_COMPLETED = "turn.completed"
    TURN_FAILED = "turn.failed"
    
    # Item lifecycle
    ITEM_STARTED = "item.started"
    ITEM_UPDATED = "item.updated"
    ITEM_COMPLETED = "item.completed"
    
    # Progress
    PROGRESS_UPDATE = "progress.update"
    
    # Token tracking
    TOKEN_COUNT = "token.count"

class ItemType(str, Enum):
    AGENT_MESSAGE = "agent_message"
    REASONING = "reasoning"
    SQL_EXECUTION = "sql_execution"
    DOC_RETRIEVAL = "doc_retrieval"
    CHART_GENERATION = "chart_generation"
    GUARDRAIL_CHECK = "guardrail_check"

class EventItem(BaseModel):
    id: str
    type: ItemType
    description: str                         # Brief human-readable description
    status: Literal["in_progress", "completed", "failed"]
    output: Any | None = None
    error: str | None = None
    started_at: datetime
    completed_at: datetime | None = None

class Event(BaseModel):
    type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    thread_id: str
    turn_id: str | None = None
    item: EventItem | None = None
    usage: dict | None = None                # Token usage
    error: dict | None = None

class EventEmitter:
    """Manages event emission and subscription"""
    
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self._subscribers: list[asyncio.Queue] = []
        self._item_counter = 0
    
    def subscribe(self) -> asyncio.Queue:
        """Create new event subscription"""
        queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue
    
    def unsubscribe(self, queue: asyncio.Queue):
        self._subscribers.remove(queue)
    
    async def emit(self, event: Event):
        """Emit event to all subscribers"""
        for queue in self._subscribers:
            await queue.put(event)
    
    async def start_item(self, item_type: ItemType, description: str) -> str:
        """Start tracking a new item"""
        self._item_counter += 1
        item_id = f"item_{self._item_counter}"
        
        item = EventItem(
            id=item_id,
            type=item_type,
            description=description,
            status="in_progress",
            started_at=datetime.utcnow()
        )
        
        await self.emit(Event(
            type=EventType.ITEM_STARTED,
            thread_id=self.thread_id,
            item=item
        ))
        
        return item_id
    
    async def complete_item(self, item_id: str, output: Any = None):
        """Mark item as completed"""
        await self.emit(Event(
            type=EventType.ITEM_COMPLETED,
            thread_id=self.thread_id,
            item=EventItem(
                id=item_id,
                type=ItemType.AGENT_MESSAGE,  # Will be updated
                description="Completed",
                status="completed",
                output=output,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
        ))

# FastAPI streaming endpoint
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    emitter = EventEmitter(request.thread_id)
    queue = emitter.subscribe()
    
    # Start agent execution in background
    asyncio.create_task(run_agent_with_events(request, emitter))
    
    async def event_generator():
        while True:
            event = await queue.get()
            yield f"data: {event.model_dump_json()}\n\n"
            if event.type in [EventType.TURN_COMPLETED, EventType.TURN_FAILED]:
                break
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )