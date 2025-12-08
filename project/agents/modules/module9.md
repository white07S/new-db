Module 9: Session Management
Why this approach
Codex stores sessions in JSONL files with auto-compaction at ~220k tokens. We use PostgreSQL for persistence and Redis for real-time state, enabling session resume and conversation history.
Code Citation
File: codex-rs/ (session storage)
~/.codex/sessions/$YEAR/$MONTH/$DAY/rollout-*.jsonl
Implementation
python# agents/session/models.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any

class Message(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_count: int = 0
    metadata: dict = {}

class Session(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    messages: list[Message] = Field(default_factory=list)
    total_tokens: int = 0
    compaction_count: int = 0
    metadata: dict = {}

# agents/session/postgres_store.py
import asyncpg
from typing import Optional

class PostgresSessionStore:
    """Persistent session storage"""
    
    def __init__(self, connection_string: str):
        self.conn_string = connection_string
        self.pool: asyncpg.Pool | None = None
    
    async def initialize(self):
        self.pool = await asyncpg.create_pool(self.conn_string)
        await self._create_tables()
    
    async def _create_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id UUID PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    total_tokens INTEGER DEFAULT 0,
                    compaction_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}'
                );
                
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    session_id UUID REFERENCES sessions(session_id),
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    token_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}'
                );
                
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages(session_id, timestamp);
            """)
    
    async def create_session(self, user_id: str) -> Session:
        session_id = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sessions (session_id, user_id) VALUES ($1, $2)",
                session_id, user_id
            )
        return Session(session_id=session_id, user_id=user_id)
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM sessions WHERE session_id = $1",
                session_id
            )
            if not row:
                return None
            
            messages = await conn.fetch(
                "SELECT * FROM messages WHERE session_id = $1 ORDER BY timestamp",
                session_id
            )
            
            return Session(
                session_id=row["session_id"],
                user_id=row["user_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                total_tokens=row["total_tokens"],
                compaction_count=row["compaction_count"],
                messages=[Message(**dict(m)) for m in messages],
                metadata=row["metadata"]
            )
    
    async def add_message(self, session_id: str, message: Message):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO messages (session_id, role, content, timestamp, token_count, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, session_id, message.role, message.content, 
                message.timestamp, message.token_count, message.metadata)
            
            await conn.execute("""
                UPDATE sessions 
                SET updated_at = NOW(), total_tokens = total_tokens + $2
                WHERE session_id = $1
            """, session_id, message.token_count)

# agents/session/redis_cache.py
import redis.asyncio as redis

class RedisSessionCache:
    """Real-time session state caching"""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour cache
    
    async def cache_session(self, session: Session):
        key = f"session:{session.session_id}"
        await self.redis.setex(
            key,
            self.ttl,
            session.model_dump_json()
        )
    
    async def get_cached_session(self, session_id: str) -> Optional[Session]:
        key = f"session:{session_id}"
        data = await self.redis.get(key)
        if data:
            return Session.model_validate_json(data)
        return None
    
    async def cache_scratchpad(self, session_id: str, scratchpad: Scratchpad):
        key = f"scratchpad:{session_id}"
        await self.redis.setex(key, self.ttl, scratchpad.model_dump_json())
    
    async def invalidate(self, session_id: str):
        await self.redis.delete(
            f"session:{session_id}",
            f"scratchpad:{session_id}"
        )

# agents/session/compaction.py
class SessionCompactor:
    """Compress conversation history when approaching token limits"""
    
    COMPACTION_THRESHOLD = 100000  # Tokens before compaction
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def should_compact(self, session: Session) -> bool:
        return session.total_tokens > self.COMPACTION_THRESHOLD
    
    async def compact(self, session: Session) -> Session:
        """Summarize older messages, keep recent ones"""
        if len(session.messages) < 10:
            return session
        
        # Keep last 5 messages, summarize the rest
        to_summarize = session.messages[:-5]
        to_keep = session.messages[-5:]
        
        # Generate summary
        summary_prompt = "Summarize this conversation history concisely:\n\n"
        summary_prompt += "\n".join([
            f"{m.role}: {m.content}" for m in to_summarize
        ])
        
        summary = await self.llm.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        # Create compacted session
        summary_message = Message(
            role="system",
            content=f"[Conversation Summary]: {summary.choices[0].message.content}",
            token_count=len(summary.choices[0].message.content) // 4
        )
        
        session.messages = [summary_message] + to_keep
        session.compaction_count += 1
        session.total_tokens = sum(m.token_count for m in session.messages)
        
        return session
