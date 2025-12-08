# Conversational AI Agent System: Complete Reference Document

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Concepts](#2-core-concepts)
3. [Request Lifecycle](#3-request-lifecycle)
4. [Module Deep Dives](#4-module-deep-dives)
5. [Data Flow Patterns](#5-data-flow-patterns)
6. [State Management](#6-state-management)
7. [Event System](#7-event-system)
8. [Error Handling](#8-error-handling)
9. [Extension Points](#9-extension-points)
10. [Operational Patterns](#10-operational-patterns)

---

## 1. System Overview

### 1.1 What This System Does

This system is a **conversational AI agent** that answers natural language questions by:
- Converting questions to SQL queries and executing them against a database
- Retrieving relevant documents from a vector store
- Generating visualizations (charts) from data
- Maintaining conversation context across multiple turns
- Validating inputs/outputs to prevent hallucination and misuse

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│                    (Web UI / Mobile App / API Client)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼ HTTP/SSE
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                       │
│                         (FastAPI + SSE Streaming)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ /chat       │  │ /chat/stream│  │ /sessions   │  │ /health     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  Guardrail       │  │  Task Planner    │  │  Event Emitter   │          │
│  │  Runner          │  │  Engine          │  │                  │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                     SCRATCHPAD                                │          │
│  │              (Context Assembly & Tracking)                    │          │
│  └──────────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RETRIEVAL LAYER                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  SQL RAG         │  │  Doc RAG         │  │  Future...       │          │
│  │  Retriever       │  │  Retriever       │  │  (API, Graph)    │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│           │                    │                                            │
│           ▼                    ▼                                            │
│  ┌──────────────────┐  ┌──────────────────┐                                │
│  │  Truncation      │  │  Chunking        │                                │
│  │  Policy          │  │  Strategy        │                                │
│  └──────────────────┘  └──────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PERSISTENCE LAYER                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  PostgreSQL      │  │  Redis           │  │  Vector Store    │          │
│  │  (Sessions,      │  │  (Cache,         │  │  (Embeddings)    │          │
│  │   Messages)      │  │   Real-time)     │  │                  │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL SERVICES                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  OpenAI /        │  │  Target          │  │  Azure Doc       │          │
│  │  Azure OpenAI    │  │  Database        │  │  Intelligence    │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Event-Driven** | Every operation emits events for real-time UI updates and debugging |
| **Modular** | Each component has single responsibility, clear interfaces |
| **Fail-Safe** | Guardrails prevent bad inputs; truncation prevents context overflow |
| **Extensible** | New retrieval methods plug in without core changes |
| **Observable** | Full event trace for every request |
| **Stateful** | Conversation context persists across turns |

---

## 2. Core Concepts

### 2.1 The Agent

The **Agent** is the central orchestrator. It:
- Receives user queries
- Runs input guardrails
- Creates execution plans
- Delegates to retrieval modules
- Assembles final responses
- Runs output guardrails

```python
# Conceptual Agent structure
Agent:
    name: str                      # "DataAssistant"
    instructions: Callable         # Dynamic system prompt
    tools: List[Tool]              # Available tools (sql_rag, doc_rag, chart_gen)
    handoffs: List[Agent]          # Sub-agents for delegation
    input_guardrails: List         # Pre-processing validation
    output_guardrails: List        # Post-processing validation
    output_type: Type              # Structured output schema
```

### 2.2 Sessions and Messages

A **Session** represents a conversation thread. It contains:
- Unique identifier
- User identifier
- List of messages (user + assistant)
- Token count (for compaction decisions)
- Metadata (timestamps, compaction count)

```python
Session:
    session_id: UUID
    user_id: str
    messages: List[Message]
    total_tokens: int
    compaction_count: int
    
Message:
    role: "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: datetime
    token_count: int
    metadata: dict  # source, tool_call_id, etc.
```

### 2.3 Tasks and Plans

A **Task** is a single atomic operation. A **TaskPlan** is a directed acyclic graph (DAG) of tasks.

```python
Task:
    id: str
    task_description: str          # What to do
    tool_feature: str              # Which tool to use
    reasoning: str                 # Why this task exists
    depends_on: List[str]          # Task IDs that must complete first
    condition: Optional[str]       # For conditional execution
    
TaskPlan:
    plan_id: str
    tasks: List[Task]
    execution_mode: parallel | sequential | conditional
```

**Execution Order Resolution:**
```
Given tasks: A (no deps), B (no deps), C (depends on A, B), D (depends on C)

Execution Waves:
  Wave 1: [A, B]     ← Execute in parallel
  Wave 2: [C]        ← Wait for A, B, then execute
  Wave 3: [D]        ← Wait for C, then execute
```

### 2.4 Scratchpad

The **Scratchpad** is the context assembly workspace. It:
- Tracks what context is available
- Identifies missing context (gaps)
- Assembles final context for LLM calls
- Manages token budget

```python
Scratchpad:
    session_id: str
    items: Dict[str, ContextItem]  # Available context
    gaps: List[ContextGap]         # Missing context
    reasoning_trace: List[str]     # Audit log
    total_tokens: int
    max_tokens: int                # Budget limit
    
ContextItem:
    key: str                       # "schema_context", "query_results"
    value: Any
    source: str                    # "sql_rag", "doc_rag", "user_input"
    status: available | missing | stale | pending
    token_count: int
```

### 2.5 Response Views

All responses are **dict-based views** that can be extended without breaking clients.

```python
BaseView:
    text: str                      # Always present
    view_type: "base"

DataView(BaseView):
    view_type: "data"
    data_dict: Dict[str, List]     # Column-oriented data
    columns: List[str]
    row_count: int
    truncated: bool
    truncation_reason: Optional[str]

ChartView(DataView):
    view_type: "chart"
    chart_spec: dict               # Vega-Lite specification
    chart_type: str                # "bar", "line", etc.

ErrorView(BaseView):
    view_type: "error"
    error_code: str
    suggestions: List[str]
```

### 2.6 Events

Every operation emits **Events** for observability and real-time UI updates.

```python
Event:
    type: EventType                # turn.started, item.completed, etc.
    timestamp: datetime
    thread_id: str
    turn_id: Optional[str]
    item: Optional[EventItem]
    usage: Optional[dict]          # Token usage
    
EventItem:
    id: str
    type: ItemType                 # sql_execution, doc_retrieval, etc.
    description: str               # Human-readable status
    status: in_progress | completed | failed
    output: Optional[Any]
```

---

## 3. Request Lifecycle

### 3.1 Complete Request Flow

This section traces a user query from arrival to response, showing every step.

```
USER QUERY: "Show me top 5 customers by revenue and create a bar chart"
```

#### Phase 1: Request Reception

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. HTTP Request arrives at /chat/stream                         │
│                                                                 │
│    POST /chat/stream                                            │
│    {                                                            │
│      "session_id": "abc-123",                                   │
│      "message": "Show me top 5 customers by revenue..."         │
│    }                                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Session Loading                                              │
│                                                                 │
│    a. Check Redis cache for session                             │
│       └─ Cache HIT → Use cached session                         │
│       └─ Cache MISS → Load from PostgreSQL                      │
│                                                                 │
│    b. Load conversation history                                 │
│    c. Check if compaction needed (tokens > threshold)           │
│    d. Initialize Scratchpad with session context                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Event Emitter Initialization                                 │
│                                                                 │
│    EventEmitter created for this thread                         │
│    SSE stream opened to client                                  │
│                                                                 │
│    EMIT: {type: "turn.started", turn_id: "turn_1"}             │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 2: Input Validation (Guardrails)

```
┌─────────────────────────────────────────────────────────────────┐
│ 4. Input Guardrails (Run in Parallel)                           │
│                                                                 │
│    ┌─────────────────────┐  ┌─────────────────────┐            │
│    │ AmbiguousReference  │  │ OutOfContext        │            │
│    │ Guardrail           │  │ Guardrail           │            │
│    │                     │  │                     │            │
│    │ Check for:          │  │ Check if query      │            │
│    │ - "this", "that"    │  │ relates to schema   │            │
│    │ - "it", "they"      │  │                     │            │
│    │ - "the previous"    │  │ Uses LLM to         │            │
│    │                     │  │ classify            │            │
│    └─────────────────────┘  └─────────────────────┘            │
│              │                        │                         │
│              └────────────┬───────────┘                         │
│                           ▼                                     │
│    ┌─────────────────────────────────────────┐                 │
│    │ MisleadingQuery Guardrail               │                 │
│    │                                         │                 │
│    │ Check for:                              │                 │
│    │ - Future data requests                  │                 │
│    │ - Impossible constraints                │                 │
│    │ - Non-existent entities                 │                 │
│    └─────────────────────────────────────────┘                 │
│                                                                 │
│    EMIT: {type: "item.started", item: {type: "guardrail_check",│
│           description: "Validating query"}}                     │
│                                                                 │
│    Result: ALL PASSED → Continue                                │
│            ANY FAILED → Return ErrorView with suggestions       │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 3: Task Planning

```
┌─────────────────────────────────────────────────────────────────┐
│ 5. Task Plan Generation                                         │
│                                                                 │
│    Input: "Show me top 5 customers by revenue and create        │
│            a bar chart"                                         │
│                                                                 │
│    LLM Call (Structured Output):                                │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ System: You are a task planner. Decompose queries into  │ │
│    │         atomic tasks with dependencies.                  │ │
│    │                                                          │ │
│    │ User: {query}                                            │ │
│    │                                                          │ │
│    │ Response Format: TaskPlan                                │ │
│    └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│    Generated Plan:                                              │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ TaskPlan:                                                │ │
│    │   tasks:                                                 │ │
│    │     - id: "task_1"                                       │ │
│    │       description: "Query top 5 customers by revenue"    │ │
│    │       tool_feature: "sql_rag"                            │ │
│    │       reasoning: "Need customer revenue data"            │ │
│    │       depends_on: []                                     │ │
│    │                                                          │ │
│    │     - id: "task_2"                                       │ │
│    │       description: "Generate bar chart from results"     │ │
│    │       tool_feature: "chart_gen"                          │ │
│    │       reasoning: "User requested visualization"          │ │
│    │       depends_on: ["task_1"]                             │ │
│    └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│    EMIT: {type: "item.completed", item: {type: "reasoning",    │
│           description: "Created execution plan: 2 tasks"}}      │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 4: Task Execution

```
┌─────────────────────────────────────────────────────────────────┐
│ 6. Execute Wave 1: SQL RAG (task_1)                             │
│                                                                 │
│    EMIT: {type: "item.started", item: {type: "sql_execution",  │
│           description: "Generating SQL query"}}                 │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 6a. SQL Generation                                       │ │
│    │                                                          │ │
│    │ Input:                                                   │ │
│    │   - Query: "Query top 5 customers by revenue"            │ │
│    │   - Schema: {tables, columns, relationships}             │ │
│    │   - Examples: Similar SQL pairs from index               │ │
│    │                                                          │ │
│    │ LLM Call → Generated SQL:                                │ │
│    │   SELECT c.customer_id, c.name,                          │ │
│    │          SUM(o.total) as revenue                         │ │
│    │   FROM customers c                                       │ │
│    │   JOIN orders o ON c.customer_id = o.customer_id         │ │
│    │   GROUP BY c.customer_id, c.name                         │ │
│    │   ORDER BY revenue DESC                                  │ │
│    │   LIMIT 5                                                │ │
│    └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│    EMIT: {type: "item.updated", item: {description:            │
│           "Executing SQL query"}}                               │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 6b. SQL Execution                                        │ │
│    │                                                          │ │
│    │ Execute against target database                          │ │
│    │ Result: DataFrame with 5 rows                            │ │
│    │                                                          │ │
│    │ │ customer_id │ name      │ revenue   │                  │ │
│    │ │─────────────│───────────│───────────│                  │ │
│    │ │ 101         │ Acme Corp │ 150,000   │                  │ │
│    │ │ 203         │ TechStart │ 125,000   │                  │ │
│    │ │ ...         │ ...       │ ...       │                  │ │
│    └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 6c. Truncation Check                                     │ │
│    │                                                          │ │
│    │ DataFrame size: 5 rows × 3 columns = ~500 bytes          │ │
│    │ Threshold: 50KB                                          │ │
│    │ Result: NO TRUNCATION NEEDED                             │ │
│    └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 6d. Update Scratchpad                                    │ │
│    │                                                          │ │
│    │ scratchpad.add_context(                                  │ │
│    │   key="query_results",                                   │ │
│    │   value={dataframe as dict},                             │ │
│    │   source="sql_rag",                                      │ │
│    │   tokens=estimated_tokens                                │ │
│    │ )                                                        │ │
│    └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│    EMIT: {type: "item.completed", item: {type: "sql_execution",│
│           description: "Retrieved 5 customers",                 │
│           output: {rows: 5, columns: 3}}}                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Execute Wave 2: Chart Generation (task_2)                    │
│                                                                 │
│    EMIT: {type: "item.started", item: {type: "chart_generation",│
│           description: "Creating bar chart"}}                   │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 7a. Assemble Context from Scratchpad                     │ │
│    │                                                          │ │
│    │ context = scratchpad.assemble_context()                  │ │
│    │ # Returns: query_results, schema_context, etc.           │ │
│    └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 7b. Vega-Lite Generation                                 │ │
│    │                                                          │ │
│    │ LLM Call with data + chart request:                      │ │
│    │                                                          │ │
│    │ Generated Spec:                                          │ │
│    │ {                                                        │ │
│    │   "$schema": "https://vega.github.io/schema/...",        │ │
│    │   "mark": "bar",                                         │ │
│    │   "encoding": {                                          │ │
│    │     "x": {"field": "name", "type": "nominal"},           │ │
│    │     "y": {"field": "revenue", "type": "quantitative"}    │ │
│    │   },                                                     │ │
│    │   "data": {"values": [...]}                              │ │
│    │ }                                                        │ │
│    └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│    EMIT: {type: "item.completed", item: {type: "chart_generation",│
│           description: "Bar chart created"}}                    │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 5: Response Assembly

```
┌─────────────────────────────────────────────────────────────────┐
│ 8. Response Generation                                          │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 8a. Generate Natural Language Response                   │ │
│    │                                                          │ │
│    │ LLM Call:                                                │ │
│    │   System: Generate a response summarizing the data.      │ │
│    │           DO NOT fabricate any numbers.                  │ │
│    │   Context: {query_results from scratchpad}               │ │
│    │   User: Summarize these results                          │ │
│    │                                                          │ │
│    │ Response: "Here are the top 5 customers by revenue:      │ │
│    │            Acme Corp leads with $150,000..."             │ │
│    └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 8b. Build Response View                                  │ │
│    │                                                          │ │
│    │ ChartView:                                               │ │
│    │   text: "Here are the top 5 customers..."                │ │
│    │   view_type: "chart"                                     │ │
│    │   data_dict: {customer_id: [...], name: [...], ...}      │ │
│    │   columns: ["customer_id", "name", "revenue"]            │ │
│    │   row_count: 5                                           │ │
│    │   truncated: false                                       │ │
│    │   chart_spec: {vega-lite spec}                           │ │
│    │   chart_type: "bar"                                      │ │
│    └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 6: Output Validation

```
┌─────────────────────────────────────────────────────────────────┐
│ 9. Output Guardrails                                            │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ NoFabricatedData Guardrail                               │ │
│    │                                                          │ │
│    │ Check: Does the text response contain numbers that       │ │
│    │        weren't in the SQL results?                       │ │
│    │                                                          │ │
│    │ LLM Call:                                                │ │
│    │   Compare response text against SQL result data          │ │
│    │   Flag any numbers not found in source data              │ │
│    │                                                          │ │
│    │ Result: PASSED (all numbers match SQL results)           │ │
│    └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│    If FAILED:                                                   │
│      - Remove fabricated content                                │
│      - Regenerate response OR return error                      │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 7: Persistence and Response

```
┌─────────────────────────────────────────────────────────────────┐
│ 10. Persist Session State                                       │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 10a. Save Messages to PostgreSQL                         │ │
│    │                                                          │ │
│    │ INSERT INTO messages (session_id, role, content, ...)    │ │
│    │ VALUES                                                   │ │
│    │   ($session_id, 'user', $user_message, ...),             │ │
│    │   ($session_id, 'assistant', $response_text, ...)        │ │
│    │                                                          │ │
│    │ UPDATE sessions SET                                      │ │
│    │   updated_at = NOW(),                                    │ │
│    │   total_tokens = total_tokens + $new_tokens              │ │
│    │ WHERE session_id = $session_id                           │ │
│    └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 10b. Update Redis Cache                                  │ │
│    │                                                          │ │
│    │ SETEX session:{session_id} 3600 {serialized_session}     │ │
│    │ SETEX scratchpad:{session_id} 3600 {serialized_scratchpad}│ │
│    └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 11. Send Response                                               │
│                                                                 │
│    EMIT: {type: "turn.completed", turn_id: "turn_1",           │
│           usage: {prompt_tokens: 1500, completion_tokens: 300}} │
│                                                                 │
│    Final SSE Event:                                             │
│    data: {                                                      │
│      "type": "response",                                        │
│      "view": {                                                  │
│        "text": "Here are the top 5 customers...",               │
│        "view_type": "chart",                                    │
│        "data_dict": {...},                                      │
│        "chart_spec": {...}                                      │
│      }                                                          │
│    }                                                            │
│                                                                 │
│    SSE Stream Closed                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Request Flow Diagram (Simplified)

```
Request → Session Load → Input Guardrails → Task Planning → Task Execution → Response Assembly → Output Guardrails → Persist → Response
    │          │               │                 │                │                  │                 │           │         │
    │          │               │                 │                │                  │                 │           │         │
    ▼          ▼               ▼                 ▼                ▼                  ▼                 ▼           ▼         ▼
 Events     Events          Events            Events           Events             Events            Events      Events    Events
    │          │               │                 │                │                  │                 │           │         │
    └──────────┴───────────────┴─────────────────┴────────────────┴──────────────────┴─────────────────┴───────────┴─────────┘
                                                            │
                                                            ▼
                                                    SSE Stream to Client
```

---

## 4. Module Deep Dives

### 4.1 Guardrails Module

#### Purpose
Guardrails validate inputs and outputs to ensure the system behaves predictably and safely.

#### Input Guardrails

| Guardrail | Detection Method | Response on Failure |
|-----------|------------------|---------------------|
| `AmbiguousReferenceGuardrail` | Pattern matching for pronouns | "Could you clarify what 'X' refers to?" |
| `OutOfContextGuardrail` | LLM classification against schema | "I can only answer questions about [tables]" |
| `MisleadingQueryGuardrail` | LLM analysis for impossible requests | "This query asks for [X] which doesn't exist" |

#### Execution Model

```python
# Guardrails run in parallel for speed
async def check_input_guardrails(query: str, context: dict) -> List[GuardrailResult]:
    parallel_guardrails = [
        AmbiguousReferenceGuardrail(),
        OutOfContextGuardrail(llm),
        MisleadingQueryGuardrail(llm)
    ]
    
    results = await asyncio.gather(*[
        g.check(query, context) for g in parallel_guardrails
    ])
    
    # Short-circuit: if any failed, return immediately
    failures = [r for r in results if not r.passed]
    if failures:
        return failures[0]  # Return first failure
    
    return None  # All passed
```

#### Output Guardrails

| Guardrail | Detection Method | Response on Failure |
|-----------|------------------|---------------------|
| `NoFabricatedDataGuardrail` | Compare response numbers against SQL results | Regenerate without fabricated content |

### 4.2 Task Planner Module

#### Purpose
Decomposes complex queries into atomic, dependency-aware tasks.

#### Input/Output

```
Input:  "Find top customers AND their orders AND create a chart"
Output: TaskPlan with 3 tasks:
        - Task 1: Find top customers (sql_rag)
        - Task 2: Find their orders (sql_rag, depends on Task 1)
        - Task 3: Create chart (chart_gen, depends on Task 2)
```

#### Dependency Resolution Algorithm

```python
def get_execution_order(tasks: List[Task]) -> List[List[Task]]:
    """
    Topological sort with parallel grouping.
    
    Returns waves where tasks within a wave can execute in parallel,
    but waves execute sequentially.
    """
    executed = set()
    waves = []
    remaining = {t.id: t for t in tasks}
    
    while remaining:
        # Find all tasks whose dependencies are satisfied
        ready = [
            t for t in remaining.values()
            if all(dep in executed for dep in t.depends_on)
        ]
        
        if not ready:
            raise CircularDependencyError()
        
        # Sort by priority within wave
        ready.sort(key=lambda t: t.priority)
        waves.append(ready)
        
        # Mark as executed
        for t in ready:
            executed.add(t.id)
            del remaining[t.id]
    
    return waves
```

#### Conditional Execution

```python
Task:
    id: "task_3"
    condition: "task_1.row_count > 0"  # Only execute if Task 1 returned data
    
# During execution:
if task.condition:
    condition_met = evaluate_condition(task.condition, prior_results)
    if not condition_met:
        return SkippedResult(reason="Condition not met")
```

### 4.3 Scratchpad Module

#### Purpose
Tracks context assembly state and manages the information available for LLM calls.

#### Context Lifecycle

```
1. PENDING   → Context requested but retrieval not started
2. AVAILABLE → Context retrieved and ready
3. STALE     → Context exists but may be outdated
4. MISSING   → Context not available
```

#### Gap Analysis

```python
def identify_gaps(required: List[str]) -> List[ContextGap]:
    """
    Check if all required context is available.
    Returns list of gaps with suggested actions.
    """
    gaps = []
    for key in required:
        item = self.items.get(key)
        
        if not item:
            gaps.append(ContextGap(
                key=key,
                reason=f"'{key}' not loaded",
                suggested_action=self._suggest_action(key)
            ))
        elif item.status == ContextStatus.STALE:
            gaps.append(ContextGap(
                key=key,
                reason=f"'{key}' may be outdated",
                suggested_action=f"refresh_{key}"
            ))
    
    return gaps
```

#### Token Budget Management

```python
MAX_TOKENS = 100_000

def add_context(key: str, value: Any, tokens: int):
    if self.total_tokens + tokens > MAX_TOKENS:
        # Need to make room
        self._evict_low_priority_context(tokens_needed=tokens)
    
    self.items[key] = ContextItem(...)
    self.total_tokens += tokens
```

### 4.4 Truncation Module

#### Purpose
Intelligently reduces large SQL results to fit within token limits while preserving critical data.

#### Three-Step Algorithm

```
Step 1: Remove Sparse Columns
        ├─ Calculate null/empty percentage per column
        ├─ Remove columns with >80% null/empty
        └─ Rationale: Sparse columns provide little value

Step 2: Remove by Priority
        ├─ Priority from schema: high, medium, low
        ├─ Remove low priority first, then medium
        ├─ NEVER remove high priority
        └─ Rationale: Schema designer knows what's important

Step 3: Truncate Rows
        ├─ Preserve ALL remaining columns
        ├─ Keep first N rows (head truncation)
        └─ Rationale: Partial data is better than no data
```

#### Priority Mapping

```json
// Schema definition
{
  "tables": [{
    "name": "orders",
    "columns": [
      {"name": "order_id", "priority": "high"},      // NEVER remove
      {"name": "total", "priority": "high"},         // NEVER remove
      {"name": "customer_id", "priority": "high"},   // NEVER remove
      {"name": "created_at", "priority": "medium"},  // Remove if needed
      {"name": "internal_notes", "priority": "low"}, // Remove first
      {"name": "debug_data", "priority": "low"}      // Remove first
    ]
  }]
}
```

#### Truncation Decision Tree

```
                    Is DataFrame > max_bytes?
                            │
                    ┌───────┴───────┐
                    │ NO            │ YES
                    ▼               ▼
               Return as-is    Step 1: Remove sparse
                                      │
                               Still > max_bytes?
                                      │
                               ┌──────┴──────┐
                               │ NO          │ YES
                               ▼             ▼
                          Return        Step 2: Remove low/medium priority
                                              │
                                       Still > max_bytes?
                                              │
                                       ┌──────┴──────┐
                                       │ NO          │ YES
                                       ▼             ▼
                                   Return        Step 3: Truncate rows
                                                        │
                                                        ▼
                                                    Return
```

### 4.5 Retrieval Modules

#### Base Interface

```python
class BaseRetriever(ABC):
    name: str
    
    @abstractmethod
    async def retrieve(self, query: str, context: dict) -> RetrievalResult:
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        pass
```

#### SQL RAG Retriever

```
Flow:
1. Receive natural language query
2. Load schema context
3. Retrieve similar SQL examples (if indexed)
4. Generate SQL via LLM
5. Execute SQL against database
6. Apply truncation if needed
7. Return RetrievalResult
```

```python
async def retrieve(self, query: str, context: dict) -> RetrievalResult:
    # 1. Build prompt with schema
    prompt = self._build_sql_prompt(query, self.schema)
    
    # 2. Generate SQL
    sql = await self._generate_sql(prompt)
    
    # 3. Execute
    df = await self._execute_sql(sql)
    
    # 4. Truncate if needed
    truncated = self.truncator.truncate(df)
    
    # 5. Return
    return RetrievalResult(
        source="sql_rag",
        content=truncated.data,
        metadata={
            "sql": sql,
            "truncated": truncated.truncated,
            "original_rows": truncated.original_rows
        }
    )
```

#### Doc RAG Retriever

```
Flow:
1. Receive natural language query
2. Embed query
3. Search vector store
4. Retrieve top-k chunks
5. Assemble context
6. Return RetrievalResult
```

```python
async def retrieve(self, query: str, context: dict) -> RetrievalResult:
    # 1. Embed query
    embedding = await self.embedder.embed(query)
    
    # 2. Vector search
    chunks = await self.vector_store.search(embedding, top_k=5)
    
    # 3. Assemble
    content = "\n\n".join([c.text for c in chunks])
    
    return RetrievalResult(
        source="doc_rag",
        content=content,
        metadata={
            "num_chunks": len(chunks),
            "sources": [c.source for c in chunks]
        }
    )
```

#### Retriever Registry

```python
class RetrieverRegistry:
    """
    Central registry for all retrieval modules.
    Enables dynamic routing based on query type.
    """
    
    def __init__(self):
        self._retrievers: Dict[str, BaseRetriever] = {}
    
    def register(self, retriever: BaseRetriever):
        self._retrievers[retriever.name] = retriever
    
    def route(self, query: str, context: dict) -> BaseRetriever:
        """
        Select appropriate retriever based on query characteristics.
        """
        # Check if query mentions documents/policies
        doc_keywords = ["policy", "procedure", "document", "manual"]
        if any(kw in query.lower() for kw in doc_keywords):
            return self._retrievers.get("doc_rag")
        
        # Default to SQL for data queries
        return self._retrievers.get("sql_rag")
```

### 4.6 Session Management

#### Session Lifecycle

```
1. CREATE   → New conversation starts
2. LOAD     → Existing conversation resumed
3. UPDATE   → New message added
4. COMPACT  → History summarized (when tokens > threshold)
5. ARCHIVE  → Conversation ends (moved to cold storage)
```

#### Storage Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                      HOT PATH (Real-time)                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      REDIS                               │   │
│  │                                                          │   │
│  │  session:{id}     → Full session JSON (TTL: 1 hour)     │   │
│  │  scratchpad:{id}  → Current scratchpad (TTL: 1 hour)    │   │
│  │  lock:{id}        → Distributed lock for updates        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Cache miss / Write-through
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     WARM PATH (Persistent)                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    POSTGRESQL                            │   │
│  │                                                          │   │
│  │  sessions table:                                         │   │
│  │    - session_id (PK)                                     │   │
│  │    - user_id                                             │   │
│  │    - created_at, updated_at                              │   │
│  │    - total_tokens                                        │   │
│  │    - compaction_count                                    │   │
│  │    - metadata (JSONB)                                    │   │
│  │                                                          │   │
│  │  messages table:                                         │   │
│  │    - id (PK)                                             │   │
│  │    - session_id (FK)                                     │   │
│  │    - role                                                │   │
│  │    - content                                             │   │
│  │    - timestamp                                           │   │
│  │    - token_count                                         │   │
│  │    - metadata (JSONB)                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### Compaction Process

```
Trigger: session.total_tokens > COMPACTION_THRESHOLD (100k tokens)

Process:
1. Identify messages to summarize (all except last 5)
2. Generate summary via LLM
3. Create new session with:
   - Summary as system message
   - Last 5 messages preserved
4. Update compaction_count
5. Invalidate Redis cache

Result:
  Before: 50 messages, 120k tokens
  After:  6 messages (1 summary + 5 recent), ~20k tokens
```

---

## 5. Data Flow Patterns

### 5.1 Context Flow

```
┌─────────────┐    Schema      ┌─────────────┐
│   Schema    │───────────────▶│             │
│   Store     │                │             │
└─────────────┘                │             │
                               │  Scratchpad │
┌─────────────┐   SQL Result   │             │
│  SQL RAG    │───────────────▶│             │
│             │                │             │
└─────────────┘                │             │
                               │             │
┌─────────────┐   Doc Chunks   │             │     Assembled     ┌─────────────┐
│  Doc RAG    │───────────────▶│             │────────Context───▶│     LLM     │
│             │                │             │                   │             │
└─────────────┘                │             │                   └─────────────┘
                               │             │
┌─────────────┐   History      │             │
│   Session   │───────────────▶│             │
│             │                │             │
└─────────────┘                └─────────────┘
```

### 5.2 Token Flow Tracking

```python
# Every component tracks tokens

class TokenTracker:
    def __init__(self, budget: int):
        self.budget = budget
        self.used = 0
        self.breakdown = {}
    
    def add(self, component: str, tokens: int):
        self.used += tokens
        self.breakdown[component] = self.breakdown.get(component, 0) + tokens
    
    def remaining(self) -> int:
        return self.budget - self.used
    
    def report(self) -> dict:
        return {
            "total_budget": self.budget,
            "total_used": self.used,
            "remaining": self.remaining(),
            "breakdown": self.breakdown
        }

# Usage throughout the pipeline:
tracker = TokenTracker(budget=100000)
tracker.add("system_prompt", 500)
tracker.add("schema_context", 2000)
tracker.add("session_history", 5000)
tracker.add("sql_results", 1500)
# ... etc
```

### 5.3 Error Propagation

```
┌─────────────────────────────────────────────────────────────────┐
│                    Error Categories                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RECOVERABLE                    NON-RECOVERABLE                 │
│  ─────────────                  ─────────────────               │
│                                                                 │
│  • SQL syntax error             • Database connection lost      │
│    → Retry with fix             • LLM API timeout               │
│                                 • Token budget exceeded         │
│  • Empty result set             • Circular task dependency      │
│    → Return with message                                        │
│                                                                 │
│  • Guardrail soft fail          PARTIAL SUCCESS                 │
│    → Suggest clarification      ─────────────────               │
│                                                                 │
│  • Truncation applied           • Some tasks failed             │
│    → Return partial data          → Return successful results   │
│                                   → Note failures               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. State Management

### 6.1 State Locations

| State Type | Location | Lifetime | Purpose |
|------------|----------|----------|---------|
| Session | PostgreSQL + Redis | Persistent | Conversation history |
| Scratchpad | Redis | Per-request | Context assembly |
| Task Plan | Memory | Per-request | Execution orchestration |
| Events | SSE Stream | Per-request | Real-time updates |
| Schema Cache | Redis | 1 hour TTL | Avoid repeated DB introspection |

### 6.2 State Transitions

```
SESSION STATES:
  CREATED → ACTIVE → COMPACTED → ARCHIVED
              │          │
              ▼          ▼
           ACTIVE     ACTIVE
         (continue) (continue post-compaction)

TASK STATES:
  PENDING → RUNNING → COMPLETED
              │           │
              ▼           ▼
           FAILED     SKIPPED
                    (condition not met)

SCRATCHPAD ITEM STATES:
  PENDING → AVAILABLE → STALE
              │
              ▼
           EVICTED
         (token budget)
```

### 6.3 Concurrency Handling

```python
# Distributed lock for session updates
async def update_session_with_lock(session_id: str, update_fn):
    lock_key = f"lock:session:{session_id}"
    
    # Acquire lock
    acquired = await redis.set(lock_key, "1", nx=True, ex=30)
    if not acquired:
        raise ConcurrentModificationError("Session is being updated")
    
    try:
        session = await load_session(session_id)
        updated = update_fn(session)
        await save_session(updated)
        await invalidate_cache(session_id)
    finally:
        await redis.delete(lock_key)
```

---

## 7. Event System

### 7.1 Event Types

```python
class EventType(Enum):
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
    
    # Tokens
    TOKEN_COUNT = "token.count"
```

### 7.2 Event Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Event Producer (Agent)                       │
│                                                                 │
│  await emitter.emit(Event(type=TURN_STARTED, ...))             │
│  await emitter.emit(Event(type=ITEM_STARTED, ...))             │
│  await emitter.emit(Event(type=ITEM_COMPLETED, ...))           │
│  await emitter.emit(Event(type=TURN_COMPLETED, ...))           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Event Emitter                                │
│                                                                 │
│  subscribers: List[asyncio.Queue]                               │
│                                                                 │
│  async def emit(event):                                         │
│      for queue in subscribers:                                  │
│          await queue.put(event)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Event Consumers                              │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ SSE Stream  │  │ Logger      │  │ Metrics     │             │
│  │ (to client) │  │ (audit)     │  │ (monitoring)│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 SSE Message Format

```javascript
// Client receives these events:

// Turn started
data: {"type": "turn.started", "turn_id": "turn_123", "timestamp": "2024-..."}

// SQL execution started
data: {"type": "item.started", "item": {"id": "item_1", "type": "sql_execution", "description": "Generating SQL query", "status": "in_progress"}}

// SQL execution completed
data: {"type": "item.completed", "item": {"id": "item_1", "type": "sql_execution", "description": "Retrieved 5 rows", "status": "completed", "output": {"rows": 5}}}

// Turn completed with response
data: {"type": "turn.completed", "turn_id": "turn_123", "response": {...}, "usage": {"prompt_tokens": 1500, "completion_tokens": 300}}
```

### 7.4 Event-Driven UI Updates

```javascript
// Client-side event handling
const eventSource = new EventSource('/chat/stream?session_id=abc');

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch (data.type) {
        case 'turn.started':
            showLoadingIndicator();
            break;
            
        case 'item.started':
            addProgressItem(data.item.description);
            break;
            
        case 'item.completed':
            updateProgressItem(data.item.id, 'completed');
            break;
            
        case 'turn.completed':
            hideLoadingIndicator();
            renderResponse(data.response);
            break;
            
        case 'turn.failed':
            showError(data.error);
            break;
    }
};
```

---

## 8. Error Handling

### 8.1 Error Categories

```python
class BaseAgentError(Exception):
    """Base class for all agent errors"""
    error_code: str
    recoverable: bool
    suggestions: List[str]

class InputValidationError(BaseAgentError):
    """Guardrail failures"""
    error_code = "INPUT_VALIDATION_FAILED"
    recoverable = True

class SQLGenerationError(BaseAgentError):
    """Failed to generate valid SQL"""
    error_code = "SQL_GENERATION_FAILED"
    recoverable = True

class SQLExecutionError(BaseAgentError):
    """SQL execution failed"""
    error_code = "SQL_EXECUTION_FAILED"
    recoverable = True  # May retry with fixed SQL

class TokenBudgetExceededError(BaseAgentError):
    """Context too large"""
    error_code = "TOKEN_BUDGET_EXCEEDED"
    recoverable = False

class CircularDependencyError(BaseAgentError):
    """Task plan has cycles"""
    error_code = "CIRCULAR_DEPENDENCY"
    recoverable = False

class RetrievalError(BaseAgentError):
    """Retrieval module failed"""
    error_code = "RETRIEVAL_FAILED"
    recoverable = True
```

### 8.2 Error Handling Strategy

```python
async def execute_with_retry(
    func: Callable,
    max_retries: int = 3,
    retry_on: List[Type[Exception]] = [SQLExecutionError]
):
    """
    Execute function with exponential backoff retry for recoverable errors.
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except tuple(retry_on) as e:
            last_error = e
            if not e.recoverable or attempt == max_retries - 1:
                raise
            
            # Exponential backoff
            wait_time = 2 ** attempt
            await asyncio.sleep(wait_time)
            
            # Apply fix if available
            if hasattr(e, 'suggested_fix'):
                apply_fix(e.suggested_fix)
    
    raise last_error
```

### 8.3 Error Response Format

```python
# All errors return ErrorView
ErrorView:
    text: str                  # Human-readable message
    view_type: "error"
    error_code: str            # Machine-readable code
    error_details: dict        # Additional context
    suggestions: List[str]     # How to fix

# Example
ErrorView(
    text="I couldn't understand what 'this' refers to. Could you be more specific?",
    view_type="error",
    error_code="AMBIGUOUS_REFERENCE",
    error_details={
        "ambiguous_term": "this",
        "has_session_history": False
    },
    suggestions=[
        "Try: 'Show me the orders from the previous query'",
        "Or: 'Show me the customer orders'"
    ]
)
```

### 8.4 Graceful Degradation

```python
async def process_with_fallback(query: str, context: dict) -> ResponseView:
    """
    Attempt primary flow, fall back gracefully on failures.
    """
    try:
        # Primary: Full task planning + execution
        plan = await task_planner.create_plan(query, context)
        results = await task_planner.execute_plan(plan)
        return build_response(results)
    
    except TaskPlanningError:
        # Fallback 1: Direct single retrieval
        try:
            retriever = registry.route(query, context)
            result = await retriever.retrieve(query, context)
            return build_simple_response(result)
        
        except RetrievalError:
            # Fallback 2: Return what we know
            return BaseView(
                text="I encountered an issue processing your request. "
                     "Please try rephrasing your question.",
                view_type="base"
            )
```

---

## 9. Extension Points

### 9.1 Adding a New Retrieval Module

```python
# 1. Create retriever class
class APIRetriever(BaseRetriever):
    name = "api_rag"
    
    def __init__(self, api_client):
        self.client = api_client
    
    async def retrieve(self, query: str, context: dict) -> RetrievalResult:
        # Your implementation
        response = await self.client.search(query)
        return RetrievalResult(
            source="api_rag",
            content=response.data,
            metadata={"api_call_id": response.id}
        )
    
    def get_capabilities(self) -> List[str]:
        return ["external_api", "real_time_data"]

# 2. Register
registry.register(APIRetriever(api_client))

# 3. Update routing logic (optional)
# The registry.route() method can be extended to recognize API queries
```

### 9.2 Adding a New Guardrail

```python
# 1. Create guardrail class
class ProfanityGuardrail(InputGuardrail):
    name = "profanity_check"
    run_in_parallel = True  # Can run alongside others
    
    async def check(self, user_input: str, context: dict) -> GuardrailResult:
        # Your implementation
        if contains_profanity(user_input):
            return GuardrailResult(
                passed=False,
                reason="Query contains inappropriate language",
                suggested_response="Please rephrase your question."
            )
        return GuardrailResult(passed=True)

# 2. Add to guardrail runner
runner = GuardrailRunner(
    input_guardrails=[
        AmbiguousReferenceGuardrail(),
        OutOfContextGuardrail(llm),
        MisleadingQueryGuardrail(llm),
        ProfanityGuardrail()  # New
    ],
    output_guardrails=[...]
)
```

### 9.3 Adding a New Response View

```python
# 1. Define new view
class MapView(DataView):
    """Response with geographic visualization"""
    view_type: Literal["map"] = "map"
    map_spec: dict              # Leaflet/Mapbox config
    center: Tuple[float, float] # Lat, lng
    zoom: int

# 2. Update ResponseBuilder
class ResponseBuilder:
    @staticmethod
    def map(text: str, df_dict: dict, map_config: dict) -> MapView:
        return MapView(
            text=text,
            data_dict=df_dict,
            columns=list(df_dict.keys()),
            row_count=len(df_dict[next(iter(df_dict))]),
            map_spec=map_config,
            center=map_config.get("center", (0, 0)),
            zoom=map_config.get("zoom", 10)
        )

# 3. Client handles new view_type
```

### 9.4 Adding a New Event Type

```python
# 1. Extend EventType enum
class EventType(str, Enum):
    # ... existing ...
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"

# 2. Emit in relevant code
async def get_cached_or_fetch(key: str):
    cached = await redis.get(key)
    if cached:
        await emitter.emit(Event(type=EventType.CACHE_HIT, ...))
        return cached
    
    await emitter.emit(Event(type=EventType.CACHE_MISS, ...))
    data = await fetch_fresh(key)
    await redis.set(key, data)
    return data
```

---

## 10. Operational Patterns

### 10.1 Monitoring and Observability

```
┌─────────────────────────────────────────────────────────────────┐
│                      Metrics to Track                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LATENCY                        THROUGHPUT                      │
│  ─────────                      ──────────                      │
│  • Total request time           • Requests per second           │
│  • SQL generation time          • Concurrent sessions           │
│  • SQL execution time           • Messages per session          │
│  • LLM call time                                                │
│                                                                 │
│  QUALITY                        RESOURCES                       │
│  ─────────                      ───────────                     │
│  • Guardrail trigger rate       • Token usage per request       │
│  • Truncation frequency         • Redis memory                  │
│  • SQL error rate               • PostgreSQL connections        │
│  • Compaction frequency         • LLM API costs                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Logging Strategy

```python
# Structured logging throughout
import structlog

logger = structlog.get_logger()

async def process_request(request):
    logger.info("request_started",
        session_id=request.session_id,
        query_length=len(request.message)
    )
    
    try:
        result = await agent.run(request)
        logger.info("request_completed",
            session_id=request.session_id,
            response_type=result.view_type,
            tokens_used=result.metadata.get("tokens")
        )
        return result
    except Exception as e:
        logger.error("request_failed",
            session_id=request.session_id,
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise
```

### 10.3 Health Checks

```python
@app.get("/health")
async def health_check():
    checks = {}
    
    # PostgreSQL
    try:
        await postgres_pool.fetchval("SELECT 1")
        checks["postgres"] = "healthy"
    except Exception as e:
        checks["postgres"] = f"unhealthy: {e}"
    
    # Redis
    try:
        await redis.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {e}"
    
    # LLM API
    try:
        await llm.models.list()
        checks["llm_api"] = "healthy"
    except Exception as e:
        checks["llm_api"] = f"unhealthy: {e}"
    
    # Target database
    try:
        await target_db.execute("SELECT 1")
        checks["target_db"] = "healthy"
    except Exception as e:
        checks["target_db"] = f"unhealthy: {e}"
    
    overall = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"
    
    return {"status": overall, "checks": checks}
```

### 10.4 Configuration Management

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    postgres_url: str
    redis_url: str
    target_db_url: str
    
    # LLM
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    default_model: str = "gpt-4o"
    
    # Limits
    max_tokens_per_request: int = 100000
    max_rows_per_query: int = 500
    session_compaction_threshold: int = 100000
    
    # Cache
    redis_ttl_seconds: int = 3600
    
    # Features
    enable_doc_rag: bool = True
    enable_chart_generation: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Appendix A: Quick Reference

### A.1 Request Processing Checklist

```
□ Session loaded (Redis → PostgreSQL fallback)
□ Event emitter initialized
□ Input guardrails passed
□ Task plan generated
□ Each task executed with events
□ Results added to scratchpad
□ Response generated
□ Output guardrails passed
□ Session updated
□ Response sent
```

### A.2 Common Failure Points

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| Slow responses | LLM latency | Check model, reduce context |
| "Ambiguous reference" errors | Missing session history | Check Redis cache, PostgreSQL |
| SQL errors | Schema mismatch | Refresh schema cache |
| Truncation too aggressive | Wrong priorities in schema | Review schema priority settings |
| Token budget exceeded | Large SQL results | Adjust truncation config |

### A.3 Key File Locations

```
agents/
├── models/agent_config.py       # Core data models
├── prompts/system_prompt.py     # Dynamic system prompt
├── planner/task_planner.py      # Task decomposition
├── scratchpad/context_tracker.py # Context assembly
├── response/views.py            # Response types
├── truncation/dataframe_truncator.py # Data reduction
├── guardrails/                  # Input/output validation
├── retrieval/                   # SQL RAG, Doc RAG
├── events/protocol.py           # Event system
└── session/                     # Persistence layer
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Agent** | The orchestrating component that processes user queries |
| **Scratchpad** | Working memory for assembling context |
| **Task Plan** | DAG of atomic operations to fulfill a query |
| **Guardrail** | Validation component for inputs or outputs |
| **Retriever** | Module that fetches data (SQL, documents, etc.) |
| **View** | Response format (BaseView, DataView, ChartView) |
| **Session** | Persistent conversation state |
| **Compaction** | Summarizing old messages to reduce tokens |
| **Truncation** | Reducing large SQL results intelligently |
| **Event** | Real-time notification of system activity |

---

*Document Version: 1.0*
*Last Updated: [Current Date]*