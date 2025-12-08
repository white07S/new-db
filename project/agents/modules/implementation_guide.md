---

## Phase 0: Foundation 
**Goal:** Pure data models and interfaces that everything else depends on

```
Implementation Order:
1. agents/models/agent_config.py      → Pydantic models
2. agents/response/views.py           → BaseView, DataView, ChartView
3. agents/session/models.py           → Session, Message models
4. agents/events/protocol.py          → Event, EventItem models (just the models, not emitter)
```

**Why first:** These are pure data structures with zero dependencies. Every other module imports from these. Get the contracts right before building implementations.

**Testing at this phase:**
```python
# tests/test_models.py
def test_response_view_serialization():
    view = DataView(text="test", data_dict={"a": [1,2,3]}, columns=["a"], row_count=3)
    assert view.model_dump()["view_type"] == "data"

def test_task_dependency_validation():
    task = Task(id="1", task_description="x", tool_feature="sql", reasoning="y", depends_on=["invalid"])
    # Should serialize cleanly
```

---

## Phase 1: Storage Layer 
**Goal:** Persistent state before any LLM calls

```
Implementation Order:
1. agents/session/postgres_store.py   → Database schema + CRUD
2. agents/session/redis_cache.py      → Cache layer
3. config/settings.py                 → Environment config
```

**Why second:** Session management is the backbone. Without it, you can't test multi-turn conversations, can't persist state, can't debug what happened. Build this before LLM integration.

**Testing at this phase:**
```python
# tests/test_session_store.py
@pytest.mark.asyncio
async def test_session_crud(postgres_store):
    session = await postgres_store.create_session("user_1")
    assert session.session_id
    
    await postgres_store.add_message(session.session_id, Message(role="user", content="hello"))
    
    loaded = await postgres_store.get_session(session.session_id)
    assert len(loaded.messages) == 1

# tests/test_redis_cache.py
@pytest.mark.asyncio
async def test_cache_invalidation(redis_cache):
    session = Session(session_id="test", user_id="u1")
    await redis_cache.cache_session(session)
    
    cached = await redis_cache.get_cached_session("test")
    assert cached.user_id == "u1"
    
    await redis_cache.invalidate("test")
    assert await redis_cache.get_cached_session("test") is None
```

**Milestone:** You can create sessions, store messages, cache state. No LLM yet.

---

## Phase 2: Single Retrieval Path - SQL RAG 
**Goal:** One complete end-to-end path working

```
Implementation Order:
1. agents/retrieval/base.py           → Abstract BaseRetriever
2. agents/retrieval/sql_rag.py        → SQL generation + execution
3. agents/retrieval/registry.py       → Simple registry (just SQL for now)
4. agents/prompts/system_prompt.py    → Basic system prompt (no dynamic features yet)
```

**Why this order:** 
- `base.py` defines the interface all retrievers follow
- `sql_rag.py` is your core functionality - get this working first
- Registry is simple when you only have one retriever
- System prompt can be static initially

**Testing at this phase:**
```python
# tests/test_sql_rag.py
@pytest.mark.asyncio
async def test_sql_generation(llm_client, test_db):
    retriever = SQLRAGRetriever(test_db, TEST_SCHEMA, embedder)
    
    result = await retriever.retrieve(
        "Show total orders by customer",
        {}
    )
    
    assert result.source == "sql_rag"
    assert "customer" in str(result.metadata.get("sql", "")).lower()
    assert isinstance(result.content, dict)  # DataFrame as dict

# Integration test - actually runs SQL
@pytest.mark.asyncio
async def test_sql_execution_integration(llm_client, real_db):
    retriever = SQLRAGRetriever(real_db, REAL_SCHEMA, embedder)
    result = await retriever.retrieve("Count all orders", {})
    
    assert result.content  # Got actual data back
```

**Milestone:** User types query → SQL generated → SQL executed → Data returned. One complete path.

---

## Phase 3: Event System 
**Goal:** Observability before adding complexity

```
Implementation Order:
1. agents/events/protocol.py          → EventEmitter (the async logic)
2. api/main.py                        → FastAPI with SSE endpoint
```

**Why now:** Before adding guardrails, task planner, etc., you need visibility into what's happening. Events let you debug the system as you add complexity.

**Testing at this phase:**
```python
# tests/test_events.py
@pytest.mark.asyncio
async def test_event_emission():
    emitter = EventEmitter("thread_1")
    queue = emitter.subscribe()
    
    item_id = await emitter.start_item(ItemType.SQL_EXECUTION, "Running query")
    
    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event.type == EventType.ITEM_STARTED
    assert event.item.description == "Running query"

# Integration: wrap SQL RAG with events
@pytest.mark.asyncio
async def test_sql_rag_with_events(llm_client, test_db):
    emitter = EventEmitter("test")
    queue = emitter.subscribe()
    
    # SQL RAG now emits events
    retriever = SQLRAGRetriever(test_db, schema, embedder, emitter=emitter)
    await retriever.retrieve("Count orders", {})
    
    events = []
    while not queue.empty():
        events.append(await queue.get())
    
    assert any(e.item.type == ItemType.SQL_EXECUTION for e in events)
```

**Milestone:** Every operation emits events. You can see what's happening in real-time.

---

## Phase 4: Guardrails 
**Goal:** Input/output validation before complex orchestration

```
Implementation Order:
1. agents/guardrails/input_guardrails.py   → One guardrail at a time:
   a. AmbiguousReferenceGuardrail          (pattern matching, no LLM)
   b. OutOfContextGuardrail                (LLM-based)
   c. MisleadingQueryGuardrail             (LLM-based)
2. agents/guardrails/output_guardrails.py  → NoFabricatedDataGuardrail
```

**Why this order within guardrails:**
- `AmbiguousReferenceGuardrail` is pure Python pattern matching - test without LLM
- `OutOfContextGuardrail` is simpler LLM call (binary classification)
- `MisleadingQueryGuardrail` is more nuanced - do last

**Testing at this phase:**
```python
# tests/test_guardrails.py

# Test 1: No LLM needed
def test_ambiguous_reference_patterns():
    guardrail = AmbiguousReferenceGuardrail()
    
    # Sync check for patterns
    assert guardrail._has_ambiguous_pattern("show me this")
    assert guardrail._has_ambiguous_pattern("what about that one")
    assert not guardrail._has_ambiguous_pattern("show all orders")

# Test 2: LLM-based but isolated
@pytest.mark.asyncio
async def test_out_of_context_classification(llm_client):
    guardrail = OutOfContextGuardrail(llm_client)
    
    result = await guardrail.check(
        "What's the weather?",
        {"schema_summary": "orders, customers, products"}
    )
    assert not result.passed

# Test 3: Integration with retrieval
@pytest.mark.asyncio
async def test_guardrail_blocks_bad_query(full_pipeline):
    response = await full_pipeline.process("Show me this thing")
    
    assert response.view_type == "error"
    assert "clarify" in response.text.lower()
```

**Milestone:** Bad queries are caught before wasting LLM calls on SQL generation.

---

## Phase 5: Scratchpad 
**Goal:** Context tracking between operations

```
Implementation Order:
1. agents/scratchpad/context_tracker.py   → Scratchpad class
2. Integrate with SQLRAGRetriever         → Auto-add results to scratchpad
3. Integrate with Session                 → Load prior context
```

**Why now:** Scratchpad needs retrieval (to track results) and session (to load history). Both exist now.

**Testing at this phase:**
```python
# tests/test_scratchpad.py
def test_scratchpad_gap_detection():
    scratchpad = Scratchpad(session_id="test")
    
    # Nothing added yet
    gaps = scratchpad.identify_gaps(["schema_context", "query_results"])
    assert len(gaps) == 2
    
    # Add one
    scratchpad.add_context("schema_context", "CREATE TABLE...", "sql_rag", 100)
    
    gaps = scratchpad.identify_gaps(["schema_context", "query_results"])
    assert len(gaps) == 1
    assert gaps[0].key == "query_results"

# Integration
@pytest.mark.asyncio
async def test_scratchpad_with_retrieval(sql_rag, scratchpad):
    result = await sql_rag.retrieve("Count orders", {})
    scratchpad.add_context("query_results", result.content, "sql_rag", result.token_count)
    
    assert scratchpad.is_complete(["query_results"])
    assert "query_results" in scratchpad.assemble_context()
```

**Milestone:** System tracks what context it has, identifies gaps, assembles context for LLM.

---

## Phase 6: Truncation Policy 
**Goal:** Handle large SQL results gracefully

```
Implementation Order:
1. agents/truncation/dataframe_truncator.py   → All three steps
2. Integrate with SQLRAGRetriever             → Auto-truncate large results
3. Update DataView                            → Include truncation metadata
```

**Why now:** You have working SQL RAG. Now handle edge case of large results.

**Testing at this phase:**
```python
# tests/test_truncation.py

# Unit tests for each step
def test_step1_sparse_column_removal():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "sparse": [None, None, None, None, "x"]  # 80% null
    })
    truncator = DataFrameTruncator(SCHEMA)
    result = truncator._step1_remove_sparse(df)
    assert "sparse" not in result.columns

def test_step2_priority_removal():
    df = pd.DataFrame({
        "high_col": range(100),
        "low_col": range(100)
    })
    schema = {"tables": [{"columns": [
        {"name": "high_col", "priority": "high"},
        {"name": "low_col", "priority": "low"}
    ]}]}
    
    truncator = DataFrameTruncator(schema, TruncationConfig(max_bytes=50))
    result = truncator._step2_remove_by_priority(df)
    
    assert "high_col" in result.columns
    # low_col may be removed

def test_step3_row_truncation():
    df = pd.DataFrame({"a": range(1000)})
    truncator = DataFrameTruncator({}, TruncationConfig(max_rows=100))
    result = truncator._step3_truncate_rows(df)
    assert len(result) == 100

# Integration
@pytest.mark.asyncio
async def test_large_result_truncation(sql_rag_with_truncation, large_table_db):
    result = await sql_rag_with_truncation.retrieve("SELECT * FROM big_table", {})
    
    assert result.metadata["truncated"]
    assert result.metadata["original_rows"] > len(result.content[next(iter(result.content))])
```

**Milestone:** Large SQL results are intelligently truncated without losing critical data.

---

## Phase 7: Task Planner 
**Goal:** Multi-step query decomposition

```
Implementation Order:
1. agents/planner/task_planner.py    → Task, TaskPlan models
2. TaskPlannerEngine.create_plan()   → LLM-based plan generation
3. TaskPlannerEngine.execute_plan()  → Dependency-aware execution
4. Integrate with retrieval          → Plans call retrieval modules
```

**Why last among core modules:** Task planner orchestrates everything else. It needs:
- Retrieval modules (to call)
- Events (to emit progress)
- Scratchpad (to track context)
- Guardrails (to validate sub-queries)

**Testing at this phase:**
```python
# tests/test_task_planner.py

# Unit: Dependency resolution
def test_execution_order():
    plan = TaskPlan(
        plan_id="test",
        tasks=[
            Task(id="1", task_description="a", tool_feature="sql", reasoning="r", depends_on=[]),
            Task(id="2", task_description="b", tool_feature="sql", reasoning="r", depends_on=[]),
            Task(id="3", task_description="c", tool_feature="sql", reasoning="r", depends_on=["1", "2"]),
        ],
        execution_mode=ExecutionMode.PARALLEL
    )
    
    waves = plan.get_execution_order()
    assert len(waves) == 2
    assert {t.id for t in waves[0]} == {"1", "2"}  # Parallel
    assert waves[1][0].id == "3"  # Sequential after

# Integration: LLM generates valid plan
@pytest.mark.asyncio
async def test_plan_generation(llm_client):
    planner = TaskPlannerEngine(llm_client)
    
    plan = await planner.create_plan(
        "Find top customers AND their recent orders AND create a chart",
        context
    )
    
    assert len(plan.tasks) >= 3
    # Chart task should depend on data tasks
    chart_task = next(t for t in plan.tasks if "chart" in t.tool_feature)
    assert len(chart_task.depends_on) > 0

# Full integration
@pytest.mark.asyncio
async def test_plan_execution_e2e(full_system):
    result = await full_system.process(
        "Find top 5 products by revenue and show a bar chart"
    )
    
    assert result.view_type == "chart"
    assert result.chart_spec  # Vega-Lite generated
```

**Milestone:** Complex queries decompose into parallel/sequential tasks automatically.

---

## Phase 8: Second Retrieval - Doc RAG 
**Goal:** Prove the abstraction works

```
Implementation Order:
1. agents/retrieval/doc_rag.py       → Document retrieval
2. Update registry                   → Register both retrievers
3. Update system prompt              → Route based on query type
```

**Why last:** By now your retrieval abstraction is battle-tested with SQL RAG. Adding Doc RAG should be straightforward - if it's not, your abstraction is wrong.

**Testing at this phase:**
```python
# tests/test_doc_rag.py
@pytest.mark.asyncio
async def test_doc_retrieval(vector_store):
    retriever = DocRAGRetriever(vector_store, embedder)
    result = await retriever.retrieve("What is the refund policy?", {})
    
    assert result.source == "doc_rag"
    assert "refund" in result.content.lower()

# Integration: System routes correctly
@pytest.mark.asyncio
async def test_retrieval_routing(full_system):
    # SQL query
    sql_result = await full_system.process("Count all orders")
    assert sql_result.metadata.get("source") == "sql_rag"
    
    # Doc query
    doc_result = await full_system.process("What are the shipping policies?")
    assert doc_result.metadata.get("source") == "doc_rag"
```

**Milestone:** Two retrieval methods working, system routes automatically.

---

## Phase 9: Session Compaction 
**Goal:** Long conversation support

```
Implementation Order:
1. agents/session/compaction.py      → Summarization logic
2. Integrate with session store      → Auto-compact when threshold hit
```

**Testing:**
```python
@pytest.mark.asyncio
async def test_compaction_preserves_recent(llm_client):
    compactor = SessionCompactor(llm_client)
    
    session = Session(
        session_id="test",
        user_id="u1",
        messages=[Message(role="user", content=f"msg {i}") for i in range(20)],
        total_tokens=150000  # Over threshold
    )
    
    compacted = await compactor.compact(session)
    
    assert compacted.compaction_count == 1
    assert len(compacted.messages) < 20
    assert compacted.messages[-1].content == "msg 19"  # Recent preserved
```

---


---

## Key Principles Behind This Order

1. **Data models first** - Get contracts right before implementations
2. **Storage before compute** - You need state to test anything meaningful
3. **One vertical slice early** - SQL RAG end-to-end by day 10
4. **Observability before complexity** - Events before guardrails/planner
5. **Validation before orchestration** - Guardrails before task planner
6. **Prove abstractions late** - Doc RAG tests if retrieval abstraction is right

---

## Integration Test Checkpoints

After each phase, run this integration suite:

```python
# tests/integration/test_checkpoints.py

@pytest.mark.asyncio
async def test_phase2_checkpoint():
    """After SQL RAG: query → SQL → data"""
    result = await sql_rag.retrieve("Count orders", {})
    assert isinstance(result.content, dict)

@pytest.mark.asyncio  
async def test_phase4_checkpoint():
    """After Guardrails: bad query → blocked"""
    result = await pipeline.process("Show me that thing")
    assert result.view_type == "error"

@pytest.mark.asyncio
async def test_phase7_checkpoint():
    """After Task Planner: complex query → decomposed → executed"""
    result = await pipeline.process("Top customers AND their orders AND chart")
    assert result.view_type == "chart"

@pytest.mark.asyncio
async def test_phase8_checkpoint():
    """After Doc RAG: routes correctly"""
    sql_r = await pipeline.process("Count orders")
    doc_r = await pipeline.process("What's the return policy?")
    assert sql_r.metadata["source"] != doc_r.metadata["source"]
```
