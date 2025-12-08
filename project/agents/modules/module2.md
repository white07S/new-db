Module 2: Task Planner with CRUD
Why this approach
WrenAI's pipeline orchestration demonstrates that complex queries often require parallel retrieval followed by sequential analysis. github The task planner enforces execution order while enabling parallelism where dependencies allow.
Code Citation
File: WrenAI/wren-ai-service/src/web/v1/services/ask.py
python# Parallel retrieval pattern
sql_samples_task, instructions_task = await asyncio.gather(
    self._pipelines["sql_pairs_retrieval"].run(...),
    self._pipelines["instructions_retrieval"].run(...),
)
Implementation
python# agents/planner/task_planner.py
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum
import asyncio

class ExecutionMode(str, Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"

class Task(BaseModel):
    """Single executable task in the plan"""
    id: str
    task_description: str
    tool_feature: str                           # Tool to execute
    reasoning: str                              # Why this task exists
    depends_on: list[str] = Field(default_factory=list)  # Task IDs
    condition: str | None = None                # For conditional execution
    priority: int = 0                           # Lower = higher priority
    
class TaskPlan(BaseModel):
    """Execution plan with dependency graph"""
    plan_id: str
    tasks: list[Task]
    execution_mode: ExecutionMode
    
    def get_execution_order(self) -> list[list[Task]]:
        """Returns tasks grouped by execution wave (parallel within wave)"""
        executed = set()
        waves = []
        remaining = {t.id: t for t in self.tasks}
        
        while remaining:
            # Find tasks with all dependencies satisfied
            wave = [
                t for t in remaining.values()
                if all(dep in executed for dep in t.depends_on)
            ]
            if not wave:
                raise ValueError("Circular dependency detected")
            
            wave.sort(key=lambda t: t.priority)
            waves.append(wave)
            
            for t in wave:
                executed.add(t.id)
                del remaining[t.id]
        
        return waves

class TaskPlannerEngine:
    """Creates and manages task execution plans"""
    
    async def create_plan(self, user_query: str, context: AgentContext) -> TaskPlan:
        """LLM-generated task plan from natural language"""
        # Use structured outputs for reliable plan generation
        plan_response = await self.llm.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": user_query}
            ],
            response_format=TaskPlan
        )
        return plan_response
    
    async def execute_plan(self, plan: TaskPlan, tool_registry: dict) -> dict:
        """Execute plan respecting dependencies"""
        results = {}
        
        for wave in plan.get_execution_order():
            # Execute wave in parallel
            wave_tasks = [
                self._execute_task(task, tool_registry, results)
                for task in wave
            ]
            wave_results = await asyncio.gather(*wave_tasks, return_exceptions=True)
            
            for task, result in zip(wave, wave_results):
                results[task.id] = result
        
        return results
    
    async def _execute_task(self, task: Task, registry: dict, prior_results: dict):
        """Execute single task with context from prior results"""
        tool = registry.get(task.tool_feature)
        if not tool:
            raise ValueError(f"Unknown tool: {task.tool_feature}")
        
        # Inject prior results as context
        context = {dep: prior_results[dep] for dep in task.depends_on}
        return await tool.execute(task.task_description, context)
Example Query Decomposition
User Query: "Find the top 5 customers by revenue AND their most purchased products, then create a chart"

TaskPlan:
├── Task 1: "Find top 5 customers by revenue" (sql_rag, parallel)
├── Task 2: "Find most purchased products per customer" (sql_rag, parallel)
├── Task 3: "Analyze customer-product relationship" (depends_on: [1, 2], sequential)
└── Task 4: "Generate Vega-Lite chart" (depends_on: [3], sequential)