
import asyncio
import json
import shutil
import os
from typing import Optional, Dict, Any

from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    SystemPromptTemplate,
    SQLOutputParser,
)
from dbgpt.core.awel import (
    DAG,
    InputOperator,
    InputSource,
    JoinOperator,
    MapOperator,
)
from dbgpt.core.operators import PromptBuilderOperator, RequestBuilderOperator
from dbgpt.model.operators import LLMOperator
from dbgpt_ext.rag import ChunkParameters
from dbgpt_ext.rag.operators.db_schema import (
    DBSchemaAssemblerOperator,
)
from dbgpt_ext.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig

from .connector import CustomSQLiteConnector
from .retriever import InMemoryBM25Retriever
from .agent import ScratchpadSchemaAgent
from .operators import RobustSQLOperator
from .chart_operators import ChartGenerationOperator
from . import logger

from dbgpt.rag.embedding import DefaultEmbeddingFactory
from dbgpt.model.proxy import OpenAILLMClient

# --- Configuration Constants (Can be overridden via kwargs if needed) ---
# Keeping them close for self-containment or passed in init

RESPONSE_FORMAT_SIMPLE = {
    "thoughts": "thoughts summary to say to user",
    "sql": "SQL Query to run",
}

SYSTEM_PROMPT = """You are a database expert. Please answer the user's question based on the database selected by the user and some of the available table structure definitions of the database.
Database name:
    {db_name}
Table structure definition:
    {table_info}
    
Constraint:
1.Please understand the user's intention based on the user's question, and use the given table structure definition to create a grammatically correct {dialect} sql. If sql is not required, answer the user's question directly.
2.You can only use the tables provided in the table structure information to generate sql. If you cannot generate sql based on the provided table structure, please say: "The table structure information provided is not enough to generate sql queries." It is prohibited to fabricate information at will.
3.Please be careful not to mistake the relationship between tables and columns when generating SQL.
4.Please check the correctness of the SQL and ensure that the query performance is optimized under correct conditions.
 
User Question:
    {user_input}
Please think step by step and respond according to the following JSON format:
    {response}
Ensure the response is correct json and can be parsed by Python json.loads.
"""

class SQLRAGPipeline:
    def __init__(
        self, 
        openai_api_key: str, 
        llm_model: str, 
        embedding_model: str,
        sqlite_path: str,
        vector_store_path: str
    ):
        self.openai_api_key = openai_api_key
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.sqlite_path = sqlite_path
        self.vector_store_path = vector_store_path
        
        # Clean up vector store to avoid duplicates/errors, MUST be done BEFORE initializing ChromaStore
        if os.path.exists(self.vector_store_path):
            shutil.rmtree(self.vector_store_path)
        
        # Initialize Components
        self.embeddings = DefaultEmbeddingFactory.openai(
             api_key=openai_api_key,
             model_name=embedding_model,
        )
        
        self.db_conn = CustomSQLiteConnector.from_file_path(sqlite_path)
        
        # Initialize Vector Store
        self.vector_store = ChromaStore(
            ChromaVectorConfig(persist_path=vector_store_path),
            name="ltm_vector_store",
            embedding_fn=self.embeddings,
        )
        
        # Initialize LLM Client
        self.llm_client = OpenAILLMClient(model=llm_model, api_key=openai_api_key)
        
        # Initialize Agent Helpers (to be set up in build_index or lazily)
        self.bm25_retriever = None
        self.agent = None

    async def build_index(self):
        """Builds valid vector store index and initializes retrievers."""
        # Directory cleanup moved to __init__ to avoid breaking open Chroma connection

        with DAG("load_schema_dag") as load_schema_dag:
            input_task = InputOperator.dummy_input()
            assembler_task = DBSchemaAssemblerOperator(
                connector=self.db_conn,
                table_vector_store_connector=self.vector_store,
                chunk_parameters=ChunkParameters(
                    chunk_strategy="CHUNK_BY_SEPARATOR",
                    separator="WARNING_DO_NOT_SPLIT",
                    enable_merge=False,
                )
            )
            input_task >> assembler_task
        
        # Run Indexing
        chunks = await assembler_task.call()
        
        # Initialize Retrievers
        self.bm25_retriever = InMemoryBM25Retriever(chunks, top_k=3)
        
        # We need a temporary retriever for the agent to access vector store
        # In main.py it relied on 'retriever_task' from a dag, but we can access vector_store directly?
        # Actually ScratchpadSchemaAgent expects an object with 'aretrieve' method.
        # We can wrap the vector_store in a simple adapter or use the DBSchemaRetrieverOperator logic manually.
        # For simplicity, let's create a dummy adapter since the Agent calls `vector_retriever.aretrieve(query)`
        
        class VectorAdapter:
            def __init__(self, store):
                self.store = store
            async def aretrieve(self, query):
                 # This return format must match what agent expects (List[Chunk])
                 # ChromaStore.query returns List[Chunk] usually?
                 # No, ChromaStore.query returns List[Chunk]
                 return await self.store.aqueury(query, top_k=4) 
                 # Wait, 'aqueury' might be a typo or specific method. 
                 # Let's check DBSchemaRetrieverOperator usage.
                 # It calls self._table_vector_store_connector.query(query, top_k, filters)
        
        # Correct approach: replicate DBSchemaRetrieverOperator's retriever.
        # Use dbgpt_ext for RAG components
        from dbgpt_ext.rag.retriever.db_schema import DBSchemaRetriever
        # Wait, the import in main.py was DBSchemaRetrieverOperator
        # The operator wraps a retriever.
        
        # Re-using the operator logic is safest if we want to stay in AWEL land,
        # but the agent is a python class.
        # The agent in main.py was passed `retriever_task._retriever`.
        # So let's instantiate the operator just to get the retriever, or instantiate retriever directly.
        from dbgpt_ext.rag.retriever.db_schema import DBSchemaRetriever
        
        # Note: DBSchemaRetriever is not directly importable from `dbgpt_ext.rag.operators.db_schema`
        # It's usually in `dbgpt_ext.rag.retriever.db_schema` but let's check imports
        # Actually main.py used: `retriever_task = DBSchemaRetrieverOperator(...)`
        # And passed `retriever_task._retriever`.
        
        # So let's emulate that:
        schema_retriever = DBSchemaRetriever(
             top_k=3,
             table_vector_store_connector=self.vector_store,
             field_vector_store_connector=self.vector_store
        )
        
        self.agent = ScratchpadSchemaAgent(
            llm_client=self.llm_client,
            model_name=self.llm_model,
            vector_retriever=schema_retriever,
            bm25_retriever=self.bm25_retriever
        )


    async def run_pipeline(self, query: str):
        if not self.agent:
            await self.build_index()

        # Define DAG
        with DAG("chat_data_dag") as chat_data_dag:
            input_task = InputOperator(input_source=InputSource.from_callable())
            
            # 1. Retrieval (Scratchpad)
            async def scratchpad_retrieval(user_input: str):
                 logger.info(f"Triggering Scratchpad", extra={"props": {"user_input": user_input}})
                 return await self.agent.run(user_input)

            retriever_task = MapOperator(scratchpad_retrieval)
            content_task = MapOperator(lambda cks: "\n\n".join([c.content for c in cks])) 
            
            # 2. Prompting
            merge_task = JoinOperator(lambda table_info, ext_dict: {**ext_dict, "table_info": table_info}) 
            
            prompt = ChatPromptTemplate(
                messages=[
                    SystemPromptTemplate.from_template(
                        SYSTEM_PROMPT,
                        response_format=json.dumps(
                            RESPONSE_FORMAT_SIMPLE, ensure_ascii=False, indent=4
                        ),
                    ),
                    HumanPromptTemplate.from_template("{user_input}"),
                ]
            )
            
            prompt_task = PromptBuilderOperator(prompt)
            req_build_task = RequestBuilderOperator(model=self.llm_model)
            llm_task = LLMOperator(llm_client=self.llm_client) 
            sql_parse_task = SQLOutputParser()
            
            # 3. Robust Execution (The new part)
            # RobustSQLOperator expects {sql, user_input, table_info, thoughts...}
            # sql_parse_task returns a dict, likely {sql: "...", thoughts: "..."}
            # We need to preserve user_input and table_info from upstream for the correction agent context.
            
            # We join the output of sql_parse_task with the input_data (or merge_task output)
            # merge_task output is {user_input, ..., table_info}
            
            # input to RobustSQL: 
            # Needs to combine sql_parse output + merge_task output.
            
            combine_context_task = JoinOperator(
                lambda parse_res, context_dict: {**context_dict, **parse_res}
            )
            
            robust_exec_task = RobustSQLOperator(
                connector=self.db_conn,
                llm_client=self.llm_client,
                model_name=self.llm_model
            )

            # 4. Chart Generation (After SQL execution)
            chart_gen_task = ChartGenerationOperator(
                llm_client=self.llm_client,
                model_name=self.llm_model,
                vega_schema_path="chart_schema/vega-lite-schema-v5.json",
                output_dir="chart_outputs"
            )

            # Wiring
            (input_task 
             >> MapOperator(lambda x: x["user_input"]) 
             >> retriever_task 
             >> content_task)
            
            # JoinOperator Args order depends on connection order
            # Arg1: table_info (from content_task)
            # Arg2: ext_dict (from input_task)
            content_task >> merge_task
            input_task >> merge_task
            
            (merge_task 
             >> prompt_task 
             >> req_build_task 
             >> llm_task 
             >> sql_parse_task)
             
            # Join parsed SQL with original context to feed into Robust Executor
            # merge_task contains 'user_input' and 'table_info'
            # sql_parse_task contains 'sql' and 'thoughts'
            
            # We want: sql_parse_task -> (join with merge_task) -> robust_exec_task
            merge_task >> combine_context_task
            sql_parse_task >> combine_context_task

            combine_context_task >> robust_exec_task

            # Connect chart generation after SQL execution
            robust_exec_task >> chart_gen_task

        # Execute
        logger.info(f"Executing Query", extra={"props": {"query": query}})
        input_data = {
            "user_input": query,
            "db_name": self.db_conn.get_current_db_name(), # dynamic or fixed
            "dialect": "SQLite",
            "response": json.dumps(RESPONSE_FORMAT_SIMPLE, ensure_ascii=False, indent=4)
        }
        
        return await chart_gen_task.call(input_data)
