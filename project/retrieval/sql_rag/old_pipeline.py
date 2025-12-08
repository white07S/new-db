
import asyncio
import json
import shutil
import os
from pathlib import Path
from typing import Dict, Any

from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    SystemPromptTemplate,
    SQLOutputParser,
)
from dbgpt.core.awel import DAG, InputOperator, InputSource, JoinOperator
from dbgpt.core.operators import PromptBuilderOperator, RequestBuilderOperator
from dbgpt.model.operators import LLMOperator
from dbgpt_ext.datasource.rdbms.conn_sqlite import SQLiteConnector
from dbgpt_ext.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig

from .agent import ScratchpadSchemaAgent
from .operators import RobustSQLOperator
from .chart_operators import ChartGenerationOperator
from .schema_search import SchemaSearchEngine
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
        vector_store_path: str,
    ):
        self.openai_api_key = openai_api_key
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.sqlite_path = sqlite_path
        self.vector_store_path = vector_store_path

        if os.path.exists(self.vector_store_path):
            shutil.rmtree(self.vector_store_path)

        self.embeddings = DefaultEmbeddingFactory.openai(
            api_key=openai_api_key,
            model_name=embedding_model,
        )

        self.db_conn = SQLiteConnector.from_file_path(sqlite_path)

        self.vector_store = ChromaStore(
            ChromaVectorConfig(persist_path=vector_store_path),
            name="ltm_vector_store",
            embedding_fn=self.embeddings,
        )

        self.schema_file = (
            Path(__file__).resolve().parent / "test_data_schema" / "output_schema.json"
        )

        self.llm_client = OpenAILLMClient(model=llm_model, api_key=openai_api_key)

        self.schema_search_engine: SchemaSearchEngine | None = None
        self.agent: ScratchpadSchemaAgent | None = None

    async def build_index(self):
        """Initialize schema search engine and agent."""
        if not self.schema_search_engine:
            self.schema_search_engine = SchemaSearchEngine(
                schema_path=self.schema_file,
                vector_store=self.vector_store,
            )

        if not self.agent:
            self.agent = ScratchpadSchemaAgent(
                llm_client=self.llm_client,
                model_name=self.llm_model,
                schema_search_engine=self.schema_search_engine,
            )


    async def run_pipeline(self, query: str):
        if not self.agent:
            await self.build_index()

        try:
            schema_context = await self.agent.run(query)
        except RuntimeError as exc:
            logger.error(
                "Schema context assembly failed",
                extra={"props": {"error": str(exc), "query": query}},
            )
            return {
                "error": "Failed to assemble required schema context. "
                "Please refine your question.",
                "details": str(exc),
            }

        if not schema_context:
            return {
                "error": "Schema context is empty after retrieval. Unable to proceed.",
                "details": "No schema rows matched the query intent.",
            }

        # Define DAG
        with DAG("chat_data_dag") as chat_data_dag:
            input_task = InputOperator(input_source=InputSource.from_callable())

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
            
            combine_context_task = JoinOperator(
                lambda context_dict, parse_res: {**context_dict, **parse_res}
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
            input_task >> prompt_task >> req_build_task >> llm_task >> sql_parse_task

            input_task >> combine_context_task
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
            "response": json.dumps(RESPONSE_FORMAT_SIMPLE, ensure_ascii=False, indent=4),
            "table_info": schema_context,
        }

        return await chart_gen_task.call(input_data)
