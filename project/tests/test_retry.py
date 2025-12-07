
import asyncio
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sql_rag.pipeline import SQLRAGPipeline
from dbgpt.core.awel import DAG, InputOperator, InputSource, MapOperator
from dbgpt.core.operators import PromptBuilderOperator, RequestBuilderOperator
from dbgpt.model.operators import LLMOperator
from sql_rag.operators import RobustSQLOperator
from sql_rag.logger import setup_logger

logger = setup_logger("test_retry")

# Re-use config from main
OPENAI_API_KEY = ""
LLM_MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
SQLITE_DB_PATH = "/Users/preetam/Develop/data_testing/brazilian_ecommerce.sqlite"
# CHANGED to local path in project root
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db_store_test")

async def test_retry_flow():
    logger.info("Testing Retry Logic...")
    logger.info(f"Using Vector Store Path: {VECTOR_STORE_PATH}")
    
    pipeline = SQLRAGPipeline(
        openai_api_key=OPENAI_API_KEY,
        llm_model=LLM_MODEL_NAME,
        embedding_model=EMBEDDING_MODEL_NAME,
        sqlite_path=SQLITE_DB_PATH,
        vector_store_path=VECTOR_STORE_PATH
    )
    
    # Manually constructing a broken SQL scenario
    # We will instantiate the operator directly and feed it a bad SQL first
    
    # data_testing/brazilian_ecommerce.sqlite has 'olist_orders_dataset'
    
    bad_sql = "SELLECT * FROM olist_orders_dataset LIMIT 5" # Syntax error
    user_query = "Show me 5 orders"
    
    operator = RobustSQLOperator(
        connector=pipeline.db_conn,
        llm_client=pipeline.llm_client,
        model_name=LLM_MODEL_NAME
    )
    
    input_data = {
        "sql": bad_sql,
        "user_input": user_query,
        "table_info": "Table: olist_orders_dataset, Columns: order_id, customer_id, order_status...",
        "thoughts": "Initial thought"
    }
    

    print(f"Injecting Bad SQL: {bad_sql}")
    
    # Debug: Check actual tables
    print(f"Available tables: {pipeline.db_conn.get_table_names()}")

    try:
        # AWEL operators are called via .call() usually in a DAG, 
        # but MapOperator.call() acts on input_value
        result = await operator.call(input_data)
        
        print("\nTest Result: SUCCESS")
        print(f"Original SQL: {bad_sql}")
        print(f"Final SQL: {result['sql']}")
        print(f"Correction Log:\n{result['thoughts']}")
        
    except Exception as e:
        print(f"\nTest Result: FAILED. Exception raised: {e}")

if __name__ == "__main__":
    asyncio.run(test_retry_flow())
