
import asyncio
import os
from sql_rag.pipeline import SQLRAGPipeline
from sql_rag import logger


# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================================================

# OpenAI API Key - Replace with actual key or load from env
OPENAI_API_KEY = ""

# LLM Model name
LLM_MODEL_NAME = "gpt-4o"

# Embedding Model name
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# Path to your SQLite database file
SQLITE_DB_PATH = "/Users/preetam/Develop/data_testing/brazilian_ecommerce.sqlite"

# Vector store path - CHANGED to local path in project root
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "chroma_db_store")

# ============================================================================

async def main():
    logger.info("Starting Text-to-SQL Flow with SQL RAG Pipeline...")
    logger.info(f"Using Vector Store Path: {VECTOR_STORE_PATH}")
    
    pipeline = SQLRAGPipeline(
        openai_api_key=OPENAI_API_KEY,
        llm_model=LLM_MODEL_NAME,
        embedding_model=EMBEDDING_MODEL_NAME,
        sqlite_path=SQLITE_DB_PATH,
        vector_store_path=VECTOR_STORE_PATH
    )
    
    # Initialize components
    await pipeline.build_index()
    
    query = "Do orders with low review scores (1–2) use different payment types compared to high review scores (4–5)? Show the distribution of payment_type for low vs high review_score."
    
    try:
        result = await pipeline.run_pipeline(query)
        logger.info("FINAL RESULT", extra={"props": {"result": str(result)}})

        # Display data results
        if "data" in result:
             print("\nData Preview:")
             print(result["data"])

        # Display SQL reasoning
        if "thoughts" in result:
             print("\nExecution Thoughts & Corrections:")
             print(result["thoughts"])

        # Display chart generation results
        if "chart_1" in result and result["chart_1"]:
            print("\n=== Chart 1 ===")
            print(f"Chart Type: {result['chart_1'].get('chart_type', 'N/A')}")
            print(f"Reasoning: {result['chart_1'].get('reasoning', 'N/A')}")
            if result['chart_1'].get('chart_schema'):
                print("Chart schema generated successfully")

        if "chart_2" in result and result["chart_2"]:
            print("\n=== Chart 2 ===")
            print(f"Chart Type: {result['chart_2'].get('chart_type', 'N/A')}")
            print(f"Reasoning: {result['chart_2'].get('reasoning', 'N/A')}")
            if result['chart_2'].get('chart_schema'):
                print("Chart schema generated successfully")

        if "charts_saved_to" in result and result["charts_saved_to"]:
            print("\n=== Charts Saved To ===")
            for file_path in result["charts_saved_to"]:
                print(f"  - {file_path}")

        if "chart_generation_error" in result:
            print(f"\nChart Generation Error: {result['chart_generation_error']}")

    except Exception as e:
        logger.error(f"Pipeline Failed", extra={"props": {"error": str(e)}})

if __name__ == "__main__":
    asyncio.run(main())