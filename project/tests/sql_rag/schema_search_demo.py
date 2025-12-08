import asyncio
import tempfile
from pathlib import Path

from dbgpt.rag.embedding import DefaultEmbeddingFactory
from dbgpt.model.proxy import OpenAILLMClient
from dbgpt_ext.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig

from sql_rag.agent import ScratchpadSchemaAgent
from sql_rag.schema_search import SchemaSearchEngine
from sql_rag import logger


# Hard-coded configuration (matching other sql_rag tests)
OPENAI_API_KEY = ""
LLM_MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

POSITIVE_QUERY = "Show total payment_value grouped by customer_state"
NEGATIVE_QUERY = "How many stars are in the Andromeda galaxy compared to Milky Way?"


async def run_demo():
    logger.info("Bootstrapping schema search demo...")
    persist_dir = tempfile.mkdtemp(prefix="schema_demo_")

    embeddings = DefaultEmbeddingFactory.openai(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL_NAME,
    )
    vector_store = ChromaStore(
        ChromaVectorConfig(persist_path=persist_dir),
        name="schema_demo_store",
        embedding_fn=embeddings,
    )

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "sql_rag"
        / "test_data_schema"
        / "output_schema.json"
    )
    schema_engine = SchemaSearchEngine(
        schema_path=schema_path,
        vector_store=vector_store,
    )

    llm_client = OpenAILLMClient(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL_NAME,
        model_alias=LLM_MODEL_NAME,
    )
    agent = ScratchpadSchemaAgent(
        llm_client=llm_client,
        model_name=LLM_MODEL_NAME,
        schema_search_engine=schema_engine,
    )

    print("\n=== Positive Query ===")
    print(f"Question: {POSITIVE_QUERY}")
    schema_context = await agent.run(POSITIVE_QUERY)
    print("Schema Context:")
    print(schema_context)

    print("\n=== Negative Query ===")
    print(f"Question: {NEGATIVE_QUERY}")
    try:
        await agent.run(NEGATIVE_QUERY)
    except RuntimeError as exc:
        print("Received expected failure:")
        print(str(exc))
    else:
        raise RuntimeError("Negative query unexpectedly succeeded.")


if __name__ == "__main__":
    asyncio.run(run_demo())
