import asyncio
import shutil
import tempfile

from doc_rag.pipeline import DocRAGPipeline
from doc_rag import logger

OPENAI_API_KEY = ""
AZURE_ENDPOINT = ""
AZURE_KEY = ""
PDF_PATH = "doc_rag/test_data/2q25-media-release-en.pdf"
LLM_MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

POSITIVE_QUERY = "What net profit did UBS report in 2Q25?"
NEGATIVE_QUERY = "How many electric vehicles did Tesla deliver in Q2 2025?"


async def main():
    logger.info("Starting Doc RAG pipeline test...", extra={"props": {"pdf": PDF_PATH}})
    vector_store_dir = tempfile.mkdtemp(prefix="doc_rag_store_")

    pipeline = DocRAGPipeline(
        openai_api_key=OPENAI_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        azure_api_key=AZURE_KEY,
        llm_model=LLM_MODEL_NAME,
        embedding_model=EMBEDDING_MODEL_NAME,
        vector_store_path=vector_store_dir,
    )

    try:
        await pipeline.build_index(PDF_PATH)

        print("\n=== Positive Query ===")
        print(f"Question: {POSITIVE_QUERY}")
        positive_result = await pipeline.query(POSITIVE_QUERY)
        print("Answer:")
        print(positive_result.get("answer", "<no answer>"))
        print("Sources:")
        for chunk in positive_result.get("source_chunks", []):
            print(f"- Page {chunk.get('page')} :: Section {chunk.get('section')}")
            print(f"  Preview: {chunk.get('content_preview', '')}")

        print("\n=== Negative Query ===")
        print(f"Question: {NEGATIVE_QUERY}")
        negative_result = await pipeline.query(NEGATIVE_QUERY)
        answer_text = negative_result.get("answer", "").lower()
        print("Answer:")
        print(negative_result.get("answer", "<no answer>"))

        failure_tokens = [
            "not contain",
            "does not provide",
            "insufficient",
            "do not have",
            "cannot find",
            "not available",
        ]
        if not any(token in answer_text for token in failure_tokens):
            raise RuntimeError("Negative query unexpectedly succeeded with a confident answer.")

    finally:
        shutil.rmtree(vector_store_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
