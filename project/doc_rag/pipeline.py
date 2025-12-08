"""Document RAG AWEL pipeline with metadata-aware responses.

Main pipeline for querying PDF documents with hybrid search and reranking.
"""

import asyncio
import json
import os
import shutil
from typing import Dict, Any, List

from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    SystemPromptTemplate,
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
from dbgpt.model.proxy import OpenAILLMClient
from dbgpt.rag.embedding import DefaultEmbeddingFactory
from dbgpt_ext.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig

from .pdf_loader import MetadataAwarePDFLoader
from .retriever import HybridDocRetriever
from .reranker import DefaultReranker
from . import logger


RESPONSE_FORMAT = {
    "answer": "Your detailed answer based on the context",
    "sources": [
        {
            "page": "page number",
            "section": "section name",
            "relevance": "brief explanation of relevance"
        }
    ]
}

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on the provided document context.

Document Context:
{context}

Document Metadata:
- Title: {doc_title}
- Source: {source_file}

Instructions:
1. Answer the user's question using ONLY the provided context
2. If the context doesn't contain enough information, say so clearly
3. Always cite the page numbers and sections from the source documents
4. Be precise and factual

User Question:
{question}

Respond in the following JSON format:
{response_format}

Ensure the response is valid JSON that can be parsed by Python json.loads.
"""


class DocRAGPipeline:
    """Document RAG pipeline with hybrid search and metadata-aware responses.
    
    Features:
    - PDF loading with page-level metadata
    - Hybrid retrieval (vector + BM25)
    - RRF fusion for result combination
    - LLM-powered answer generation with source citations
    """
    
    def __init__(
        self,
        openai_api_key: str,
        azure_endpoint: str,
        azure_api_key: str,
        llm_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        vector_store_path: str = "./doc_rag_vector_store",
        azure_model_id: str = "prebuilt-layout",
        pdf_language: str = "en",
    ):
        """Initialize the Document RAG pipeline.
        
        Args:
            openai_api_key: OpenAI API key
            azure_endpoint: Azure Document Intelligence endpoint
            azure_api_key: Azure Document Intelligence API key
            llm_model: LLM model name
            embedding_model: Embedding model name
            vector_store_path: Path to store vector embeddings
            azure_model_id: Azure Document Intelligence model ID
            pdf_language: Language hint for document analysis
        """
        self.openai_api_key = openai_api_key
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.vector_store_path = vector_store_path
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self.azure_model_id = azure_model_id
        self.pdf_language = pdf_language
        
        # Initialize components
        self.embeddings = DefaultEmbeddingFactory.openai(
            api_key=openai_api_key,
            model_name=embedding_model,
        )
        
        self.llm_client = OpenAILLMClient(model=llm_model, api_key=openai_api_key)
        self.pdf_loader = MetadataAwarePDFLoader(
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            azure_model_id=azure_model_id,
            language=pdf_language,
        )
        
        # Will be initialized during build_index
        self.vector_store = None
        self.retriever = None
        self.chunks = None
        self.doc_metadata = None
        
        logger.info(f"Initialized DocRAGPipeline", extra={"props": {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "azure_model_id": azure_model_id,
        }})
    
    async def build_index(
        self,
        pdf_path: str,
        force_rebuild: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """Load PDF and build vector index.
        
        Args:
            pdf_path: Path to PDF file
            force_rebuild: Whether to clear existing index
            chunk_size: Chunk size to use during parsing
            chunk_overlap: Character overlap between chunks
        """
        logger.info(f"Building index", extra={"props": {
            "pdf_path": pdf_path,
            "force_rebuild": force_rebuild
        }})
        
        # Clean up vector store if force_rebuild
        if force_rebuild and os.path.exists(self.vector_store_path):
            shutil.rmtree(self.vector_store_path)
        
        # Load PDF content and metadata via Azure Document Intelligence
        self.chunks, self.doc_metadata = self.pdf_loader.load(
            pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # Initialize vector store
        self.vector_store = ChromaStore(
            vector_store_config=ChromaVectorConfig(persist_path=self.vector_store_path),
            name="doc_rag_store",
            embedding_fn=self.embeddings,
        )
        
        # Index chunks in vector store
        logger.info(f"Indexing chunks", extra={"props": {"count": len(self.chunks)}})
        await self._index_chunks(self.chunks)
        
        # Initialize hybrid retriever
        self.retriever = HybridDocRetriever(
            vector_store=self.vector_store,
            chunks=self.chunks,
            bm25_top_k=10,
            vector_top_k=10,
            final_top_k=5
        )
        
        logger.info(f"Index built successfully", extra={"props": {"total_chunks": len(self.chunks)}})
    
    async def _index_chunks(self, chunks):
        """Index chunks in vector store.
        
        Args:
            chunks: List of chunks to index
        """
        # Use vector store's load method
        if hasattr(self.vector_store, 'aload_document'):
            await self.vector_store.aload_document(chunks)
        elif hasattr(self.vector_store, 'load_document'):
            self.vector_store.load_document(chunks)
        else:
            # Fallback: add documents one by one
            for chunk in chunks:
                if hasattr(self.vector_store, 'add'):
                    self.vector_store.add([chunk.content], metadatas=[chunk.metadata])
    
    async def query(self, question: str) -> Dict[str, Any]:
        """Query the document and get answer with source metadata.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source metadata
        """
        if not self.retriever:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        logger.info(f"Processing query", extra={"props": {"question": question[:100]}})
        
        # Define DAG
        with DAG("doc_rag_pipeline") as dag:
            input_task = InputOperator(input_source=InputSource.from_callable())
            
            # Retrieval using hybrid search
            async def hybrid_retrieve(query_data: Dict[str, Any]) -> List:
                question = query_data["question"]
                chunks = await self.retriever.retrieve(question)
                return chunks
            
            retriever_task = MapOperator(hybrid_retrieve)
            
            # Format context with metadata
            def format_context(chunks) -> str:
                context_parts = []
                for i, chunk in enumerate(chunks):
                    meta = chunk.metadata if hasattr(chunk, 'metadata') else {}
                    page = meta.get('page', 'N/A')
                    section = meta.get('section', 'N/A')
                    
                    context_parts.append(
                        f"[Source {i+1}] (Page {page}, Section: {section})\n{chunk.content}\n"
                    )
                return "\n---\n".join(context_parts)
            
            context_task = MapOperator(format_context)
            
            # Prepare chunks metadata for response
            def extract_source_metadata(chunks) -> List[Dict]:
                return [
                    {
                        "page": chunk.metadata.get('page', 'N/A') if hasattr(chunk, 'metadata') else 'N/A',
                        "section": chunk.metadata.get('section', 'N/A') if hasattr(chunk, 'metadata') else 'N/A',
                        "score": chunk.score if hasattr(chunk, 'score') else None,
                        "content_preview": chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                    }
                    for chunk in chunks
                ]
            
            metadata_task = MapOperator(extract_source_metadata)
            
            # Merge context with input data
            merge_task = JoinOperator(
                lambda context, input_data: {
                    **input_data,
                    "context": context
                }
            )
            
            # Also capture source metadata
            merge_meta_task = JoinOperator(
                lambda sources, prev_data: {
                    **prev_data,
                    "source_chunks": sources
                }
            )
            
            # Build prompt
            prompt = ChatPromptTemplate(
                messages=[
                    SystemPromptTemplate.from_template(
                        SYSTEM_PROMPT
                    ),
                    HumanPromptTemplate.from_template("{question}"),
                ]
            )
            
            prompt_task = PromptBuilderOperator(prompt)
            req_build_task = RequestBuilderOperator(model=self.llm_model)
            llm_task = LLMOperator(llm_client=self.llm_client)
            
            # Parse LLM response
            def parse_response(llm_output) -> Dict[str, Any]:
                text = llm_output.text if hasattr(llm_output, 'text') else str(llm_output)
                try:
                    start = text.find('{')
                    end = text.rfind('}') + 1
                    if start != -1 and end > start:
                        return json.loads(text[start:end])
                except:
                    pass
                return {"answer": text, "sources": []}
            
            parse_task = MapOperator(parse_response)
            
            # Final merge with source metadata
            def final_output(parsed, source_chunks) -> Dict[str, Any]:
                return {
                    "answer": parsed.get("answer", ""),
                    "sources": parsed.get("sources", []),
                    "source_chunks": source_chunks,
                    "doc_metadata": self.doc_metadata
                }
            
            final_task = JoinOperator(final_output)
            
            # Wire the DAG
            input_task >> retriever_task
            retriever_task >> context_task >> merge_task
            input_task >> merge_task
            
            retriever_task >> metadata_task >> merge_meta_task
            merge_task >> merge_meta_task
            
            merge_meta_task >> prompt_task >> req_build_task >> llm_task >> parse_task
            
            # Final merge
            parse_task >> final_task
            merge_meta_task >> MapOperator(lambda x: x.get("source_chunks", [])) >> final_task
        
        # Execute
        metadata = self.doc_metadata or {}
        input_data = {
            "question": question,
            "doc_title": metadata.get("title", "Unknown"),
            "source_file": metadata.get("source", "Unknown"),
            "response_format": json.dumps(RESPONSE_FORMAT, indent=2)
        }
        
        result = await final_task.call(input_data)
        
        logger.info(f"Query complete", extra={"props": {
            "sources_count": len(result.get("source_chunks", []))
        }})
        
        return result


async def run_doc_rag_demo():
    """Demo function to run the document RAG pipeline."""
    import os
    
    # Configuration
    API_KEY = os.environ.get("OPENAI_API_KEY", "")
    AZURE_ENDPOINT = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
    AZURE_KEY = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
    PDF_PATH = "doc_rag/test_data/2q25-media-release-en.pdf"
    
    if not API_KEY:
        print("Please set OPENAI_API_KEY environment variable")
        return
    if not AZURE_ENDPOINT or not AZURE_KEY:
        print("Please set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY environment variables")
        return
    
    # Initialize pipeline
    pipeline = DocRAGPipeline(
        openai_api_key=API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        azure_api_key=AZURE_KEY,
        llm_model="gpt-4o",
        embedding_model="text-embedding-3-small",
        vector_store_path="./doc_rag_chroma_store"
    )
    
    # Build index
    await pipeline.build_index(PDF_PATH)
    
    # Query
    question = "What net profit did UBS report for Q2 2025?"
    result = await pipeline.query(question)
    
    print("\n=== Answer ===")
    print(result.get("answer", "No answer"))
    
    print("\n=== Sources ===")
    for source in result.get("source_chunks", []):
        print(f"Page {source['page']}, Section: {source['section']}")
        print(f"  Preview: {source['content_preview']}")
        print()


if __name__ == "__main__":
    asyncio.run(run_doc_rag_demo())
