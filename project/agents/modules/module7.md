Module 7: Retrieval Module Abstraction
Why this approach
WrenAI's pipeline architecture demonstrates pluggable retrieval modules. Each retrieval method is a self-contained pipeline that can be added without refactoring the core system. github
Implementation
python# agents/retrieval/base.py
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any

class RetrievalResult(BaseModel):
    """Standard retrieval result format"""
    source: str                              # "sql_rag", "doc_rag", etc.
    content: Any                             # Retrieved content
    metadata: dict = {}
    confidence: float = 1.0
    token_count: int = 0

class BaseRetriever(ABC):
    """Abstract base for all retrieval modules"""
    name: str
    
    @abstractmethod
    async def retrieve(self, query: str, context: dict) -> RetrievalResult:
        """Execute retrieval"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """Return list of query types this retriever handles"""
        pass

# agents/retrieval/sql_rag.py
class SQLRAGRetriever(BaseRetriever):
    name = "sql_rag"
    
    def __init__(self, db_connection, schema: dict, embedder):
        self.db = db_connection
        self.schema = schema
        self.embedder = embedder
    
    async def retrieve(self, query: str, context: dict) -> RetrievalResult:
        # 1. Generate SQL from natural language
        sql = await self._generate_sql(query, context)
        
        # 2. Execute SQL
        result_df = await self._execute_sql(sql)
        
        # 3. Apply truncation if needed
        truncator = DataFrameTruncator(self.schema)
        truncated = truncator.truncate(result_df)
        
        return RetrievalResult(
            source="sql_rag",
            content=truncated.data,
            metadata={
                "sql": sql,
                "truncated": truncated.truncated,
                "original_rows": truncated.original_rows
            }
        )
    
    def get_capabilities(self) -> list[str]:
        return ["structured_data", "aggregations", "filtering", "joins"]

# agents/retrieval/doc_rag.py
class DocRAGRetriever(BaseRetriever):
    name = "doc_rag"
    
    def __init__(self, vector_store, embedder, doc_intelligence_client=None):
        self.vector_store = vector_store
        self.embedder = embedder
        self.doc_client = doc_intelligence_client  # Azure Doc Intelligence
    
    async def retrieve(self, query: str, context: dict) -> RetrievalResult:
        # 1. Embed query
        query_embedding = await self.embedder.embed(query)
        
        # 2. Search vector store
        results = await self.vector_store.search(query_embedding, top_k=5)
        
        # 3. Assemble context
        content = "\n\n".join([r.text for r in results])
        
        return RetrievalResult(
            source="doc_rag",
            content=content,
            metadata={
                "num_chunks": len(results),
                "sources": [r.source for r in results]
            }
        )
    
    def get_capabilities(self) -> list[str]:
        return ["unstructured_text", "documents", "policies", "procedures"]

# agents/retrieval/registry.py
class RetrieverRegistry:
    """Manages pluggable retrieval modules"""
    
    def __init__(self):
        self._retrievers: dict[str, BaseRetriever] = {}
    
    def register(self, retriever: BaseRetriever):
        """Add a retriever to the registry"""
        self._retrievers[retriever.name] = retriever
    
    def get(self, name: str) -> BaseRetriever | None:
        return self._retrievers.get(name)
    
    def get_for_capability(self, capability: str) -> list[BaseRetriever]:
        """Find retrievers that handle a specific capability"""
        return [
            r for r in self._retrievers.values()
            if capability in r.get_capabilities()
        ]
    
    def list_all(self) -> list[str]:
        return list(self._retrievers.keys())

# Usage
registry = RetrieverRegistry()
registry.register(SQLRAGRetriever(db, schema, embedder))
registry.register(DocRAGRetriever(vector_store, embedder))
# Future: registry.register(APIRetriever(...))
