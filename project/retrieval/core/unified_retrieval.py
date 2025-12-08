"""
Unified Retrieval System combining SQL and Document RAG.

This module provides a unified interface for both SQL and document-based
retrieval augmented generation, using Qdrant as the vector store.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass
import toml

from providers.config import load_config, get_config
from providers.qdrant import QdrantProvider, QdrantConfig
from providers.embeddings import EmbeddingProvider
from providers.database import DatabaseProvider
from providers.llm import LLMProvider

from retrieval.sql_rag.sql_retrieval import SQLRetrieval
from retrieval.doc_rag.doc_retrieval import DocumentRetrieval

from common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for the unified retrieval system."""

    # Qdrant configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: Optional[str] = None
    qdrant_collection_prefix: str = "fullrag_"

    # Model configuration
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    provider: str = "openai"

    # Database configuration
    sqlite_path: Optional[str] = None

    # Document configuration
    doc_chunk_size: int = 1024
    doc_chunk_overlap: int = 128

    # Output configuration
    output_dir: str = "output"
    log_dir: str = "output/logs"
    chart_dir: str = "output/charts"

    # Context provider paths
    schema_path: str = "retrieval/context_provider/test_data_schema/output_schema.json"
    vega_schema_path: str = "retrieval/context_provider/chart_schema/vega-lite-schema-v5.json"

    @classmethod
    def from_config_file(cls, config_path: str = None, env_file: str = None) -> 'RetrievalConfig':
        """
        Load configuration from config.toml and environment variables.

        Args:
            config_path: Path to config.toml file
            env_file: Path to .env file to load

        Returns:
            RetrievalConfig instance
        """
        # Load environment variables if env_file provided
        if env_file and Path(env_file).exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")

        # Load config.toml
        config = load_config(config_path)

        # Extract values from config
        instance = cls()

        # Qdrant settings
        if hasattr(config, 'qdrant') and config.qdrant:
            qdrant_url = config.qdrant.url or 'http://localhost:6333'
            if qdrant_url.startswith('http://'):
                host = qdrant_url.replace('http://', '').split(':')[0]
                port = int(qdrant_url.split(':')[-1])
            else:
                host = 'localhost'
                port = 6333

            instance.qdrant_host = host
            instance.qdrant_port = port
            instance.qdrant_api_key = config.qdrant.api_key or None
            instance.qdrant_collection_prefix = config.qdrant.collection_prefix or 'fullrag_'

        # Provider settings
        if hasattr(config, 'providers'):
            instance.provider = config.providers.default_provider

            # Get model names based on provider
            if instance.provider == 'azure' and config.providers.azure:
                instance.llm_model = config.providers.azure.models.llm
                instance.embedding_model = config.providers.azure.models.embedding
            else:
                instance.llm_model = config.providers.openai.models.llm
                instance.embedding_model = config.providers.openai.models.embedding

        # Testing configuration
        if hasattr(config, 'testing') and config.testing:
            instance.sqlite_path = config.testing.test_db_url

        logger.info(f"Loaded configuration from {config_path or 'default path'}")
        logger.info(f"Using provider: {instance.provider}")
        logger.info(f"LLM model: {instance.llm_model}")
        logger.info(f"Embedding model: {instance.embedding_model}")

        return instance


class UnifiedRetrievalSystem:
    """
    Unified system for SQL and Document retrieval augmented generation.

    This class provides a single interface to both SQL and document RAG systems,
    managing configuration, providers, and execution.
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        config_path: Optional[str] = None,
        env_file: Optional[str] = None
    ):
        """
        Initialize the unified retrieval system.

        Args:
            config: RetrievalConfig instance
            config_path: Path to config.toml
            env_file: Path to .env file
        """
        # Load configuration
        if config is None:
            # Default paths if not provided
            if config_path is None:
                config_path = "providers/config.toml"
            if env_file is None:
                env_file = "providers/.env.dev"

            config = RetrievalConfig.from_config_file(config_path, env_file)

        self.config = config

        # Create output directories
        Path(self.config.output_dir).mkdir(exist_ok=True)
        Path(self.config.log_dir).mkdir(exist_ok=True)
        Path(self.config.chart_dir).mkdir(exist_ok=True)

        # Initialize providers
        self._initialize_providers()

        # Initialize retrieval systems
        self.sql_retrieval = None
        self.doc_retrieval = None

        logger.info("Unified Retrieval System initialized")

    def _initialize_providers(self):
        """Initialize all providers from configuration."""

        # Get app config for API keys
        app_config = get_config()

        # Qdrant provider
        qdrant_config = QdrantConfig(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            grpc_port=self.config.qdrant_grpc_port,
            api_key=self.config.qdrant_api_key,
            prefer_grpc=True
        )
        self.qdrant_provider = QdrantProvider(qdrant_config)

        # Embedding provider
        self.embedding_provider = EmbeddingProvider(
            provider=self.config.provider,
            model=self.config.embedding_model
        )

        # LLM provider
        self.llm_provider = LLMProvider(
            provider=self.config.provider,
            model=self.config.llm_model
        )

        # Database provider (if SQLite path provided)
        if self.config.sqlite_path and Path(self.config.sqlite_path).exists():
            self.db_provider = DatabaseProvider(
                db_type="sqlite",
                db_path=self.config.sqlite_path
            )
            logger.info(f"Database provider initialized: {self.config.sqlite_path}")
        else:
            self.db_provider = None
            logger.warning(f"Database not found: {self.config.sqlite_path}")

    async def initialize_sql_rag(self, recreate_collection: bool = False) -> 'SQLRetrieval':
        """
        Initialize SQL RAG system.

        Args:
            recreate_collection: Whether to recreate the collection

        Returns:
            SQLRetrieval instance
        """
        if self.db_provider is None:
            raise ValueError("Database provider not initialized. Check sqlite_path in config.")

        collection_name = f"{self.config.qdrant_collection_prefix}sql_schema"

        self.sql_retrieval = SQLRetrieval(
            db_provider=self.db_provider,
            qdrant_provider=self.qdrant_provider,
            embedding_provider=self.embedding_provider,
            llm_provider=self.llm_provider,
            collection_name=collection_name,
            schema_path=self.config.schema_path,
            vega_schema_path=self.config.vega_schema_path,
            chart_output_dir=self.config.chart_dir,
            recreate_collection=recreate_collection
        )

        await self.sql_retrieval.build_index()
        logger.info("SQL RAG system initialized")

        return self.sql_retrieval

    async def initialize_doc_rag(self, recreate_collection: bool = False) -> 'DocumentRetrieval':
        """
        Initialize Document RAG system.

        Args:
            recreate_collection: Whether to recreate the collection

        Returns:
            DocumentRetrieval instance
        """
        collection_name = f"{self.config.qdrant_collection_prefix}documents"

        self.doc_retrieval = DocumentRetrieval(
            qdrant_provider=self.qdrant_provider,
            embedding_provider=self.embedding_provider,
            llm_provider=self.llm_provider,
            collection_name=collection_name,
            chunk_size=self.config.doc_chunk_size,
            chunk_overlap=self.config.doc_chunk_overlap,
            recreate_collection=recreate_collection
        )

        logger.info("Document RAG system initialized")

        return self.doc_retrieval

    async def query(
        self,
        query: str,
        mode: Literal["sql", "doc", "auto"] = "auto"
    ) -> Dict[str, Any]:
        """
        Execute a query against the appropriate RAG system.

        Args:
            query: Natural language query
            mode: Which RAG system to use (sql, doc, or auto-detect)

        Returns:
            Query results
        """
        # Auto-detect mode based on query content
        if mode == "auto":
            mode = self._detect_query_mode(query)
            logger.info(f"Auto-detected query mode: {mode}")

        if mode == "sql":
            if self.sql_retrieval is None:
                await self.initialize_sql_rag()
            return await self.sql_retrieval.query(query)

        elif mode == "doc":
            if self.doc_retrieval is None:
                await self.initialize_doc_rag()
            return await self.doc_retrieval.query(query)

        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _detect_query_mode(self, query: str) -> str:
        """
        Detect whether query is for SQL or document RAG.

        Args:
            query: Natural language query

        Returns:
            "sql" or "doc"
        """
        query_lower = query.lower()

        # SQL indicators
        sql_keywords = [
            'table', 'database', 'sql', 'query', 'revenue', 'sales',
            'customer', 'order', 'product', 'seller', 'aggregate',
            'sum', 'count', 'average', 'rank', 'top', 'bottom',
            'growth', 'trend', 'analysis', 'metric', 'kpi'
        ]

        # Document indicators
        doc_keywords = [
            'document', 'pdf', 'text', 'file', 'paper', 'report',
            'article', 'content', 'paragraph', 'section', 'page',
            'summary', 'explain', 'describe', 'tell me about'
        ]

        sql_score = sum(1 for kw in sql_keywords if kw in query_lower)
        doc_score = sum(1 for kw in doc_keywords if kw in query_lower)

        return "sql" if sql_score >= doc_score else "doc"

    async def index_documents(
        self,
        file_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Index documents for document RAG.

        Args:
            file_paths: List of document paths
            metadata: Optional metadata

        Returns:
            Indexing results
        """
        if self.doc_retrieval is None:
            await self.initialize_doc_rag()

        return await self.doc_retrieval.index_documents(file_paths, metadata)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval systems.

        Returns:
            Statistics dictionary
        """
        stats = {
            "configuration": {
                "provider": self.config.provider,
                "llm_model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
                "qdrant_host": f"{self.config.qdrant_host}:{self.config.qdrant_port}"
            },
            "collections": {}
        }

        # Get Qdrant collections
        try:
            collections = self.qdrant_provider.list_collections()
            for collection in collections:
                if collection.startswith(self.config.qdrant_collection_prefix):
                    count = self.qdrant_provider.count(collection)
                    stats["collections"][collection] = {
                        "document_count": count
                    }
        except Exception as e:
            logger.error(f"Failed to get collection statistics: {e}")

        # Add SQL RAG status
        stats["sql_rag"] = {
            "initialized": self.sql_retrieval is not None,
            "database": self.config.sqlite_path if self.db_provider else None
        }

        # Add Doc RAG status
        stats["doc_rag"] = {
            "initialized": self.doc_retrieval is not None
        }

        return stats

    async def close(self):
        """Close all connections and cleanup resources."""
        if self.qdrant_provider:
            self.qdrant_provider.close()

        if self.db_provider:
            self.db_provider.close()

        logger.info("Unified Retrieval System closed")