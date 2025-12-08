"""
Embeddings Provider

This module provides a unified interface for generating embeddings
using different providers (OpenAI, Azure, etc.).
"""

from typing import List, Optional, Union
import asyncio
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from providers.config import get_config, AppConfig
from common.logger import get_logger

logger = get_logger(__name__)


class EmbeddingProvider:
    """
    Unified embedding provider that supports multiple backends.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        batch_size: int = 100
    ):
        """
        Initialize the embedding provider.

        Args:
            api_key: Optional API key override
            model: Optional model override
            provider: Optional provider override ("openai" or "azure")
            batch_size: Maximum number of texts to embed in one API call
        """
        self.config: AppConfig = get_config()
        self._api_key_override = api_key
        self._model_override = model
        self._provider_override = provider
        self.batch_size = batch_size

        # Initialize clients
        self.sync_client = self._get_sync_client()
        self.async_client = self._get_async_client()

        # Get model configuration
        self.model_name = self._get_model_name()
        self.dimensions = self._get_dimensions()

        logger.info(f"Initialized embedding provider", extra={"props": {
            "provider": self._get_provider(),
            "model": self.model_name,
            "dimensions": self.dimensions
        }})

    def _get_provider(self) -> str:
        """Get the provider to use."""
        if self._provider_override:
            return self._provider_override
        return self.config.providers.default_provider

    def _get_sync_client(self) -> Union[OpenAI, AzureOpenAI]:
        """Initialize the synchronous client."""
        provider = self._get_provider()

        if provider == "azure":
            azure_conf = self.config.providers.azure
            if not azure_conf:
                raise ValueError("Azure configuration is missing")

            api_key = self._api_key_override or azure_conf.get_api_key()
            return AzureOpenAI(
                api_key=api_key,
                api_version=azure_conf.api_version,
                azure_endpoint=azure_conf.api_base
            )
        else:
            openai_conf = self.config.providers.openai
            api_key = self._api_key_override or openai_conf.get_api_key()
            return OpenAI(
                api_key=api_key,
                base_url=openai_conf.api_base
            )

    def _get_async_client(self) -> Union[AsyncOpenAI, AsyncAzureOpenAI]:
        """Initialize the asynchronous client."""
        provider = self._get_provider()

        if provider == "azure":
            azure_conf = self.config.providers.azure
            if not azure_conf:
                raise ValueError("Azure configuration is missing")

            api_key = self._api_key_override or azure_conf.get_api_key()
            return AsyncAzureOpenAI(
                api_key=api_key,
                api_version=azure_conf.api_version,
                azure_endpoint=azure_conf.api_base
            )
        else:
            openai_conf = self.config.providers.openai
            api_key = self._api_key_override or openai_conf.get_api_key()
            return AsyncOpenAI(
                api_key=api_key,
                base_url=openai_conf.api_base
            )

    def _get_model_name(self) -> str:
        """Get the embedding model name."""
        if self._model_override:
            return self._model_override

        provider = self._get_provider()
        if provider == "azure":
            if self.config.providers.azure:
                return self.config.providers.azure.models.embedding
        return self.config.providers.openai.models.embedding

    def _get_dimensions(self) -> int:
        """Get the embedding dimensions based on the model."""
        model = self.model_name.lower()

        # OpenAI models
        if "text-embedding-3-large" in model:
            return 3072
        elif "text-embedding-3-small" in model:
            return 1536
        elif "text-embedding-ada-002" in model:
            return 1536

        # Default dimension
        logger.warning(f"Unknown model {self.model_name}, defaulting to 1536 dimensions")
        return 1536

    # ==================== Synchronous Methods ====================

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = self.sync_client.embeddings.create(
                model=self.model_name,
                input=text
            )
            embedding = response.data[0].embedding

            logger.debug(f"Generated embedding for text (length: {len(text)})")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            all_embeddings = []

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]

                response = self.sync_client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )

                # Extract embeddings maintaining order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Generated embeddings for batch {i//self.batch_size + 1}")

            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    # ==================== Asynchronous Methods ====================

    async def aembed_text(self, text: str) -> List[float]:
        """
        Asynchronously generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = await self.async_client.embeddings.create(
                model=self.model_name,
                input=text
            )
            embedding = response.data[0].embedding

            logger.debug(f"Generated embedding for text (length: {len(text)})")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            all_embeddings = []

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]

                response = await self.async_client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )

                # Extract embeddings maintaining order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Generated embeddings for batch {i//self.batch_size + 1}")

            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    async def aembed_texts_parallel(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in parallel batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            # Split into batches
            batches = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batches.append(batch)

            # Process batches in parallel
            tasks = []
            for batch in batches:
                task = self.async_client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

            # Extract and flatten embeddings
            all_embeddings = []
            for response in responses:
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            logger.info(f"Generated {len(all_embeddings)} embeddings in parallel")
            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings in parallel: {e}")
            raise


# Convenience function
def create_embedding_provider(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None
) -> EmbeddingProvider:
    """
    Create an embedding provider instance.

    Args:
        api_key: Optional API key override
        model: Optional model override
        provider: Optional provider override

    Returns:
        EmbeddingProvider instance
    """
    return EmbeddingProvider(
        api_key=api_key,
        model=model,
        provider=provider
    )