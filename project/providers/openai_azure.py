"""
OpenAI and Azure OpenAI async client configuration module.

This module provides factory functions to create AsyncOpenAI and AsyncAzureOpenAI
clients with proper configuration from config.toml file.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from openai import AsyncAzureOpenAI, AsyncOpenAI

from config import (
    AppConfig,
    AzureConfig,
    OpenAIConfig,
    get_config,
)


class APIProvider(str, Enum):
    """Supported API providers."""

    OPENAI = "openai"
    AZURE = "azure"


def create_async_openai_client() -> AsyncOpenAI:
    """
    Create an AsyncOpenAI client using configuration from config.toml.

    Returns:
        AsyncOpenAI: Configured async OpenAI client.

    Raises:
        ValueError: If OpenAI API key is not configured.
    """
    config: AppConfig = get_config()
    openai_config: OpenAIConfig = config.providers.openai

    return AsyncOpenAI(
        api_key=openai_config.get_api_key(),
        base_url=openai_config.api_base,
    )


def create_async_azure_openai_client(
    deployment_type: Literal["llm", "embedding"] = "llm",
) -> AsyncAzureOpenAI:
    """
    Create an AsyncAzureOpenAI client using configuration from config.toml.

    Args:
        deployment_type: Type of deployment to use ("llm" or "embedding").

    Returns:
        AsyncAzureOpenAI: Configured async Azure OpenAI client.

    Raises:
        ValueError: If Azure configuration is not available.
    """
    config: AppConfig = get_config()
    azure_config: AzureConfig | None = config.providers.azure

    if azure_config is None:
        raise ValueError("Azure configuration not found in config.toml")

    deployment: str = (
        azure_config.models.llm
        if deployment_type == "llm"
        else azure_config.models.embedding
    )

    return AsyncAzureOpenAI(
        api_key=azure_config.get_api_key(),
        api_version=azure_config.api_version,
        azure_endpoint=azure_config.api_base,
        azure_deployment=deployment,
    )


def get_client(
    provider: APIProvider | None = None,
    deployment_type: Literal["llm", "embedding"] = "llm",
) -> AsyncOpenAI | AsyncAzureOpenAI:
    """
    Get an async OpenAI client based on the specified provider.

    Args:
        provider: The API provider to use (openai or azure).
                  If None, uses the default provider from config.
        deployment_type: Type of deployment for Azure ("llm" or "embedding").

    Returns:
        AsyncOpenAI | AsyncAzureOpenAI: Configured async client.

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    if provider is None:
        config: AppConfig = get_config()
        provider_str: str = config.providers.default_provider
        provider = APIProvider(provider_str)

    if provider == APIProvider.OPENAI:
        return create_async_openai_client()
    elif provider == APIProvider.AZURE:
        return create_async_azure_openai_client(deployment_type=deployment_type)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_llm_client(
    provider: APIProvider | None = None,
) -> AsyncOpenAI | AsyncAzureOpenAI:
    """
    Get an async client configured for LLM operations.

    Args:
        provider: The API provider to use. If None, uses default from config.

    Returns:
        AsyncOpenAI | AsyncAzureOpenAI: Configured async client for LLM.
    """
    return get_client(provider=provider, deployment_type="llm")


def get_embedding_client(
    provider: APIProvider | None = None,
) -> AsyncOpenAI | AsyncAzureOpenAI:
    """
    Get an async client configured for embedding operations.

    Args:
        provider: The API provider to use. If None, uses default from config.

    Returns:
        AsyncOpenAI | AsyncAzureOpenAI: Configured async client for embeddings.
    """
    return get_client(provider=provider, deployment_type="embedding")
