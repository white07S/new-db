"""
Configuration reader for the fullrag server.

This module reads and parses the config.toml file and provides
typed configuration objects using Pydantic models.

Supports environment variable interpolation using ${env:VAR_NAME} syntax.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import toml
from pydantic import BaseModel, Field


# Pattern to match ${env:VAR_NAME} or ${env:VAR_NAME:-default}
ENV_VAR_PATTERN: re.Pattern[str] = re.compile(
    r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"
)


def resolve_env_vars(value: str) -> str:
    """
    Resolve environment variable references in a string.

    Supports the following syntax:
    - ${env:VAR_NAME} - Required env var, raises error if not set
    - ${env:VAR_NAME:-default} - Optional env var with default value

    Args:
        value: String potentially containing env var references.

    Returns:
        str: String with env vars resolved.

    Raises:
        ValueError: If a required env var is not set.
    """

    def replace_env_var(match: re.Match[str]) -> str:
        var_name: str = match.group(1)
        default_value: str | None = match.group(2)

        env_value: str | None = os.environ.get(var_name)

        if env_value is not None:
            return env_value
        elif default_value is not None:
            return default_value
        else:
            raise ValueError(
                f"Environment variable '{var_name}' is not set and no default provided."
            )

    return ENV_VAR_PATTERN.sub(replace_env_var, value)


def resolve_env_vars_in_dict(data: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively resolve environment variables in a dictionary.

    Args:
        data: Dictionary potentially containing env var references.

    Returns:
        dict: Dictionary with all env vars resolved.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = resolve_env_vars(value)
        elif isinstance(value, dict):
            result[key] = resolve_env_vars_in_dict(value)
        elif isinstance(value, list):
            result[key] = [
                resolve_env_vars_in_dict(item) if isinstance(item, dict)
                else resolve_env_vars(item) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


class ModelsConfig(BaseModel):
    """Configuration for model names."""

    llm: str = Field(default="gpt-4o", description="LLM model name")
    embedding: str = Field(
        default="text-embedding-3-large", description="Embedding model name"
    )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding API."""

    api_url: str = Field(description="Embedding API URL")


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI provider."""

    api_base: str = Field(
        default="https://api.openai.com/v1", description="OpenAI API base URL"
    )
    api_key: str = Field(default="", description="OpenAI API key")
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    embedding: EmbeddingConfig = Field(
        default_factory=lambda: EmbeddingConfig(
            api_url="https://api.openai.com/v1/embeddings"
        )
    )

    def get_api_key(self) -> str:
        """
        Get API key.

        Returns:
            str: The API key.

        Raises:
            ValueError: If no API key is configured.
        """
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )
        return self.api_key


class AzureConfig(BaseModel):
    """Configuration for Azure OpenAI provider."""

    api_base: str = Field(description="Azure OpenAI API base URL")
    api_key: str = Field(default="", description="Azure OpenAI API key")
    api_version: str = Field(
        default="2025-03-01-preview", description="Azure OpenAI API version"
    )
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    embedding: EmbeddingConfig = Field(
        default_factory=lambda: EmbeddingConfig(
            api_url="https://api.openai.com/v1/embeddings"
        )
    )

    def get_api_key(self) -> str:
        """
        Get API key.

        Returns:
            str: The API key.

        Raises:
            ValueError: If no API key is configured.
        """
        if not self.api_key:
            raise ValueError(
                "Azure API key not found. Set AZURE_OPENAI_API_KEY environment variable."
            )
        return self.api_key


class ProvidersConfig(BaseModel):
    """Configuration for all providers."""

    default_provider: Literal["openai", "azure"] = Field(
        default="azure", description="Default provider to use"
    )
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    azure: AzureConfig | None = Field(default=None)


class RedisConfig(BaseModel):
    """Configuration for Redis."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    url: str = Field(default="redis://localhost:6379/0", description="Redis URL")


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector store."""

    url: str = Field(default="http://localhost:6333", description="Qdrant URL")
    api_key: str = Field(default="", description="Qdrant API key")
    collection_prefix: str = Field(default="fullrag_", description="Collection name prefix")


class PostgresConfig(BaseModel):
    """Configuration for PostgreSQL."""

    host: str = Field(default="localhost", description="Postgres host")
    port: int = Field(default=5432, description="Postgres port")
    user: str = Field(default="postgres", description="Postgres user")
    password: str = Field(default="", description="Postgres password")
    database: str = Field(default="app_db", description="Postgres database name")
    dsn: str = Field(default="", description="Postgres DSN")


class TestingConfig(BaseModel):
    """Configuration for testing."""

    test_db_url: str = Field(default="", description="Test SQLite database URL")
    test_doc_pdf: str = Field(default="", description="Test PDF document path")


class AppConfig(BaseModel):
    """Root application configuration."""

    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)


def _find_config_file() -> Path:
    """
    Find the config.toml file.

    Searches in the following order:
    1. Current working directory
    2. Server directory (relative to this file)
    3. Parent directories up to 3 levels

    Returns:
        Path: Path to the config.toml file.

    Raises:
        FileNotFoundError: If config.toml is not found.
    """
    # Check current working directory
    cwd_config: Path = Path.cwd() / "config.toml"
    if cwd_config.exists():
        return cwd_config

    # Check providers directory (same directory as this module)
    module_dir: Path = Path(__file__).parent
    providers_config: Path = module_dir / "config.toml"
    if providers_config.exists():
        return providers_config

    # Check server directory (relative to this module)
    server_config: Path = module_dir.parent.parent / "config.toml"
    if server_config.exists():
        return server_config

    # Check project directory (parent of providers)
    project_config: Path = module_dir.parent / "config.toml"
    if project_config.exists():
        return project_config

    # Check parent directories
    current: Path = Path.cwd()
    for _ in range(3):
        config_path: Path = current / "config.toml"
        if config_path.exists():
            return config_path
        current = current.parent

    raise FileNotFoundError(
        "config.toml not found. Please create one in the providers or project directory."
    )


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """
    Load configuration from TOML file.

    Resolves environment variable references using ${env:VAR_NAME} syntax.

    Args:
        config_path: Path to the config.toml file. If None, auto-detect.

    Returns:
        AppConfig: Parsed configuration object.

    Raises:
        FileNotFoundError: If config file is not found.
        ValueError: If config file is invalid or required env vars are missing.
    """
    if config_path is None:
        path: Path = _find_config_file()
    else:
        path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    config_data: dict[str, Any] = toml.load(path)
    resolved_data: dict[str, Any] = resolve_env_vars_in_dict(config_data)
    return AppConfig.model_validate(resolved_data)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Get cached application configuration.

    Returns:
        AppConfig: Cached configuration object.
    """
    return load_config()
