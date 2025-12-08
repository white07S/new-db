
from typing import Any, Iterable, Literal

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.responses import Response

from providers.config import AppConfig, AzureConfig, OpenAIConfig, get_config


class LLMProvider:
    """
    LLM Provider class to handle chat completions using OpenAI/Azure Responses API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        provider: Literal["openai", "azure"] | None = None,
    ) -> None:
        """
        Initialize the LLM Provider.
        
        Args:
            api_key: Optional API key override. If provided, uses this instead of config.
            model: Optional model override. If provided, uses this instead of config.
            provider: Optional provider override ("openai" or "azure").
        """
        self.config: AppConfig = get_config()
        self._api_key_override: str | None = api_key
        self._model_override: str | None = model
        self._provider_override: Literal["openai", "azure"] | None = provider
        self.client: AsyncOpenAI | AsyncAzureOpenAI = self._get_client()

    def _get_provider(self) -> Literal["openai", "azure"]:
        """Get the provider to use."""
        if self._provider_override:
            return self._provider_override
        return self.config.providers.default_provider

    def _get_client(self) -> AsyncOpenAI | AsyncAzureOpenAI:
        """
        Initialize the appropriate client based on configuration.
        """
        provider = self._get_provider()
        
        if provider == "azure":
            azure_conf: AzureConfig | None = self.config.providers.azure
            if not azure_conf:
                raise ValueError("Azure configuration is missing.")
            api_key = self._api_key_override or azure_conf.get_api_key()
            return AsyncAzureOpenAI(
                api_key=api_key,
                api_version=azure_conf.api_version,
                azure_endpoint=azure_conf.api_base,
            )
        else:
            openai_conf: OpenAIConfig = self.config.providers.openai
            api_key = self._api_key_override or openai_conf.get_api_key()
            return AsyncOpenAI(
                api_key=api_key, base_url=openai_conf.api_base
            )

    def _get_model_name(self, model: str | None = None) -> str:
        """
        Get the model name from config if not provided.
        """
        if model:
            return model
        
        # Use override if set
        if self._model_override:
            return self._model_override

        provider = self._get_provider()
        if provider == "azure":
            if self.config.providers.azure:
                return self.config.providers.azure.models.llm
        else:
            return self.config.providers.openai.models.llm
        
        # Fallback (should be covered by config validation usually)
        return "gpt-4o"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Response:
        """
        Standard chat interaction using Responses API.
        
        Args:
            messages: List of message dictionaries (role, content).
            model: Optional model override.
            stream: Whether to stream the response (currently responses API might have different streaming semantics).
            **kwargs: Additional arguments passed to responses.create.
            
        Returns:
            Response object from OpenAI API.
        """
        # Note: The Responses API uses 'input' instead of 'messages'
        # and it expects a list of dictionaries or specific types.
        # We assume 'messages' follows the standard chat format which is compatible.
        
        # Responses API currently doesn't support 'stream' in the same way as chat completions
        # explicitly in the signature shown earlier (it had stream=Literal[True] options but let's stick to basics first).
        
        return await self.client.responses.create(
            model=self._get_model_name(model),
            input=messages,
            **kwargs
        )

    async def structured_chat(
        self,
        messages: list[dict[str, Any]],
        response_format: Any,
        model: str | None = None,
        **kwargs: Any,
    ) -> Response:
        """
        Structured output chat interaction.
        
        Args:
            messages: List of message dictionaries.
            response_format: The structure definition (e.g. JSON schema or Pydantic model).
            model: Optional model override.
            **kwargs: Additional arguments.
        """
        # For Responses API, structured output is handled via the 'text' parameter's 'format'.
        # We map response_format to the expected structure.
        # If response_format is a dict with 'type': 'json_schema', passing it might work directly 
        # or it needs to be wrapped.
        
        # Based on research, `text` param takes `ResponseTextConfigParam` which has `format`.
        # `format` takes `ResponseFormatTextConfigParam`.
        
        return await self.client.responses.create(
            model=self._get_model_name(model),
            input=messages,
            text={"format": response_format},
            **kwargs
        )

    async def tool_chat(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[Any],
        model: str | None = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> Response:
        """
        Chat interaction with function calling/tools.
        
        Args:
            messages: List of message dictionaries.
            tools: List of tool definitions.
            model: Optional model override.
            tool_choice: Tool choice strategy.
            **kwargs: Additional arguments.
        """
        return await self.client.responses.create(
            model=self._get_model_name(model),
            input=messages,
            tools=tools,
            tool_choice=tool_choice, # type: ignore # checking if simple string works or needs object
            **kwargs
        )

