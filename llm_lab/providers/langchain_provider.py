# llm_lab/providers/langchain_provider.py
import os
import time

from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import RateLimitError as AnthropicRateLimitError
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import RateLimitError as OpenAIRateLimitError

from llm_lab.errors import ProviderUnavailableError
from llm_lab.models import ChatResponse
from llm_lab.providers.base import Provider

API_KEY_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

RETRYABLE_ERRORS = (
    AnthropicRateLimitError,
    AnthropicAPIConnectionError,
    OpenAIRateLimitError,
    OpenAIAPIConnectionError,
)


class LangChainProvider(Provider):
    def __init__(self, name: str, model: str) -> None:
        env_var = API_KEY_ENV_VARS[name]
        api_key = os.environ.get(env_var)
        if not api_key:
            raise RuntimeError(f"{env_var} is not set")
        self.name = name
        self.model = model
        self._client = init_chat_model(
            model,
            model_provider=name,
            max_tokens=1024,
            max_retries=3,
            api_key=api_key,
        )

    async def _create(self, prompt: str, system: str | None):
        messages = []
        if system is not None:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        return await self._client.ainvoke(messages)

    async def chat(self, prompt: str, system: str | None = None) -> ChatResponse:
        start = time.perf_counter()
        try:
            message = await self._create(prompt, system)
        except RETRYABLE_ERRORS as exc:
            raise ProviderUnavailableError(self.name, exc) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return ChatResponse(
            text=message.content,
            provider=self.name,
            model=self.model,
            input_tokens=message.usage_metadata["input_tokens"],
            output_tokens=message.usage_metadata["output_tokens"],
            latency_ms=latency_ms,
        )
