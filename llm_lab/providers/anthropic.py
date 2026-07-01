# llm_lab/providers/anthropic.py
import os
import time

from anthropic import APIConnectionError, RateLimitError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from llm_lab.errors import ProviderUnavailableError
from llm_lab.models import ChatResponse
from llm_lab.providers.base import Provider

MODEL = "claude-sonnet-4-6"


class AnthropicProvider(Provider):
    name = "anthropic"
    model = MODEL

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = ChatAnthropic(
            model=MODEL, max_tokens=1024, max_retries=3, api_key=api_key
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
        except (RateLimitError, APIConnectionError) as exc:
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
