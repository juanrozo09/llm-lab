# llm_lab/providers/anthropic.py
import os
import time

from anthropic import APIConnectionError, AsyncAnthropic, RateLimitError

from llm_lab.errors import ProviderUnavailableError
from llm_lab.models import ChatResponse
from llm_lab.providers.base import Provider, retry_on_transient

MODEL = "claude-sonnet-4-6"


class AnthropicProvider(Provider):
    name = "anthropic"
    model = MODEL

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = AsyncAnthropic(api_key=api_key)

    @retry_on_transient(RateLimitError, APIConnectionError)
    async def _create(self, prompt: str, system: str | None):
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system
        return await self._client.messages.create(**kwargs)

    async def chat(self, prompt: str, system: str | None = None) -> ChatResponse:
        start = time.perf_counter()
        try:
            message = await self._create(prompt, system)
        except (RateLimitError, APIConnectionError) as exc:
            raise ProviderUnavailableError(self.name, exc) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return ChatResponse(
            text=message.content[0].text,
            provider=self.name,
            model=self.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=latency_ms,
        )
