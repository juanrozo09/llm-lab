# llm_lab/providers/openai.py
import os
import time

from openai import APIConnectionError, AsyncOpenAI, RateLimitError

from llm_lab.errors import ProviderUnavailableError
from llm_lab.models import ChatResponse
from llm_lab.providers.base import Provider, retry_on_transient

MODEL = "gpt-4o-mini"


class OpenAIProvider(Provider):
    name = "openai"
    model = MODEL

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = AsyncOpenAI(api_key=api_key)

    @retry_on_transient(RateLimitError, APIConnectionError)
    async def _create(self, prompt: str, system: str | None):
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=1024,
            messages=messages,
        )

    async def chat(self, prompt: str, system: str | None = None) -> ChatResponse:
        start = time.perf_counter()
        try:
            completion = await self._create(prompt, system)
        except (RateLimitError, APIConnectionError) as exc:
            raise ProviderUnavailableError(self.name, exc) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return ChatResponse(
            text=completion.choices[0].message.content,
            provider=self.name,
            model=self.model,
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            latency_ms=latency_ms,
        )
