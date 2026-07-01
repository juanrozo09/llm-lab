# tests/test_providers.py
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import APIConnectionError
from openai import APIConnectionError as OpenAIAPIConnectionError

from llm_lab.errors import ProviderUnavailableError
from llm_lab.providers.anthropic import AnthropicProvider
from llm_lab.providers.openai import OpenAIProvider


def make_ai_message(text="hello", input_tokens=5, output_tokens=7):
    message = MagicMock()
    message.content = text
    message.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return message


def make_anthropic_connection_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return APIConnectionError(request=request)


async def test_anthropic_chat_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider()
    # ChatAnthropic is a pydantic v2 model with strict field validation, so a
    # plain `provider._client.ainvoke = ...` assignment raises ValueError
    # ("no field 'ainvoke'"). Use object.__setattr__ to bypass pydantic's
    # overridden __setattr__ and stub the instance method directly.
    object.__setattr__(
        provider._client, "ainvoke", AsyncMock(return_value=make_ai_message())
    )

    response = await provider.chat("hi")

    assert response.text == "hello"
    assert response.provider == "anthropic"
    assert response.model == "claude-sonnet-4-6"
    assert response.input_tokens == 5
    assert response.output_tokens == 7
    assert response.latency_ms >= 0


async def test_anthropic_chat_raises_provider_unavailable_after_retries(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider()
    object.__setattr__(
        provider._client,
        "ainvoke",
        AsyncMock(side_effect=make_anthropic_connection_error()),
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.chat("hi")

    assert exc_info.value.provider == "anthropic"


def test_anthropic_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        AnthropicProvider()


def make_openai_completion(text="hello", prompt_tokens=5, completion_tokens=7):
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=text))]
    completion.usage = MagicMock(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return completion


def make_openai_connection_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return OpenAIAPIConnectionError(request=request)


async def test_openai_chat_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIProvider()
    provider._client.chat.completions.create = AsyncMock(
        return_value=make_openai_completion()
    )

    response = await provider.chat("hi")

    assert response.text == "hello"
    assert response.provider == "openai"
    assert response.model == "gpt-4o-mini"
    assert response.input_tokens == 5
    assert response.output_tokens == 7


async def test_openai_chat_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIProvider()
    provider._client.chat.completions.create = AsyncMock(
        side_effect=[make_openai_connection_error(), make_openai_completion()]
    )

    response = await provider.chat("hi")

    assert response.text == "hello"
    assert provider._client.chat.completions.create.call_count == 2


async def test_openai_chat_raises_provider_unavailable_after_retries(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIProvider()
    provider._client.chat.completions.create = AsyncMock(
        side_effect=make_openai_connection_error()
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.chat("hi")

    assert exc_info.value.provider == "openai"
    assert provider._client.chat.completions.create.call_count == 3


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        OpenAIProvider()
