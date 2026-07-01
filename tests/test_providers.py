# tests/test_providers.py
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import APIConnectionError

from llm_lab.errors import ProviderUnavailableError
from llm_lab.providers.anthropic import AnthropicProvider


def make_anthropic_message(text="hello", input_tokens=5, output_tokens=7):
    message = MagicMock()
    message.content = [MagicMock(text=text)]
    message.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return message


def make_connection_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return APIConnectionError(request=request)


async def test_anthropic_chat_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider()
    provider._client.messages.create = AsyncMock(
        return_value=make_anthropic_message()
    )

    response = await provider.chat("hi")

    assert response.text == "hello"
    assert response.provider == "anthropic"
    assert response.model == "claude-sonnet-4-6"
    assert response.input_tokens == 5
    assert response.output_tokens == 7
    assert response.latency_ms >= 0


async def test_anthropic_chat_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider()
    provider._client.messages.create = AsyncMock(
        side_effect=[make_connection_error(), make_anthropic_message()]
    )

    response = await provider.chat("hi")

    assert response.text == "hello"
    assert provider._client.messages.create.call_count == 2


async def test_anthropic_chat_raises_provider_unavailable_after_retries(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider()
    provider._client.messages.create = AsyncMock(side_effect=make_connection_error())

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.chat("hi")

    assert exc_info.value.provider == "anthropic"
    assert provider._client.messages.create.call_count == 3


def test_anthropic_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        AnthropicProvider()
