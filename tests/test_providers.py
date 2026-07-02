# tests/test_providers.py
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from google.genai.errors import ServerError as GeminiServerError
from groq import APIConnectionError as GroqAPIConnectionError
from httpx import ConnectError as OllamaConnectError
from openai import APIConnectionError as OpenAIAPIConnectionError

from llm_lab.errors import ProviderUnavailableError
from llm_lab.providers.langchain_provider import LangChainProvider


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
    return AnthropicAPIConnectionError(request=request)


def make_openai_connection_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return OpenAIAPIConnectionError(request=request)


def make_groq_connection_error():
    request = httpx.Request(
        "POST", "https://api.groq.com/openai/v1/chat/completions"
    )
    return GroqAPIConnectionError(request=request)


def make_gemini_connection_error():
    return GeminiServerError(503, {"message": "Service unavailable"})


def make_ollama_connection_error():
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    return OllamaConnectError("Connection refused", request=request)


PROVIDER_CASES = [
    ("anthropic", "claude-sonnet-4-6", "ANTHROPIC_API_KEY", make_anthropic_connection_error),
    ("openai", "gpt-4o-mini", "OPENAI_API_KEY", make_openai_connection_error),
    ("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY", make_groq_connection_error),
    ("gemini", "gemini-2.5-flash", "GOOGLE_API_KEY", make_gemini_connection_error),
    ("ollama", "llama3.2", None, make_ollama_connection_error),
]

KEYED_PROVIDER_CASES = [case for case in PROVIDER_CASES if case[2] is not None]


@pytest.mark.parametrize("name,model,env_var,make_connection_error", PROVIDER_CASES)
async def test_chat_success(monkeypatch, name, model, env_var, make_connection_error):
    if env_var is not None:
        monkeypatch.setenv(env_var, "test-key")
    provider = LangChainProvider(name, model)
    # The underlying LangChain chat model (built by init_chat_model) is a
    # pydantic v2 model with strict field validation, so a plain
    # `provider._client.ainvoke = ...` assignment raises ValueError ("no
    # field 'ainvoke'"). Use object.__setattr__ to bypass pydantic's
    # overridden __setattr__ and stub the instance method directly.
    object.__setattr__(
        provider._client, "ainvoke", AsyncMock(return_value=make_ai_message())
    )

    response = await provider.chat("hi")

    assert response.text == "hello"
    assert response.provider == name
    assert response.model == model
    assert response.input_tokens == 5
    assert response.output_tokens == 7
    assert response.latency_ms >= 0


@pytest.mark.parametrize("name,model,env_var,make_connection_error", PROVIDER_CASES)
async def test_chat_raises_provider_unavailable_after_retries(
    monkeypatch, name, model, env_var, make_connection_error
):
    if env_var is not None:
        monkeypatch.setenv(env_var, "test-key")
    provider = LangChainProvider(name, model)
    object.__setattr__(
        provider._client, "ainvoke", AsyncMock(side_effect=make_connection_error())
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.chat("hi")

    assert exc_info.value.provider == name


@pytest.mark.parametrize(
    "name,model,env_var,make_connection_error", KEYED_PROVIDER_CASES
)
def test_provider_requires_api_key(
    monkeypatch, name, model, env_var, make_connection_error
):
    monkeypatch.delenv(env_var, raising=False)

    with pytest.raises(RuntimeError):
        LangChainProvider(name, model)


def test_ollama_provider_does_not_require_api_key():
    # Ollama has no API key at all — construction must succeed regardless
    # of any other provider's env vars being set or not.
    provider = LangChainProvider("ollama", "llama3.2")

    assert provider.name == "ollama"
    assert provider.model == "llama3.2"
