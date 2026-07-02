import pytest

from llm_lab.errors import ProviderUnavailableError
from llm_lab.models import ChatResponse, cost, resolve_model


def test_chat_response_holds_all_fields():
    response = ChatResponse(
        text="hello",
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_tokens=10,
        output_tokens=20,
        latency_ms=123.4,
    )
    assert response.text == "hello"
    assert response.provider == "anthropic"
    assert response.model == "claude-sonnet-4-6"
    assert response.input_tokens == 10
    assert response.output_tokens == 20
    assert response.latency_ms == 123.4


def test_cost_for_anthropic_model():
    response = ChatResponse(
        text="hi",
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_ms=1.0,
    )
    assert cost(response) == pytest.approx(3.00 + 15.00)


def test_cost_for_openai_model():
    response = ChatResponse(
        text="hi",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_ms=1.0,
    )
    assert cost(response) == pytest.approx(0.15 + 0.60)


def test_cost_scales_with_partial_tokens():
    response = ChatResponse(
        text="hi",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=500_000,
        output_tokens=0,
        latency_ms=1.0,
    )
    assert cost(response) == pytest.approx(0.075)


def test_provider_unavailable_error_holds_provider_and_cause():
    original = RuntimeError("boom")
    error = ProviderUnavailableError("anthropic", original)

    assert error.provider == "anthropic"
    assert error.original_error is original
    assert "anthropic" in str(error)


def test_resolve_model_returns_default_when_none():
    assert resolve_model("anthropic", None) == "claude-sonnet-4-6"
    assert resolve_model("openai", None) == "gpt-4o-mini"


def test_resolve_model_returns_valid_override():
    assert resolve_model("anthropic", "claude-opus-4-6") == "claude-opus-4-6"


def test_resolve_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model("anthropic", "not-a-real-model")


def test_resolve_model_rejects_provider_mismatch():
    with pytest.raises(ValueError, match="belongs to provider 'openai'"):
        resolve_model("anthropic", "gpt-4o-mini")


def test_resolve_model_returns_default_for_new_providers():
    assert resolve_model("groq", None) == "llama-3.3-70b-versatile"
    assert resolve_model("gemini", None) == "gemini-2.5-flash"
    assert resolve_model("ollama", None) == "llama3.2"


def test_cost_for_groq_model():
    response = ChatResponse(
        text="hi",
        provider="groq",
        model="llama-3.3-70b-versatile",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_ms=1.0,
    )
    assert cost(response) == pytest.approx(0.59 + 0.79)


def test_cost_for_ollama_model_is_zero():
    response = ChatResponse(
        text="hi",
        provider="ollama",
        model="llama3.2",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_ms=1.0,
    )
    assert cost(response) == 0.0
