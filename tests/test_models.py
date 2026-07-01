import pytest

from llm_lab.models import ChatResponse, cost


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
