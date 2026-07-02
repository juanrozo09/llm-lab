# tests/test_cli.py
from typer.testing import CliRunner

from llm_lab.cli import PROVIDERS, app
from llm_lab.errors import ProviderUnavailableError
from llm_lab.models import ChatResponse

runner = CliRunner()


class FakeProvider:
    def __init__(self, name: str, model: str, response=None, error=None):
        self.name = name
        self.model = model
        self._response = response
        self._error = error

    async def chat(self, prompt: str, system: str | None = None) -> ChatResponse:
        if self._error is not None:
            raise self._error
        return self._response


def test_chat_command_prints_response_and_cost_table(monkeypatch):
    response = ChatResponse(
        text="hi there",
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_tokens=10,
        output_tokens=20,
        latency_ms=100.0,
    )
    monkeypatch.setitem(
        PROVIDERS,
        "anthropic",
        lambda model: FakeProvider("anthropic", model, response=response),
    )

    result = runner.invoke(app, ["chat", "hello", "--provider", "anthropic"])

    assert result.exit_code == 0
    assert "hi there" in result.output
    assert "claude-sonnet-4-6" in result.output


def test_chat_command_fallback_switches_provider(monkeypatch):
    fallback_response = ChatResponse(
        text="from openai",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=5,
        output_tokens=5,
        latency_ms=50.0,
    )
    monkeypatch.setitem(
        PROVIDERS,
        "anthropic",
        lambda model: FakeProvider(
            "anthropic",
            model,
            error=ProviderUnavailableError("anthropic", RuntimeError("down")),
        ),
    )
    monkeypatch.setitem(
        PROVIDERS,
        "openai",
        lambda model: FakeProvider("openai", model, response=fallback_response),
    )

    result = runner.invoke(
        app, ["chat", "hello", "--provider", "anthropic", "--fallback"]
    )

    assert result.exit_code == 0
    assert "from openai" in result.output
    assert "fallback" in result.output.lower()


def test_chat_command_fails_without_fallback(monkeypatch):
    monkeypatch.setitem(
        PROVIDERS,
        "anthropic",
        lambda model: FakeProvider(
            "anthropic",
            model,
            error=ProviderUnavailableError("anthropic", RuntimeError("down")),
        ),
    )

    result = runner.invoke(app, ["chat", "hello", "--provider", "anthropic"])

    assert result.exit_code == 1


def test_compare_command_shows_both_providers(monkeypatch):
    anthropic_response = ChatResponse(
        text="anthropic says hi",
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_tokens=10,
        output_tokens=10,
        latency_ms=80.0,
    )
    openai_response = ChatResponse(
        text="openai says hi",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=8,
        output_tokens=8,
        latency_ms=40.0,
    )
    monkeypatch.setitem(
        PROVIDERS,
        "anthropic",
        lambda model: FakeProvider(
            "anthropic", model, response=anthropic_response
        ),
    )
    monkeypatch.setitem(
        PROVIDERS,
        "openai",
        lambda model: FakeProvider("openai", model, response=openai_response),
    )

    result = runner.invoke(app, ["compare", "hello"])

    assert result.exit_code == 0
    assert "anthropic says hi" in result.output
    assert "openai says hi" in result.output


def test_compare_command_shows_error_for_failing_provider(monkeypatch):
    openai_response = ChatResponse(
        text="openai says hi",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=8,
        output_tokens=8,
        latency_ms=40.0,
    )
    monkeypatch.setitem(
        PROVIDERS,
        "anthropic",
        lambda model: FakeProvider(
            "anthropic",
            model,
            error=ProviderUnavailableError("anthropic", RuntimeError("down")),
        ),
    )
    monkeypatch.setitem(
        PROVIDERS,
        "openai",
        lambda model: FakeProvider("openai", model, response=openai_response),
    )

    result = runner.invoke(app, ["compare", "hello"])

    assert result.exit_code == 0
    assert "openai says hi" in result.output
    assert "Error" in result.output


def test_chat_command_fails_cleanly_when_both_providers_unavailable(monkeypatch):
    monkeypatch.setitem(
        PROVIDERS,
        "anthropic",
        lambda model: FakeProvider(
            "anthropic",
            model,
            error=ProviderUnavailableError("anthropic", RuntimeError("down")),
        ),
    )
    monkeypatch.setitem(
        PROVIDERS,
        "openai",
        lambda model: FakeProvider(
            "openai",
            model,
            error=ProviderUnavailableError("openai", RuntimeError("also down")),
        ),
    )

    result = runner.invoke(
        app, ["chat", "hello", "--provider", "anthropic", "--fallback"]
    )

    assert result.exit_code == 1
    assert "Error" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_chat_command_fails_cleanly_when_provider_missing_api_key(monkeypatch):
    def raise_missing_key(model):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setitem(PROVIDERS, "anthropic", raise_missing_key)

    result = runner.invoke(app, ["chat", "hello", "--provider", "anthropic"])

    assert result.exit_code == 1
    assert "Error" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_compare_command_shows_error_row_when_provider_missing_api_key(monkeypatch):
    openai_response = ChatResponse(
        text="openai says hi",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=8,
        output_tokens=8,
        latency_ms=40.0,
    )

    def raise_missing_key(model):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setitem(PROVIDERS, "anthropic", raise_missing_key)
    monkeypatch.setitem(
        PROVIDERS,
        "openai",
        lambda model: FakeProvider("openai", model, response=openai_response),
    )

    result = runner.invoke(app, ["compare", "hello"])

    assert result.exit_code == 0
    assert "openai says hi" in result.output
    assert "Error" in result.output


def test_benchmark_command_rejects_n_less_than_one():
    result = runner.invoke(
        app, ["benchmark", "hello", "--provider", "anthropic", "--n", "0"]
    )

    assert result.exit_code == 1
    assert "Error" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_benchmark_command_fails_cleanly_when_provider_missing_api_key(monkeypatch):
    def raise_missing_key(model):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setitem(PROVIDERS, "anthropic", raise_missing_key)

    result = runner.invoke(app, ["benchmark", "hello", "--provider", "anthropic"])

    assert result.exit_code == 1
    assert "Error" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_benchmark_command_runs_n_times_and_shows_summary(monkeypatch):
    call_count = 0

    class CountingFakeProvider(FakeProvider):
        async def chat(self, prompt: str, system: str | None = None) -> ChatResponse:
            nonlocal call_count
            call_count += 1
            return ChatResponse(
                text=f"run {call_count}",
                provider="anthropic",
                model="claude-sonnet-4-6",
                input_tokens=10,
                output_tokens=10,
                latency_ms=float(call_count * 10),
            )

    monkeypatch.setitem(
        PROVIDERS,
        "anthropic",
        lambda model: CountingFakeProvider("anthropic", model),
    )

    result = runner.invoke(
        app, ["benchmark", "hello", "--provider", "anthropic", "--n", "3"]
    )

    assert result.exit_code == 0
    assert call_count == 3
    assert "Mean" in result.output


def test_chat_command_model_override_reaches_provider(monkeypatch):
    monkeypatch.setitem(
        PROVIDERS,
        "anthropic",
        lambda model: FakeProvider(
            "anthropic",
            model,
            response=ChatResponse(
                text="hi",
                provider="anthropic",
                model=model,
                input_tokens=1,
                output_tokens=1,
                latency_ms=1.0,
            ),
        ),
    )

    result = runner.invoke(
        app,
        ["chat", "hello", "--provider", "anthropic", "--model", "claude-opus-4-6"],
    )

    assert result.exit_code == 0
    assert "claude-opus-4-6" in result.output


def test_chat_command_rejects_unknown_model():
    result = runner.invoke(
        app,
        ["chat", "hello", "--provider", "anthropic", "--model", "not-a-real-model"],
    )

    assert result.exit_code == 1
    assert "Error" in result.output


def test_chat_command_rejects_model_provider_mismatch():
    result = runner.invoke(
        app,
        ["chat", "hello", "--provider", "anthropic", "--model", "gpt-4o-mini"],
    )

    assert result.exit_code == 1
    assert "Error" in result.output


def test_chat_command_fallback_uses_secondary_default_model(monkeypatch):
    captured_models = {}

    def anthropic_factory(model):
        captured_models["anthropic"] = model
        return FakeProvider(
            "anthropic",
            model,
            error=ProviderUnavailableError("anthropic", RuntimeError("down")),
        )

    def openai_factory(model):
        captured_models["openai"] = model
        return FakeProvider(
            "openai",
            model,
            response=ChatResponse(
                text="from openai",
                provider="openai",
                model=model,
                input_tokens=5,
                output_tokens=5,
                latency_ms=50.0,
            ),
        )

    monkeypatch.setitem(PROVIDERS, "anthropic", anthropic_factory)
    monkeypatch.setitem(PROVIDERS, "openai", openai_factory)

    result = runner.invoke(
        app,
        [
            "chat",
            "hello",
            "--provider",
            "anthropic",
            "--model",
            "claude-opus-4-6",
            "--fallback",
        ],
    )

    assert result.exit_code == 0
    assert captured_models["anthropic"] == "claude-opus-4-6"
    assert captured_models["openai"] == "gpt-4o-mini"


def test_compare_command_model_overrides_reach_each_provider(monkeypatch):
    captured_models = {}

    def make_factory(name):
        def factory(model):
            captured_models[name] = model
            return FakeProvider(
                name,
                model,
                response=ChatResponse(
                    text=f"{name} says hi",
                    provider=name,
                    model=model,
                    input_tokens=1,
                    output_tokens=1,
                    latency_ms=1.0,
                ),
            )

        return factory

    monkeypatch.setitem(PROVIDERS, "anthropic", make_factory("anthropic"))
    monkeypatch.setitem(PROVIDERS, "openai", make_factory("openai"))

    result = runner.invoke(
        app,
        [
            "compare",
            "hello",
            "--anthropic-model",
            "claude-opus-4-6",
            "--openai-model",
            "gpt-4o",
        ],
    )

    assert result.exit_code == 0
    assert captured_models["anthropic"] == "claude-opus-4-6"
    assert captured_models["openai"] == "gpt-4o"


def test_compare_command_rejects_model_provider_mismatch():
    result = runner.invoke(
        app, ["compare", "hello", "--anthropic-model", "gpt-4o-mini"]
    )

    assert result.exit_code == 1
    assert "Error" in result.output


def test_benchmark_command_model_override_reaches_provider(monkeypatch):
    captured_models = {}

    def factory(model):
        captured_models["model"] = model
        return FakeProvider(
            "anthropic",
            model,
            response=ChatResponse(
                text="hi",
                provider="anthropic",
                model=model,
                input_tokens=1,
                output_tokens=1,
                latency_ms=1.0,
            ),
        )

    monkeypatch.setitem(PROVIDERS, "anthropic", factory)

    result = runner.invoke(
        app,
        [
            "benchmark",
            "hello",
            "--provider",
            "anthropic",
            "--model",
            "claude-opus-4-6",
            "--n",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert captured_models["model"] == "claude-opus-4-6"


def test_benchmark_command_rejects_unknown_model():
    result = runner.invoke(
        app,
        [
            "benchmark",
            "hello",
            "--provider",
            "anthropic",
            "--model",
            "not-a-real-model",
        ],
    )

    assert result.exit_code == 1
    assert "Error" in result.output
