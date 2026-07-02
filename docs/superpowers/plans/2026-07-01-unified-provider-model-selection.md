# Unified Provider + Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `AnthropicProvider`/`OpenAIProvider` into one `LangChainProvider` class built on `init_chat_model`, and let users pick a model per provider on the CLI, validated against `PRICING`.

**Architecture:** `llm_lab/models.py` gains a `provider` tag per `PRICING` entry, a `DEFAULT_MODELS` dict, and a `resolve_model()` validator. `llm_lab/providers/langchain_provider.py` replaces the two provider files with one class parameterized by `(name, model)`. `llm_lab/cli.py`'s `PROVIDERS` registry becomes model-parameterized (`Callable[[str], Provider]`), and `chat`/`benchmark` gain `--model`, `compare` gains `--anthropic-model`/`--openai-model`.

**Tech Stack:** `langchain.chat_models.init_chat_model` (new dependency: `langchain`), reusing already-installed `langchain-anthropic`/`langchain-openai`/`langchain-core`.

## Global Constraints

- `PRICING` entries each gain a `"provider"` key (`"anthropic"` or `"openai"`) alongside existing `"input"`/`"output"` floats.
- `DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o-mini"}` — today's hardcoded defaults, centralized.
- `resolve_model(provider: str, model: str | None) -> str`: returns the default when `model is None`; raises `ValueError` if `model` isn't in `PRICING` or belongs to a different provider.
- `LangChainProvider(name: str, model: str)`: builds its client via `init_chat_model(model, model_provider=name, max_tokens=1024, max_retries=3, api_key=api_key)`; reads its API key from `API_KEY_ENV_VARS[name]` (`{"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}`), raising plain `RuntimeError` if unset.
- `chat()` catches all four of `anthropic.RateLimitError`, `anthropic.APIConnectionError`, `openai.RateLimitError`, `openai.APIConnectionError` and wraps into `ProviderUnavailableError(self.name, exc)` — unchanged contract with `llm_lab/cli.py`.
- `chat --fallback`'s secondary provider always uses `resolve_model(other_name, None)` — its own default — never the primary's `--model` override.
- `PROVIDERS: dict[str, Callable[[str], Provider]]` — every call site is `PROVIDERS[provider](model_name)`, not `PROVIDERS[provider]()`.
- Every task in this plan must leave `uv run pytest -v` fully green — no task leaves a broken intermediate state.

---

### Task 1: `resolve_model` and `DEFAULT_MODELS` in `llm_lab/models.py`

**Files:**
- Modify: `llm_lab/models.py` (full rewrite)
- Modify: `tests/test_models.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `PRICING: dict[str, dict[str, str | float]]` (now with a `"provider"` key per entry), `DEFAULT_MODELS: dict[str, str]`, `resolve_model(provider: str, model: str | None) -> str`. Task 3's CLI changes call `resolve_model` directly.

- [ ] **Step 1: Write the failing tests**

Modify the import line at the top of `tests/test_models.py` from:

```python
from llm_lab.models import ChatResponse, cost
```

to:

```python
from llm_lab.models import ChatResponse, cost, resolve_model
```

Then append these tests to the end of `tests/test_models.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v -k resolve_model`
Expected: FAIL with `ImportError: cannot import name 'resolve_model'`

- [ ] **Step 3: Rewrite `llm_lab/models.py`**

```python
from pydantic import BaseModel


class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


PRICING: dict[str, dict[str, str | float]] = {
    # Anthropic Claude
    "claude-opus-4-6": {"provider": "anthropic", "input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"provider": "anthropic", "input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"provider": "anthropic", "input": 1.00, "output": 5.00},

    # OpenAI GPT-5 family
    "gpt-5-2": {"provider": "openai", "input": 1.75, "output": 14.00},
    "gpt-5-1": {"provider": "openai", "input": 1.25, "output": 10.00},
    "gpt-5": {"provider": "openai", "input": 1.25, "output": 10.00},
    "gpt-5-mini": {"provider": "openai", "input": 0.25, "output": 2.00},
    "gpt-5-nano": {"provider": "openai", "input": 0.05, "output": 0.40},

    # OpenAI GPT-4 family
    "gpt-4.1": {"provider": "openai", "input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"provider": "openai", "input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"provider": "openai", "input": 0.10, "output": 0.40},

    "gpt-4o": {"provider": "openai", "input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"provider": "openai", "input": 0.15, "output": 0.60},

    # OpenAI reasoning models
    "o3": {"provider": "openai", "input": 2.00, "output": 8.00},
    "o4-mini": {"provider": "openai", "input": 1.10, "output": 4.40},
    "o3-mini": {"provider": "openai", "input": 1.10, "output": 4.40},
    "o1": {"provider": "openai", "input": 15.00, "output": 60.00},
    "o1-mini": {"provider": "openai", "input": 1.10, "output": 4.40},
}


DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
}


def cost(response: ChatResponse) -> float:
    rates = PRICING[response.model]
    return (
        response.input_tokens / 1_000_000 * rates["input"]
        + response.output_tokens / 1_000_000 * rates["output"]
    )


def resolve_model(provider: str, model: str | None) -> str:
    if model is None:
        return DEFAULT_MODELS[provider]
    entry = PRICING.get(model)
    if entry is None:
        raise ValueError(f"Unknown model '{model}'")
    if entry["provider"] != provider:
        raise ValueError(
            f"Model '{model}' belongs to provider '{entry['provider']}', not '{provider}'"
        )
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: 9 passed (was 5)

- [ ] **Step 5: Run the full suite to confirm nothing else broke**

Run: `uv run pytest -v`
Expected: 26 passed (was 22 — only `test_models.py` grew; `PRICING`'s new `"provider"` key doesn't affect `cost()`'s existing `rates["input"]`/`rates["output"]` lookups, so `llm_lab/cli.py` and the provider tests are unaffected).

- [ ] **Step 6: Commit**

```bash
git add llm_lab/models.py tests/test_models.py
git commit -m "feat: add resolve_model and per-model provider tagging to PRICING"
```

---

### Task 2: `LangChainProvider` — one class for both providers

**Files:**
- Create: `llm_lab/providers/langchain_provider.py`
- Modify: `tests/test_providers.py` (full rewrite)
- Modify: `pyproject.toml` (add `langchain` dependency)

**Interfaces:**
- Consumes: `Provider` from `llm_lab.providers.base` (unchanged ABC). `ChatResponse` from `llm_lab.models`. `ProviderUnavailableError` from `llm_lab.errors`.
- Produces: `LangChainProvider(name: str, model: str)` — constructor takes both a provider name (`"anthropic"`/`"openai"`) and a model string; exposes `.name`, `.model` (instance attributes now, not class attributes), `async def chat(...)` matching the `Provider` ABC. Task 3's `cli.py` constructs it as `LangChainProvider(provider_name, model_name)`.
- Note: `llm_lab/providers/anthropic.py` and `llm_lab/providers/openai.py` are NOT deleted in this task (Task 4 removes them, once `cli.py` stops importing them in Task 3). This task only stops *testing* them — `llm_lab/cli.py` still imports and uses the old classes unchanged until Task 3, so the full suite stays green throughout.

- [ ] **Step 1: Add the `langchain` dependency**

In `pyproject.toml`, replace the `dependencies` list:

```toml
dependencies = [
    "typer>=0.15",
    "rich>=13.9",
    "pydantic>=2.9",
    "anthropic>=0.40",
    "openai>=1.60",
    "langchain>=0.3",
    "langchain-anthropic>=0.3",
    "langchain-openai>=0.3",
]
```

Run: `uv sync`
Expected: resolves `langchain` (and any new transitive deps) with no errors.

- [ ] **Step 2: Write the failing tests**

Replace the full contents of `tests/test_providers.py` with:

```python
# tests/test_providers.py
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import APIConnectionError as AnthropicAPIConnectionError
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


PROVIDER_CASES = [
    ("anthropic", "claude-sonnet-4-6", "ANTHROPIC_API_KEY", make_anthropic_connection_error),
    ("openai", "gpt-4o-mini", "OPENAI_API_KEY", make_openai_connection_error),
]


@pytest.mark.parametrize("name,model,env_var,make_connection_error", PROVIDER_CASES)
async def test_chat_success(monkeypatch, name, model, env_var, make_connection_error):
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
    monkeypatch.setenv(env_var, "test-key")
    provider = LangChainProvider(name, model)
    object.__setattr__(
        provider._client, "ainvoke", AsyncMock(side_effect=make_connection_error())
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.chat("hi")

    assert exc_info.value.provider == name


@pytest.mark.parametrize("name,model,env_var,make_connection_error", PROVIDER_CASES)
def test_provider_requires_api_key(monkeypatch, name, model, env_var, make_connection_error):
    monkeypatch.delenv(env_var, raising=False)

    with pytest.raises(RuntimeError):
        LangChainProvider(name, model)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_providers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_lab.providers.langchain_provider'`

- [ ] **Step 4: Write the implementation**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_providers.py -v`
Expected: 6 passed (3 parametrized test functions × 2 provider cases)

- [ ] **Step 6: Run the full suite to confirm nothing else broke**

Run: `uv run pytest -v`
Expected: 26 passed (unchanged from Task 1 — `llm_lab/cli.py` still uses the old `AnthropicProvider`/`OpenAIProvider` classes directly, untouched by this task, so its 11 tests are unaffected).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock llm_lab/providers/langchain_provider.py tests/test_providers.py
git commit -m "feat: add LangChainProvider, a single class for both providers"
```

---

### Task 3: CLI model selection (`--model`, `--anthropic-model`, `--openai-model`)

**Files:**
- Modify: `llm_lab/cli.py` (full rewrite)
- Modify: `tests/test_cli.py` (full rewrite)

**Interfaces:**
- Consumes: `LangChainProvider` from `llm_lab.providers.langchain_provider` (Task 2). `resolve_model` from `llm_lab.models` (Task 1). `Provider` from `llm_lab.providers.base` (for the `PROVIDERS` type hint).
- Produces: `PROVIDERS: dict[str, Callable[[str], Provider]]` — same name as before, now model-parameterized. `app`, `console`, `PROVIDER_ERRORS`, `_render_cost_table` unchanged in shape/behavior.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `tests/test_cli.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — the existing tests fail with `TypeError: <lambda>() takes 0 positional arguments but 1 was given` (current `cli.py` still calls `PROVIDERS[provider]()` with no args) and the new `--model`/`--anthropic-model`/`--openai-model` tests fail with a Typer usage error (those options don't exist yet).

- [ ] **Step 3: Rewrite `llm_lab/cli.py`**

```python
# llm_lab/cli.py
import asyncio
from collections.abc import Callable

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from llm_lab.errors import ProviderUnavailableError
from llm_lab.models import ChatResponse, cost, resolve_model
from llm_lab.providers.base import Provider
from llm_lab.providers.langchain_provider import LangChainProvider

app = typer.Typer(help="Chat with, compare, and benchmark LLM providers.")
console = Console()

PROVIDERS: dict[str, Callable[[str], Provider]] = {
    "anthropic": lambda model: LangChainProvider("anthropic", model),
    "openai": lambda model: LangChainProvider("openai", model),
}

# Provider construction raises a plain RuntimeError when an API key is
# missing; chat()/gather() calls raise ProviderUnavailableError after
# retries are exhausted. Both should surface as a clean CLI error rather
# than a raw traceback.
PROVIDER_ERRORS = (RuntimeError, ProviderUnavailableError)


@app.callback()
def _callback() -> None:
    """Chat with, compare, and benchmark LLM providers."""


def _render_cost_table(response: ChatResponse) -> Table:
    table = Table(title="Cost Summary")
    table.add_column("Model")
    table.add_column("Latency (ms)")
    table.add_column("Input tokens")
    table.add_column("Output tokens")
    table.add_column("Cost ($)")
    table.add_row(
        response.model,
        f"{response.latency_ms:.1f}",
        str(response.input_tokens),
        str(response.output_tokens),
        f"{cost(response):.6f}",
    )
    return table


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="Prompt to send"),
    provider: str = typer.Option(
        "anthropic", help="Provider to use: anthropic or openai"
    ),
    model: str | None = typer.Option(
        None, "--model", help="Override the model for the selected provider"
    ),
    system: str | None = typer.Option(None, help="Optional system prompt"),
    fallback: bool = typer.Option(
        False, "--fallback", help="Retry with the other provider on failure"
    ),
) -> None:
    if provider not in PROVIDERS:
        console.print(f"[red]Error:[/red] unknown provider '{provider}'")
        raise typer.Exit(code=1)
    try:
        model_name = resolve_model(provider, model)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    async def run() -> tuple[ChatResponse, bool]:
        try:
            instance = PROVIDERS[provider](model_name)
            return await instance.chat(prompt, system), False
        except PROVIDER_ERRORS:
            if not fallback:
                raise
            other_name = "openai" if provider == "anthropic" else "anthropic"
            other_model = resolve_model(other_name, None)
            other = PROVIDERS[other_name](other_model)
            return await other.chat(prompt, system), True

    try:
        response, used_fallback = asyncio.run(run())
    except PROVIDER_ERRORS as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    title = f"{response.provider} ({response.model})"
    if used_fallback:
        title += " — fallback used"
    console.print(Panel(escape(response.text), title=title))
    console.print(_render_cost_table(response))


@app.command()
def compare(
    prompt: str = typer.Argument(..., help="Prompt to send to both providers"),
    system: str | None = typer.Option(None, help="Optional system prompt"),
    anthropic_model: str | None = typer.Option(
        None, "--anthropic-model", help="Override Anthropic's model"
    ),
    openai_model: str | None = typer.Option(
        None, "--openai-model", help="Override OpenAI's model"
    ),
) -> None:
    names = list(PROVIDERS.keys())
    overrides = {"anthropic": anthropic_model, "openai": openai_model}
    models: dict[str, str] = {}
    for name in names:
        try:
            models[name] = resolve_model(name, overrides.get(name))
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)

    async def call(name: str) -> ChatResponse:
        return await PROVIDERS[name](models[name]).chat(prompt, system)

    async def run() -> list[ChatResponse | BaseException]:
        return await asyncio.gather(
            *(call(name) for name in names),
            return_exceptions=True,
        )

    results = asyncio.run(run())

    table = Table(title="Provider Comparison")
    table.add_column("Provider")
    table.add_column("Text", no_wrap=True)
    table.add_column("Latency (ms)")
    table.add_column("Input tokens")
    table.add_column("Output tokens")
    table.add_column("Cost ($)")
    for name, result in zip(names, results):
        if isinstance(result, BaseException):
            error_text = f"[red]Error: {escape(str(result))}[/red]"
            table.add_row(name, error_text, "-", "-", "-", "-")
        else:
            table.add_row(
                name,
                escape(result.text),
                f"{result.latency_ms:.1f}",
                str(result.input_tokens),
                str(result.output_tokens),
                f"{cost(result):.6f}",
            )
    console.print(table)


@app.command()
def benchmark(
    prompt: str = typer.Argument(..., help="Prompt to run repeatedly"),
    provider: str = typer.Option(
        "anthropic", help="Provider to use: anthropic or openai"
    ),
    model: str | None = typer.Option(
        None, "--model", help="Override the model for the selected provider"
    ),
    n: int = typer.Option(5, "--n", help="Number of sequential runs"),
) -> None:
    if provider not in PROVIDERS:
        console.print(f"[red]Error:[/red] unknown provider '{provider}'")
        raise typer.Exit(code=1)
    if n < 1:
        console.print(f"[red]Error:[/red] --n must be at least 1, got {n}")
        raise typer.Exit(code=1)
    try:
        model_name = resolve_model(provider, model)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    async def run() -> list[ChatResponse]:
        instance = PROVIDERS[provider](model_name)
        responses = []
        for _ in range(n):
            responses.append(await instance.chat(prompt))
        return responses

    try:
        responses = asyncio.run(run())
    except PROVIDER_ERRORS as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    table = Table(title=f"Benchmark: {provider} ({n} runs)")
    table.add_column("Run")
    table.add_column("Latency (ms)")
    table.add_column("Input tokens")
    table.add_column("Output tokens")
    table.add_column("Cost ($)")
    for i, response in enumerate(responses, start=1):
        table.add_row(
            str(i),
            f"{response.latency_ms:.1f}",
            str(response.input_tokens),
            str(response.output_tokens),
            f"{cost(response):.6f}",
        )

    mean_latency = sum(r.latency_ms for r in responses) / len(responses)
    total_cost = sum(cost(r) for r in responses)
    table.add_row("Mean / Total", f"{mean_latency:.1f}", "-", "-", f"{total_cost:.6f}")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 19 passed (was 11)

- [ ] **Step 5: Run the full suite to confirm nothing else broke**

Run: `uv run pytest -v`
Expected: 34 passed (was 26 — only `test_cli.py` grew; `test_models.py` and `test_providers.py` are untouched by this task).

- [ ] **Step 6: Commit**

```bash
git add llm_lab/cli.py tests/test_cli.py
git commit -m "feat: add --model/--anthropic-model/--openai-model CLI flags"
```

---

### Task 4: Delete the now-dead per-provider files

**Files:**
- Delete: `llm_lab/providers/anthropic.py`
- Delete: `llm_lab/providers/openai.py`

**Interfaces:**
- Consumes: nothing — by this point neither file is imported anywhere (`tests/test_providers.py` was rewritten in Task 2 to test `LangChainProvider`; `llm_lab/cli.py` was rewritten in Task 3 to use `LangChainProvider`).
- Produces: nothing — this is the final cleanup task.

- [ ] **Step 1: Confirm nothing still imports the old files**

Run: `grep -rn "providers.anthropic\|providers.openai" llm_lab/ tests/`
Expected: no output (no matches) — if this prints anything, STOP and report BLOCKED rather than deleting, since something still depends on the old files.

- [ ] **Step 2: Delete the old provider files**

```bash
git rm llm_lab/providers/anthropic.py llm_lab/providers/openai.py
```

- [ ] **Step 3: Run the full suite to confirm nothing broke**

Run: `uv run pytest -v`
Expected: 34 passed, 0 warnings (unchanged from Task 3 — this task only deletes now-unused files).

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove dead per-provider files, superseded by LangChainProvider"
```
