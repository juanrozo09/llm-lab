# Groq, Gemini, and Ollama Providers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Groq, Gemini, and Ollama as fully supported providers, including Ollama's no-API-key local case, and let `compare` select a subset of the (now 5) registered providers.

**Architecture:** `LangChainProvider` gains an optional API-key requirement (`API_KEY_ENV_VARS[name]` can be `None`) and a provider-ID translation table for `init_chat_model`, since LangChain's own registry key for Gemini differs from our short name. `RETRYABLE_ERRORS` grows to include each new SDK's transient-error classes. `PROVIDERS`/`PRICING`/`DEFAULT_MODELS` grow to 5 entries each. `compare` gains a `--providers` selector (default unchanged) and three new per-provider model-override flags.

**Tech Stack:** `langchain-groq`, `langchain-google-genai`, `langchain-ollama` (new dependencies), reusing the existing `init_chat_model`-based `LangChainProvider`.

## Global Constraints

- `API_KEY_ENV_VARS: dict[str, str | None]` — `{"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY", "ollama": None}`. `LangChainProvider.__init__` only checks/requires a key when the value isn't `None`; for `None` it omits `api_key` from the `init_chat_model(...)` call entirely (never passes `api_key=None`).
- `MODEL_PROVIDER_IDS: dict[str, str]` translates our short name to LangChain's `model_provider` identifier — `{"anthropic": "anthropic", "openai": "openai", "groq": "groq", "gemini": "google_genai", "ollama": "ollama"}`.
- `DEFAULT_MODELS` additions: `"groq": "llama-3.3-70b-versatile"`, `"gemini": "gemini-2.5-flash"`, `"ollama": "llama3.2"`.
- `PRICING` additions (USD per 1M tokens, as of 2026-07-01): `"llama-3.3-70b-versatile": {"provider": "groq", "input": 0.59, "output": 0.79}`, `"gemini-2.5-flash": {"provider": "gemini", "input": 0.30, "output": 2.50}`, `"llama3.2": {"provider": "ollama", "input": 0.0, "output": 0.0}`.
- `compare`'s new `--providers` option: comma-separated, default `"anthropic,openai"` — identical behavior to today when omitted. Unknown name → `[red]Error:[/red]` + exit 1, same pattern as existing checks.
- `compare` gains `--groq-model`, `--gemini-model`, `--ollama-model` alongside existing `--anthropic-model`/`--openai-model`. An override for a provider not in `--providers`' selected set is simply unused (not an error).
- **Exact exception class names/import paths for Groq, Gemini, and Ollama's connection-failure behavior are not confirmed** — Task 2 requires verifying these against the actually-installed packages before finalizing `RETRYABLE_ERRORS` and the test helpers, exactly like every other provider integration built in this codebase so far. Do not trust this plan's guesses blindly.
- Every task must leave `uv run pytest -v` fully green.

---

### Task 1: `DEFAULT_MODELS`/`PRICING` entries for the three new providers

**Files:**
- Modify: `llm_lab/models.py` (full rewrite)
- Modify: `tests/test_models.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `PRICING`/`DEFAULT_MODELS` entries for `"groq"`, `"gemini"`, `"ollama"`. `resolve_model` and `cost` are unchanged (both already operate generically over whatever's in these dicts). Task 2/3 rely on `resolve_model("groq"/"gemini"/"ollama", ...)` working correctly once this task lands.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v -k "new_providers or groq_model or ollama_model_is_zero"`
Expected: FAIL — `resolve_model("groq", None)` raises `KeyError` (no `"groq"` entry in `DEFAULT_MODELS` yet), and the cost tests fail with `KeyError` on `PRICING[response.model]`.

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

    # Groq
    "llama-3.3-70b-versatile": {"provider": "groq", "input": 0.59, "output": 0.79},

    # Google Gemini
    "gemini-2.5-flash": {"provider": "gemini", "input": 0.30, "output": 2.50},

    # Ollama (local, no billing)
    "llama3.2": {"provider": "ollama", "input": 0.0, "output": 0.0},
}


DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.2",
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
Expected: 12 passed (was 9)

- [ ] **Step 5: Run the full suite to confirm nothing else broke**

Run: `uv run pytest -v`
Expected: 37 passed (was 34 — only `test_models.py` grew; nothing else references the new keys yet).

- [ ] **Step 6: Commit**

```bash
git add llm_lab/models.py tests/test_models.py
git commit -m "feat: add DEFAULT_MODELS/PRICING entries for groq, gemini, and ollama"
```

---

### Task 2: `LangChainProvider` support for Groq, Gemini, and Ollama

**Files:**
- Modify: `llm_lab/providers/langchain_provider.py` (full rewrite)
- Modify: `tests/test_providers.py` (full rewrite)
- Modify: `pyproject.toml` (add three dependencies)

**Interfaces:**
- Consumes: `Provider` from `llm_lab.providers.base` (unchanged). `ChatResponse` from `llm_lab.models` (unchanged). `ProviderUnavailableError` from `llm_lab.errors` (unchanged).
- Produces: `LangChainProvider("groq"/"gemini"/"ollama", model)` construction works — `.name`, `.model`, `async def chat(...)` unchanged in shape. Task 3's `cli.py` constructs these the same way it already constructs `"anthropic"`/`"openai"` instances.

- [ ] **Step 0: Verify the real exception types before writing anything**

This plan's guesses for import paths below are NOT confirmed. Before writing `tests/test_providers.py` or `llm_lab/providers/langchain_provider.py`, add the three new dependencies (Step 1) and then check, e.g.:

```bash
uv run python -c "from groq import RateLimitError, APIConnectionError; print('groq ok')"
uv run python -c "from google.genai.errors import ClientError, ServerError; print('genai ok')"
uv run python -c "import inspect; from langchain_ollama import ChatOllama; print(inspect.signature(ChatOllama.__init__))"
```

For Ollama specifically, there is no "rate limit" concept (it's a local server) — the realistic failure mode is the server not running. Check what exception type surfaces when `ChatOllama(...).ainvoke(...)` can't connect (e.g. try invoking against `http://localhost:11434` with nothing listening, and read the traceback) — it's likely `httpx.ConnectError` bubbling up through `langchain-ollama`'s async client, but confirm rather than assume. Adjust the code in Steps 1–4 below to match whatever you find, and document your findings in your report.

- [ ] **Step 1: Add the three new dependencies**

In `pyproject.toml`, replace the `dependencies` list:

```toml
dependencies = [
    "typer>=0.15",
    "rich>=13.9",
    "pydantic>=2.9",
    "anthropic>=0.40",
    "openai>=1.60",
    "langchain>=1.0,<2",
    "langchain-anthropic>=0.3",
    "langchain-openai>=0.3",
    "langchain-groq>=0.2",
    "langchain-google-genai>=2.0",
    "langchain-ollama>=0.2",
]
```

Run: `uv sync`
Expected: resolves the three new packages with no errors.

- [ ] **Step 2: Write the failing tests**

Replace the full contents of `tests/test_providers.py` with (adjust the exception imports/constructions per your Step 0 findings if they differ):

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_providers.py -v`
Expected: FAIL — `LangChainProvider` doesn't yet support `"groq"`/`"gemini"`/`"ollama"` (`KeyError` on `API_KEY_ENV_VARS[name]` for the new names).

- [ ] **Step 4: Rewrite `llm_lab/providers/langchain_provider.py`**

(Adjust the exception imports per your Step 0 findings if they differ from this.)

```python
# llm_lab/providers/langchain_provider.py
import os
import time

from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import RateLimitError as AnthropicRateLimitError
from google.genai.errors import ClientError as GeminiClientError
from google.genai.errors import ServerError as GeminiServerError
from groq import APIConnectionError as GroqAPIConnectionError
from groq import RateLimitError as GroqRateLimitError
from httpx import ConnectError as OllamaConnectError
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import RateLimitError as OpenAIRateLimitError

from llm_lab.errors import ProviderUnavailableError
from llm_lab.models import ChatResponse
from llm_lab.providers.base import Provider

API_KEY_ENV_VARS: dict[str, str | None] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "ollama": None,
}

MODEL_PROVIDER_IDS: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "groq": "groq",
    "gemini": "google_genai",
    "ollama": "ollama",
}

RETRYABLE_ERRORS = (
    AnthropicRateLimitError,
    AnthropicAPIConnectionError,
    OpenAIRateLimitError,
    OpenAIAPIConnectionError,
    GroqRateLimitError,
    GroqAPIConnectionError,
    GeminiClientError,
    GeminiServerError,
    OllamaConnectError,
)


class LangChainProvider(Provider):
    def __init__(self, name: str, model: str) -> None:
        env_var = API_KEY_ENV_VARS[name]
        kwargs: dict = {
            "model_provider": MODEL_PROVIDER_IDS[name],
            "max_tokens": 1024,
            "max_retries": 3,
        }
        if env_var is not None:
            api_key = os.environ.get(env_var)
            if not api_key:
                raise RuntimeError(f"{env_var} is not set")
            kwargs["api_key"] = api_key
        self.name = name
        self.model = model
        self._client = init_chat_model(model, **kwargs)

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
Expected: 15 passed (was 6 — `PROVIDER_CASES` now has 5 entries used by 2 test functions = 10, `KEYED_PROVIDER_CASES` has 4 entries used by 1 test function = 4, plus 1 standalone Ollama test = 15 total).

- [ ] **Step 6: Run the full suite to confirm nothing else broke**

Run: `uv run pytest -v`
Expected: 46 passed (was 37 — `llm_lab/cli.py` still only registers `"anthropic"`/`"openai"` in `PROVIDERS`, untouched by this task, so its tests are unaffected).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock llm_lab/providers/langchain_provider.py tests/test_providers.py
git commit -m "feat: add Groq, Gemini, and Ollama support to LangChainProvider"
```

---

### Task 3: CLI support for the 3 new providers + `compare --providers`

**Files:**
- Modify: `llm_lab/cli.py` (full rewrite)
- Modify: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `LangChainProvider` from `llm_lab.providers.langchain_provider` (Task 2, now supports `"groq"`/`"gemini"`/`"ollama"`). `resolve_model`/`DEFAULT_MODELS`/`PRICING` from `llm_lab.models` (Task 1, now has entries for the 3 new providers).
- Produces: `PROVIDERS` grows to 5 entries. `compare` gains `--providers` (default `"anthropic,openai"`) and `--groq-model`/`--gemini-model`/`--ollama-model`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_compare_command_providers_flag_selects_subset(monkeypatch):
    groq_response = ChatResponse(
        text="groq says hi",
        provider="groq",
        model="llama-3.3-70b-versatile",
        input_tokens=6,
        output_tokens=6,
        latency_ms=30.0,
    )
    gemini_response = ChatResponse(
        text="gemini says hi",
        provider="gemini",
        model="gemini-2.5-flash",
        input_tokens=7,
        output_tokens=7,
        latency_ms=35.0,
    )
    monkeypatch.setitem(
        PROVIDERS,
        "groq",
        lambda model: FakeProvider("groq", model, response=groq_response),
    )
    monkeypatch.setitem(
        PROVIDERS,
        "gemini",
        lambda model: FakeProvider("gemini", model, response=gemini_response),
    )

    result = runner.invoke(app, ["compare", "hello", "--providers", "groq,gemini"])

    assert result.exit_code == 0
    assert "groq says hi" in result.output
    assert "gemini says hi" in result.output
    assert "anthropic" not in result.output.lower()


def test_compare_command_rejects_unknown_provider_in_providers_flag():
    result = runner.invoke(
        app, ["compare", "hello", "--providers", "anthropic,not-a-real-provider"]
    )

    assert result.exit_code == 1
    assert "Error" in result.output


def test_compare_command_new_provider_model_overrides_reach_provider(monkeypatch):
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

    monkeypatch.setitem(PROVIDERS, "groq", make_factory("groq"))
    monkeypatch.setitem(PROVIDERS, "gemini", make_factory("gemini"))
    monkeypatch.setitem(PROVIDERS, "ollama", make_factory("ollama"))

    result = runner.invoke(
        app,
        [
            "compare",
            "hello",
            "--providers",
            "groq,gemini,ollama",
            "--groq-model",
            "llama-3.3-70b-versatile",
            "--gemini-model",
            "gemini-2.5-flash",
            "--ollama-model",
            "llama3.2",
        ],
    )

    assert result.exit_code == 0
    assert captured_models["groq"] == "llama-3.3-70b-versatile"
    assert captured_models["gemini"] == "gemini-2.5-flash"
    assert captured_models["ollama"] == "llama3.2"


def test_chat_command_works_with_ollama_provider(monkeypatch):
    monkeypatch.setitem(
        PROVIDERS,
        "ollama",
        lambda model: FakeProvider(
            "ollama",
            model,
            response=ChatResponse(
                text="hi from ollama",
                provider="ollama",
                model=model,
                input_tokens=3,
                output_tokens=3,
                latency_ms=5.0,
            ),
        ),
    )

    result = runner.invoke(app, ["chat", "hello", "--provider", "ollama"])

    assert result.exit_code == 0
    assert "hi from ollama" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — the new tests fail because `PROVIDERS` doesn't have `"groq"`/`"gemini"`/`"ollama"` keys yet, and `compare`/`chat` don't recognize the new flags/provider names yet.

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
    "groq": lambda model: LangChainProvider("groq", model),
    "gemini": lambda model: LangChainProvider("gemini", model),
    "ollama": lambda model: LangChainProvider("ollama", model),
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
        "anthropic",
        help="Provider to use: anthropic, openai, groq, gemini, or ollama",
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
    prompt: str = typer.Argument(..., help="Prompt to send to the selected providers"),
    system: str | None = typer.Option(None, help="Optional system prompt"),
    providers: str = typer.Option(
        "anthropic,openai",
        "--providers",
        help="Comma-separated list of providers to compare",
    ),
    anthropic_model: str | None = typer.Option(
        None, "--anthropic-model", help="Override Anthropic's model"
    ),
    openai_model: str | None = typer.Option(
        None, "--openai-model", help="Override OpenAI's model"
    ),
    groq_model: str | None = typer.Option(
        None, "--groq-model", help="Override Groq's model"
    ),
    gemini_model: str | None = typer.Option(
        None, "--gemini-model", help="Override Gemini's model"
    ),
    ollama_model: str | None = typer.Option(
        None, "--ollama-model", help="Override Ollama's model"
    ),
) -> None:
    names = [name.strip() for name in providers.split(",")]
    for name in names:
        if name not in PROVIDERS:
            console.print(f"[red]Error:[/red] unknown provider '{name}'")
            raise typer.Exit(code=1)

    overrides = {
        "anthropic": anthropic_model,
        "openai": openai_model,
        "groq": groq_model,
        "gemini": gemini_model,
        "ollama": ollama_model,
    }
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
        "anthropic",
        help="Provider to use: anthropic, openai, groq, gemini, or ollama",
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
Expected: 23 passed (was 19)

- [ ] **Step 5: Run the full suite to confirm nothing else broke**

Run: `uv run pytest -v`
Expected: 50 passed (was 46)

- [ ] **Step 6: Commit**

```bash
git add llm_lab/cli.py tests/test_cli.py
git commit -m "feat: add groq/gemini/ollama providers and compare --providers flag"
```
