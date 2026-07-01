# llm-lab CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `llm-lab` CLI — a typer/rich tool that wraps Anthropic and
OpenAI chat APIs behind a single `Provider` abstraction, with `chat`,
`compare`, and `benchmark` commands.

**Architecture:** `Provider` is an ABC with an async `chat()` method. Two
concrete subclasses (`AnthropicProvider`, `OpenAIProvider`) each wrap their
SDK's async client, decorate the raw API call with a tenacity retry, and
convert exhausted retries into a shared `ProviderUnavailableError`. The CLI
layer (typer) drives these providers via `asyncio.run`/`asyncio.gather` and
renders results with rich.

**Tech Stack:** Python 3.11+, uv, typer, rich, pydantic v2, `anthropic` SDK
(`AsyncAnthropic`), `openai` SDK (`AsyncOpenAI`), tenacity, pytest,
pytest-asyncio.

## Global Constraints

- Python >= 3.11 (spec: use `str | None` union syntax, no `typing.Optional`).
- Packaging via `uv` — `pyproject.toml` + `uv.lock`, build backend `hatchling`.
- `AnthropicProvider` uses model `claude-sonnet-4-6`, `max_tokens=1024`.
- `OpenAIProvider` uses model `gpt-4o-mini`, `max_completion_tokens=1024`.
- API keys read only from env vars `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` —
  no CLI flag for keys.
- Retry policy: tenacity, `stop_after_attempt(3)`, exponential backoff,
  retrying on that SDK's `RateLimitError` / `APIConnectionError`. Exhausted
  retries raise `ProviderUnavailableError(provider_name, original_exception)`.
- `chat --fallback`: on `ProviderUnavailableError` from the primary provider,
  retry once against the other provider; output must indicate when fallback
  was used.
- `benchmark` runs N calls **sequentially**, never concurrently.
- Pricing (USD per 1M tokens, as of 2026-07-01): `claude-sonnet-4-6` = $3.00
  input / $15.00 output; `gpt-4o-mini` = $0.15 input / $0.60 output.
- No streaming, no CLI-configurable models/token limits, no history
  persistence — out of scope per spec.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `llm_lab/__init__.py`
- Create: `llm_lab/providers/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: an installable `llm_lab` package with `llm-lab` console script
  entry point (wired to `llm_lab.cli:main`, created in Task 7), and a working
  `uv sync` / `uv run pytest` toolchain for every later task.

- [ ] **Step 1: Create the directory structure and empty package files**

```bash
mkdir -p llm_lab/providers tests
touch llm_lab/__init__.py llm_lab/providers/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "llm-lab"
version = "0.1.0"
description = "CLI to chat, compare, and benchmark Anthropic and OpenAI models"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.15",
    "rich>=13.9",
    "pydantic>=2.9",
    "anthropic>=0.40",
    "openai>=1.60",
    "tenacity>=9.0",
]

[project.scripts]
llm-lab = "llm_lab.cli:main"

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["llm_lab"]
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Sync dependencies**

Run: `uv sync`
Expected: creates `.venv/` and `uv.lock`, resolves all dependencies with no
errors. (This will fail with "No module named llm_lab.cli" if you try
`uv run llm-lab` right now — that's expected until Task 7.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore uv.lock llm_lab tests
git commit -m "chore: scaffold llm-lab project with uv"
```

---

### Task 2: `ChatResponse` model and cost calculation

**Files:**
- Create: `llm_lab/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `ChatResponse` (pydantic `BaseModel` with fields `text: str`,
  `provider: str`, `model: str`, `input_tokens: int`, `output_tokens: int`,
  `latency_ms: float`), `PRICING: dict[str, dict[str, float]]`, and
  `cost(response: ChatResponse) -> float`. Every later task that builds a
  `ChatResponse` or computes cost imports from here.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_lab.models'`

- [ ] **Step 3: Write the implementation**

```python
# llm_lab/models.py
from pydantic import BaseModel


class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def cost(response: ChatResponse) -> float:
    rates = PRICING[response.model]
    return (
        response.input_tokens / 1_000_000 * rates["input"]
        + response.output_tokens / 1_000_000 * rates["output"]
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add llm_lab/models.py tests/test_models.py
git commit -m "feat: add ChatResponse model and cost calculation"
```

---

### Task 3: `ProviderUnavailableError`

**Files:**
- Create: `llm_lab/errors.py`
- Test: `tests/test_models.py` (append)

**Interfaces:**
- Produces: `ProviderUnavailableError(provider: str, original_error: Exception)`,
  an `Exception` subclass with `.provider` and `.original_error` attributes.
  Task 5 and 6 raise this from provider `chat()` methods; Task 7 catches it
  in the CLI's fallback logic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py (append to the end of the file)
from llm_lab.errors import ProviderUnavailableError


def test_provider_unavailable_error_holds_provider_and_cause():
    original = RuntimeError("boom")
    error = ProviderUnavailableError("anthropic", original)

    assert error.provider == "anthropic"
    assert error.original_error is original
    assert "anthropic" in str(error)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py::test_provider_unavailable_error_holds_provider_and_cause -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_lab.errors'`

- [ ] **Step 3: Write the implementation**

```python
# llm_lab/errors.py
class ProviderUnavailableError(Exception):
    def __init__(self, provider: str, original_error: Exception) -> None:
        self.provider = provider
        self.original_error = original_error
        super().__init__(
            f"{provider} provider unavailable after retries: {original_error}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add llm_lab/errors.py tests/test_models.py
git commit -m "feat: add ProviderUnavailableError"
```

---

### Task 4: `Provider` ABC and retry decorator

**Files:**
- Create: `llm_lab/providers/base.py`

**Interfaces:**
- Consumes: `ChatResponse` from `llm_lab.models`.
- Produces: `Provider` (ABC with class attrs `name: str`, `model: str`, and
  abstract `async def chat(self, prompt: str, system: str | None = None) ->
  ChatResponse`), and `retry_on_transient(*exception_types: type[Exception])`
  — a factory returning a tenacity `retry` decorator
  (`stop_after_attempt(3)`, `wait_exponential`, `retry_if_exception_type`,
  `reraise=True`). Task 5 and 6 subclass `Provider` and decorate their raw
  API call with `retry_on_transient(<SDK RateLimitError>, <SDK
  APIConnectionError>)`.

No standalone test file for this task — its behavior (retry-then-succeed,
retry-exhausted) is exercised through the concrete providers in Tasks 5 and 6,
which is a more realistic integration point than testing the decorator with a
synthetic exception.

- [ ] **Step 1: Write the implementation**

```python
# llm_lab/providers/base.py
from abc import ABC, abstractmethod

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_lab.models import ChatResponse


def retry_on_transient(*exception_types: type[Exception]):
    return retry(
        retry=retry_if_exception_type(exception_types),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0, max=2),
        reraise=True,
    )


class Provider(ABC):
    name: str
    model: str

    @abstractmethod
    async def chat(self, prompt: str, system: str | None = None) -> ChatResponse:
        raise NotImplementedError
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `uv run python -c "from llm_lab.providers.base import Provider, retry_on_transient; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add llm_lab/providers/base.py
git commit -m "feat: add Provider ABC and retry_on_transient decorator"
```

---

### Task 5: `AnthropicProvider`

**Files:**
- Create: `llm_lab/providers/anthropic.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `Provider`, `retry_on_transient` from `llm_lab.providers.base`;
  `ChatResponse` from `llm_lab.models`; `ProviderUnavailableError` from
  `llm_lab.errors`.
- Produces: `AnthropicProvider()` — no-arg constructor, reads
  `ANTHROPIC_API_KEY` from env (raises `RuntimeError` if unset), exposes
  `.name == "anthropic"`, `.model == "claude-sonnet-4-6"`, and `._client`
  (an `anthropic.AsyncAnthropic` instance — tests patch
  `provider._client.messages.create` directly). Task 7's `PROVIDERS` registry
  maps `"anthropic"` to this class.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_providers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_lab.providers.anthropic'`

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_providers.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add llm_lab/providers/anthropic.py tests/test_providers.py
git commit -m "feat: add AnthropicProvider with retry and fallback support"
```

---

### Task 6: `OpenAIProvider`

**Files:**
- Create: `llm_lab/providers/openai.py`
- Test: `tests/test_providers.py` (append)

**Interfaces:**
- Consumes: same as Task 5, from `llm_lab.providers.base`, `llm_lab.models`,
  `llm_lab.errors`.
- Produces: `OpenAIProvider()` — no-arg constructor, reads `OPENAI_API_KEY`
  from env (raises `RuntimeError` if unset), exposes `.name == "openai"`,
  `.model == "gpt-4o-mini"`, `._client` (an `openai.AsyncOpenAI` instance).
  Task 7's `PROVIDERS` registry maps `"openai"` to this class.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_providers.py (append to the end of the file)
from openai import APIConnectionError as OpenAIAPIConnectionError

from llm_lab.providers.openai import OpenAIProvider


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_providers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_lab.providers.openai'`

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_providers.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add llm_lab/providers/openai.py tests/test_providers.py
git commit -m "feat: add OpenAIProvider with retry and fallback support"
```

---

### Task 7: `chat` CLI command

**Files:**
- Create: `llm_lab/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `AnthropicProvider` from `llm_lab.providers.anthropic`,
  `OpenAIProvider` from `llm_lab.providers.openai`, `ProviderUnavailableError`
  from `llm_lab.errors`, `ChatResponse`/`cost` from `llm_lab.models`.
- Produces: `app` (the `typer.Typer` instance), `PROVIDERS: dict[str,
  Callable[[], Provider]]` mapping `"anthropic"`/`"openai"` to their provider
  classes — Tasks 8 and 9 add commands to this same `app` and reuse
  `PROVIDERS`. Also produces `main() -> None` (the console-script entry
  point). Tests monkeypatch entries in `PROVIDERS` to avoid real API calls.

- [ ] **Step 1: Write the failing tests**

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
        lambda: FakeProvider("anthropic", "claude-sonnet-4-6", response=response),
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
        lambda: FakeProvider(
            "anthropic",
            "claude-sonnet-4-6",
            error=ProviderUnavailableError("anthropic", RuntimeError("down")),
        ),
    )
    monkeypatch.setitem(
        PROVIDERS,
        "openai",
        lambda: FakeProvider("openai", "gpt-4o-mini", response=fallback_response),
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
        lambda: FakeProvider(
            "anthropic",
            "claude-sonnet-4-6",
            error=ProviderUnavailableError("anthropic", RuntimeError("down")),
        ),
    )

    result = runner.invoke(app, ["chat", "hello", "--provider", "anthropic"])

    assert result.exit_code == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_lab.cli'`

- [ ] **Step 3: Write the implementation**

```python
# llm_lab/cli.py
import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_lab.errors import ProviderUnavailableError
from llm_lab.models import ChatResponse, cost
from llm_lab.providers.anthropic import AnthropicProvider
from llm_lab.providers.openai import OpenAIProvider

app = typer.Typer(help="Chat with, compare, and benchmark LLM providers.")
console = Console()

PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


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
    system: str | None = typer.Option(None, help="Optional system prompt"),
    fallback: bool = typer.Option(
        False, "--fallback", help="Retry with the other provider on failure"
    ),
) -> None:
    if provider not in PROVIDERS:
        console.print(f"[red]Error:[/red] unknown provider '{provider}'")
        raise typer.Exit(code=1)

    async def run() -> tuple[ChatResponse, bool]:
        instance = PROVIDERS[provider]()
        try:
            return await instance.chat(prompt, system), False
        except ProviderUnavailableError:
            if not fallback:
                raise
            other_name = "openai" if provider == "anthropic" else "anthropic"
            other = PROVIDERS[other_name]()
            return await other.chat(prompt, system), True

    try:
        response, used_fallback = asyncio.run(run())
    except ProviderUnavailableError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    title = f"{response.provider} ({response.model})"
    if used_fallback:
        title += " — fallback used"
    console.print(Panel(response.text, title=title))
    console.print(_render_cost_table(response))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add llm_lab/cli.py tests/test_cli.py
git commit -m "feat: add chat CLI command with fallback support"
```

---

### Task 8: `compare` CLI command

**Files:**
- Modify: `llm_lab/cli.py` (append command, no changes to existing code)
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `PROVIDERS`, `_render_cost_table` helper pattern (not reused
  directly — `compare` builds its own table), `console`, `app` from Task 7.
- Produces: `compare` command registered on `app`. No new names exported for
  later tasks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py (append to the end of the file)
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
        lambda: FakeProvider(
            "anthropic", "claude-sonnet-4-6", response=anthropic_response
        ),
    )
    monkeypatch.setitem(
        PROVIDERS,
        "openai",
        lambda: FakeProvider("openai", "gpt-4o-mini", response=openai_response),
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
        lambda: FakeProvider(
            "anthropic",
            "claude-sonnet-4-6",
            error=ProviderUnavailableError("anthropic", RuntimeError("down")),
        ),
    )
    monkeypatch.setitem(
        PROVIDERS,
        "openai",
        lambda: FakeProvider("openai", "gpt-4o-mini", response=openai_response),
    )

    result = runner.invoke(app, ["compare", "hello"])

    assert result.exit_code == 0
    assert "openai says hi" in result.output
    assert "Error" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k compare -v`
Expected: FAIL with `AssertionError` (no "compare" subcommand exists yet —
typer exits non-zero with a usage error).

- [ ] **Step 3: Append the implementation to `llm_lab/cli.py`**

```python
# llm_lab/cli.py (add after the chat command, before `def main()`)
@app.command()
def compare(
    prompt: str = typer.Argument(..., help="Prompt to send to both providers"),
    system: str | None = typer.Option(None, help="Optional system prompt"),
) -> None:
    names = list(PROVIDERS.keys())

    async def run() -> list[ChatResponse | BaseException]:
        instances = [PROVIDERS[name]() for name in names]
        return await asyncio.gather(
            *(instance.chat(prompt, system) for instance in instances),
            return_exceptions=True,
        )

    results = asyncio.run(run())

    table = Table(title="Provider Comparison")
    table.add_column("Provider")
    table.add_column("Text")
    table.add_column("Latency (ms)")
    table.add_column("Input tokens")
    table.add_column("Output tokens")
    table.add_column("Cost ($)")
    for name, result in zip(names, results):
        if isinstance(result, BaseException):
            table.add_row(name, f"[red]Error: {result}[/red]", "-", "-", "-", "-")
        else:
            table.add_row(
                name,
                result.text,
                f"{result.latency_ms:.1f}",
                str(result.input_tokens),
                str(result.output_tokens),
                f"{cost(result):.6f}",
            )
    console.print(table)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add llm_lab/cli.py tests/test_cli.py
git commit -m "feat: add compare CLI command"
```

---

### Task 9: `benchmark` CLI command

**Files:**
- Modify: `llm_lab/cli.py` (append command, no changes to existing code)
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `PROVIDERS`, `console`, `app` from Task 7.
- Produces: `benchmark` command registered on `app`. No new names exported.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py (append to the end of the file)
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
        lambda: CountingFakeProvider("anthropic", "claude-sonnet-4-6"),
    )

    result = runner.invoke(
        app, ["benchmark", "hello", "--provider", "anthropic", "--n", "3"]
    )

    assert result.exit_code == 0
    assert call_count == 3
    assert "Mean" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -k benchmark -v`
Expected: FAIL — no "benchmark" subcommand exists yet.

- [ ] **Step 3: Append the implementation to `llm_lab/cli.py`**

```python
# llm_lab/cli.py (add after the compare command, before `def main()`)
@app.command()
def benchmark(
    prompt: str = typer.Argument(..., help="Prompt to run repeatedly"),
    provider: str = typer.Option(
        "anthropic", help="Provider to use: anthropic or openai"
    ),
    n: int = typer.Option(5, "--n", help="Number of sequential runs"),
) -> None:
    if provider not in PROVIDERS:
        console.print(f"[red]Error:[/red] unknown provider '{provider}'")
        raise typer.Exit(code=1)

    async def run() -> list[ChatResponse]:
        instance = PROVIDERS[provider]()
        responses = []
        for _ in range(n):
            responses.append(await instance.chat(prompt))
        return responses

    responses = asyncio.run(run())

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add llm_lab/cli.py tests/test_cli.py
git commit -m "feat: add benchmark CLI command"
```

---

### Task 10: Full test suite check, README, and manual smoke test

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: nothing new — documents the `chat`/`compare`/`benchmark`
  commands built in Tasks 7–9.
- Produces: nothing consumed by other tasks (this is the final task).

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests from Tasks 2, 3, 5, 6, 7, 8, 9 pass (23 passed).

- [ ] **Step 2: Write `README.md`**

```markdown
# llm-lab

A CLI to chat with, compare, and benchmark Anthropic and OpenAI chat models
through a single `Provider` abstraction.

## Install

    uv sync
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...

## Commands

### chat

Send a prompt to one provider and print the response plus a cost summary.

    uv run llm-lab chat "What is the capital of France?" --provider anthropic

With a system prompt and automatic fallback to the other provider if the
primary one fails after retries:

    uv run llm-lab chat "Summarize this repo" \
        --provider anthropic --system "You are a terse assistant" --fallback

### compare

Send the same prompt to both providers concurrently and print a
side-by-side comparison table (text, latency, tokens, cost).

    uv run llm-lab compare "Explain recursion in one sentence"

### benchmark

Run the same prompt N times sequentially against one provider and print
per-run stats plus mean latency and total cost.

    uv run llm-lab benchmark "Say hello" --provider openai --n 10

## Environment variables

- `ANTHROPIC_API_KEY` — required for `AnthropicProvider` (model:
  `claude-sonnet-4-6`).
- `OPENAI_API_KEY` — required for `OpenAIProvider` (model: `gpt-4o-mini`).

## Development

    uv sync
    uv run pytest -v
```

- [ ] **Step 3: Manual smoke test (requires real API keys)**

Run:
```bash
export ANTHROPIC_API_KEY=<your key>
export OPENAI_API_KEY=<your key>
uv run llm-lab chat "Say hello in five words" --provider anthropic
uv run llm-lab compare "Say hello in five words"
uv run llm-lab benchmark "Say hello in five words" --provider openai --n 2
```
Expected: `chat` prints a panel with response text and a cost table;
`compare` prints a two-row table with both providers' responses; `benchmark`
prints a 2-row + summary table. If you don't have both API keys handy, at
minimum run `uv run llm-lab chat "hi" --provider anthropic` (or `openai`)
with whichever key you have.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add README with usage examples"
```
