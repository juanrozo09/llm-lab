# LangChain Provider Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the direct `anthropic`/`openai` SDK calls inside `AnthropicProvider`/`OpenAIProvider` with LangChain's `ChatAnthropic`/`ChatOpenAI` chat models, dropping tenacity in favor of LangChain's built-in retry.

**Architecture:** Both providers construct a LangChain chat model instead of a raw SDK async client, call `await self._client.ainvoke(messages)`, and read `.content`/`.usage_metadata` off the returned `AIMessage` — a shape now identical across both providers. `chat()` still catches each SDK's own `(RateLimitError, APIConnectionError)` and wraps into `ProviderUnavailableError`, unchanged from today.

**Tech Stack:** `langchain-anthropic` (`ChatAnthropic`), `langchain-openai` (`ChatOpenAI`), `langchain-core` (`SystemMessage`/`HumanMessage`), `anthropic`/`openai` (kept only for their exception classes).

## Global Constraints

- `AnthropicProvider`: model `claude-sonnet-4-6`, `max_tokens=1024`, `max_retries=3` on `ChatAnthropic`.
- `OpenAIProvider`: model `gpt-4o-mini`, `max_completion_tokens=1024`, `max_retries=3` on `ChatOpenAI`.
- API keys read only from `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` env vars — constructors raise plain `RuntimeError` if unset (unchanged from today).
- `chat()` catches `(RateLimitError, APIConnectionError)` imported from the top-level `anthropic`/`openai` packages (NOT re-exported by LangChain) and wraps into `ProviderUnavailableError(self.name, exc)` — unchanged contract with `llm_lab/cli.py`.
- Retry is now entirely internal to `ainvoke()` — no `retry_on_transient`/tenacity anywhere in the final state. No "retry-then-succeeds" test exists anymore; only success and exhaustion-wrapping tests per provider.
- `llm_lab/models.py`, `llm_lab/errors.py`, `llm_lab/cli.py`, and `tests/test_cli.py` are NOT modified by this plan.
- Every task in this plan must leave `uv run pytest -v` fully green — dependency and code changes are sequenced so no task leaves a broken intermediate state.

---

### Task 1: Add LangChain dependencies

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `langchain-anthropic` and `langchain-openai` installed and importable. No code changes yet — `tenacity` stays in place until Task 4, since `llm_lab/providers/openai.py` still depends on it until Task 3 is done.

- [ ] **Step 1: Add the two new dependencies to `pyproject.toml`**

Replace the `dependencies` list:

```toml
dependencies = [
    "typer>=0.15",
    "rich>=13.9",
    "pydantic>=2.9",
    "anthropic>=0.40",
    "openai>=1.60",
    "tenacity>=9.0",
    "langchain-anthropic>=0.3",
    "langchain-openai>=0.3",
]
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`
Expected: resolves `langchain-anthropic`/`langchain-openai` (and their transitive deps) with no errors, updates `uv.lock`.

- [ ] **Step 3: Verify the full suite is still green**

Run: `uv run pytest -v`
Expected: 24 passed (unchanged — no source code has changed yet, this just confirms the dependency bump didn't break anything).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add langchain-anthropic and langchain-openai dependencies"
```

---

### Task 2: Migrate `AnthropicProvider` to LangChain

**Files:**
- Modify: `llm_lab/providers/anthropic.py` (full rewrite)
- Modify: `tests/test_providers.py` (full rewrite — new shared helper + Anthropic tests; OpenAI section stays byte-for-byte identical to today since `llm_lab/providers/openai.py` isn't touched in this task)

**Interfaces:**
- Consumes: `Provider` from `llm_lab.providers.base` (still exports `retry_on_transient` at this point — `openai.py` still uses it, untouched by this task). `ChatResponse` from `llm_lab.models`. `ProviderUnavailableError` from `llm_lab.errors`.
- Produces: `AnthropicProvider` with the same public shape as before (`.name`, `.model`, `async def chat(...)`, constructor raising `RuntimeError` if `ANTHROPIC_API_KEY` unset) but backed by `self._client` = a `ChatAnthropic` instance instead of `AsyncAnthropic`. Introduces `make_ai_message(text, input_tokens, output_tokens)` test helper in `tests/test_providers.py`, which Task 3 will reuse for the OpenAI tests.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `tests/test_providers.py` with:

```python
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
    provider._client.ainvoke = AsyncMock(return_value=make_ai_message())

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
    provider._client.ainvoke = AsyncMock(side_effect=make_anthropic_connection_error())

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
```

Note: only the three `test_anthropic_*` tests and the new `make_ai_message`/`make_anthropic_connection_error` helpers are new/changed. Everything from `make_openai_completion` onward is byte-for-byte identical to the current file — copied verbatim because `llm_lab/providers/openai.py` isn't touched until Task 3.

- [ ] **Step 2: Run tests to verify the new Anthropic tests fail**

Run: `uv run pytest tests/test_providers.py -v -k anthropic`
Expected: FAIL — `provider._client.ainvoke` doesn't exist yet on an `AsyncAnthropic`-backed client (current `AnthropicProvider` still uses `self._client.messages.create`).

- [ ] **Step 3: Rewrite `llm_lab/providers/anthropic.py`**

```python
# llm_lab/providers/anthropic.py
import os
import time

from anthropic import APIConnectionError, RateLimitError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from llm_lab.errors import ProviderUnavailableError
from llm_lab.models import ChatResponse
from llm_lab.providers.base import Provider

MODEL = "claude-sonnet-4-6"


class AnthropicProvider(Provider):
    name = "anthropic"
    model = MODEL

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = ChatAnthropic(
            model=MODEL, max_tokens=1024, max_retries=3, api_key=api_key
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
        except (RateLimitError, APIConnectionError) as exc:
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

- [ ] **Step 4: Run the full suite to verify it passes**

Run: `uv run pytest -v`
Expected: 23 passed (was 24 — `test_providers.py` now has 7 tests instead of 8, since the Anthropic retry-then-succeed test was dropped; the 4 OpenAI tests are untouched and still pass against the unmodified `llm_lab/providers/openai.py`).

- [ ] **Step 5: Commit**

```bash
git add llm_lab/providers/anthropic.py tests/test_providers.py
git commit -m "feat: migrate AnthropicProvider to langchain-anthropic"
```

---

### Task 3: Migrate `OpenAIProvider` to LangChain

**Files:**
- Modify: `llm_lab/providers/openai.py` (full rewrite)
- Modify: `tests/test_providers.py` (full rewrite — reuses `make_ai_message` from Task 2, replaces the OpenAI section, drops the retry-then-succeed test)

**Interfaces:**
- Consumes: `Provider` from `llm_lab.providers.base` (still exports `retry_on_transient` at this point — nothing imports it anymore after this task, but it isn't removed until Task 4). `make_ai_message` helper introduced in Task 2, now reused here for OpenAI's success/exhaustion tests.
- Produces: `OpenAIProvider` with the same public shape as before, backed by `self._client` = a `ChatOpenAI` instance instead of `AsyncOpenAI`.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `tests/test_providers.py` with:

```python
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


def make_openai_connection_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return OpenAIAPIConnectionError(request=request)


async def test_anthropic_chat_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider()
    provider._client.ainvoke = AsyncMock(return_value=make_ai_message())

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
    provider._client.ainvoke = AsyncMock(side_effect=make_anthropic_connection_error())

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.chat("hi")

    assert exc_info.value.provider == "anthropic"


def test_anthropic_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        AnthropicProvider()


async def test_openai_chat_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIProvider()
    provider._client.ainvoke = AsyncMock(return_value=make_ai_message())

    response = await provider.chat("hi")

    assert response.text == "hello"
    assert response.provider == "openai"
    assert response.model == "gpt-4o-mini"
    assert response.input_tokens == 5
    assert response.output_tokens == 7


async def test_openai_chat_raises_provider_unavailable_after_retries(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIProvider()
    provider._client.ainvoke = AsyncMock(side_effect=make_openai_connection_error())

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.chat("hi")

    assert exc_info.value.provider == "openai"


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        OpenAIProvider()
```

Note: the three `test_anthropic_*` tests and their helpers (`make_ai_message`, `make_anthropic_connection_error`) are unchanged from Task 2 — copied verbatim. The OpenAI section is fully replaced: `make_openai_completion` is deleted (no longer needed — `make_ai_message` now covers both providers), `make_openai_connection_error` is kept, and the retry-then-succeed test is dropped.

- [ ] **Step 2: Run tests to verify the new OpenAI tests fail**

Run: `uv run pytest tests/test_providers.py -v -k openai`
Expected: FAIL — `provider._client.ainvoke` doesn't exist on an `AsyncOpenAI`-backed client (current `OpenAIProvider` still uses `self._client.chat.completions.create`).

- [ ] **Step 3: Rewrite `llm_lab/providers/openai.py`**

```python
# llm_lab/providers/openai.py
import os
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, RateLimitError

from llm_lab.errors import ProviderUnavailableError
from llm_lab.models import ChatResponse
from llm_lab.providers.base import Provider

MODEL = "gpt-4o-mini"


class OpenAIProvider(Provider):
    name = "openai"
    model = MODEL

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = ChatOpenAI(
            model=MODEL, max_completion_tokens=1024, max_retries=3, api_key=api_key
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
        except (RateLimitError, APIConnectionError) as exc:
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

- [ ] **Step 4: Run the full suite to verify it passes**

Run: `uv run pytest -v`
Expected: 22 passed (was 23 — `test_providers.py` now has 6 tests instead of 7, since the OpenAI retry-then-succeed test was dropped).

- [ ] **Step 5: Commit**

```bash
git add llm_lab/providers/openai.py tests/test_providers.py
git commit -m "feat: migrate OpenAIProvider to langchain-openai"
```

---

### Task 4: Remove tenacity

**Files:**
- Modify: `llm_lab/providers/base.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing new. By this point neither `AnthropicProvider` nor `OpenAIProvider` imports `retry_on_transient` (both were migrated off it in Tasks 2 and 3).
- Produces: `llm_lab/providers/base.py` exporting only `Provider` (the `retry_on_transient` factory is deleted). No later task depends on this — it's the final cleanup.

- [ ] **Step 1: Remove `retry_on_transient` from `llm_lab/providers/base.py`**

Replace the full contents of `llm_lab/providers/base.py` with:

```python
from abc import ABC, abstractmethod

from llm_lab.models import ChatResponse


class Provider(ABC):
    name: str
    model: str

    @abstractmethod
    async def chat(self, prompt: str, system: str | None = None) -> ChatResponse:
        raise NotImplementedError
```

- [ ] **Step 2: Remove `tenacity` from `pyproject.toml`**

Replace the `dependencies` list:

```toml
dependencies = [
    "typer>=0.15",
    "rich>=13.9",
    "pydantic>=2.9",
    "anthropic>=0.40",
    "openai>=1.60",
    "langchain-anthropic>=0.3",
    "langchain-openai>=0.3",
]
```

- [ ] **Step 3: Sync dependencies**

Run: `uv sync`
Expected: `tenacity` is removed from the environment; no errors (nothing imports it anymore).

- [ ] **Step 4: Run the full suite to verify it still passes**

Run: `uv run pytest -v`
Expected: 22 passed, 0 warnings (unchanged count from Task 3 — this task only removes now-unused code and a now-unused dependency).

- [ ] **Step 5: Commit**

```bash
git add llm_lab/providers/base.py pyproject.toml uv.lock
git commit -m "chore: remove tenacity now that retries are handled by LangChain"
```
