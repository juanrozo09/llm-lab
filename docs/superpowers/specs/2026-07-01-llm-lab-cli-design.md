# llm-lab CLI — Design

Date: 2026-07-01

## Purpose

A Python CLI (`llm-lab`) that wraps the Anthropic and OpenAI chat APIs behind a
single `Provider` abstraction, letting a user send a prompt to one or both
providers, compare responses side by side, or benchmark latency/cost across
repeated runs.

## Project structure

```
llm-lab/
├── pyproject.toml          # uv-managed, Python >=3.11
├── README.md
├── llm_lab/
│   ├── __init__.py
│   ├── cli.py              # typer app: chat, compare, benchmark
│   ├── models.py           # ChatResponse (pydantic), pricing table, cost calc
│   ├── errors.py           # ProviderUnavailableError
│   └── providers/
│       ├── __init__.py
│       ├── base.py         # Provider ABC + retry decorator
│       ├── anthropic.py    # AnthropicProvider
│       └── openai.py       # OpenAIProvider
└── tests/
    ├── test_models.py
    ├── test_providers.py
    └── test_cli.py
```

Packaging via `uv` (pyproject.toml + uv-managed venv/lock). Python >= 3.11.

## Core abstraction

`Provider` is an **ABC** (not a `Protocol`) — `AnthropicProvider` and
`OpenAIProvider` share initialization logic (reading the API key from an env
var, building an async SDK client), so a base class avoids duplicating that.

```python
class Provider(ABC):
    name: str  # "anthropic" | "openai"
    model: str

    @abstractmethod
    async def chat(self, prompt: str, system: str | None = None) -> ChatResponse: ...
```

`ChatResponse` (pydantic `BaseModel`):

```python
class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
```

Each concrete provider hardcodes its model and token-limit parameter name
internally (not CLI-configurable):

- `AnthropicProvider`: `claude-sonnet-4-6`, uses `anthropic.AsyncAnthropic`,
  `max_tokens=1024`.
- `OpenAIProvider`: `gpt-4o-mini`, uses `openai.AsyncOpenAI`,
  `max_completion_tokens=1024`.

`chat()` is `async def` on both, using each SDK's async client. This lets
`compare` run both providers concurrently via `asyncio.gather`, and lets
`chat`/`benchmark` just `asyncio.run()` a single call.

## Retry, errors, fallback

- `@retry_on_transient` decorator (tenacity), applied around each provider's
  underlying SDK call: `stop_after_attempt(3)`, `wait_exponential`, retrying on
  that SDK's `RateLimitError` / `APIConnectionError`.
- When retries are exhausted, wrap the final exception in
  `ProviderUnavailableError(provider_name, original_exception)` (custom
  exception in `errors.py`) and raise that instead of tenacity's `RetryError`.
- `chat --fallback`: try the primary provider; on `ProviderUnavailableError`,
  automatically retry the prompt against the other provider. The CLI output
  indicates which provider actually served the response (e.g. a note in the
  output panel if fallback was used).

## CLI commands (typer + rich)

- `llm-lab chat "<prompt>" --provider anthropic|openai [--system TEXT] [--fallback]`
  → rich `Panel` with the response text, then a small table with model,
  latency, input/output tokens, and cost.
- `llm-lab compare "<prompt>" [--system TEXT]`
  → runs both providers concurrently via `asyncio.gather`, renders a rich
  side-by-side `Table` (columns: Provider, Text, Latency ms, Input tok,
  Output tok, Cost).
- `llm-lab benchmark "<prompt>" --provider anthropic|openai --n 10`
  → runs N calls **sequentially** against the chosen provider, prints a rich
  table of per-run stats plus a summary row with mean latency and total cost.

API keys come from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` env vars only — no
CLI flag for keys.

## Pricing / cost calculation

Neither SDK's response includes a dollar cost, only token usage. `models.py`
holds a small hardcoded pricing table (USD per 1M tokens, sourced from each
provider's published pricing as of 2026-07-01):

```python
PRICING = {
    "claude-sonnet-4-6": {"input": ..., "output": ...},
    "gpt-4o-mini": {"input": ..., "output": ...},
}
```

`cost(response: ChatResponse) -> float` computes
`input_tokens/1e6 * input_rate + output_tokens/1e6 * output_rate`. Exact rates
to be filled in during implementation from current provider pricing pages.

## Testing (pytest)

- `test_models.py`: `ChatResponse` validation, cost calculation helper.
- `test_providers.py`: mock `AsyncAnthropic`/`AsyncOpenAI` clients to cover:
  success path, retry-then-succeed, retry-exhausted →
  `ProviderUnavailableError`, and the fallback path in the CLI layer.
- `test_cli.py`: typer `CliRunner` tests for `chat`, `compare`, `benchmark`
  with mocked providers — no real network/API calls in tests.

## README

Usage examples for all three commands, env var setup
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`), install instructions via `uv`.

## Out of scope

- Streaming responses.
- Configurable models/token limits via CLI flags.
- Persisting run history to disk.
- Concurrent benchmark runs.
