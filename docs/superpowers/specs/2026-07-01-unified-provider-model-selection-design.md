# llm-lab: Unified Provider + Model Selection — Design

Date: 2026-07-01

## Purpose

1. Replace `AnthropicProvider`/`OpenAIProvider` with a single generic
   provider class built on LangChain's `init_chat_model`, since its
   standard params (`api_key`, `max_tokens`, `max_retries`, etc.) are
   documented as unified across providers — no need for two near-identical
   classes anymore.
2. Let the user pick which model to use per provider on the command line,
   validated against the expanded `PRICING` table already in
   `llm_lab/models.py`, with today's hardcoded models as defaults.

## Scope

- `llm_lab/providers/anthropic.py`, `llm_lab/providers/openai.py` — deleted,
  replaced by `llm_lab/providers/langchain_provider.py`.
- `llm_lab/models.py` — `PRICING` gains a `provider` field per entry; new
  `DEFAULT_MODELS` dict and `resolve_model()` function.
- `llm_lab/cli.py` — `--model` on `chat`/`benchmark`, `--anthropic-model`/
  `--openai-model` on `compare`; `PROVIDERS` becomes model-parameterized.
- `tests/test_providers.py`, `tests/test_cli.py`, `tests/test_models.py` —
  updated for the new shapes.
- `llm_lab/providers/base.py`, `llm_lab/errors.py` — unchanged.

## Provider layer

`llm_lab/providers/langchain_provider.py`:

```python
class LangChainProvider(Provider):
    def __init__(self, name: str, model: str) -> None: ...
    async def _create(self, prompt: str, system: str | None): ...
    async def chat(self, prompt: str, system: str | None = None) -> ChatResponse: ...
```

- `name` and `model` become instance attributes (set in `__init__`), not
  class attributes — the same class now serves both providers.
- `API_KEY_ENV_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}`
  drives which env var `__init__` reads; raises `RuntimeError` if unset,
  same as today.
- The client is built via
  `init_chat_model(model, model_provider=name, max_tokens=1024, max_retries=3, api_key=api_key)`
  — `model_provider` is passed explicitly (not relying on LangChain's
  name-prefix inference), since our custom model names aren't guaranteed
  to match its built-in inference table.
- `chat()` catches a 4-tuple of exception types — both SDKs'
  `RateLimitError`/`APIConnectionError` (`anthropic.RateLimitError`,
  `anthropic.APIConnectionError`, `openai.RateLimitError`,
  `openai.APIConnectionError`) — and wraps into
  `ProviderUnavailableError(self.name, exc)`, same contract as today.
  Catching all four regardless of which provider is active is harmless:
  only the two relevant to `self.name`'s actual client can ever be raised.
- `message.content` / `message.usage_metadata["input_tokens"/"output_tokens"]`
  reading is unchanged from the current LangChain-based providers.

## Model selection & validation

`llm_lab/models.py` additions:

```python
PRICING: dict[str, dict[str, str | float]] = {
    "claude-sonnet-4-6": {"provider": "anthropic", "input": 3.00, "output": 15.00},
    # ... (existing entries, each gains a "provider" key)
}

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
}

def resolve_model(provider: str, model: str | None) -> str:
    """Return the model to use, defaulting and validating against PRICING."""
```

`resolve_model`:
- `model is None` → returns `DEFAULT_MODELS[provider]`.
- `model` given but not a key in `PRICING` → `ValueError(f"Unknown model '{model}'")`.
- `model` given and in `PRICING`, but `PRICING[model]["provider"] != provider`
  → `ValueError` naming the mismatch (e.g. picking a `gpt-*` model under
  `--provider anthropic`).
- Otherwise returns `model` unchanged.

`cost()` is unchanged — `rates["input"]`/`rates["output"]` lookups still
work with the extra `"provider"` key present.

## CLI changes

- `chat` and `benchmark` each add `--model` (`str | None`, default `None`).
  Resolved via `resolve_model(provider, model)` synchronously, before any
  async call — same up-front-validation pattern as the existing "unknown
  provider" check. A `ValueError` prints `[red]Error:[/red] {msg}` and exits
  1, exactly like today's provider/`--n` validation.
- `compare` adds `--anthropic-model` and `--openai-model` (each
  `str | None`), resolved the same way per provider before the concurrent
  `asyncio.gather`.
- `chat --fallback`: the fallback provider's model is always
  `resolve_model(other_name, None)` — its own default — never the primary
  provider's `--model` value, since that string belongs to the primary
  provider's model namespace.
- `PROVIDERS` changes from `dict[str, type[Provider]]` (zero-arg
  constructors) to `dict[str, Callable[[str], Provider]]`:
  ```python
  PROVIDERS: dict[str, Callable[[str], Provider]] = {
      "anthropic": lambda model: LangChainProvider("anthropic", model),
      "openai": lambda model: LangChainProvider("openai", model),
  }
  ```
  Every call site changes from `PROVIDERS[provider]()` to
  `PROVIDERS[provider](model_name)`.

## Testing changes

- `tests/test_models.py` gains tests for `resolve_model`: default when
  `model=None`, valid override, unknown model, provider/model mismatch.
- `tests/test_providers.py` is rewritten around `LangChainProvider`,
  parameterized by `(name, model, env_var)` for both providers — replacing
  the two near-duplicate provider test suites with one parameterized suite
  (or two thin instantiations of shared test functions, implementer's
  choice) covering: success, exhaustion-wrapping, missing-API-key, for both
  "anthropic" and "openai".
- `tests/test_cli.py`'s `FakeProvider`-based `PROVIDERS` monkeypatches
  change from `lambda: FakeProvider(...)` to `lambda model: FakeProvider(...)`.
  New tests: `--model` override reaches the provider correctly, unknown/
  mismatched `--model` prints a clean error and exits 1 (for `chat`,
  `compare`, and `benchmark`), and `chat --fallback --model <primary-model>`
  falls back using the secondary provider's default (not the override).

## Out of scope

- Changing `ChatResponse`, `ProviderUnavailableError`, or the cost-table
  values themselves (already populated by the user).
- Any change to `--system` handling, retry counts, or token limits.
