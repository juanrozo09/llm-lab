# llm-lab: Add Groq, Gemini, and Ollama Providers — Design

Date: 2026-07-01

## Purpose

Add three new providers — Groq, Gemini, and Ollama — so the CLI can be
tested against free/local models instead of requiring paid Anthropic/OpenAI
credits. Ollama in particular needs no API key at all (it's a local server),
which the current provider architecture doesn't yet accommodate.

## Scope

- `llm_lab/providers/langchain_provider.py` — key-optional construction,
  provider-ID translation, expanded retryable-error set.
- `llm_lab/models.py` — `PRICING`/`DEFAULT_MODELS` entries for the three new
  providers.
- `llm_lab/cli.py` — `PROVIDERS` registry grows to 5 entries; `compare` gains
  `--providers` (subset selection) and three new per-provider model-override
  flags.
- `pyproject.toml` — add `langchain-groq`, `langchain-google-genai`,
  `langchain-ollama`.
- `tests/test_providers.py`, `tests/test_cli.py`, `tests/test_models.py` —
  extended coverage for the new providers.

`chat`/`benchmark`'s `--provider` option is unchanged in shape (already a
generic string checked against `PROVIDERS.keys()`) — it just now accepts
`"groq"`/`"gemini"`/`"ollama"` as valid values once those keys exist.

## Provider layer

`API_KEY_ENV_VARS` becomes `dict[str, str | None]`:

```python
API_KEY_ENV_VARS: dict[str, str | None] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "ollama": None,
}
```

`LangChainProvider.__init__` only performs the "is the key set?" `RuntimeError`
check when `API_KEY_ENV_VARS[name]` is not `None`. When it's `None` (Ollama),
no env var is read and `api_key` is omitted entirely from the
`init_chat_model(...)` call rather than passed as `None` — safer than
assuming every LangChain integration gracefully accepts an explicit `None`
for a kwarg it may not even declare.

A new `MODEL_PROVIDER_IDS` dict translates our internal short provider name
to whatever identifier LangChain's `init_chat_model(..., model_provider=...)`
actually expects, since these don't always match 1:1:

```python
MODEL_PROVIDER_IDS: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "groq": "groq",
    "gemini": "google_genai",
    "ollama": "ollama",
}
```

`RETRYABLE_ERRORS` grows to include each new SDK's transient-error classes.
Groq's Python SDK mirrors OpenAI's naming (`groq.RateLimitError`,
`groq.APIConnectionError`) since its API is OpenAI-compatible. Google's
`google-genai` SDK uses a different hierarchy (`google.genai.errors.ClientError`
for 4xx including rate limits, `ServerError` for 5xx) rather than named
rate-limit classes. Ollama has no auth/rate-limit concept — failure means the
local server isn't reachable, surfacing as a connection error from the
underlying HTTP client. **The implementer must verify the exact class names
against the installed `langchain-groq`/`langchain-google-genai`/
`langchain-ollama` versions** rather than trust this description blindly —
same practice already used successfully for every other provider integration
in this codebase.

## Model selection & pricing

`DEFAULT_MODELS` gains:

```python
"groq": "llama-3.3-70b-versatile",
"gemini": "gemini-2.5-flash",
"ollama": "llama3.2",
```

`PRICING` gains matching entries (USD per 1M tokens, as of 2026-07-01):

```python
"llama-3.3-70b-versatile": {"provider": "groq", "input": 0.59, "output": 0.79},
"gemini-2.5-flash": {"provider": "gemini", "input": 0.30, "output": 2.50},
"llama3.2": {"provider": "ollama", "input": 0.0, "output": 0.0},
```

Ollama's `$0.0`/`$0.0` is factually correct — there's no per-token billing
for local compute. `resolve_model()` and `cost()` are otherwise unchanged;
both already operate generically over whatever's in `PRICING`.

## CLI changes

`PROVIDERS` gains three entries following the existing pattern:

```python
"groq": lambda model: LangChainProvider("groq", model),
"gemini": lambda model: LangChainProvider("gemini", model),
"ollama": lambda model: LangChainProvider("ollama", model),
```

`chat`/`benchmark` need no structural change — `--provider groq`,
`--provider gemini`, `--provider ollama` work immediately once the dict
entries exist.

`compare` changes:

- New `--providers` option: comma-separated provider names, default
  `"anthropic,openai"` (identical to today's behavior when the flag is
  omitted — no existing workflow breaks). An unknown name in the list
  produces a clean `[red]Error:[/red]` message and exit code 1, same
  pattern as the existing unknown-provider checks in `chat`/`benchmark`.
- Three new options: `--groq-model`, `--gemini-model`, `--ollama-model`,
  alongside the existing `--anthropic-model`/`--openai-model`. Only
  providers actually named in `--providers` get their override resolved and
  validated; an override flag for a provider not in the selected set is
  simply unused (not an error).

## Dependencies

Add to `pyproject.toml`: `langchain-groq`, `langchain-google-genai`,
`langchain-ollama`. `anthropic`/`openai`/`langchain`/`langchain-anthropic`/
`langchain-openai` are unchanged.

## Testing changes

- `tests/test_providers.py`: `PROVIDER_CASES` (used for success/exhaustion
  tests, which apply to all 5 providers) grows to 5 entries. A separate,
  smaller case list excludes Ollama for the "requires API key" test, since
  Ollama has no key to require.
- `tests/test_cli.py`: new tests for `compare --providers <subset>` (custom
  subset runs only those providers), the default-subset-when-omitted case,
  an unknown name in `--providers` producing a clean error, and the three
  new per-provider model-override flags reaching the right provider (same
  pattern as the existing `--anthropic-model`/`--openai-model` tests).
- `tests/test_models.py`: assertions that `resolve_model`/`DEFAULT_MODELS`
  work correctly for the three new provider names.

## Out of scope

- Any change to `chat --fallback` (still only ever falls back between
  whichever provider was primary and the other named one — extending
  fallback to a chain of more than two providers is a separate, unrequested
  feature).
- Streaming, tool calling, or any capability beyond plain single-turn chat
  for the new providers.
- Auto-detecting whether Ollama's local server is actually running/has the
  model pulled before attempting a call — errors surface the same way any
  other `ProviderUnavailableError`-wrapped failure does.
