# llm-lab: Migrate Provider Calls to LangChain — Design

Date: 2026-07-01

## Purpose

Replace the direct `anthropic`/`openai` SDK calls inside `AnthropicProvider`
and `OpenAIProvider` with LangChain's chat model wrappers
(`langchain_anthropic.ChatAnthropic`, `langchain_openai.ChatOpenAI`), so
`llm-lab` calls both LLMs through LangChain instead of the raw provider
SDKs directly.

## Scope

Only the provider layer changes:

- `llm_lab/providers/base.py`
- `llm_lab/providers/anthropic.py`
- `llm_lab/providers/openai.py`
- `tests/test_providers.py`
- `pyproject.toml` (dependencies)

`llm_lab/models.py` (`ChatResponse`, pricing, `cost()`), `llm_lab/errors.py`
(`ProviderUnavailableError`), and `llm_lab/cli.py` (all three commands) are
**not modified** — the existing `Provider` ABC → `ChatResponse` boundary
already isolates this change to the provider implementations.

## Dependency changes

- Remove: `tenacity`
- Add: `langchain-anthropic`, `langchain-openai`
- Keep: `anthropic`, `openai` — still imported directly for the
  `RateLimitError`/`APIConnectionError` exception classes our code catches.
  LangChain does not redefine equivalent exception types; these SDKs remain
  transitive dependencies of the LangChain packages regardless, so keeping
  them explicit just pins them clearly.

## Provider changes

Both `AnthropicProvider` and `OpenAIProvider` construct a LangChain chat
model instead of a raw SDK async client:

- `AnthropicProvider`: `ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024, max_retries=3, api_key=api_key)`
- `OpenAIProvider`: `ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=1024, max_retries=3, api_key=api_key)`

Both constructors keep reading their API key explicitly from
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` and raising a plain `RuntimeError` if
unset — unchanged from today.

LangChain normalizes both providers' request/response shape, so the two
classes converge more than they do today:

- **Messages:** both build the same message list —
  `[SystemMessage(content=system)]` (only if `system is not None`) followed
  by `HumanMessage(content=prompt)` — imported from `langchain_core.messages`.
  (Previously Anthropic used a separate `system=` kwarg while OpenAI used a
  system message in its list; LangChain unifies this.)
- **Response parsing:** both call `await self._client.ainvoke(messages)`,
  then read `response.content` for text and
  `response.usage_metadata["input_tokens"]` /
  `response.usage_metadata["output_tokens"]` for token counts. (Previously
  OpenAI's raw SDK used differently-named `prompt_tokens`/`completion_tokens`
  — LangChain's `usage_metadata` uses the same key names for every provider.)

`chat()` keeps its existing structure: time the call, catch
`(RateLimitError, APIConnectionError)` (imported from the top-level
`anthropic`/`openai` packages, same as today) and wrap into
`ProviderUnavailableError(self.name, exc)`, otherwise build a `ChatResponse`.
The `_create()` / `chat()` split is kept for structural consistency with the
rest of the codebase, but `_create()` no longer carries a retry decorator —
it's now a thin wrapper around `self._client.ainvoke(messages)`.

## Retry behavior change

Retries move from our own tenacity decorator to LangChain's built-in retry
(`max_retries=3` on the client), per explicit choice. Two consequences:

1. **Attempt-count semantics may differ from today's tests.** Tenacity's
   `stop_after_attempt(3)` meant 3 total attempts. LangChain's `max_retries=3`
   most likely means 3 retries *in addition to* the first attempt (4 total).
   The implementer must verify the actual behavior against the installed
   `langchain-anthropic`/`langchain-openai` versions and write the exhaustion
   test to match whatever that turns out to be — not assume the old count.
2. **The "retry-then-succeeds" test is dropped.** Retry now happens inside
   `ainvoke()`, invisible to our code. Mocking `ainvoke()` directly can still
   verify "eventually raises → we wrap it in `ProviderUnavailableError`" and
   "succeeds → we return a `ChatResponse`", but not the retry mechanics
   themselves — those are now LangChain's tested responsibility, not ours.

## Testing changes

`tests/test_providers.py` is rewritten for the new call shape:

- A shared fake-response helper (`.content` as a string, `.usage_metadata`
  as a dict with `input_tokens`/`output_tokens`) replaces the two
  differently-shaped helpers used today, since both providers now share the
  same response shape.
- Per provider: one success test, one exhaustion-wrapping test (mock
  `ainvoke` raising the SDK's own exception type, assert it becomes
  `ProviderUnavailableError`), one missing-API-key test (unchanged from
  today). No retry-then-succeed test (see above).
- `tests/test_cli.py` is unaffected — CLI tests already mock at the
  `Provider`/`PROVIDERS` level via `FakeProvider`, never touching the real
  provider internals.

## Out of scope

- Unifying `AnthropicProvider`/`OpenAIProvider` into a single generic class
  via `init_chat_model` (considered, explicitly declined — keeps the
  existing, already-reviewed two-subclass structure).
- Any change to CLI behavior, cost calculation, or the `ProviderUnavailableError`
  contract exposed to `llm_lab/cli.py`.
