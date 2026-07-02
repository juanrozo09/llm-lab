# llm-lab

A Python CLI to **chat with**, **compare**, and **benchmark** LLMs across five
providers — Anthropic, OpenAI, Groq, Gemini, and Ollama — through a single,
unified `Provider` abstraction built on [LangChain](https://python.langchain.com/).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples](#examples)
- [Supported Providers & Models](#supported-providers--models)
- [Development](#development)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)

## Introduction

`llm-lab` is a small command-line tool for working with chat-completion LLMs
from the terminal, without writing a script for every provider you want to
try. It exists to answer three practical questions quickly:

- **"What does this model say?"** — `chat` sends one prompt to one provider.
- **"Which model handles this better?"** — `compare` sends the same prompt to
  several providers at once and prints their answers side by side.
- **"How fast/expensive is this, really?"** — `benchmark` runs a prompt N
  times against one provider and reports mean latency and total cost.

Every response is normalized into the same shape (text, token counts,
latency, cost) regardless of which provider produced it, so the three
commands work identically whether you're pointed at a paid model like
`claude-sonnet-4-6` or a free/local one like Ollama's `llama3.2`.

## Features

- **Five providers, one interface** — Anthropic, OpenAI, Groq, Gemini, and
  Ollama, selected with a single `--provider` flag.
- **Free/local testing path** — Groq and Gemini both have real free tiers,
  and Ollama runs entirely locally with no API key at all.
- **Automatic cost tracking** — every response is priced from a built-in
  per-model rate table and shown in every command's output.
- **Fallback on failure** — `chat --fallback` automatically retries against
  a second provider if the first is unavailable after retries.
- **Model overrides with validation** — override the default model per
  provider via `--model` (or per-provider flags in `compare`); an unknown or
  mismatched model name is rejected before any network call is made.
- **Clean failure modes** — missing API keys, unknown providers/models, and
  exhausted retries all produce a short, readable error and a non-zero exit
  code — never a raw stack trace.

## Architecture

```
                     ┌─────────────┐
   CLI (typer)  ───▶ │  llm_lab/   │
   chat / compare /  │   cli.py    │
   benchmark         └──────┬──────┘
                             │  PROVIDERS: dict[name -> factory]
                             ▼
                    ┌────────────────────┐
                    │  Provider (ABC)     │  llm_lab/providers/base.py
                    │  .chat(prompt, sys) │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────────┐
                    │   LangChainProvider      │  llm_lab/providers/langchain_provider.py
                    │   (name, model)           │
                    │   → init_chat_model(...)  │  one class for all 5 providers
                    └─────────┬────────────────┘
                              │ ainvoke()
                ┌─────────────┼──────────────┬───────────────┬─────────────┐
                ▼             ▼              ▼               ▼             ▼
          ChatAnthropic  ChatOpenAI     ChatGroq   ChatGoogleGenerativeAI  ChatOllama
           (Anthropic)    (OpenAI)      (Groq)           (Gemini)          (Ollama)
```

- **`llm_lab/cli.py`** — three Typer commands (`chat`, `compare`,
  `benchmark`) plus the `PROVIDERS` registry mapping a provider name to a
  factory function. All provider selection, model validation, and error
  formatting happens here; the commands know nothing about any specific
  LLM SDK.
- **`llm_lab/providers/base.py`** — the `Provider` abstract base class:
  `name`, `model`, and one async method, `chat(prompt, system) -> ChatResponse`.
  Anything satisfying this contract can be dropped into `PROVIDERS`.
- **`llm_lab/providers/langchain_provider.py`** — `LangChainProvider` is the
  single concrete implementation for all five providers. It builds its
  client via LangChain's `init_chat_model(model, model_provider=name, ...)`
  factory rather than importing five separate SDK-specific classes, reads
  the right API-key environment variable per provider (or none, for
  Ollama), and lets LangChain's built-in retry logic (`max_retries=3`)
  handle transient failures. A flat tuple of each provider's
  rate-limit/connection-error exception types is caught and re-raised as a
  single `ProviderUnavailableError`.
- **`llm_lab/models.py`** — `ChatResponse` (the pydantic model every
  provider returns: text, provider, model, token counts, latency), the
  `PRICING` table (cost per 1M tokens per model), `DEFAULT_MODELS`, and
  `resolve_model(provider, model)` — the single place that validates a
  `--model` override against the provider it's supposed to belong to.
- **`llm_lab/errors.py`** — `ProviderUnavailableError`, the one custom
  exception the whole CLI understands; everything else (missing API key,
  bad model name) is a plain `RuntimeError`/`ValueError` caught at the CLI
  boundary.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo-url>
cd llm-lab
uv sync
```

`uv sync` creates a `.venv/` and installs every provider's SDK
(`anthropic`, `openai`, `langchain`, `langchain-anthropic`,
`langchain-openai`, `langchain-groq`, `langchain-google-genai`,
`langchain-ollama`) from `uv.lock`. You only need credentials for the
provider(s) you actually plan to use — see [Configuration](#configuration).

## Configuration

Each provider (except Ollama) needs its API key set as an environment
variable:

| Provider | Env var | Get a key |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | console.anthropic.com |
| OpenAI | `OPENAI_API_KEY` | platform.openai.com |
| Groq | `GROQ_API_KEY` | console.groq.com (free tier) |
| Gemini | `GOOGLE_API_KEY` | aistudio.google.com (free tier) |
| Ollama | — none — | run a local server, no key needed |

You only need the keys for the providers you're actually going to use.

**Option 1 — export in your shell:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=gsk_...
```

**Option 2 — a `.env` file** (already in `.gitignore` — never commit this
file):

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=...
```

Then either pass it explicitly per command:

```bash
uv run --env-file .env llm-lab chat "hello" --provider groq
```

or set it once per terminal session so every `uv run` picks it up
automatically:

```bash
export UV_ENV_FILE=.env
uv run llm-lab chat "hello" --provider groq
```

**Ollama** needs the [Ollama app](https://ollama.com) running locally and
at least one model pulled, e.g.:

```bash
ollama pull llama3.2
```

## Usage

```
uv run llm-lab --help
```

### `chat` — send one prompt to one provider

```
uv run llm-lab chat PROMPT [OPTIONS]

  --provider   TEXT   anthropic, openai, groq, gemini, or ollama  [default: anthropic]
  --model      TEXT   Override the model for the selected provider
  --system     TEXT   Optional system prompt
  --fallback          Retry with the other provider on failure
```

### `compare` — send one prompt to several providers at once

```
uv run llm-lab compare PROMPT [OPTIONS]

  --system           TEXT   Optional system prompt
  --providers        TEXT   Comma-separated list of providers to compare  [default: anthropic,openai]
  --anthropic-model  TEXT   Override Anthropic's model
  --openai-model     TEXT   Override OpenAI's model
  --groq-model       TEXT   Override Groq's model
  --gemini-model     TEXT   Override Gemini's model
  --ollama-model     TEXT   Override Ollama's model
```

### `benchmark` — run one prompt N times against one provider

```
uv run llm-lab benchmark PROMPT [OPTIONS]

  --provider   TEXT      anthropic, openai, groq, gemini, or ollama  [default: anthropic]
  --model      TEXT      Override the model for the selected provider
  --n          INTEGER   Number of sequential runs  [default: 5]
```

## Examples

> Run the commands below and paste your terminal output into each block.

#### `chat`

```bash
uv run llm-lab chat "What is the capital of France?" --provider gemini
```

```
╭───────────────────────────────────────────────────────────────── gemini (gemini-2.5-flash) ─────────────────────────────────────────────────────────────────╮
│ The capital of France is **Paris**.                                                                                                                         │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                Cost Summary                                 
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Model            ┃ Latency (ms) ┃ Input tokens ┃ Output tokens ┃ Cost ($) ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ gemini-2.5-flash │ 752.8        │ 8            │ 33            │ 0.000085 │
└──────────────────┴──────────────┴──────────────┴───────────────┴──────────┘

```

#### `chat` with a system prompt and fallback

```bash
uv run llm-lab chat "Summarize this repo in one sentence" \
    --provider groq --system "You are a terse assistant" --fallback
```

```
╭────────────────────────────────────────────────────────────── groq (llama-3.3-70b-versatile) ───────────────────────────────────────────────────────────────╮
│ Yes.                                                                                                                                                        │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                    Cost Summary                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Model                   ┃ Latency (ms) ┃ Input tokens ┃ Output tokens ┃ Cost ($) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ llama-3.3-70b-versatile │ 635.0        │ 40           │ 3             │ 0.000026 │
└─────────────────────────┴──────────────┴──────────────┴───────────────┴──────────┘

```


#### `compare` with a custom subset and model overrides

```bash
uv run llm-lab compare "Explain recursion in one sentence" \
    --providers groq,gemini \
    --groq-model llama-3.3-70b-versatile
```

```
                                                                      Provider Comparison                                                                      
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Provider ┃ Text                                                                                    ┃ Latency (ms) ┃ Input tokens ┃ Output tokens ┃ Cost ($) ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ groq     │ Recursion is a programming concept where a function calls itself repeatedly until it    │ 777.7        │ 41           │ 43            │ 0.000058 │
│          │ reaches a base case that stops the recursion, allowing the function to solve a problem  │              │              │               │          │
│          │ by breaking it down into smaller instances of the same problem.                         │              │              │               │          │
│ gemini   │ Recursion is a programming technique where a function solves a problem by calling       │ 3962.4       │ 6            │ 751           │ 0.001879 │
│          │ itself with a simpler version of the problem until a base case is reached.              │              │              │               │          │
└──────────┴─────────────────────────────────────────────────────────────────────────────────────────┴──────────────┴──────────────┴───────────────┴──────────┘
```

#### `benchmark`

```bash
uv run llm-lab benchmark "Say hello" --provider gemini --n 6
```

```
                       Benchmark: gemini (6 runs)                        
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Run          ┃ Latency (ms) ┃ Input tokens ┃ Output tokens ┃ Cost ($) ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ 1            │ 706.7        │ 3            │ 18            │ 0.000046 │
│ 2            │ 602.0        │ 3            │ 21            │ 0.000053 │
│ 3            │ 726.7        │ 3            │ 21            │ 0.000053 │
│ 4            │ 633.9        │ 3            │ 18            │ 0.000046 │
│ 5            │ 5724.1       │ 3            │ 17            │ 0.000043 │
│ 6            │ 2118.5       │ 3            │ 18            │ 0.000046 │
│ Mean / Total │ 1752.0       │ -            │ -             │ 0.000288 │
└──────────────┴──────────────┴──────────────┴───────────────┴──────────┘
```

## Supported Providers & Models

Every `PRICING` entry is tagged with its provider, so `--model`/
`--<provider>-model` overrides are validated against the right one. Prices
are USD per 1M tokens.

| Provider | Default model | Input | Output |
|---|---|---|---|
| anthropic | `claude-sonnet-4-6` | $3.00 | $15.00 |
| openai | `gpt-4o-mini` | $0.15 | $0.60 |
| groq | `llama-3.3-70b-versatile` | $0.59 | $0.79 |
| gemini | `gemini-2.5-flash` | $0.30 | $2.50 |
| ollama | `llama3.2` | $0.00 | $0.00 (local) |

Other supported model overrides include `claude-opus-4-6`,
`claude-haiku-4-5`, and the full OpenAI GPT-5/GPT-4.1/GPT-4o/reasoning
lineup (`gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4o`, `o3`,
`o1`, and more) — see `PRICING` in `llm_lab/models.py` for the full,
current list.

## Development

```bash
uv sync
uv run pytest -v
```

The test suite mocks every provider's HTTP layer — no API keys or network
access are required to run it, and no real API calls are ever made in CI.

## Project Structure

```
llm_lab/
├── cli.py                       # chat / compare / benchmark commands
├── models.py                    # ChatResponse, PRICING, resolve_model
├── errors.py                    # ProviderUnavailableError
└── providers/
    ├── base.py                  # Provider ABC
    └── langchain_provider.py    # LangChainProvider (all 5 providers)
tests/
├── test_cli.py
├── test_models.py
└── test_providers.py
```

## Known Limitations

- `chat --fallback` only ever falls back between Anthropic and OpenAI
  specifically — it doesn't chain across all five providers.
- Ollama's `max_tokens` cap isn't currently honored (Ollama uses a
  different parameter, `num_predict`, for this), so its responses aren't
  length-capped the same way the other four providers' are.
- Pricing in `PRICING` is a hardcoded snapshot (as of 2026-07-01) and isn't
  fetched live from any provider — check the provider's own pricing page if
  you need up-to-the-minute rates.
