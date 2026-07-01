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
