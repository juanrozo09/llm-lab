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
    table.add_column("Provider", no_wrap=True)
    table.add_column("Text", overflow="fold", ratio=1)
    table.add_column("Latency (ms)", no_wrap=True)
    table.add_column("Input tokens", no_wrap=True)
    table.add_column("Output tokens", no_wrap=True)
    table.add_column("Cost ($)", no_wrap=True)
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
