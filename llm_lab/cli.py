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
    table.add_column("Text", no_wrap=True)
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
