from pydantic import BaseModel


class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


PRICING: dict[str, dict[str, str | float]] = {
    # Anthropic Claude
    "claude-opus-4-6": {"provider": "anthropic", "input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"provider": "anthropic", "input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"provider": "anthropic", "input": 1.00, "output": 5.00},

    # OpenAI GPT-5 family
    "gpt-5-2": {"provider": "openai", "input": 1.75, "output": 14.00},
    "gpt-5-1": {"provider": "openai", "input": 1.25, "output": 10.00},
    "gpt-5": {"provider": "openai", "input": 1.25, "output": 10.00},
    "gpt-5-mini": {"provider": "openai", "input": 0.25, "output": 2.00},
    "gpt-5-nano": {"provider": "openai", "input": 0.05, "output": 0.40},

    # OpenAI GPT-4 family
    "gpt-4.1": {"provider": "openai", "input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"provider": "openai", "input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"provider": "openai", "input": 0.10, "output": 0.40},

    "gpt-4o": {"provider": "openai", "input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"provider": "openai", "input": 0.15, "output": 0.60},

    # OpenAI reasoning models
    "o3": {"provider": "openai", "input": 2.00, "output": 8.00},
    "o4-mini": {"provider": "openai", "input": 1.10, "output": 4.40},
    "o3-mini": {"provider": "openai", "input": 1.10, "output": 4.40},
    "o1": {"provider": "openai", "input": 15.00, "output": 60.00},
    "o1-mini": {"provider": "openai", "input": 1.10, "output": 4.40},

    # Groq
    "llama-3.3-70b-versatile": {"provider": "groq", "input": 0.59, "output": 0.79},

    # Google Gemini
    "gemini-2.5-flash": {"provider": "gemini", "input": 0.30, "output": 2.50},

    # Ollama (local, no billing)
    "llama3.2": {"provider": "ollama", "input": 0.0, "output": 0.0},
}


DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.2",
}


def cost(response: ChatResponse) -> float:
    rates = PRICING[response.model]
    return (
        response.input_tokens / 1_000_000 * rates["input"]
        + response.output_tokens / 1_000_000 * rates["output"]
    )


def resolve_model(provider: str, model: str | None) -> str:
    if model is None:
        return DEFAULT_MODELS[provider]
    entry = PRICING.get(model)
    if entry is None:
        raise ValueError(f"Unknown model '{model}'")
    if entry["provider"] != provider:
        raise ValueError(
            f"Model '{model}' belongs to provider '{entry['provider']}', not '{provider}'"
        )
    return model
