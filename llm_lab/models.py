from pydantic import BaseModel


class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def cost(response: ChatResponse) -> float:
    rates = PRICING[response.model]
    return (
        response.input_tokens / 1_000_000 * rates["input"]
        + response.output_tokens / 1_000_000 * rates["output"]
    )
