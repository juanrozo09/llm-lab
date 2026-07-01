from abc import ABC, abstractmethod

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_lab.models import ChatResponse


def retry_on_transient(*exception_types: type[Exception]):
    return retry(
        retry=retry_if_exception_type(exception_types),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0, max=2),
        reraise=True,
    )


class Provider(ABC):
    name: str
    model: str

    @abstractmethod
    async def chat(self, prompt: str, system: str | None = None) -> ChatResponse:
        raise NotImplementedError
