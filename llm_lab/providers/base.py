from abc import ABC, abstractmethod

from llm_lab.models import ChatResponse


class Provider(ABC):
    name: str
    model: str

    @abstractmethod
    async def chat(self, prompt: str, system: str | None = None) -> ChatResponse:
        raise NotImplementedError
