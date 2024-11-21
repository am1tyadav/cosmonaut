from abc import ABC, abstractmethod

from pydantic import BaseModel

from cosmonaut.data_models import AIClientConfig


class BaseProcessor(ABC):
    def __init__(self, config: AIClientConfig):
        self._config = config

    @abstractmethod
    def build_messages(self, prompt: str, instructions: str) -> list[dict]: ...

    @abstractmethod
    def build_json(
        self, messages: list[dict], temperature: float, instructions: str
    ) -> dict: ...

    @abstractmethod
    def extract_text(self, response: dict) -> str: ...

    @abstractmethod
    def parse_outputs(self, text: str, response_format: BaseModel) -> BaseModel: ...
