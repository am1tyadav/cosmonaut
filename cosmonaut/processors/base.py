from abc import ABC, abstractmethod

from pydantic import BaseModel

from cosmonaut.data_models import AIClientConfig


class BaseProcessor(ABC):
    def __init__(self, config: AIClientConfig):
        self._config = config

    @abstractmethod
    def build_messages(self, prompt: str, instructions: str) -> list[dict]: ...

    @abstractmethod
    def build_request_data(
        self, messages: list[dict], temperature: float, instructions: str
    ) -> dict: ...

    @abstractmethod
    def extract_output(self, response: dict) -> str | dict: ...

    @abstractmethod
    def parse_output(
        self, text_or_dict: str | dict, response_format: BaseModel
    ) -> BaseModel: ...
