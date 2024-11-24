from abc import ABC, abstractmethod

from pydantic import BaseModel

from cosmonaut.data_models import AIClientConfig, PredictionResponse
from cosmonaut.processors.base import BaseProcessor


class BaseClient(ABC):
    def __init__(self, config: AIClientConfig, processor: BaseProcessor):
        self._config = config
        self._processor = processor

    @property
    def config(self) -> AIClientConfig:
        return self._config

    @property
    @abstractmethod
    def headers(self) -> dict: ...

    @property
    @abstractmethod
    def endpoint(self) -> str: ...

    @abstractmethod
    def completion(
        self,
        prompt: str,
        instructions: str,
        temperature: float = 0.5,
    ) -> dict: ...

    @abstractmethod
    def predict(
        self, prompt: str, response_format: BaseModel, instructions: str
    ) -> PredictionResponse: ...
