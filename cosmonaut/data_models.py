from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

"""Enums"""


class AIServiceProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


class ResponseInfo(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class LabelConfig(BaseModel):
    name: str
    description: str


class LabelDetails(BaseModel):
    name: str = Field(description="Name of the label")
    reason: str = Field(description="Reason for predicting the label")


class PredictionDetail(BaseModel):
    name: str = Field(description="Name of the category")
    labels: dict[int, LabelDetails] = Field(
        description="Dictionary mapping label IDs to their prediction details"
    )


class Prediction(BaseModel):
    prediction: dict[int, PredictionDetail] = Field(
        description="Dictionary mapping category IDs to their predictions"
    )


class CategoryConfig(BaseModel):
    name: str
    description: str
    max_predictions: int
    labels: dict[int, LabelConfig]


class ClassifierConfig(BaseModel):
    instructions_filename: Path | None = None
    instructions: str | None = None
    """If None, instructions are created when loading config"""
    categories: dict[int, CategoryConfig]
    examples: list[Prediction]


class AIClientConfig(BaseModel):
    ai_provider: AIServiceProvider
    name: str
    api_key_var: str
    base_url: str | None = None
    timeout: int = 30
    max_tokens: int = 2048


class DataConfig(BaseModel):
    result_column: str


class Config(BaseModel):
    classifier: ClassifierConfig
    ai_client: AIClientConfig
    data: DataConfig


class PredictionResponse(BaseModel):
    success: bool
    info: str | None = None
    detail: str | None = None
    prediction: Prediction | None = None
