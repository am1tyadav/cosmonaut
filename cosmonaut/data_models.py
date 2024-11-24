from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

"""Enums"""


class AIServiceProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


class KnownBaseURLs(str, Enum):
    ANTHROPIC = "https://api.anthropic.com/v1"
    OPENAI = "https://api.openai.com/v1"
    GEMINI = "https://generativelanguage.googleapis.com/v1beta"


class OpenAIResponseFormat(str, Enum):
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
    TEXT = "text"


"""Prediction Schemas"""


class LabelDetails(BaseModel):
    label: str = Field(description="Name of the label")
    reason: str = Field(description="Reason for predicting the label")


class PredictionDetail(BaseModel):
    category: str
    labels: list[LabelDetails] | list[str]


class Predictions(BaseModel):
    predictions: list[PredictionDetail]


"""Config Schemas"""


class LabelConfig(BaseModel):
    label: str
    description: str


class CategoryConfig(BaseModel):
    category: str
    description: str
    max_predictions: int
    labels: list[LabelConfig] | list[str]


class ClassifierConfig(BaseModel):
    instructions_filename: Path | None = None
    instructions: str | None = None
    require_reason: bool = True
    label_descriptions_provided: bool = True
    """If None, instructions are created when loading config"""
    categories: list[CategoryConfig]
    examples: list[Predictions]


class AIClientConfig(BaseModel):
    ai_provider: AIServiceProvider
    name: str
    api_key_var: str
    base_url: str | None = None
    timeout: int = 30
    max_tokens: int = 2048
    prediction_schema: dict | None = None
    openai_response_format: OpenAIResponseFormat = OpenAIResponseFormat.TEXT


class DataConfig(BaseModel):
    result_column: str


class Config(BaseModel):
    classifier: ClassifierConfig
    ai_client: AIClientConfig
    data: DataConfig


"""Response Schemas"""


class PredictionResponse(BaseModel):
    success: bool
    detail: str | None = None
    predictions: Predictions | None = None
