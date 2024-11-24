import ast

from pydantic import BaseModel

from cosmonaut.data_models import AIClientConfig, OpenAIResponseFormat
from cosmonaut.logging import get_logger
from cosmonaut.processors.base import BaseProcessor

logger = get_logger(__name__)


def get_openai_response_format(
    openai_response_format: OpenAIResponseFormat, schema: dict
) -> dict | None:
    match openai_response_format:
        case OpenAIResponseFormat.JSON_OBJECT:
            return {
                "type": "json_object",
            }
        case OpenAIResponseFormat.JSON_SCHEMA:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "Predictions",
                    "strict": True,
                    "schema": schema,
                },
            }
        case OpenAIResponseFormat.TEXT:
            return None


class OpenAIProcessor(BaseProcessor):
    def __init__(self, config: AIClientConfig):
        super().__init__(config)

        self._response_format = get_openai_response_format(
            self._config.openai_response_format, self._config.prediction_schema
        )

    def build_messages(self, prompt: str, instructions: str) -> list[dict]:
        messages = [
            {
                "role": "system",
                "content": instructions,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        return messages

    def build_request_data(
        self, messages: list[dict], temperature: float, instructions: str
    ) -> dict:

        data = {
            "model": self._config.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }

        if self._response_format is not None:
            data["response_format"] = self._response_format

        return data

    def extract_output(self, response: dict) -> str | dict:
        return response["choices"][0]["message"]["content"]

    def parse_output(
        self, text_or_dict: str | dict, response_format: BaseModel
    ) -> BaseModel:
        """Needed when the model doesn't support tools/ response_format directly"""
        outputs = text_or_dict  # is text in this case
        cleaned = outputs.replace("```json", "").replace("```", "").replace("\n", "")

        try:
            return response_format.model_validate_json(cleaned)
        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")

        try:
            return response_format.model_validate(ast.literal_eval(cleaned))
        except Exception as e:
            logger.warning(f"Failed to parse with ast: {e}")

        try:
            return response_format.model_validate(ast.literal_eval(cleaned + "}"))
        except Exception as e:
            logger.warning(f"Failed to parse with ast fix: {e}")

        raise ValueError(f"Failed to parse response for outputs: ({outputs})")
