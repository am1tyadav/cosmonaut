import ast

from pydantic import BaseModel

from cosmonaut.logging import get_logger
from cosmonaut.processors.base import BaseProcessor

logger = get_logger(__name__)


class OpenAIProcessor(BaseProcessor):
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

    def build_json(
        self, messages: list[dict], temperature: float, instructions: str
    ) -> dict:
        data = {
            "model": self._config.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }
        return data

    def extract_output(self, response: dict) -> str | dict:
        return response["choices"][0]["message"]["content"]

    def parse_outputs(
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
