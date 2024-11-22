import ast

from loguru import logger
from pydantic import BaseModel

from cosmonaut.processors.base import BaseProcessor


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

    def extract_text(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]

    def parse_outputs(self, outputs: str, response_format: BaseModel) -> BaseModel:
        """Needed when the model doesn't support tools/ response_format directly"""
        cleaned = outputs.replace("```json", "").replace("```", "")

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
