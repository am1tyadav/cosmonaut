from pydantic import BaseModel

from cosmonaut.processors.base import BaseProcessor


class AnthropicProcessor(BaseProcessor):
    def build_messages(self, prompt: str, instructions: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    def build_json(
        self, messages: list[dict], temperature: float, instructions: str
    ) -> dict:
        data = {
            "model": self._config.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
            "system": instructions,
            "tool_choice": {"type": "tool", "name": "predicted_output"},
            "tools": [
                {
                    "name": "predicted_output",
                    "description": "Predicted output using well-structured JSON.",
                    "input_schema": self._config.prediction_schema,
                }
            ],
        }
        return data

    def extract_output(self, response: dict) -> str | dict:
        return response["content"][0]["input"]

    def parse_outputs(
        self, text_or_dict: str | dict, response_format: BaseModel
    ) -> BaseModel:
        return response_format.model_validate(text_or_dict)
