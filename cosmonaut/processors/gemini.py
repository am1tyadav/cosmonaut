from cosmonaut.processors.openai import OpenAIProcessor


class GeminiProcessor(OpenAIProcessor):
    def build_messages(self, prompt: str, instructions: str) -> list[dict]:
        messages = [
            {
                "parts": {"text": prompt},
            },
        ]
        return messages

    def build_json(
        self, messages: list[dict], temperature: float, instructions: str
    ) -> dict:
        generation_config = {
            "temperature": temperature,
            "maxOutputTokens": self._config.max_tokens,
        }

        data = {
            "contents": messages,
            "generationConfig": generation_config,
            "system_instruction": {"parts": {"text": instructions}},
        }
        return data

    def extract_text(self, response: dict) -> str:
        return response["candidates"][0]["content"]["parts"][0]["text"]
