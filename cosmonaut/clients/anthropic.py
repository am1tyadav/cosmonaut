import os

from cosmonaut.clients.openai import OpenAIClient


class AnthropicClient(OpenAIClient):
    @property
    def headers(self) -> dict:
        api_key = os.getenv(self.config.api_key_var)

        return {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": api_key,
        }

    @property
    def endpoint(self) -> str:
        return "/messages"
