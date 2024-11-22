import os

from cosmonaut.clients.openai import OpenAIRESTClient


class GeminiRESTClient(OpenAIRESTClient):
    @property
    def headers(self) -> dict:
        return {"Content-Type": "application/json"}

    @property
    def endpoint(self) -> str:
        api_key = os.getenv(self.config.api_key_var)
        model_name = self.config.name
        return f"/models/{model_name}:generateContent?key={api_key}"
