import os

import httpx
from pydantic import BaseModel

from cosmonaut.clients.base import BaseClient
from cosmonaut.data_models import PredictionResponse
from cosmonaut.logging import get_logger

logger = get_logger(__name__)


class OpenAIClient(BaseClient):
    @property
    def headers(self) -> dict:
        api_key = os.getenv(self.config.api_key_var)

        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @property
    def endpoint(self) -> str:
        return "/chat/completions"

    async def completion(
        self,
        prompt: str,
        instructions: str,
        temperature: float = 0.5,
    ) -> dict:
        messages = self._processor.build_messages(prompt, instructions)
        data = self._processor.build_request_data(messages, temperature, instructions)

        async with httpx.AsyncClient(
            base_url=self._config.base_url, timeout=self.config.timeout
        ) as client:
            response = await client.post(
                self.endpoint,
                json=data,
                headers=self.headers,
                timeout=self.config.timeout,
            )
        response.raise_for_status()
        return response.json()

    async def predict(
        self, prompt: str, response_format: BaseModel, instructions: str
    ) -> PredictionResponse:
        try:
            response = await self.completion(prompt, instructions)
            text = self._processor.extract_output(response)
            predictions = self._processor.parse_output(text, response_format)
            return PredictionResponse(success=True, predictions=predictions)
        except Exception as e:
            exception_type = type(e).__name__
            logger.error(e)
            detail = f"{exception_type}: {e}"
            return PredictionResponse(success=False, detail=detail)
