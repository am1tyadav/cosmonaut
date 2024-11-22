import os

import httpx
from loguru import logger
from pydantic import BaseModel

from cosmonaut.clients.base import BaseRESTClient
from cosmonaut.data_models import PredictionResponse, ResponseInfo


class OpenAIRESTClient(BaseRESTClient):
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
        try:
            messages = self._processor.build_messages(prompt, instructions)
            data = self._processor.build_json(messages, temperature, instructions)

            async with httpx.AsyncClient(
                base_url=self._config.base_url, timeout=self.config.timeout
            ) as client:
                response = await client.post(
                    self.endpoint,
                    json=data,
                    headers=self.headers,
                    timeout=self.config.timeout,
                )
        except Exception as e:
            exception_type = type(e).__name__
            error_message = f"Exception type: {exception_type}. Request failed: {e}"
            logger.error(error_message)
            raise Exception(error_message) from e

        response.raise_for_status()
        return response.json()

    async def predict(
        self, prompt: str, response_format: BaseModel, instructions: str
    ) -> PredictionResponse:
        try:
            response = await self.completion(prompt, instructions)
            text = self._processor.extract_text(response)
            predictions = self._processor.parse_outputs(text, response_format)
            return PredictionResponse(
                success=True, info=ResponseInfo.SUCCESS.value, predictions=predictions
            )
        except Exception as e:
            logger.error(e)
            return PredictionResponse(
                success=False, info=ResponseInfo.ERROR.value, detail=str(e)
            )
