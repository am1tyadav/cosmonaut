import asyncio
from pathlib import Path
from typing import Callable

import pandas as pd

from cosmonaut.clients.selector import get_client
from cosmonaut.config_loader import load_config
from cosmonaut.data_models import Config, Predictions


class Cosmonaut:
    def __init__(
        self,
        config_or_config_filepath: dict | Path,
        fn_create_prompt: Callable[[pd.Series], str],
    ):
        self._config_or_config_filepath = config_or_config_filepath
        self._config = load_config(config_or_config_filepath)
        self._fn_create_prompt = fn_create_prompt
        self._ai_client = get_client(self._config.ai_client)

    @property
    def config(self) -> Config:
        return self._config

    async def _run(self, inputs: pd.DataFrame) -> pd.DataFrame:
        prompts = inputs.apply(self._fn_create_prompt, axis=1)

        tasks = [
            self._ai_client.predict(
                prompt, Predictions, self._config.classifier.instructions
            )
            for prompt in prompts.values
        ]

        responses = await asyncio.gather(*tasks)

        inputs[self._config.data.result_column] = [
            response.model_dump() for response in responses
        ]
        return inputs

    def run(self, inputs: pd.DataFrame) -> pd.DataFrame:
        return asyncio.run(self._run(inputs))
