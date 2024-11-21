from cosmonaut.clients.anthropic import AnthropicRESTClient
from cosmonaut.clients.base import BaseRESTClient
from cosmonaut.clients.gemini import GeminiRESTClient
from cosmonaut.clients.openai import OpenAIRESTClient
from cosmonaut.data_models import AIClientConfig, AIServiceProvider
from cosmonaut.processors.anthropic import AnthropicProcessor
from cosmonaut.processors.gemini import GeminiProcessor
from cosmonaut.processors.openai import OpenAIProcessor


def get_client(config: AIClientConfig) -> BaseRESTClient:
    ai_provider = config.ai_provider

    match ai_provider:
        case AIServiceProvider.ANTHROPIC:
            return AnthropicRESTClient(config, AnthropicProcessor(config))
        case AIServiceProvider.OPENAI:
            return OpenAIRESTClient(config, OpenAIProcessor(config))
        case AIServiceProvider.GEMINI:
            return GeminiRESTClient(config, GeminiProcessor(config))
        case _:
            raise ValueError(f"Unsupported AI Provider type: {ai_provider}")
