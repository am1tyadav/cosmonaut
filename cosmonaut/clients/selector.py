from cosmonaut.clients.anthropic import AnthropicClient
from cosmonaut.clients.base import BaseClient
from cosmonaut.clients.gemini import GeminiClient
from cosmonaut.clients.openai import OpenAIClient
from cosmonaut.data_models import AIClientConfig, AIServiceProvider
from cosmonaut.processors.anthropic import AnthropicProcessor
from cosmonaut.processors.gemini import GeminiProcessor
from cosmonaut.processors.openai import OpenAIProcessor


def get_client(config: AIClientConfig) -> BaseClient:
    ai_provider = config.ai_provider

    match ai_provider:
        case AIServiceProvider.ANTHROPIC:
            return AnthropicClient(config, AnthropicProcessor(config))
        case AIServiceProvider.OPENAI:
            return OpenAIClient(config, OpenAIProcessor(config))
        case AIServiceProvider.GEMINI:
            return GeminiClient(config, GeminiProcessor(config))
        case _:
            raise ValueError(f"Unsupported AI Provider type: {ai_provider}")
