from app.services.llm.base import BaseLLMProvider
from app.services.llm.provider import OpenAICompatibleLLMProvider


def get_llm_provider() -> BaseLLMProvider:
    """Dependency provider factory returning configured LLM provider."""
    return OpenAICompatibleLLMProvider()
