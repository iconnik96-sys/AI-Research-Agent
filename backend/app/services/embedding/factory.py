from app.services.embedding.base import BaseEmbeddingProvider
from app.services.embedding.provider import OpenAICompatibleEmbeddingProvider


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Dependency provider returning the configured embedding provider."""
    return OpenAICompatibleEmbeddingProvider()
