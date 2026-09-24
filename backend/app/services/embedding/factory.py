from app.services.embedding.base import BaseEmbeddingProvider
from app.services.embedding.provider import SupabaseEmbeddingProvider


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Dependency provider returning the configured Supabase embedding provider."""
    return SupabaseEmbeddingProvider()
