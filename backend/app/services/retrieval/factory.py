from fastapi import Depends

from app.services.embedding.base import BaseEmbeddingProvider
from app.services.embedding.factory import get_embedding_provider
from app.services.persistence.base import BaseResearchRepository
from app.services.persistence.factory import get_research_repository
from app.services.retrieval.base import BaseRetriever
from app.services.retrieval.pgvector import PgVectorRetriever


def get_retriever(
    embedding_provider: BaseEmbeddingProvider = Depends(get_embedding_provider),
    repository: BaseResearchRepository = Depends(get_research_repository),
) -> BaseRetriever:
    """Dependency provider returning configured PgVectorRetriever."""
    return PgVectorRetriever(
        embedding_provider=embedding_provider,
        repository=repository,
    )
