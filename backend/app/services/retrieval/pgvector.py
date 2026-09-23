import logging
from typing import List, Optional

from app.core.config import settings
from app.schemas.retrieval import RetrievedChunk
from app.services.embedding.base import BaseEmbeddingProvider
from app.services.embedding.exceptions import EmbeddingConfigError, EmbeddingError
from app.services.persistence.base import BaseResearchRepository
from app.services.persistence.exceptions import DatabaseConfigError, DatabaseError
from app.services.retrieval.base import BaseRetriever
from app.services.retrieval.exceptions import (
    RetrievalConfigError,
    RetrievalDatabaseError,
    RetrievalEmbeddingError,
)

logger = logging.getLogger(__name__)


class PgVectorRetriever(BaseRetriever):
    """Semantic retriever using query embeddings and Supabase pgvector cosine similarity."""

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        repository: BaseResearchRepository,
        default_top_k: Optional[int] = None,
        default_similarity_threshold: Optional[float] = None,
    ):
        self.embedding_provider = embedding_provider
        self.repository = repository
        self.default_top_k = (
            default_top_k if default_top_k is not None else settings.RETRIEVAL_TOP_K
        )
        self.default_similarity_threshold = (
            default_similarity_threshold
            if default_similarity_threshold is not None
            else settings.RETRIEVAL_SIMILARITY_THRESHOLD
        )

    async def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve top-K similar chunks using pgvector cosine similarity."""
        if not query or not query.strip():
            return []

        effective_top_k = top_k if top_k is not None else self.default_top_k
        if effective_top_k <= 0:
            raise RetrievalConfigError("top_k must be greater than 0")

        effective_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.default_similarity_threshold
        )

        # 1. Generate query embedding using the configured embedding provider
        try:
            query_embedding = await self.embedding_provider.embed_text(query.strip())
        except EmbeddingConfigError as exc:
            logger.error("Embedding configuration error during query embedding: %s", exc)
            raise RetrievalConfigError(str(exc)) from exc
        except EmbeddingError as exc:
            logger.error("Failed to generate query embedding: %s", exc)
            raise RetrievalEmbeddingError(
                f"Error generating embedding for query: {str(exc)}"
            ) from exc

        # 2. Query stored chunks in PostgreSQL using pgvector cosine similarity
        try:
            results = await self.repository.search_similar_chunks(
                query_embedding=query_embedding,
                session_id=session_id,
                top_k=effective_top_k,
                similarity_threshold=effective_threshold,
            )
            return results
        except DatabaseConfigError as exc:
            logger.error("Database configuration error during vector search: %s", exc)
            raise RetrievalConfigError(str(exc)) from exc
        except DatabaseError as exc:
            logger.error("Database error during vector similarity search: %s", exc)
            raise RetrievalDatabaseError(
                f"Database error during vector retrieval: {str(exc)}"
            ) from exc
