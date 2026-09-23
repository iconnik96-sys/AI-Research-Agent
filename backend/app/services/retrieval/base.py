from abc import ABC, abstractmethod
from typing import List, Optional

from app.schemas.retrieval import RetrievedChunk


class BaseRetriever(ABC):
    """Abstract interface for semantic retrieval of relevant document chunks."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve the most relevant document chunks for a natural-language query.

        Args:
            query: The research question or search query.
            session_id: Optional research session UUID to scope retrieval.
            top_k: Maximum number of chunks to return. Defaults to service setting.
            similarity_threshold: Minimum cosine similarity threshold. Defaults to service setting.

        Returns:
            List of RetrievedChunk objects ordered from highest to lowest similarity.

        Raises:
            RetrievalConfigError: If retrieval or embedding configuration is invalid.
            RetrievalEmbeddingError: If generating the query embedding fails.
            RetrievalDatabaseError: If database execution fails.
            RetrievalError: For general retrieval failures.
        """
        pass
