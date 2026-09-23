from abc import ABC, abstractmethod
from typing import Optional

from app.schemas.document import Document


class BaseExtractor(ABC):
    """Abstract base class for document and webpage extractors."""

    @abstractmethod
    async def extract(
        self,
        url: str,
        title: Optional[str] = None,
        score: Optional[float] = None,
    ) -> Document:
        """Fetch and extract clean readable text from a source URL.

        Args:
            url: The URL of the source to extract.
            title: Optional default or fallback title from search results.
            score: Optional search relevance score.

        Returns:
            Document containing cleaned text and metadata.

        Raises:
            FetchTimeoutError: If the request times out.
            FetchNetworkError: If connection/network fails.
            InvalidResponseError: If the server returns an HTTP error status.
            ExtractionContentError: If the page has no extractable content.
        """
        pass
