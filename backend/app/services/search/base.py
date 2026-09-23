from abc import ABC, abstractmethod
from typing import List

from app.schemas.research import SourceItem


class SearchError(Exception):
    """Base exception for search provider operations."""
    pass


class SearchConfigError(SearchError):
    """Raised when search provider configuration (e.g. API key) is missing or invalid."""
    pass


class SearchTimeoutError(SearchError):
    """Raised when the search provider request times out."""
    pass


class SearchProviderError(SearchError):
    """Raised when the upstream search provider returns an error or invalid response."""
    pass


class BaseSearchProvider(ABC):
    """Abstract base class for all search providers."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[SourceItem]:
        """Execute a search query and return a list of structured source items.

        Args:
            query: The research or search question.
            max_results: Maximum number of sources to return.

        Returns:
            List of structured SourceItem objects.

        Raises:
            SearchConfigError: If credentials or configuration are missing.
            SearchTimeoutError: If the search request times out.
            SearchProviderError: If the upstream provider fails.
        """
        pass
