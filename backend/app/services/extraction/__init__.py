"""Document and webpage extraction services."""
from app.services.extraction.base import BaseExtractor
from app.services.extraction.exceptions import (
    ExtractionContentError,
    ExtractionError,
    FetchNetworkError,
    FetchTimeoutError,
    InvalidResponseError,
)
from app.services.extraction.webpage import WebpageExtractor


def get_webpage_extractor() -> WebpageExtractor:
    """Dependency provider returning a WebpageExtractor instance."""
    return WebpageExtractor()


__all__ = [
    "BaseExtractor",
    "WebpageExtractor",
    "ExtractionError",
    "FetchTimeoutError",
    "FetchNetworkError",
    "InvalidResponseError",
    "ExtractionContentError",
    "get_webpage_extractor",
]
