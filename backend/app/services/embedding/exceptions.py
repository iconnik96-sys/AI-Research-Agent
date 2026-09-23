from typing import Optional


class EmbeddingError(Exception):
    """Base exception for all embedding service errors."""
    pass


class EmbeddingConfigError(EmbeddingError):
    """Raised when embedding configuration or credentials (e.g. API key) are missing or invalid."""
    pass


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when an embedding request times out."""
    pass


class EmbeddingNetworkError(EmbeddingError):
    """Raised when network connectivity fails while communicating with the embedding API."""
    pass


class EmbeddingProviderError(EmbeddingError):
    """Raised when the upstream embedding provider returns an HTTP error or rate limit."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class EmbeddingResponseError(EmbeddingError):
    """Raised when the embedding response is malformed or vector dimensions do not match."""
    pass
