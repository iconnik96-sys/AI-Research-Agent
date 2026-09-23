from typing import Optional


class ExtractionError(Exception):
    """Base exception for all document extraction errors."""
    pass


class FetchTimeoutError(ExtractionError):
    """Raised when fetching a source document or webpage times out."""
    pass


class FetchNetworkError(ExtractionError):
    """Raised when network connectivity fails while fetching a document."""
    pass


class InvalidResponseError(ExtractionError):
    """Raised when the server returns an HTTP error (e.g. 403, 404, 500) or non-HTML content."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class ExtractionContentError(ExtractionError):
    """Raised when a webpage has no meaningful or extractable text content."""
    pass
