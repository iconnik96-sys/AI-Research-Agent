from typing import Optional


class LLMError(Exception):
    """Base exception for all LLM service errors."""
    pass


class LLMConfigError(LLMError):
    """Raised when LLM configuration or credentials (e.g. API key) are missing or invalid."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM completion request times out."""
    pass


class LLMNetworkError(LLMError):
    """Raised when network connectivity fails while communicating with the LLM API."""
    pass


class LLMProviderError(LLMError):
    """Raised when the upstream LLM provider returns an HTTP error or rate limit."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class LLMResponseError(LLMError):
    """Raised when the LLM response is malformed, fails schema validation, or contains invalid citations."""
    pass
