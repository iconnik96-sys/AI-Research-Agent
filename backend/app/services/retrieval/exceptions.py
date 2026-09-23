class RetrievalError(Exception):
    """Base exception for all retrieval service errors."""
    pass


class RetrievalConfigError(RetrievalError):
    """Raised when retrieval configuration is invalid or missing."""
    pass


class RetrievalEmbeddingError(RetrievalError):
    """Raised when query embedding generation fails during retrieval."""
    pass


class RetrievalDatabaseError(RetrievalError):
    """Raised when database query execution fails during retrieval."""
    pass
