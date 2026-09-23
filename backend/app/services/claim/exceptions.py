class ClaimError(Exception):
    """Base exception for claim extraction and verification errors."""
    pass


class ClaimExtractionError(ClaimError):
    """Raised when claim extraction fails due to provider or network errors."""
    pass


class ClaimResponseError(ClaimExtractionError):
    """Raised when extracted claims violate schema or validation rules."""
    pass


class ClaimVerificationError(ClaimError):
    """Raised when claim verification fails."""
    pass


class ClaimConfigError(ClaimError):
    """Raised when claim service configuration is invalid or missing."""
    pass
