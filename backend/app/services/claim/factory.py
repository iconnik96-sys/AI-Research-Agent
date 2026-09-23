from app.services.claim.base import BaseClaimExtractor, BaseClaimVerifier
from app.services.claim.extractor import LLMClaimExtractor
from app.services.claim.verifier import LLMClaimVerifier


def get_claim_extractor() -> BaseClaimExtractor:
    """FastAPI dependency factory returning configured claim extractor instance."""
    return LLMClaimExtractor()


def get_claim_verifier() -> BaseClaimVerifier:
    """FastAPI dependency factory returning configured claim verifier instance."""
    return LLMClaimVerifier()
