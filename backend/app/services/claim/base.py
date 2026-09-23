from abc import ABC, abstractmethod
from typing import List

from app.schemas.claim import ExtractedClaim, VerifiedClaim
from app.schemas.evidence import EvidenceItem


class BaseClaimExtractor(ABC):
    """Abstract interface for extracting candidate factual claims from evidence."""

    @abstractmethod
    async def extract_claims(
        self,
        question: str,
        evidence: List[EvidenceItem],
    ) -> List[ExtractedClaim]:
        """Extract atomic factual claims with candidate evidence mappings.

        Args:
            question: The user research question.
            evidence: Formatted evidence items with deterministic identifiers (e.g. 'E1', 'E2').

        Returns:
            List of ExtractedClaim instances.
        """
        pass


class BaseClaimVerifier(ABC):
    """Abstract interface for independently verifying extracted claims against evidence."""

    @abstractmethod
    async def verify_single_claim(
        self,
        question: str,
        claim: ExtractedClaim,
        evidence_items: List[EvidenceItem],
    ) -> VerifiedClaim:
        """Independently verify a single claim against its associated evidence items.

        Args:
            question: The user research question.
            claim: ExtractedClaim containing statement and candidate evidence IDs.
            evidence_items: The specific EvidenceItem objects cited by the claim.

        Returns:
            VerifiedClaim containing status, explanation, and confirmed supporting evidence IDs.
        """
        pass

    @abstractmethod
    async def verify_claims(
        self,
        question: str,
        claims: List[ExtractedClaim],
        evidence: List[EvidenceItem],
    ) -> List[VerifiedClaim]:
        """Verify multiple extracted claims with bounded concurrency.

        Args:
            question: The user research question.
            claims: Extracted candidate claims.
            evidence: All available evidence items in the research session.

        Returns:
            List of VerifiedClaim instances.
        """
        pass
