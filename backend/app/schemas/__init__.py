"""Pydantic schemas for data validation and API responses."""
from app.schemas.claim import (
    ClaimVerificationStatus,
    ExtractedClaim,
    ExtractedClaimsResponse,
    VerifiedClaim,
    VerifyClaimsRequest,
    VerifyClaimsResponse,
)
from app.schemas.document import Document
from app.schemas.evidence import EvidenceItem
from app.schemas.report import ResearchReport, ResearchSection, SourceReference
from app.schemas.research import ResearchRequest, ResearchResponse, SourceItem

__all__ = [
    "ResearchRequest",
    "ResearchResponse",
    "SourceItem",
    "Document",
    "ResearchReport",
    "ResearchSection",
    "SourceReference",
    "EvidenceItem",
    "ClaimVerificationStatus",
    "ExtractedClaim",
    "ExtractedClaimsResponse",
    "VerifiedClaim",
    "VerifyClaimsRequest",
    "VerifyClaimsResponse",
]
