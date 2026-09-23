from enum import Enum
from typing import List, Set
from pydantic import BaseModel, Field, field_validator

from app.schemas.evidence import EvidenceItem


class ClaimVerificationStatus(str, Enum):
    """Explicit verification status for a research claim."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ExtractedClaim(BaseModel):
    """A factual claim extracted from evidence before independent verification."""

    id: str = Field(..., description="Unique claim identifier within session (e.g. 'C1', 'C2')")
    claim: str = Field(..., min_length=1, description="Factual claim statement")
    evidence_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of supporting evidence item IDs (e.g. ['E1', 'E2'])",
    )

    @field_validator("id", "claim")
    @classmethod
    def validate_non_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or blank")
        return v.strip()

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Claim must cite at least one evidence ID")
        cleaned = [e.strip() for e in v if e and e.strip()]
        if len(cleaned) != len(v):
            raise ValueError("Evidence IDs cannot contain empty or blank strings")
        # Ensure unique evidence IDs within the claim (Adjustment 3)
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Evidence IDs within a claim must be unique")
        return cleaned


class ExtractedClaimsResponse(BaseModel):
    """LLM response container for extracted claims."""

    claims: List[ExtractedClaim] = Field(
        default_factory=list,
        description="List of extracted claims",
    )

    @field_validator("claims")
    @classmethod
    def validate_unique_claim_ids(cls, v: List[ExtractedClaim]) -> List[ExtractedClaim]:
        seen_ids: Set[str] = set()
        for c in v:
            if c.id in seen_ids:
                raise ValueError(f"Duplicate claim ID detected: '{c.id}'. Claim IDs must be unique.")
            seen_ids.add(c.id)
        return v


class VerifiedClaim(BaseModel):
    """A research claim that has been independently verified against its evidence."""

    id: str = Field(..., description="Claim identifier (e.g. 'C1')")
    claim: str = Field(..., description="Factual claim statement")
    status: ClaimVerificationStatus = Field(..., description="Verification status")
    reason: str = Field(..., description="Concise justification from the verification model")
    evidence_ids: List[str] = Field(..., description="Associated evidence IDs considered during verification")
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Subset of evidence IDs confirmed by the verifier to directly support the claim",
    )


class VerifyClaimsRequest(BaseModel):
    """Request model for standalone claim verification."""

    question: str = Field(..., min_length=1, description="The research question context")
    claims: List[ExtractedClaim] = Field(..., description="List of candidate claims to verify")
    evidence: List["EvidenceItem"] = Field(..., description="Available evidence items")


class VerifyClaimsResponse(BaseModel):
    """Response model for standalone claim verification."""

    question: str = Field(..., description="The research question")
    verified_claims: List[VerifiedClaim] = Field(
        default_factory=list,
        description="List of independently verified claims",
    )

