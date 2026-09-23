import asyncio
import json
import logging
from typing import Dict, List, Optional, Set
import httpx
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.schemas.claim import ClaimVerificationStatus, ExtractedClaim, VerifiedClaim
from app.schemas.evidence import EvidenceItem
from app.services.claim.base import BaseClaimVerifier
from app.services.claim.exceptions import (
    ClaimConfigError,
    ClaimError,
    ClaimVerificationError,
)

logger = logging.getLogger(__name__)

CLAIM_VERIFIER_SYSTEM_PROMPT = """You are an independent factual claim verification component.

Your job is to independently evaluate whether a factual claim is supported by the provided evidence items.
Do NOT assume the candidate evidence supports the claim just because it was cited. Independently analyze each evidence excerpt.

Evaluation Statuses:
- "SUPPORTED": The evidence explicitly and directly supports the entire claim statement.
- "PARTIALLY_SUPPORTED": The evidence supports some aspects of the claim, but other aspects are missing, speculative, or unconfirmed.
- "UNSUPPORTED": The evidence directly contradicts the claim, or the claim is false according to the evidence.
- "INSUFFICIENT_EVIDENCE": The provided evidence does not contain enough information to confirm or refute the claim.

Return ONLY valid JSON matching the schema:
{{
  "status": "SUPPORTED",
  "reason": "Concise justification explaining why the status was assigned",
  "supporting_evidence_ids": ["E1"]
}}

Rules for "supporting_evidence_ids":
1. Include ONLY evidence IDs from the candidate evidence that genuinely and directly substantiate the claim.
2. If status is "UNSUPPORTED" or "INSUFFICIENT_EVIDENCE", "supporting_evidence_ids" should be an empty list [].
3. For "PARTIALLY_SUPPORTED", include only those evidence IDs that support the confirmed parts.
"""


class _SingleClaimVerificationResponse(BaseModel):
    """Internal validation model for verifier LLM output."""

    status: ClaimVerificationStatus = Field(..., description="Verification status")
    reason: str = Field(..., min_length=1, description="Concise justification")
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Evidence IDs confirmed to support the claim",
    )


class LLMClaimVerifier(BaseClaimVerifier):
    """Independently verifies extracted factual claims against evidence with bounded concurrency."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        concurrency: Optional[int] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.LLM_API_KEY
        self.base_url = base_url if base_url is not None else settings.LLM_BASE_URL
        self.model = model if model is not None else settings.LLM_MODEL
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.CLAIM_VERIFICATION_TIMEOUT_SECONDS
        )
        self.concurrency = (
            concurrency
            if concurrency is not None
            else settings.CLAIM_VERIFICATION_CONCURRENCY
        )

    async def verify_single_claim(
        self,
        question: str,
        claim: ExtractedClaim,
        evidence_items: List[EvidenceItem],
    ) -> VerifiedClaim:
        """Independently verify a single claim against its candidate evidence items."""
        if not evidence_items:
            return VerifiedClaim(
                id=claim.id,
                claim=claim.claim,
                status=ClaimVerificationStatus.INSUFFICIENT_EVIDENCE,
                reason="No candidate evidence items provided for verification.",
                evidence_ids=claim.evidence_ids,
                supporting_evidence_ids=[],
            )

        if not self.api_key or not self.api_key.strip():
            raise ClaimConfigError(
                "LLM API key is not configured for claim verification. Please set LLM_API_KEY."
            )

        candidate_ids: Set[str] = {item.evidence_id for item in evidence_items}
        evidence_lines = [
            f"[{item.evidence_id}] Title: {item.title}\nContent: {item.text}"
            for item in evidence_items
        ]
        evidence_block = "\n\n".join(evidence_lines)

        user_content = (
            f"Research Question: {question}\n\n"
            f"Claim Identifier: {claim.id}\n"
            f"Claim Statement: \"{claim.claim}\"\n"
            f"Candidate Evidence IDs: {', '.join(claim.evidence_ids)}\n\n"
            f"Candidate Evidence Items:\n{evidence_block}\n\n"
            f"Independently evaluate whether the claim is supported by the evidence above."
        )

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": CLAIM_VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            logger.error("Claim verification timed out for claim %s: %s", claim.id, exc)
            raise ClaimVerificationError(
                f"Claim verification timed out for claim {claim.id} after {self.timeout_seconds}s."
            ) from exc
        except httpx.NetworkError as exc:
            logger.error("Network error during claim verification for claim %s: %s", claim.id, exc)
            raise ClaimVerificationError(
                f"Network transport error during claim verification: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error during claim verification for claim %s: %s", claim.id, exc)
            raise ClaimError(f"Unexpected error calling claim verifier API: {exc}") from exc

        if response.status_code == 401:
            logger.error("Claim verification authentication failed (HTTP 401).")
            raise ClaimConfigError("LLM API authentication failed. Please check LLM_API_KEY.")
        if response.status_code != 200:
            logger.error(
                "Claim verification unexpected HTTP %d: %s",
                response.status_code,
                response.text,
            )
            raise ClaimVerificationError(
                f"Claim verifier API returned unexpected status code {response.status_code}."
            )

        try:
            body = response.json()
            raw_content = body["choices"][0]["message"]["content"]
            parsed_json = json.loads(raw_content)
            parsed_res = _SingleClaimVerificationResponse.model_validate(parsed_json)
        except (KeyError, IndexError, json.JSONDecodeError, ValidationError) as exc:
            logger.error("Failed to parse verifier response for claim %s: %s", claim.id, exc)
            raise ClaimVerificationError(
                f"Invalid or unparseable response from claim verifier: {exc}"
            ) from exc

        # Filter confirmed supporting evidence to only those that were candidate evidence items
        valid_supporting = [
            eid.strip()
            for eid in parsed_res.supporting_evidence_ids
            if eid.strip() in candidate_ids
        ]

        return VerifiedClaim(
            id=claim.id,
            claim=claim.claim,
            status=parsed_res.status,
            reason=parsed_res.reason,
            evidence_ids=claim.evidence_ids,
            supporting_evidence_ids=valid_supporting,
        )

    async def verify_claims(
        self,
        question: str,
        claims: List[ExtractedClaim],
        evidence: List[EvidenceItem],
    ) -> List[VerifiedClaim]:
        """Verify multiple claims concurrently bounded by semaphore."""
        if not claims:
            return []

        evidence_map: Dict[str, EvidenceItem] = {e.evidence_id: e for e in evidence}
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _bounded_verify(claim: ExtractedClaim) -> VerifiedClaim:
            async with semaphore:
                candidate_items = [
                    evidence_map[eid]
                    for eid in claim.evidence_ids
                    if eid in evidence_map
                ]
                try:
                    return await self.verify_single_claim(question, claim, candidate_items)
                except Exception as exc:
                    logger.warning(
                        "Verification failed for claim %s: %s. Defaulting to INSUFFICIENT_EVIDENCE.",
                        claim.id,
                        exc,
                    )
                    return VerifiedClaim(
                        id=claim.id,
                        claim=claim.claim,
                        status=ClaimVerificationStatus.INSUFFICIENT_EVIDENCE,
                        reason=f"Verification failed due to error: {exc}",
                        evidence_ids=claim.evidence_ids,
                        supporting_evidence_ids=[],
                    )

        tasks = [_bounded_verify(claim) for claim in claims]
        verified = await asyncio.gather(*tasks)
        return list(verified)
