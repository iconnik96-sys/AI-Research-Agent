import json
import logging
from typing import List, Optional, Set
import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.claim import ExtractedClaim, ExtractedClaimsResponse
from app.schemas.evidence import EvidenceItem
from app.services.claim.base import BaseClaimExtractor
from app.services.claim.exceptions import (
    ClaimConfigError,
    ClaimError,
    ClaimExtractionError,
    ClaimResponseError,
)

logger = logging.getLogger(__name__)

CLAIM_EXTRACTOR_SYSTEM_PROMPT = """You are an evidence analysis and factual claim extraction component.

Your job is to extract atomic, verifiable factual claims from the provided evidence items.

Rules:
1. Extract ONLY factual assertions directly mentioned or supported by the evidence items.
2. Return at most {max_claims} claims.
3. Each claim must have:
   - "id": A unique identifier such as "C1", "C2", "C3".
   - "claim": A clear, concise, self-contained factual statement.
   - "evidence_ids": A list of candidate evidence identifiers (e.g. ["E1"], ["E1", "E2"]) that provide source material for the claim.
4. Each claim must reference at least one evidence ID.
5. All referenced evidence IDs must exist in the provided evidence list.
6. Return ONLY valid JSON matching the schema.

Schema:
{{
  "claims": [
    {{
      "id": "C1",
      "claim": "Factual claim statement",
      "evidence_ids": ["E1"]
    }}
  ]
}}"""


class LLMClaimExtractor(BaseClaimExtractor):
    """Extract candidate factual claims from retrieved research evidence using an LLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_claims: Optional[int] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.LLM_API_KEY
        self.base_url = base_url if base_url is not None else settings.LLM_BASE_URL
        self.model = model if model is not None else settings.LLM_MODEL
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.CLAIM_EXTRACTION_TIMEOUT_SECONDS
        )
        self.max_claims = (
            max_claims
            if max_claims is not None
            else settings.CLAIM_MAX_COUNT
        )

    async def extract_claims(
        self,
        question: str,
        evidence: List[EvidenceItem],
    ) -> List[ExtractedClaim]:
        """Extract atomic factual claims and map them to evidence IDs."""
        if not evidence:
            logger.info("No evidence provided for claim extraction; returning empty list.")
            return []

        if not self.api_key or not self.api_key.strip():
            raise ClaimConfigError(
                "LLM API key is not configured for claim extraction. Please set LLM_API_KEY."
            )

        # Build evidence text block
        evidence_lines = []
        valid_evidence_ids: Set[str] = set()
        for item in evidence:
            valid_evidence_ids.add(item.evidence_id)
            evidence_lines.append(
                f"[{item.evidence_id}] Title: {item.title}\nContent: {item.text}"
            )
        evidence_block = "\n\n".join(evidence_lines)

        user_content = (
            f"Research Question: {question}\n\n"
            f"Available Evidence Items:\n{evidence_block}\n\n"
            f"Extract atomic factual claims with their candidate evidence IDs."
        )

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        system_content = CLAIM_EXTRACTOR_SYSTEM_PROMPT.format(max_claims=self.max_claims)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            logger.error("Claim extraction timed out after %.1fs: %s", self.timeout_seconds, exc)
            raise ClaimExtractionError(
                f"Claim extraction request timed out after {self.timeout_seconds}s."
            ) from exc
        except httpx.NetworkError as exc:
            logger.error("Network error during claim extraction: %s", exc)
            raise ClaimExtractionError(
                f"Network transport error during claim extraction: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error during claim extraction: %s", exc)
            raise ClaimError(f"Unexpected error calling claim extractor API: {exc}") from exc

        if response.status_code == 401:
            logger.error("Claim extraction authentication failed (HTTP 401).")
            raise ClaimConfigError("LLM API authentication failed. Please check LLM_API_KEY.")
        if response.status_code == 429:
            logger.error("Claim extraction rate limit exceeded (HTTP 429).")
            raise ClaimExtractionError("LLM API rate limit exceeded. Please try again later.")
        if response.status_code >= 500:
            logger.error(
                "Claim extraction upstream server error HTTP %d: %s",
                response.status_code,
                response.text,
            )
            raise ClaimExtractionError(
                f"Upstream LLM provider error HTTP {response.status_code}."
            )
        if response.status_code != 200:
            logger.error(
                "Claim extraction unexpected HTTP %d: %s",
                response.status_code,
                response.text,
            )
            raise ClaimResponseError(
                f"Claim extractor API returned unexpected status code {response.status_code}."
            )

        try:
            body = response.json()
            raw_content = body["choices"][0]["message"]["content"]
            parsed_json = json.loads(raw_content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse claim extractor JSON response: %s", exc)
            raise ClaimResponseError(
                f"Invalid or unparseable JSON returned by claim extractor: {exc}"
            ) from exc

        try:
            claims_response = ExtractedClaimsResponse.model_validate(parsed_json)
        except ValidationError as exc:
            logger.error("Claim extraction response failed schema validation: %s", exc)
            raise ClaimResponseError(
                f"Claim extractor response does not conform to schema: {exc}"
            ) from exc

        claims = claims_response.claims

        # Validation Rule 1: Claim count must not exceed CLAIM_MAX_COUNT (no silent truncation)
        if len(claims) > self.max_claims:
            logger.warning(
                "Claim extractor generated %d claims, exceeding limit of %d",
                len(claims),
                self.max_claims,
            )
            raise ClaimResponseError(
                f"Claim extractor returned {len(claims)} claims, exceeding maximum limit of {self.max_claims}."
            )

        # Validation Rule 2: All evidence IDs must exist
        for claim in claims:
            for eid in claim.evidence_ids:
                if eid not in valid_evidence_ids:
                    logger.warning(
                        "Claim '%s' references unknown evidence ID '%s' (valid: %s)",
                        claim.id,
                        eid,
                        valid_evidence_ids,
                    )
                    raise ClaimResponseError(
                        f"Claim '{claim.id}' references unknown evidence ID '{eid}'."
                    )

        return claims
