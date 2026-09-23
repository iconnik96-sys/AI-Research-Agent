import json
import uuid
import httpx
import pytest
from pydantic import ValidationError

from app.schemas.claim import (
    ClaimVerificationStatus,
    ExtractedClaim,
    ExtractedClaimsResponse,
    VerifiedClaim,
)
from app.schemas.evidence import EvidenceItem
from app.services.claim.exceptions import (
    ClaimConfigError,
    ClaimError,
    ClaimExtractionError,
    ClaimResponseError,
    ClaimVerificationError,
)
from app.services.claim.extractor import LLMClaimExtractor
from app.services.claim.verifier import LLMClaimVerifier


# ---------------------------------------------------------------------------
# Schema Validation Tests (Adjustment 3 & Data Integrity)
# ---------------------------------------------------------------------------

def test_extracted_claim_valid():
    claim = ExtractedClaim(
        id="C1",
        claim="Nuclear fusion produces net energy.",
        evidence_ids=["E1", "E2"],
    )
    assert claim.id == "C1"
    assert claim.claim == "Nuclear fusion produces net energy."
    assert claim.evidence_ids == ["E1", "E2"]


def test_extracted_claim_blank_id_or_statement():
    with pytest.raises(ValidationError):
        ExtractedClaim(id="", claim="Valid claim", evidence_ids=["E1"])

    with pytest.raises(ValidationError):
        ExtractedClaim(id="C1", claim="   ", evidence_ids=["E1"])


def test_extracted_claim_empty_evidence_ids():
    with pytest.raises(ValidationError):
        ExtractedClaim(id="C1", claim="Valid claim", evidence_ids=[])


def test_extracted_claim_duplicate_evidence_ids_within_claim():
    """Adjustment 3: Evidence IDs within a claim must be unique."""
    with pytest.raises(ValidationError, match="Evidence IDs within a claim must be unique"):
        ExtractedClaim(id="C1", claim="Valid claim", evidence_ids=["E1", "E1"])


def test_extracted_claims_response_unique_claim_ids():
    """Adjustment 3: Claim IDs must be unique across the extracted response."""
    # Valid
    res = ExtractedClaimsResponse(
        claims=[
            ExtractedClaim(id="C1", claim="Claim 1", evidence_ids=["E1"]),
            ExtractedClaim(id="C2", claim="Claim 2", evidence_ids=["E2"]),
        ]
    )
    assert len(res.claims) == 2

    # Duplicate claim ID C1
    with pytest.raises(ValidationError, match="Duplicate claim ID detected: 'C1'"):
        ExtractedClaimsResponse(
            claims=[
                ExtractedClaim(id="C1", claim="Claim 1", evidence_ids=["E1"]),
                ExtractedClaim(id="C1", claim="Claim 2", evidence_ids=["E2"]),
            ]
        )


def test_verified_claim_status_values():
    assert ClaimVerificationStatus.SUPPORTED == "SUPPORTED"
    assert ClaimVerificationStatus.PARTIALLY_SUPPORTED == "PARTIALLY_SUPPORTED"
    assert ClaimVerificationStatus.UNSUPPORTED == "UNSUPPORTED"
    assert ClaimVerificationStatus.INSUFFICIENT_EVIDENCE == "INSUFFICIENT_EVIDENCE"


# ---------------------------------------------------------------------------
# LLMClaimExtractor Tests
# ---------------------------------------------------------------------------

def _make_sample_evidence():
    return [
        EvidenceItem(
            evidence_id="E1",
            chunk_id=str(uuid.uuid4()),
            document_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            url="https://example.com/source1",
            title="Fusion Overview",
            text="In 2022, scientists at LLNL achieved fusion ignition for the first time.",
            similarity=0.92,
        ),
        EvidenceItem(
            evidence_id="E2",
            chunk_id=str(uuid.uuid4()),
            document_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            url="https://example.com/source2",
            title="Fusion Challenges",
            text="Commercial power plants require high repetition rates and tritium breeding.",
            similarity=0.88,
        ),
    ]


@pytest.mark.anyio
async def test_extractor_empty_evidence():
    extractor = LLMClaimExtractor(api_key="mock-key")
    claims = await extractor.extract_claims(question="Fusion status", evidence=[])
    assert claims == []


@pytest.mark.anyio
async def test_extractor_missing_api_key():
    extractor = LLMClaimExtractor(api_key="")
    with pytest.raises(ClaimConfigError, match="LLM API key is not configured"):
        await extractor.extract_claims(question="Fusion status", evidence=_make_sample_evidence())


@pytest.mark.anyio
async def test_extractor_successful_extraction(monkeypatch):
    evidence = _make_sample_evidence()
    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "claims": [
                            {
                                "id": "C1",
                                "claim": "LLNL achieved fusion ignition in 2022.",
                                "evidence_ids": ["E1"],
                            },
                            {
                                "id": "C2",
                                "claim": "Commercial fusion requires high repetition rates.",
                                "evidence_ids": ["E2"],
                            },
                        ]
                    })
                }
            }
        ]
    }

    async def mock_post(*args, **kwargs):
        return httpx.Response(200, json=mock_payload, request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    extractor = LLMClaimExtractor(api_key="mock-key")
    claims = await extractor.extract_claims(question="Fusion status", evidence=evidence)

    assert len(claims) == 2
    assert claims[0].id == "C1"
    assert claims[0].claim == "LLNL achieved fusion ignition in 2022."
    assert claims[0].evidence_ids == ["E1"]
    assert claims[1].id == "C2"
    assert claims[1].evidence_ids == ["E2"]


@pytest.mark.anyio
async def test_extractor_exceeding_max_claims_raises_error(monkeypatch):
    """Adjustment 3: Count must not exceed CLAIM_MAX_COUNT, no silent truncation."""
    evidence = _make_sample_evidence()
    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "claims": [
                            {"id": f"C{i}", "claim": f"Claim {i}", "evidence_ids": ["E1"]}
                            for i in range(1, 5)
                        ]
                    })
                }
            }
        ]
    }

    async def mock_post(*args, **kwargs):
        return httpx.Response(200, json=mock_payload, request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # Set max_claims to 2
    extractor = LLMClaimExtractor(api_key="mock-key", max_claims=2)
    with pytest.raises(ClaimResponseError, match="exceeding maximum limit of 2"):
        await extractor.extract_claims(question="Fusion status", evidence=evidence)


@pytest.mark.anyio
async def test_extractor_references_unknown_evidence_id(monkeypatch):
    """Adjustment 3: All evidence IDs must exist in provided evidence."""
    evidence = _make_sample_evidence()  # Has E1 and E2
    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "claims": [
                            {"id": "C1", "claim": "Valid statement", "evidence_ids": ["E99"]}
                        ]
                    })
                }
            }
        ]
    }

    async def mock_post(*args, **kwargs):
        return httpx.Response(200, json=mock_payload, request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    extractor = LLMClaimExtractor(api_key="mock-key")
    with pytest.raises(ClaimResponseError, match="unknown evidence ID 'E99'"):
        await extractor.extract_claims(question="Fusion status", evidence=evidence)


@pytest.mark.anyio
async def test_extractor_malformed_json(monkeypatch):
    evidence = _make_sample_evidence()

    async def mock_post(*args, **kwargs):
        return httpx.Response(200, text="Not JSON", request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    extractor = LLMClaimExtractor(api_key="mock-key")
    with pytest.raises(ClaimResponseError, match="Invalid or unparseable JSON"):
        await extractor.extract_claims(question="Fusion status", evidence=evidence)


@pytest.mark.anyio
async def test_extractor_http_errors(monkeypatch):
    evidence = _make_sample_evidence()
    extractor = LLMClaimExtractor(api_key="mock-key")

    # 401 Unauthorized
    async def mock_401(*args, **kwargs):
        return httpx.Response(401, json={"error": "unauthorized"}, request=httpx.Request("POST", "http://test"))
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_401)
    with pytest.raises(ClaimConfigError):
        await extractor.extract_claims("question", evidence)

    # 429 Rate limit
    async def mock_429(*args, **kwargs):
        return httpx.Response(429, json={"error": "rate limit"}, request=httpx.Request("POST", "http://test"))
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_429)
    with pytest.raises(ClaimExtractionError):
        await extractor.extract_claims("question", evidence)

    # 500 Upstream error
    async def mock_500(*args, **kwargs):
        return httpx.Response(500, text="Internal server error", request=httpx.Request("POST", "http://test"))
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_500)
    with pytest.raises(ClaimExtractionError):
        await extractor.extract_claims("question", evidence)


# ---------------------------------------------------------------------------
# LLMClaimVerifier Tests (Adjustment 1: Verifier Independence)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_verifier_no_evidence_items():
    verifier = LLMClaimVerifier(api_key="mock-key")
    claim = ExtractedClaim(id="C1", claim="Fusion ignited", evidence_ids=["E1"])
    result = await verifier.verify_single_claim("Fusion", claim, evidence_items=[])
    assert result.status == ClaimVerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.supporting_evidence_ids == []


@pytest.mark.anyio
async def test_verifier_missing_api_key():
    verifier = LLMClaimVerifier(api_key="")
    claim = ExtractedClaim(id="C1", claim="Fusion ignited", evidence_ids=["E1"])
    evidence = _make_sample_evidence()
    with pytest.raises(ClaimConfigError):
        await verifier.verify_single_claim("Fusion", claim, evidence_items=evidence)


@pytest.mark.anyio
async def test_verifier_independent_supported_status(monkeypatch):
    """Adjustment 1: Verifier independently validates and filters supporting evidence IDs."""
    evidence = _make_sample_evidence()
    claim = ExtractedClaim(
        id="C1",
        claim="LLNL achieved fusion ignition in 2022.",
        evidence_ids=["E1", "E2"],
    )

    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "status": "SUPPORTED",
                        "reason": "E1 explicitly documents the 2022 LLNL ignition event. E2 discusses challenges and is not needed.",
                        "supporting_evidence_ids": ["E1"],
                    })
                }
            }
        ]
    }

    async def mock_post(*args, **kwargs):
        return httpx.Response(200, json=mock_payload, request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    verifier = LLMClaimVerifier(api_key="mock-key")
    verified = await verifier.verify_single_claim("Fusion progress", claim, evidence)

    assert verified.id == "C1"
    assert verified.status == ClaimVerificationStatus.SUPPORTED
    assert verified.reason.startswith("E1 explicitly documents")
    # Verifier independently identified ONLY E1 as supporting, filtering out E2
    assert verified.supporting_evidence_ids == ["E1"]
    assert verified.evidence_ids == ["E1", "E2"]


@pytest.mark.anyio
async def test_verifier_independent_unsupported_status(monkeypatch):
    """Adjustment 1: Verifier correctly marks claim UNSUPPORTED when evidence contradicts it."""
    evidence = _make_sample_evidence()
    claim = ExtractedClaim(
        id="C2",
        claim="Commercial fusion reactors are currently powering 50% of the world.",
        evidence_ids=["E2"],
    )

    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "status": "UNSUPPORTED",
                        "reason": "Evidence states commercial power plants still face unsolved technical challenges.",
                        "supporting_evidence_ids": [],
                    })
                }
            }
        ]
    }

    async def mock_post(*args, **kwargs):
        return httpx.Response(200, json=mock_payload, request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    verifier = LLMClaimVerifier(api_key="mock-key")
    verified = await verifier.verify_single_claim("Fusion status", claim, evidence)

    assert verified.status == ClaimVerificationStatus.UNSUPPORTED
    assert verified.supporting_evidence_ids == []


@pytest.mark.anyio
async def test_verifier_batch_bounded_concurrency(monkeypatch):
    evidence = _make_sample_evidence()
    claims = [
        ExtractedClaim(id=f"C{i}", claim=f"Claim {i}", evidence_ids=["E1"])
        for i in range(1, 6)
    ]

    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "status": "SUPPORTED",
                        "reason": "Supported by evidence.",
                        "supporting_evidence_ids": ["E1"],
                    })
                }
            }
        ]
    }

    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=mock_payload, request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    verifier = LLMClaimVerifier(api_key="mock-key", concurrency=2)
    verified = await verifier.verify_claims("Fusion", claims, evidence)

    assert len(verified) == 5
    assert call_count == 5
    for v in verified:
        assert v.status == ClaimVerificationStatus.SUPPORTED
        assert v.supporting_evidence_ids == ["E1"]
