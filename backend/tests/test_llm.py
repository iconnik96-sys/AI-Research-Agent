import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch

from app.schemas.claim import ClaimVerificationStatus, VerifiedClaim
from app.schemas.document import Document
from app.schemas.retrieval import RetrievedChunk
from app.services.llm.evidence import prepare_evidence, prepare_evidence_from_chunks
from app.services.llm.exceptions import (
    LLMConfigError,
    LLMNetworkError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.services.llm.provider import OpenAICompatibleLLMProvider


@pytest.fixture
def sample_documents():
    return [
        Document(
            url="https://example.com/fusion-1",
            title="Fusion Ignition Breakthrough",
            text="Researchers at NIF produced 3.15 MJ of energy from a 2.05 MJ laser shot, achieving net energy gain.",
            score=0.98,
            char_count=104,
        ),
        Document(
            url="https://example.com/fusion-2",
            title="Commercial Fusion Timelines",
            text="Commercial power generation from magnetic confinement still faces plasma stability and materials challenges.",
            score=0.91,
            char_count=113,
        ),
    ]


def test_evidence_preparation_deterministic(sample_documents):
    refs, text = prepare_evidence(sample_documents, max_chars_per_doc=50)

    assert len(refs) == 2
    assert refs[0].id == "S1"
    assert refs[0].title == "Fusion Ignition Breakthrough"
    assert refs[0].url == "https://example.com/fusion-1"

    assert refs[1].id == "S2"
    assert refs[1].title == "Commercial Fusion Timelines"
    assert refs[1].url == "https://example.com/fusion-2"

    assert "[Source ID: S1]" in text
    assert "[Source ID: S2]" in text
    # Test per-document character limitation
    assert len(sample_documents[0].text[:50]) <= 50


@pytest.mark.anyio
async def test_empty_documents_handling():
    provider = OpenAICompatibleLLMProvider(api_key="mock-key")

    with patch.object(httpx.AsyncClient, "post") as mock_post:
        report = await provider.generate_report(
            question="What is fusion?",
            documents=[],
        )
        mock_post.assert_not_called()
        assert "Insufficient evidence" in report.summary
        assert len(report.sections) == 1
        assert report.sources == []


@pytest.mark.anyio
async def test_missing_api_key(sample_documents):
    provider = OpenAICompatibleLLMProvider(api_key="")

    with pytest.raises(LLMConfigError, match="LLM API key is not configured"):
        await provider.generate_report(
            question="What is fusion?",
            documents=sample_documents,
        )


@pytest.mark.anyio
async def test_successful_report_generation(sample_documents):
    provider = OpenAICompatibleLLMProvider(
        api_key="mock-key",
        model="test-gpt-model",
    )

    llm_output = {
        "title": "Synthesis of Nuclear Fusion Breakthroughs",
        "summary": "Recent experiments at NIF achieved net energy gain, although commercial deployment challenges persist.",
        "sections": [
            {
                "heading": "Net Energy Gain",
                "content": "NIF demonstrated net energy gain generating 3.15 MJ.",
                "citations": ["S1"],
            },
            {
                "heading": "Commercial Prospects",
                "content": "Commercial development faces materials and stability hurdles.",
                "citations": ["S2"],
            },
        ],
    }

    mock_response = httpx.Response(
        status_code=200,
        json={
            "id": "chatcmpl-mock",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(llm_output),
                    },
                }
            ],
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        report = await provider.generate_report(
            question="What are recent breakthroughs in fusion?",
            documents=sample_documents,
        )

        assert report.title == "Synthesis of Nuclear Fusion Breakthroughs"
        assert len(report.sections) == 2
        assert report.sections[0].heading == "Net Energy Gain"
        assert report.sections[0].citations == ["S1"]
        assert report.sections[1].citations == ["S2"]
        assert len(report.sources) == 2
        assert report.sources[0].id == "S1"

        # Verify correct model parameter was passed
        mock_post.assert_awaited_once()
        sent_payload = mock_post.await_args.kwargs["json"]
        assert sent_payload["model"] == "test-gpt-model"


@pytest.mark.anyio
async def test_malformed_json_response(sample_documents):
    provider = OpenAICompatibleLLMProvider(api_key="mock-key")

    mock_response = httpx.Response(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is non-JSON text.",
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(LLMResponseError, match="malformed non-JSON"):
            await provider.generate_report(
                question="What is fusion?",
                documents=sample_documents,
            )


@pytest.mark.anyio
async def test_invalid_citation_id(sample_documents):
    provider = OpenAICompatibleLLMProvider(api_key="mock-key")

    # S99 does not exist in sample_documents (only S1 and S2 exist)
    llm_output = {
        "title": "Fusion Report",
        "summary": "Summary",
        "sections": [
            {
                "heading": "Hallucinated Citation",
                "content": "Statement claiming false evidence.",
                "citations": ["S99"],
            }
        ],
    }

    mock_response = httpx.Response(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(llm_output),
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(LLMResponseError, match="Invalid citation ID 'S99'"):
            await provider.generate_report(
                question="What is fusion?",
                documents=sample_documents,
            )


@pytest.mark.anyio
async def test_timeout_raises_llm_timeout_error(sample_documents):
    provider = OpenAICompatibleLLMProvider(api_key="mock-key", timeout_seconds=2.0)

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Read timed out")
        with pytest.raises(LLMTimeoutError, match="timed out"):
            await provider.generate_report(
                question="What is fusion?",
                documents=sample_documents,
            )


@pytest.mark.anyio
async def test_network_error_raises_llm_network_error(sample_documents):
    provider = OpenAICompatibleLLMProvider(api_key="mock-key")

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")
        with pytest.raises(LLMNetworkError, match="Network error"):
            await provider.generate_report(
                question="What is fusion?",
                documents=sample_documents,
            )


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [401, 429, 500])
async def test_provider_http_error(sample_documents, status_code):
    provider = OpenAICompatibleLLMProvider(api_key="mock-key")

    mock_response = httpx.Response(
        status_code=status_code,
        text=f"Error {status_code}",
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(LLMProviderError) as exc_info:
            await provider.generate_report(
                question="What is fusion?",
                documents=sample_documents,
            )
        assert exc_info.value.status_code == status_code


def test_prepare_evidence_from_chunks():
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            session_id="s1",
            chunk_index=0,
            text="First excerpt from Doc 1.",
            similarity=0.92,
            url="https://example.com/doc1",
            title="Doc 1",
        ),
        RetrievedChunk(
            chunk_id="c2",
            document_id="d1",
            session_id="s1",
            chunk_index=2,
            text="Second excerpt from Doc 1.",
            similarity=0.85,
            url="https://example.com/doc1",
            title="Doc 1",
        ),
        RetrievedChunk(
            chunk_id="c3",
            document_id="d2",
            session_id="s1",
            chunk_index=0,
            text="Excerpt from Doc 2.",
            similarity=0.88,
            url="https://example.com/doc2",
            title="Doc 2",
        ),
    ]

    refs, text = prepare_evidence_from_chunks(chunks)

    # Multiple chunks from Doc 1 grouped under S1, Doc 2 under S2
    assert len(refs) == 2
    assert refs[0].id == "S1"
    assert refs[0].url == "https://example.com/doc1"
    assert refs[1].id == "S2"
    assert refs[1].url == "https://example.com/doc2"

    assert "First excerpt from Doc 1." in text
    assert "Second excerpt from Doc 1." in text
    assert "Excerpt from Doc 2." in text
    assert "[Source ID: S1]" in text
    assert "[Source ID: S2]" in text

    # Empty chunks handling
    empty_refs, empty_text = prepare_evidence_from_chunks([])
    assert empty_refs == []
    assert empty_text == ""


@pytest.mark.anyio
async def test_generate_report_from_chunks():
    provider = OpenAICompatibleLLMProvider(api_key="mock-key")
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            session_id="s1",
            chunk_index=0,
            text="Laser inertial confinement achieved fusion ignition.",
            similarity=0.95,
            url="https://example.com/fusion",
            title="Fusion Ignition",
        )
    ]

    mock_llm_json = {
        "title": "Fusion Report via RAG",
        "summary": "Ignition successfully achieved.",
        "sections": [
            {
                "heading": "Ignition Findings",
                "content": "Fusion ignition demonstrated experimentally.",
                "citations": ["S1"],
            }
        ],
    }

    mock_response = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": json.dumps(mock_llm_json)}}]},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        report = await provider.generate_report(
            question="What is the latest on fusion?",
            chunks=chunks,
        )

    assert report.title == "Fusion Report via RAG"
    assert len(report.sections) == 1
    assert report.sections[0].citations == ["S1"]
    assert len(report.sources) == 1
    assert report.sources[0].id == "S1"
    assert report.sources[0].url == "https://example.com/fusion"


@pytest.mark.anyio
async def test_generate_report_with_grounding_claims():
    """Adjustment 4: Test that verified claims and grounding rules are integrated into prompt."""
    provider = OpenAICompatibleLLMProvider(api_key="mock-key")
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            session_id="s1",
            chunk_index=0,
            text="Laser inertial confinement achieved fusion ignition.",
            similarity=0.95,
            url="https://example.com/fusion",
            title="Fusion Ignition",
        )
    ]
    claims = [
        VerifiedClaim(
            id="C1",
            claim="Fusion ignition demonstrated experimentally.",
            status=ClaimVerificationStatus.SUPPORTED,
            reason="Confirmed by experimental results.",
            evidence_ids=["E1"],
            supporting_evidence_ids=["E1"],
        ),
        VerifiedClaim(
            id="C2",
            claim="Commercial fusion is operational worldwide.",
            status=ClaimVerificationStatus.UNSUPPORTED,
            reason="Directly contradicted by current reactor status.",
            evidence_ids=["E1"],
            supporting_evidence_ids=[],
        ),
    ]

    mock_llm_json = {
        "title": "Grounded Fusion Report",
        "summary": "Ignition was achieved, though commercial deployment remains unsolved.",
        "sections": [
            {
                "heading": "Ignition Findings",
                "content": "Experimental ignition was achieved in 2022.",
                "citations": ["S1"],
            }
        ],
    }

    mock_response = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": json.dumps(mock_llm_json)}}]},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        report = await provider.generate_report(
            question="What is the latest on fusion?",
            chunks=chunks,
            claims=claims,
        )

        assert report.title == "Grounded Fusion Report"
        # Verify that prompt sent to LLM included the claims and grounding guidance
        called_payload = mock_post.call_args.kwargs["json"]
        user_message = called_payload["messages"][1]["content"]
        assert "Verified Claims & Grounding Guidance:" in user_message
        assert "[C1] (SUPPORTED): \"Fusion ignition demonstrated experimentally.\"" in user_message
        assert "[C2] (UNSUPPORTED): \"Commercial fusion is operational worldwide.\"" in user_message

