from typing import List, Optional
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.document import Document
from app.schemas.report import ResearchReport, ResearchSection, SourceReference
from app.schemas.research import SourceItem
from app.services.extraction import get_webpage_extractor
from app.services.extraction.base import BaseExtractor
from app.services.llm import (
    BaseLLMProvider,
    LLMConfigError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
    get_llm_provider,
)
from app.services.search.base import (
    BaseSearchProvider,
    SearchConfigError,
    SearchProviderError,
    SearchTimeoutError,
)
from app.services.search.factory import get_search_provider


class MockSearchProvider(BaseSearchProvider):
    """Mock search provider for API integration tests."""

    def __init__(self, sources: Optional[List[SourceItem]] = None, error: Optional[Exception] = None):
        self.sources = sources or [
            SourceItem(
                title="Mock Source 1",
                url="https://example.com/source1",
                content="Snippet 1 content",
                score=0.9,
            )
        ]
        self.error = error

    async def search(self, query: str, max_results: int = 5) -> List[SourceItem]:
        if self.error:
            raise self.error
        return self.sources


class MockWebpageExtractor(BaseExtractor):
    """Mock extractor for API integration tests."""

    def __init__(self, documents: Optional[List[Document]] = None):
        self.documents = documents if documents is not None else [
            Document(
                url="https://example.com/source1",
                title="Mock Source 1",
                text="Cleaned readable text for Mock Source 1 extracted cleanly.",
                score=0.9,
                char_count=58,
            )
        ]

    async def extract(
        self,
        url: str,
        title: Optional[str] = None,
        score: Optional[float] = None,
    ) -> Document:
        return self.documents[0]

    async def extract_many(self, sources: List[SourceItem]) -> List[Document]:
        return self.documents


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for API integration tests."""

    def __init__(self, report: Optional[ResearchReport] = None, error: Optional[Exception] = None):
        self.report = report or ResearchReport(
            title="Mock Research Report",
            summary="Mock executive summary of research findings.",
            sections=[
                ResearchSection(
                    heading="Key Findings",
                    content="Detailed findings backed by evidence.",
                    citations=["S1"],
                )
            ],
            sources=[
                SourceReference(
                    id="S1",
                    title="Mock Source 1",
                    url="https://example.com/source1",
                )
            ],
        )
        self.error = error

    async def generate_report(self, question: str, documents: List[Document]) -> ResearchReport:
        if self.error:
            raise self.error
        return self.report


@pytest.fixture
def client():
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_endpoint_success(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_llm = MockLLMProvider()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["question"] == payload["question"]
    assert data["status"] == "completed"

    # Verify sources, documents, and report are all present
    assert len(data["sources"]) == 1
    assert data["sources"][0]["title"] == "Mock Source 1"

    assert len(data["documents"]) == 1
    assert data["documents"][0]["url"] == "https://example.com/source1"

    assert data["report"] is not None
    assert data["report"]["title"] == "Mock Research Report"
    assert len(data["report"]["sections"]) == 1
    assert data["report"]["sections"][0]["citations"] == ["S1"]
    assert len(data["report"]["sources"]) == 1


def test_research_endpoint_extraction_failure_tolerance(client):
    mock_search = MockSearchProvider()
    # Extractor returns empty list because all page fetches failed
    mock_extractor = MockWebpageExtractor(documents=[])
    # Empty documents returns insufficient evidence report deterministically
    mock_llm = MockLLMProvider(
        report=ResearchReport(
            title="Insufficient Evidence",
            summary="No sources could be extracted.",
            sections=[],
            sources=[],
        )
    )

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "completed"
    assert len(data["sources"]) == 1
    assert data["documents"] == []
    assert data["report"]["summary"] == "No sources could be extracted."


def test_research_endpoint_missing_search_key(client):
    mock_provider = MockSearchProvider(
        error=SearchConfigError("Tavily API key is not configured. Please set TAVILY_API_KEY.")
    )
    app.dependency_overrides[get_search_provider] = lambda: mock_provider

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 503
    assert "Tavily API key is not configured" in response.json()["detail"]


def test_research_endpoint_search_timeout(client):
    mock_provider = MockSearchProvider(
        error=SearchTimeoutError("Search request timed out after 10.0s")
    )
    app.dependency_overrides[get_search_provider] = lambda: mock_provider

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 504


def test_research_endpoint_search_provider_error(client):
    mock_provider = MockSearchProvider(
        error=SearchProviderError("Upstream API error HTTP 500")
    )
    app.dependency_overrides[get_search_provider] = lambda: mock_provider

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 502


def test_research_endpoint_llm_missing_api_key(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_llm = MockLLMProvider(
        error=LLMConfigError("LLM API key is not configured. Please set LLM_API_KEY.")
    )

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 503
    assert "LLM API key is not configured" in response.json()["detail"]


def test_research_endpoint_llm_timeout(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_llm = MockLLMProvider(
        error=LLMTimeoutError("LLM request timed out after 30.0s")
    )

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 504


def test_research_endpoint_llm_provider_error(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_llm = MockLLMProvider(
        error=LLMProviderError("Upstream LLM API error 500")
    )

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 502


def test_research_endpoint_llm_response_error(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_llm = MockLLMProvider(
        error=LLMResponseError("Invalid citation ID 'S99'")
    )

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 502
    assert "Invalid citation ID" in response.json()["detail"]


def test_research_endpoint_missing_question(client):
    response = client.post("/api/research", json={})
    assert response.status_code == 422


def test_research_endpoint_empty_question(client):
    response = client.post("/api/research", json={"question": ""})
    assert response.status_code == 422
