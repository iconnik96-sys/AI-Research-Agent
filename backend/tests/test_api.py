import uuid
from typing import Dict, List, Optional
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
from app.services.persistence import (
    BaseResearchRepository,
    DatabaseConfigError,
    DatabaseConnectionError,
    get_research_repository,
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


class MockResearchRepository(BaseResearchRepository):
    """Mock repository for API integration tests."""

    def __init__(self, create_error: Optional[Exception] = None):
        self.create_error = create_error
        self.created_sessions: List[str] = []
        self.sources_saved: Dict[str, List[SourceItem]] = {}
        self.documents_saved: Dict[str, List[Document]] = {}
        self.completed_sessions: Dict[str, ResearchReport] = {}
        self.failed_sessions: Dict[str, str] = {}

    async def create_session(self, question: str) -> str:
        if self.create_error:
            raise self.create_error
        session_id = str(uuid.uuid4())
        self.created_sessions.append(session_id)
        return session_id

    async def save_sources(self, session_id: str, sources: List[SourceItem]) -> Dict[str, str]:
        self.sources_saved[session_id] = sources
        return {s.url: str(uuid.uuid4()) for s in sources}

    async def save_documents(
        self,
        session_id: str,
        documents: List[Document],
        source_id_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.documents_saved[session_id] = documents

    async def complete_session(self, session_id: str, report: ResearchReport) -> None:
        self.completed_sessions[session_id] = report

    async def fail_session(self, session_id: str, error_message: str) -> None:
        self.failed_sessions[session_id] = error_message


@pytest.fixture
def client():
    app.dependency_overrides.clear()
    app.dependency_overrides[get_search_provider] = lambda: MockSearchProvider()
    app.dependency_overrides[get_webpage_extractor] = lambda: MockWebpageExtractor()
    app.dependency_overrides[get_llm_provider] = lambda: MockLLMProvider()
    app.dependency_overrides[get_research_repository] = lambda: MockResearchRepository()
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
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["session_id"] is not None
    assert uuid.UUID(data["session_id"])  # verify valid UUID
    assert data["question"] == payload["question"]
    assert data["status"] == "completed"

    # Verify repository calls
    assert data["session_id"] in mock_repo.created_sessions
    assert data["session_id"] in mock_repo.completed_sessions
    assert len(mock_repo.sources_saved[data["session_id"]]) == 1

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
    mock_extractor = MockWebpageExtractor(documents=[])
    mock_llm = MockLLMProvider(
        report=ResearchReport(
            title="Insufficient Evidence",
            summary="No sources could be extracted.",
            sections=[],
            sources=[],
        )
    )
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["session_id"] is not None
    assert data["status"] == "completed"
    assert len(data["sources"]) == 1
    assert data["documents"] == []
    assert data["report"]["summary"] == "No sources could be extracted."
    assert data["session_id"] in mock_repo.completed_sessions


def test_research_endpoint_database_missing_config(client):
    mock_repo = MockResearchRepository(
        create_error=DatabaseConfigError("DATABASE_URL is not configured.")
    )
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 503
    assert "DATABASE_URL is not configured" in response.json()["detail"]


def test_research_endpoint_database_connection_failure(client):
    mock_repo = MockResearchRepository(
        create_error=DatabaseConnectionError("Database error creating session: connection failure")
    )
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 500
    assert "Database persistence error" in response.json()["detail"]


def test_research_endpoint_missing_search_key(client):
    mock_search = MockSearchProvider(
        error=SearchConfigError("Tavily API key is not configured. Please set TAVILY_API_KEY.")
    )
    mock_repo = MockResearchRepository()
    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 503
    assert "Tavily API key is not configured" in response.json()["detail"]

    # Verify session marked as failed
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_search_timeout(client):
    mock_search = MockSearchProvider(
        error=SearchTimeoutError("Search request timed out after 10.0s")
    )
    mock_repo = MockResearchRepository()
    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 504
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_search_provider_error(client):
    mock_search = MockSearchProvider(
        error=SearchProviderError("Upstream API error HTTP 500")
    )
    mock_repo = MockResearchRepository()
    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 502
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_llm_missing_api_key(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_llm = MockLLMProvider(
        error=LLMConfigError("LLM API key is not configured. Please set LLM_API_KEY.")
    )
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 503
    assert "LLM API key is not configured" in response.json()["detail"]
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_llm_timeout(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_llm = MockLLMProvider(
        error=LLMTimeoutError("LLM request timed out after 30.0s")
    )
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 504
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_llm_provider_error(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_llm = MockLLMProvider(
        error=LLMProviderError("Upstream LLM API error 500")
    )
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 502
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_llm_response_error(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_llm = MockLLMProvider(
        error=LLMResponseError("Invalid citation ID 'S99'")
    )
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 502
    assert "Invalid citation ID" in response.json()["detail"]
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_missing_question(client):
    response = client.post("/api/research", json={})
    assert response.status_code == 422


def test_research_endpoint_empty_question(client):
    response = client.post("/api/research", json={"question": ""})
    assert response.status_code == 422
