import uuid
from typing import Dict, List, Optional
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.chunk import DocumentChunk
from app.schemas.document import Document
from app.schemas.report import ResearchReport, ResearchSection, SourceReference
from app.schemas.research import SourceItem
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievedChunk
from app.services.embedding import (
    BaseEmbeddingProvider,
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
    get_embedding_provider,
)
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
    DatabaseError,
    get_research_repository,
)
from app.services.retrieval import (
    BaseRetriever,
    RetrievalConfigError,
    RetrievalDatabaseError,
    RetrievalEmbeddingError,
    RetrievalError,
    get_retriever,
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
        self.last_generate_call: dict = {}

    async def generate_report(
        self,
        question: str,
        documents: Optional[List[Document]] = None,
        chunks: Optional[List[RetrievedChunk]] = None,
    ) -> ResearchReport:
        if self.error:
            raise self.error
        self.last_generate_call = {
            "question": question,
            "documents": documents,
            "chunks": chunks,
        }
        return self.report


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider for API integration tests."""

    def __init__(self, dimensions: int = 1536, error: Optional[Exception] = None):
        self._dimensions = dimensions
        self.error = error
        self.embedded_texts: List[List[str]] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.error:
            raise self.error
        self.embedded_texts.append(texts)
        return [[0.1] * self._dimensions for _ in texts]

    async def embed_text(self, text: str) -> List[float]:
        if self.error:
            raise self.error
        return [0.1] * self._dimensions


class MockRetriever(BaseRetriever):
    """Mock semantic retriever for API integration tests."""

    def __init__(
        self,
        results: Optional[List[RetrievedChunk]] = None,
        error: Optional[Exception] = None,
    ):
        self.results = results
        self.error = error
        self.last_retrieve_call: dict = {}

    async def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        if self.error:
            raise self.error
        self.last_retrieve_call = {
            "query": query,
            "session_id": session_id,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }
        if self.results is not None:
            return self.results
        return [
            RetrievedChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=str(uuid.uuid4()),
                session_id=session_id or str(uuid.uuid4()),
                chunk_index=0,
                text="Mock retrieved chunk text for RAG synthesis.",
                similarity=0.92,
                url="https://example.com/source1",
                title="Mock Source 1",
            )
        ]


class MockResearchRepository(BaseResearchRepository):
    """Mock repository for API integration tests."""

    def __init__(
        self,
        create_error: Optional[Exception] = None,
        save_chunks_error: Optional[Exception] = None,
        search_similar_chunks_error: Optional[Exception] = None,
    ):
        self.create_error = create_error
        self.save_chunks_error = save_chunks_error
        self.search_similar_chunks_error = search_similar_chunks_error
        self.created_sessions: List[str] = []
        self.sources_saved: Dict[str, List[SourceItem]] = {}
        self.documents_saved: Dict[str, List[Document]] = {}
        self.chunks_saved: Dict[str, List[DocumentChunk]] = {}
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
    ) -> Dict[str, str]:
        self.documents_saved[session_id] = documents
        return {doc.url: str(uuid.uuid4()) for doc in documents}

    async def save_chunks(
        self,
        session_id: str,
        chunks: List[DocumentChunk],
    ) -> None:
        if self.save_chunks_error:
            raise self.save_chunks_error
        self.chunks_saved[session_id] = chunks

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        session_id: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        if self.search_similar_chunks_error:
            raise self.search_similar_chunks_error
        return [
            RetrievedChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=str(uuid.uuid4()),
                session_id=session_id or str(uuid.uuid4()),
                chunk_index=0,
                text="Repository retrieved chunk text.",
                similarity=0.94,
                url="https://example.com/source1",
                title="Mock Source 1",
            )
        ]

    async def complete_session(self, session_id: str, report: ResearchReport) -> None:
        self.completed_sessions[session_id] = report

    async def fail_session(self, session_id: str, error_message: str) -> None:
        self.failed_sessions[session_id] = error_message


@pytest.fixture
def client():
    app.dependency_overrides.clear()
    app.dependency_overrides[get_search_provider] = lambda: MockSearchProvider()
    app.dependency_overrides[get_webpage_extractor] = lambda: MockWebpageExtractor()
    app.dependency_overrides[get_embedding_provider] = lambda: MockEmbeddingProvider()
    app.dependency_overrides[get_retriever] = lambda: MockRetriever()
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
    mock_embedding = MockEmbeddingProvider()
    mock_retriever = MockRetriever()
    mock_llm = MockLLMProvider()
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embedding
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
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

    # Verify chunks and embeddings were generated and persisted
    assert len(mock_embedding.embedded_texts) == 1
    chunks = mock_repo.chunks_saved.get(data["session_id"], [])
    assert len(chunks) == 1
    assert chunks[0].embedding is not None
    assert len(chunks[0].embedding) == 1536
    assert chunks[0].document_id is not None
    assert chunks[0].session_id == data["session_id"]

    # Verify semantic retrieval was invoked with session_id
    assert mock_retriever.last_retrieve_call["session_id"] == data["session_id"]
    assert mock_retriever.last_retrieve_call["query"] == payload["question"]

    # Verify LLM received ONLY retrieved chunks, not raw documents
    assert mock_llm.last_generate_call["documents"] is None
    assert mock_llm.last_generate_call["chunks"] is not None
    assert len(mock_llm.last_generate_call["chunks"]) == 1

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


def test_research_endpoint_embedding_missing_api_key(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_embedding = MockEmbeddingProvider(
        error=EmbeddingConfigError("Embedding API key is not configured. Please set EMBEDDING_API_KEY.")
    )
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embedding
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 503
    assert "Embedding API key is not configured" in response.json()["detail"]
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_embedding_timeout(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_embedding = MockEmbeddingProvider(
        error=EmbeddingTimeoutError("Embedding request timed out after 30.0s")
    )
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embedding
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 504
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_embedding_provider_error(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_embedding = MockEmbeddingProvider(
        error=EmbeddingProviderError("Upstream embedding error 500")
    )
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embedding
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 502
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_embedding_response_error(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_embedding = MockEmbeddingProvider(
        error=EmbeddingResponseError("Embedding dimension mismatch")
    )
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embedding
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 502
    assert "Invalid response from embedding provider" in response.json()["detail"]
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_chunk_persistence_error(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_embedding = MockEmbeddingProvider()
    mock_repo = MockResearchRepository(
        save_chunks_error=DatabaseError("Failed to persist chunks into database")
    )

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embedding
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 500
    assert "Database persistence error" in response.json()["detail"]
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_retrieval_embedding_error(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_embedding = MockEmbeddingProvider()
    mock_retriever = MockRetriever(
        error=RetrievalEmbeddingError("Embedding service failed during retrieval")
    )
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embedding
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 502
    assert "Embedding service failed during retrieval" in response.json()["detail"]
    assert len(mock_repo.failed_sessions) == 1


def test_research_endpoint_retrieval_database_error(client):
    mock_search = MockSearchProvider()
    mock_extractor = MockWebpageExtractor()
    mock_embedding = MockEmbeddingProvider()
    mock_retriever = MockRetriever(
        error=RetrievalDatabaseError("Database failure during vector retrieval")
    )
    mock_repo = MockResearchRepository()

    app.dependency_overrides[get_search_provider] = lambda: mock_search
    app.dependency_overrides[get_webpage_extractor] = lambda: mock_extractor
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embedding
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_research_repository] = lambda: mock_repo

    payload = {"question": "What is quantum computing?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 500
    assert "Database retrieval error" in response.json()["detail"]
    assert len(mock_repo.failed_sessions) == 1


def test_retrieve_endpoint_success(client):
    mock_retriever = MockRetriever()
    app.dependency_overrides[get_retriever] = lambda: mock_retriever

    payload = {"query": "breakthroughs in battery chemistry"}
    response = client.post("/api/research/retrieve", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "breakthroughs in battery chemistry"
    assert data["top_k"] == 5
    assert len(data["results"]) == 1
    assert data["results"][0]["similarity"] == 0.92
    assert data["results"][0]["url"] == "https://example.com/source1"
    assert mock_retriever.last_retrieve_call["query"] == "breakthroughs in battery chemistry"
    assert mock_retriever.last_retrieve_call["session_id"] is None


def test_retrieve_endpoint_session_scoped(client):
    session_id = str(uuid.uuid4())
    mock_retriever = MockRetriever()
    app.dependency_overrides[get_retriever] = lambda: mock_retriever

    payload = {
        "query": "breakthroughs in battery chemistry",
        "session_id": session_id,
        "top_k": 3,
        "similarity_threshold": 0.8,
    }
    response = client.post("/api/research/retrieve", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == payload["query"]
    assert data["top_k"] == 3
    assert len(data["results"]) == 1
    assert mock_retriever.last_retrieve_call["session_id"] == session_id
    assert mock_retriever.last_retrieve_call["top_k"] == 3
    assert mock_retriever.last_retrieve_call["similarity_threshold"] == 0.8


def test_retrieve_endpoint_config_error(client):
    mock_retriever = MockRetriever(
        error=RetrievalConfigError("Retrieval configuration error: missing key")
    )
    app.dependency_overrides[get_retriever] = lambda: mock_retriever

    payload = {"query": "breakthroughs in battery chemistry"}
    response = client.post("/api/research/retrieve", json=payload)
    assert response.status_code == 503
    assert "Retrieval configuration error" in response.json()["detail"]


def test_retrieve_endpoint_embedding_error(client):
    mock_retriever = MockRetriever(
        error=RetrievalEmbeddingError("Embedding upstream failure")
    )
    app.dependency_overrides[get_retriever] = lambda: mock_retriever

    payload = {"query": "breakthroughs in battery chemistry"}
    response = client.post("/api/research/retrieve", json=payload)
    assert response.status_code == 502
    assert "Embedding upstream failure" in response.json()["detail"]


def test_retrieve_endpoint_database_error(client):
    mock_retriever = MockRetriever(
        error=RetrievalDatabaseError("Database query failed")
    )
    app.dependency_overrides[get_retriever] = lambda: mock_retriever

    payload = {"query": "breakthroughs in battery chemistry"}
    response = client.post("/api/research/retrieve", json=payload)
    assert response.status_code == 500
    assert "Database retrieval error" in response.json()["detail"]


def test_retrieve_endpoint_empty_query(client):
    payload = {"query": "   "}
    response = client.post("/api/research/retrieve", json=payload)
    assert response.status_code == 422


