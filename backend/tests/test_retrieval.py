import uuid
from typing import List, Optional
import pytest

from app.schemas.chunk import DocumentChunk
from app.schemas.document import Document
from app.schemas.report import ResearchReport
from app.schemas.research import SourceItem
from app.schemas.retrieval import RetrievedChunk
from app.services.embedding.base import BaseEmbeddingProvider
from app.services.embedding.exceptions import EmbeddingConfigError, EmbeddingTimeoutError
from app.services.persistence.base import BaseResearchRepository
from app.services.persistence.exceptions import DatabaseConfigError, DatabaseConnectionError
from app.services.retrieval.exceptions import (
    RetrievalConfigError,
    RetrievalDatabaseError,
    RetrievalEmbeddingError,
)
from app.services.retrieval.pgvector import PgVectorRetriever


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, error: Optional[Exception] = None):
        self.error = error
        self.embedded_texts: List[str] = []

    @property
    def dimensions(self) -> int:
        return 1536

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.error:
            raise self.error
        self.embedded_texts.extend(texts)
        return [[0.1] * 1536 for _ in texts]

    async def embed_text(self, text: str) -> List[float]:
        if self.error:
            raise self.error
        self.embedded_texts.append(text)
        return [0.1] * 1536


class FakeResearchRepository(BaseResearchRepository):
    def __init__(
        self,
        search_results: Optional[List[RetrievedChunk]] = None,
        search_error: Optional[Exception] = None,
    ):
        self.search_results = search_results or []
        self.search_error = search_error
        self.last_search_call: dict = {}

    async def create_session(self, question: str) -> str:
        return str(uuid.uuid4())

    async def save_sources(self, session_id: str, sources: List[SourceItem]) -> dict:
        return {}

    async def save_documents(self, session_id: str, documents: List[Document], source_id_map: Optional[dict] = None) -> dict:
        return {}

    async def save_chunks(self, session_id: str, chunks: List[DocumentChunk]) -> None:
        pass

    async def complete_session(self, session_id: str, report: ResearchReport) -> None:
        pass

    async def fail_session(self, session_id: str, error_message: str) -> None:
        pass

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        session_id: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        if self.search_error:
            raise self.search_error
        self.last_search_call = {
            "query_embedding": query_embedding,
            "session_id": session_id,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }
        return self.search_results[:top_k]


@pytest.mark.anyio
async def test_retrieve_empty_query():
    retriever = PgVectorRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        repository=FakeResearchRepository(),
    )
    assert await retriever.retrieve("") == []
    assert await retriever.retrieve("   ") == []


@pytest.mark.anyio
async def test_retrieve_invalid_top_k():
    retriever = PgVectorRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        repository=FakeResearchRepository(),
    )
    with pytest.raises(RetrievalConfigError, match="top_k must be greater than 0"):
        await retriever.retrieve("fusion energy", top_k=0)


@pytest.mark.anyio
async def test_retrieve_success_ordering_and_limiting():
    sample_chunks = [
        RetrievedChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=str(uuid.uuid4()),
            session_id="session-1",
            chunk_index=0,
            text="High similarity chunk",
            similarity=0.95,
            url="https://example.com/high",
            title="High Relevance",
        ),
        RetrievedChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=str(uuid.uuid4()),
            session_id="session-1",
            chunk_index=1,
            text="Medium similarity chunk",
            similarity=0.82,
            url="https://example.com/med",
            title="Med Relevance",
        ),
        RetrievedChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=str(uuid.uuid4()),
            session_id="session-1",
            chunk_index=2,
            text="Low similarity chunk",
            similarity=0.71,
            url="https://example.com/low",
            title="Low Relevance",
        ),
    ]

    emb_provider = FakeEmbeddingProvider()
    repo = FakeResearchRepository(search_results=sample_chunks)
    retriever = PgVectorRetriever(embedding_provider=emb_provider, repository=repo)

    results = await retriever.retrieve(
        query="fusion energy",
        session_id="session-1",
        top_k=2,
        similarity_threshold=0.8,
    )

    assert len(results) == 2
    assert results[0].similarity == 0.95
    assert results[1].similarity == 0.82
    assert emb_provider.embedded_texts == ["fusion energy"]
    assert repo.last_search_call["session_id"] == "session-1"
    assert repo.last_search_call["top_k"] == 2
    assert repo.last_search_call["similarity_threshold"] == 0.8


@pytest.mark.anyio
async def test_retrieve_embedding_config_error():
    emb_provider = FakeEmbeddingProvider(
        error=EmbeddingConfigError("Missing EMBEDDING_API_KEY")
    )
    retriever = PgVectorRetriever(
        embedding_provider=emb_provider,
        repository=FakeResearchRepository(),
    )
    with pytest.raises(RetrievalConfigError, match="Missing EMBEDDING_API_KEY"):
        await retriever.retrieve("fusion")


@pytest.mark.anyio
async def test_retrieve_embedding_timeout_error():
    emb_provider = FakeEmbeddingProvider(
        error=EmbeddingTimeoutError("Embedding request timed out")
    )
    retriever = PgVectorRetriever(
        embedding_provider=emb_provider,
        repository=FakeResearchRepository(),
    )
    with pytest.raises(RetrievalEmbeddingError, match="timed out"):
        await retriever.retrieve("fusion")


@pytest.mark.anyio
async def test_retrieve_database_config_error():
    repo = FakeResearchRepository(
        search_error=DatabaseConfigError("DATABASE_URL missing")
    )
    retriever = PgVectorRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        repository=repo,
    )
    with pytest.raises(RetrievalConfigError, match="DATABASE_URL missing"):
        await retriever.retrieve("fusion")


@pytest.mark.anyio
async def test_retrieve_database_connection_error():
    repo = FakeResearchRepository(
        search_error=DatabaseConnectionError("DB connection refused")
    )
    retriever = PgVectorRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        repository=repo,
    )
    with pytest.raises(RetrievalDatabaseError, match="DB connection refused"):
        await retriever.retrieve("fusion")


@pytest.mark.anyio
async def test_retrieve_empty_results():
    retriever = PgVectorRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        repository=FakeResearchRepository(search_results=[]),
    )
    results = await retriever.retrieve("rare topic")
    assert results == []
