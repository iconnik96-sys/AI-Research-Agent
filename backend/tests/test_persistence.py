import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.config import settings
from app.db.models import ResearchSessionModel
from app.schemas.chunk import DocumentChunk
from app.schemas.document import Document
from app.schemas.report import ResearchReport, ResearchSection, SourceReference
from app.schemas.research import SourceItem
from app.services.persistence.exceptions import (
    DatabaseConfigError,
    DatabaseConnectionError,
    DatabaseError,
    SessionNotFoundError,
)
from app.services.persistence.repository import (
    SQLAlchemyResearchRepository,
    _sanitize_error,
)


def test_sanitize_error():
    sensitive_msg = "Error connecting to postgresql+asyncpg://postgres:SuperSecretPassword123@db.supabase.co:5432/postgres"
    sanitized = _sanitize_error(Exception(sensitive_msg))
    assert "SuperSecretPassword123" not in sanitized
    assert "sensitive details masked" in sanitized

    regular_msg = "Table not found"
    assert _sanitize_error(Exception(regular_msg)) == regular_msg


def test_missing_database_url_raises_config_error(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    with pytest.raises(DatabaseConfigError, match="DATABASE_URL is not configured"):
        SQLAlchemyResearchRepository()


@pytest.fixture
def mock_db_session():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.add_all = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.get = AsyncMock()
    return mock_session


@pytest.fixture
def mock_session_factory(mock_db_session):
    factory = MagicMock()
    # Support `async with factory() as db:`
    factory.return_value.__aenter__.return_value = mock_db_session
    factory.return_value.__aexit__.return_value = None
    return factory


@pytest.mark.anyio
async def test_create_session(mock_session_factory, mock_db_session, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    session_id = await repo.create_session("What are the latest fusion breakthroughs?")

    assert session_id is not None
    # Validate it's a valid UUID
    parsed_uuid = uuid.UUID(session_id)
    assert str(parsed_uuid) == session_id

    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_save_sources(mock_session_factory, mock_db_session, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    test_session_id = str(uuid.uuid4())
    sources = [
        SourceItem(
            title="Fusion Article",
            url="https://example.com/fusion",
            content="Snippet",
            score=0.95,
        ),
        SourceItem(
            title="Second Article",
            url="https://example.com/second",
            content="Snippet 2",
            score=0.85,
        ),
    ]

    url_to_id = await repo.save_sources(test_session_id, sources)

    assert len(url_to_id) == 2
    assert "https://example.com/fusion" in url_to_id
    assert "https://example.com/second" in url_to_id

    mock_db_session.add_all.assert_called_once()
    mock_db_session.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_save_documents(mock_session_factory, mock_db_session, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    test_session_id = str(uuid.uuid4())
    source_uuid = str(uuid.uuid4())
    source_map = {"https://example.com/fusion": source_uuid}

    documents = [
        Document(
            url="https://example.com/fusion",
            title="Fusion Article",
            text="Detailed scientific text about net energy gain.",
            score=0.95,
            char_count=45,
        )
    ]

    doc_id_map = await repo.save_documents(test_session_id, documents, source_id_map=source_map)

    assert "https://example.com/fusion" in doc_id_map
    mock_db_session.add_all.assert_called_once()
    mock_db_session.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_complete_session_success(mock_session_factory, mock_db_session, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    test_session_id = str(uuid.uuid4())
    existing_session = ResearchSessionModel(
        id=uuid.UUID(test_session_id),
        question="What is fusion?",
        status="pending",
    )
    mock_db_session.get.return_value = existing_session

    report = ResearchReport(
        title="Fusion Report",
        summary="Net energy gain confirmed.",
        sections=[
            ResearchSection(
                heading="Results",
                content="3.15 MJ output.",
                citations=["S1"],
            )
        ],
        sources=[
            SourceReference(
                id="S1",
                title="Source 1",
                url="https://example.com/s1",
            )
        ],
    )

    await repo.complete_session(test_session_id, report)

    assert existing_session.status == "completed"
    assert existing_session.report is not None
    assert existing_session.report["title"] == "Fusion Report"
    mock_db_session.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_complete_session_not_found(mock_session_factory, mock_db_session, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    test_session_id = str(uuid.uuid4())
    mock_db_session.get.return_value = None

    report = ResearchReport(title="T", summary="S", sections=[], sources=[])

    with pytest.raises(SessionNotFoundError, match="not found"):
        await repo.complete_session(test_session_id, report)


@pytest.mark.anyio
async def test_fail_session(mock_session_factory, mock_db_session, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    test_session_id = str(uuid.uuid4())
    existing_session = ResearchSessionModel(
        id=uuid.UUID(test_session_id),
        question="What is fusion?",
        status="pending",
    )
    mock_db_session.get.return_value = existing_session

    await repo.fail_session(test_session_id, "Tavily rate limit exceeded")

    assert existing_session.status == "failed"
    mock_db_session.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_database_connection_failure(mock_session_factory, mock_db_session, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    mock_db_session.commit.side_effect = Exception("Connection refused by peer")

    with pytest.raises(DatabaseConnectionError, match="Database error creating session"):
        await repo.create_session("Question")


@pytest.mark.anyio
async def test_save_chunks_success(mock_session_factory, mock_db_session, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    test_session_id = str(uuid.uuid4())
    test_doc_id = str(uuid.uuid4())
    chunks = [
        DocumentChunk(
            document_id=test_doc_id,
            session_id=test_session_id,
            chunk_index=0,
            text="Chunk 1 text",
            char_count=12,
            embedding=[0.1] * 1536,
        ),
        DocumentChunk(
            document_id=test_doc_id,
            session_id=test_session_id,
            chunk_index=1,
            text="Chunk 2 text",
            char_count=12,
            embedding=[0.2] * 1536,
        ),
    ]

    await repo.save_chunks(test_session_id, chunks)

    mock_db_session.add_all.assert_called_once()
    mock_db_session.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_save_chunks_empty(mock_session_factory, mock_db_session, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    test_session_id = str(uuid.uuid4())
    await repo.save_chunks(test_session_id, [])

    mock_db_session.add_all.assert_not_called()


@pytest.mark.anyio
async def test_save_chunks_missing_doc_id(mock_session_factory, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    test_session_id = str(uuid.uuid4())
    chunks = [
        DocumentChunk(
            document_id=None,
            session_id=test_session_id,
            chunk_index=0,
            text="Missing doc id",
            char_count=14,
            embedding=[0.1] * 1536,
        )
    ]

    with pytest.raises(DatabaseError, match="missing required document_id"):
        await repo.save_chunks(test_session_id, chunks)


@pytest.mark.anyio
async def test_save_chunks_missing_embedding(mock_session_factory, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    test_session_id = str(uuid.uuid4())
    test_doc_id = str(uuid.uuid4())
    chunks = [
        DocumentChunk(
            document_id=test_doc_id,
            session_id=test_session_id,
            chunk_index=0,
            text="Missing embedding",
            char_count=17,
            embedding=None,
        )
    ]

    with pytest.raises(DatabaseError, match="missing required embedding vector"):
        await repo.save_chunks(test_session_id, chunks)


@pytest.mark.anyio
async def test_save_chunks_db_connection_failure(mock_session_factory, mock_db_session, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    test_session_id = str(uuid.uuid4())
    test_doc_id = str(uuid.uuid4())
    chunks = [
        DocumentChunk(
            document_id=test_doc_id,
            session_id=test_session_id,
            chunk_index=0,
            text="Text",
            char_count=4,
            embedding=[0.1] * 1536,
        )
    ]

    mock_db_session.commit.side_effect = Exception("DB timeout")

    with pytest.raises(DatabaseConnectionError, match="Database error persisting document chunks"):
        await repo.save_chunks(test_session_id, chunks)


@pytest.mark.anyio
async def test_search_similar_chunks_success(mock_session_factory, mock_db_session, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    test_chunk_id = uuid.uuid4()
    test_doc_id = uuid.uuid4()
    test_session_id = uuid.uuid4()

    mock_row = (
        test_chunk_id,
        test_doc_id,
        test_session_id,
        0,
        "Retrieved content",
        "https://example.com/source",
        "Source Title",
        0.912,
    )
    mock_result = MagicMock()
    mock_result.all.return_value = [mock_row]
    mock_db_session.execute.return_value = mock_result

    results = await repo.search_similar_chunks(
        query_embedding=[0.1] * 1536,
        session_id=str(test_session_id),
        top_k=3,
        similarity_threshold=0.8,
    )

    assert len(results) == 1
    chunk = results[0]
    assert chunk.chunk_id == str(test_chunk_id)
    assert chunk.document_id == str(test_doc_id)
    assert chunk.session_id == str(test_session_id)
    assert chunk.chunk_index == 0
    assert chunk.text == "Retrieved content"
    assert chunk.url == "https://example.com/source"
    assert chunk.title == "Source Title"
    assert pytest.approx(chunk.similarity, 0.001) == 0.912
    mock_db_session.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_search_similar_chunks_empty_embedding(mock_session_factory, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)
    results = await repo.search_similar_chunks(query_embedding=[])
    assert results == []


@pytest.mark.anyio
async def test_search_similar_chunks_invalid_session_uuid(mock_session_factory, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)
    with pytest.raises(DatabaseError, match="Invalid session_id format"):
        await repo.search_similar_chunks(query_embedding=[0.1] * 1536, session_id="not-a-uuid")


@pytest.mark.anyio
async def test_search_similar_chunks_db_error(mock_session_factory, mock_db_session, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
    repo = SQLAlchemyResearchRepository(session_factory=mock_session_factory)

    mock_db_session.execute.side_effect = Exception("DB query failed")

    with pytest.raises(DatabaseConnectionError, match="Database error during similarity search"):
        await repo.search_similar_chunks(query_embedding=[0.1] * 1536)
