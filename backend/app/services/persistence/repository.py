import logging
import uuid
from typing import Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.db.models import (
    ClaimEvidenceModel,
    ClaimModel,
    DocumentChunkModel,
    DocumentModel,
    ResearchSessionModel,
    SourceModel,
)
from app.db.session import get_session_factory
from app.schemas.chunk import DocumentChunk
from app.schemas.claim import VerifiedClaim
from app.schemas.document import Document
from app.schemas.evidence import EvidenceItem
from app.schemas.report import ResearchReport
from app.schemas.research import SourceItem
from app.schemas.retrieval import RetrievedChunk
from app.services.persistence.base import BaseResearchRepository
from app.services.persistence.exceptions import (
    DatabaseConfigError,
    DatabaseConnectionError,
    DatabaseError,
    SessionNotFoundError,
)

logger = logging.getLogger(__name__)


def _sanitize_error(exc: Exception) -> str:
    """Strip any credentials or connection strings from error messages."""
    msg = str(exc)
    # Mask any potential password/connection uri patterns
    if "@" in msg and "://" in msg:
        return "Database connection failure (sensitive details masked)"
    return msg


class SQLAlchemyResearchRepository(BaseResearchRepository):
    """SQLAlchemy implementation of research repository with asyncpg."""

    def __init__(
        self,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
    ):
        if not settings.DATABASE_URL or not settings.DATABASE_URL.strip():
            raise DatabaseConfigError(
                "DATABASE_URL is not configured. Please set DATABASE_URL environment variable."
            )
        self._session_factory = session_factory

    def _get_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is not None:
            return self._session_factory
        try:
            return get_session_factory()
        except Exception as exc:
            raise DatabaseConnectionError(
                f"Failed to initialize database engine: {_sanitize_error(exc)}"
            ) from exc

    async def create_session(self, question: str) -> str:
        factory = self._get_factory()
        try:
            session_id = uuid.uuid4()
            async with factory() as db:
                session_model = ResearchSessionModel(
                    id=session_id,
                    question=question,
                    status="pending",
                )
                db.add(session_model)
                await db.commit()
                return str(session_id)
        except Exception as exc:
            logger.error("Failed to create research session: %s", _sanitize_error(exc))
            raise DatabaseConnectionError(
                f"Database error creating session: {_sanitize_error(exc)}"
            ) from exc

    async def save_sources(
        self,
        session_id: str,
        sources: List[SourceItem],
    ) -> Dict[str, str]:
        if not sources:
            return {}

        factory = self._get_factory()
        try:
            session_uuid = uuid.UUID(session_id)
            url_to_id: Dict[str, str] = {}
            source_models: List[SourceModel] = []

            for s in sources:
                src_id = uuid.uuid4()
                url_to_id[s.url] = str(src_id)
                source_models.append(
                    SourceModel(
                        id=src_id,
                        session_id=session_uuid,
                        title=s.title,
                        url=s.url,
                        content=s.content,
                        score=s.score,
                    )
                )

            async with factory() as db:
                db.add_all(source_models)
                await db.commit()

            return url_to_id
        except Exception as exc:
            logger.error("Failed to persist sources for session %s: %s", session_id, _sanitize_error(exc))
            raise DatabaseConnectionError(
                f"Database error persisting sources: {_sanitize_error(exc)}"
            ) from exc

    async def save_documents(
        self,
        session_id: str,
        documents: List[Document],
        source_id_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        if not documents:
            return {}

        factory = self._get_factory()
        try:
            session_uuid = uuid.UUID(session_id)
            doc_id_map: Dict[str, str] = {}
            doc_models: List[DocumentModel] = []

            for doc in documents:
                doc_uuid = uuid.uuid4()
                doc_id_map[doc.url] = str(doc_uuid)

                source_uuid = None
                if source_id_map and doc.url in source_id_map:
                    try:
                        source_uuid = uuid.UUID(source_id_map[doc.url])
                    except (ValueError, TypeError):
                        source_uuid = None

                doc_models.append(
                    DocumentModel(
                        id=doc_uuid,
                        session_id=session_uuid,
                        source_id=source_uuid,
                        url=doc.url,
                        title=doc.title,
                        text=doc.text,
                        char_count=doc.char_count,
                        score=doc.score,
                    )
                )

            async with factory() as db:
                db.add_all(doc_models)
                await db.commit()

            return doc_id_map
        except Exception as exc:
            logger.error("Failed to persist documents for session %s: %s", session_id, _sanitize_error(exc))
            raise DatabaseConnectionError(
                f"Database error persisting documents: {_sanitize_error(exc)}"
            ) from exc

    async def save_chunks(
        self,
        session_id: str,
        chunks: List[DocumentChunk],
    ) -> None:
        if not chunks:
            return

        factory = self._get_factory()
        try:
            session_uuid = uuid.UUID(session_id)
            chunk_models: List[DocumentChunkModel] = []

            for chunk in chunks:
                if not chunk.document_id:
                    raise DatabaseError(
                        f"Chunk index {chunk.chunk_index} is missing required document_id."
                    )
                if not chunk.embedding:
                    raise DatabaseError(
                        f"Chunk index {chunk.chunk_index} is missing required embedding vector."
                    )

                doc_uuid = uuid.UUID(chunk.document_id)
                chunk_models.append(
                    DocumentChunkModel(
                        id=uuid.uuid4(),
                        session_id=session_uuid,
                        document_id=doc_uuid,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        embedding=chunk.embedding,
                    )
                )

            async with factory() as db:
                db.add_all(chunk_models)
                await db.commit()
        except DatabaseError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to persist chunks for session %s: %s",
                session_id,
                _sanitize_error(exc),
            )
            raise DatabaseConnectionError(
                f"Database error persisting document chunks: {_sanitize_error(exc)}"
            ) from exc

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        session_id: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        if not query_embedding:
            return []

        factory = self._get_factory()
        try:
            # Cosine similarity = 1 - cosine distance (<=>)
            cosine_dist = DocumentChunkModel.embedding.cosine_distance(query_embedding)
            similarity = (1 - cosine_dist).label("similarity")

            stmt = (
                select(
                    DocumentChunkModel.id,
                    DocumentChunkModel.document_id,
                    DocumentChunkModel.session_id,
                    DocumentChunkModel.chunk_index,
                    DocumentChunkModel.text,
                    DocumentModel.url,
                    DocumentModel.title,
                    similarity,
                )
                .join(DocumentModel, DocumentChunkModel.document_id == DocumentModel.id)
            )

            if session_id:
                try:
                    session_uuid = uuid.UUID(session_id)
                    stmt = stmt.where(DocumentChunkModel.session_id == session_uuid)
                except (ValueError, TypeError) as exc:
                    raise DatabaseError(f"Invalid session_id format: {session_id}") from exc

            if similarity_threshold is not None:
                stmt = stmt.where(similarity >= similarity_threshold)

            stmt = stmt.order_by(similarity.desc()).limit(top_k)

            async with factory() as db:
                result = await db.execute(stmt)
                rows = result.all()

            retrieved: List[RetrievedChunk] = []
            for row in rows:
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=str(row[0]),
                        document_id=str(row[1]),
                        session_id=str(row[2]),
                        chunk_index=row[3],
                        text=row[4],
                        url=row[5],
                        title=row[6],
                        similarity=float(row[7]),
                    )
                )
            return retrieved
        except DatabaseError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to search similar chunks (session_id=%s): %s",
                session_id,
                _sanitize_error(exc),
            )
            raise DatabaseConnectionError(
                f"Database error during similarity search: {_sanitize_error(exc)}"
            ) from exc

    async def save_claims(
        self,
        session_id: str,
        claims: List[VerifiedClaim],
        evidence_map: Dict[str, EvidenceItem],
    ) -> None:
        if not claims:
            return

        factory = self._get_factory()
        try:
            session_uuid = uuid.UUID(session_id)
            claim_models: List[ClaimModel] = []
            evidence_models: List[ClaimEvidenceModel] = []

            for c in claims:
                claim_uuid = uuid.uuid4()
                status_val = c.status.value if hasattr(c.status, "value") else str(c.status)
                claim_models.append(
                    ClaimModel(
                        id=claim_uuid,
                        session_id=session_uuid,
                        claim_identifier=c.id,
                        claim=c.claim,
                        status=status_val,
                        reason=c.reason,
                    )
                )

                supporting_set = set(c.supporting_evidence_ids)
                # Link each cited evidence ID
                for eid in c.evidence_ids:
                    if eid in evidence_map:
                        ev = evidence_map[eid]
                        chunk_uuid = uuid.UUID(ev.chunk_id)
                        evidence_models.append(
                            ClaimEvidenceModel(
                                id=uuid.uuid4(),
                                claim_id=claim_uuid,
                                chunk_id=chunk_uuid,
                                evidence_identifier=eid,
                                is_supporting=(eid in supporting_set),
                            )
                        )

            async with factory() as db:
                db.add_all(claim_models)
                db.add_all(evidence_models)
                await db.commit()
        except Exception as exc:
            logger.error("Failed to persist claims for session %s: %s", session_id, _sanitize_error(exc))
            raise DatabaseConnectionError(
                f"Database error persisting claims: {_sanitize_error(exc)}"
            ) from exc

    async def complete_session(
        self,
        session_id: str,
        report: ResearchReport,
    ) -> None:
        factory = self._get_factory()
        try:
            session_uuid = uuid.UUID(session_id)
            async with factory() as db:
                session_model = await db.get(ResearchSessionModel, session_uuid)
                if not session_model:
                    raise SessionNotFoundError(f"Research session {session_id} not found.")

                session_model.status = "completed"
                session_model.report = report.model_dump()
                await db.commit()
        except SessionNotFoundError:
            raise
        except Exception as exc:
            logger.error("Failed to complete session %s: %s", session_id, _sanitize_error(exc))
            raise DatabaseConnectionError(
                f"Database error completing session: {_sanitize_error(exc)}"
            ) from exc

    async def fail_session(
        self,
        session_id: str,
        error_message: str,
    ) -> None:
        factory = self._get_factory()
        try:
            session_uuid = uuid.UUID(session_id)
            async with factory() as db:
                session_model = await db.get(ResearchSessionModel, session_uuid)
                if session_model:
                    session_model.status = "failed"
                    await db.commit()
        except Exception as exc:
            logger.warning("Could not record failure status for session %s: %s", session_id, _sanitize_error(exc))

    async def delete_session(
        self,
        session_id: str,
    ) -> None:
        factory = self._get_factory()
        try:
            session_uuid = uuid.UUID(session_id)
            async with factory() as db:
                session_model = await db.get(ResearchSessionModel, session_uuid)
                if session_model:
                    await db.delete(session_model)
                    await db.commit()
        except Exception as exc:
            logger.error("Failed to delete session %s: %s", session_id, _sanitize_error(exc))
            raise DatabaseConnectionError(
                f"Database error deleting session: {_sanitize_error(exc)}"
            ) from exc

