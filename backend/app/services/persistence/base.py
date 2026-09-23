from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from app.schemas.chunk import DocumentChunk
from app.schemas.document import Document
from app.schemas.report import ResearchReport
from app.schemas.research import SourceItem
from app.schemas.retrieval import RetrievedChunk


class BaseResearchRepository(ABC):
    """Abstract interface for research session persistence."""

    @abstractmethod
    async def create_session(self, question: str) -> str:
        """Create a new research session in 'pending' status.

        Args:
            question: The user's research question.

        Returns:
            The created session ID string (UUID).
        """
        pass

    @abstractmethod
    async def save_sources(
        self,
        session_id: str,
        sources: List[SourceItem],
    ) -> Dict[str, str]:
        """Persist collected web sources associated with the session.

        Args:
            session_id: The session UUID string.
            sources: List of collected SourceItem objects.

        Returns:
            Mapping of source URL to persisted source UUID string.
        """
        pass

    @abstractmethod
    async def save_documents(
        self,
        session_id: str,
        documents: List[Document],
        source_id_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Persist extracted documents associated with the session and their source.

        Args:
            session_id: The session UUID string.
            documents: List of extracted Document objects.
            source_id_map: Optional mapping of URL to source UUID.

        Returns:
            Mapping of document URL to persisted document UUID string.
        """
        pass

    @abstractmethod
    async def save_chunks(
        self,
        session_id: str,
        chunks: List[DocumentChunk],
    ) -> None:
        """Persist document chunks with embeddings, associated with session and documents.

        Args:
            session_id: The session UUID string.
            chunks: List of DocumentChunk objects containing text, embeddings, and document_ids.
        """
        pass

    @abstractmethod
    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        session_id: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Search document chunks using vector cosine similarity.

        Args:
            query_embedding: Vector embedding of the search query.
            session_id: Optional session UUID string to filter chunks.
            top_k: Maximum number of chunks to return.
            similarity_threshold: Optional minimum cosine similarity threshold.

        Returns:
            List of RetrievedChunk objects ordered by similarity descending.
        """
        pass

    @abstractmethod
    async def complete_session(
        self,
        session_id: str,
        report: ResearchReport,
    ) -> None:
        """Update session status to 'completed' and store the synthesized report.

        Args:
            session_id: The session UUID string.
            report: The generated ResearchReport object.
        """
        pass

    @abstractmethod
    async def fail_session(
        self,
        session_id: str,
        error_message: str,
    ) -> None:
        """Update session status to 'failed'.

        Args:
            session_id: The session UUID string.
            error_message: Reason for the failure.
        """
        pass
