from abc import ABC, abstractmethod
from typing import List, Optional

from app.schemas.document import Document
from app.schemas.report import ResearchReport
from app.schemas.retrieval import RetrievedChunk


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers generating structured research reports."""

    @abstractmethod
    async def generate_report(
        self,
        question: str,
        documents: Optional[List[Document]] = None,
        chunks: Optional[List[RetrievedChunk]] = None,
    ) -> ResearchReport:
        """Synthesize a structured research report from research question and evidence.

        Args:
            question: The user's research question.
            documents: Optional list of cleaned, extracted source documents.
            chunks: Optional list of retrieved relevant document chunks (RAG).

        Returns:
            Structured ResearchReport containing title, summary, sections, and source citations.

        Raises:
            LLMConfigError: If API credentials or configuration are missing.
            LLMTimeoutError: If the request to the LLM times out.
            LLMNetworkError: If a network/connection error occurs.
            LLMProviderError: If the upstream provider returns an error (4xx/5xx).
            LLMResponseError: If the response is malformed or citations are invalid.
        """
        pass
