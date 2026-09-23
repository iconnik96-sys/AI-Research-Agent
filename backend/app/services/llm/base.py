from abc import ABC, abstractmethod
from typing import List

from app.schemas.document import Document
from app.schemas.report import ResearchReport


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers generating structured research reports."""

    @abstractmethod
    async def generate_report(
        self,
        question: str,
        documents: List[Document],
    ) -> ResearchReport:
        """Synthesize a structured research report from research question and documents.

        Args:
            question: The user's research question.
            documents: List of cleaned, extracted source documents.

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
