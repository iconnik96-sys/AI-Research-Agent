import json
import logging
from typing import List, Optional
import httpx

from app.core.config import settings
from app.schemas.document import Document
from app.schemas.report import ResearchReport, ResearchSection
from app.services.llm.base import BaseLLMProvider
from app.services.llm.evidence import prepare_evidence
from app.services.llm.exceptions import (
    LLMConfigError,
    LLMNetworkError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert AI research analyst.
Your task is to synthesize a rigorous, factual research report answering the user's research question using ONLY the supplied sources.

Rules:
1. Base your answer strictly on the provided evidence. Do NOT invent facts or extrapolate beyond the text.
2. Do NOT invent sources or citations. Every citation must be one of the exact Source IDs provided (e.g. S1, S2).
3. If the evidence is insufficient, contradictory, or inconclusive, explicitly state that limitation.
4. Structure your response into clear thematic sections.
5. In each section, include the list of Source IDs (e.g. ["S1", "S2"]) that support the statements in that section.
6. Return your output strictly as a JSON object matching this schema:
{
  "title": "Comprehensive title for the research report",
  "summary": "High-level executive summary of findings and evidence",
  "sections": [
    {
      "heading": "Section Heading",
      "content": "Detailed synthesis of findings supported by the sources...",
      "citations": ["S1", "S2"]
    }
  ]
}"""


class OpenAICompatibleLLMProvider(BaseLLMProvider):
    """LLM provider communicating with OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_chars_per_doc: Optional[int] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.LLM_API_KEY
        self.base_url = base_url if base_url is not None else settings.LLM_BASE_URL
        self.model = model if model is not None else settings.LLM_MODEL
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.LLM_TIMEOUT_SECONDS
        )
        self.max_chars_per_doc = (
            max_chars_per_doc
            if max_chars_per_doc is not None
            else settings.LLM_MAX_CHARS_PER_DOC
        )

    async def generate_report(
        self,
        question: str,
        documents: List[Document],
    ) -> ResearchReport:
        """Synthesize a structured research report from extracted documents."""
        # If no documents are available, return a deterministic report without calling LLM
        if not documents:
            logger.info("No documents provided for question '%s'; returning insufficient evidence report.", question)
            return ResearchReport(
                title=f"Research Report: {question}",
                summary="Insufficient evidence found. No usable documents could be retrieved to answer this research question.",
                sections=[
                    ResearchSection(
                        heading="Evidence Assessment",
                        content="No source documents were successfully extracted or available for synthesis.",
                        citations=[],
                    )
                ],
                sources=[],
            )

        if not self.api_key or not self.api_key.strip():
            raise LLMConfigError(
                "LLM API key is not configured. Please set LLM_API_KEY in your environment."
            )

        source_refs, evidence_text = prepare_evidence(
            documents=documents,
            max_chars_per_doc=self.max_chars_per_doc,
        )
        valid_source_ids = {s.id for s in source_refs}

        user_prompt = (
            f"Research Question: {question}\n\n"
            f"Supplied Sources:\n{evidence_text}\n\n"
            "Please generate the structured research report according to instructions."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"LLM request timed out after {self.timeout_seconds}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"LLM provider error HTTP {exc.response.status_code}: {exc.response.text}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise LLMNetworkError(
                f"Network error while connecting to LLM provider: {str(exc)}"
            ) from exc

        return self._parse_and_validate_response(
            data=data,
            question=question,
            source_refs=source_refs,
            valid_source_ids=valid_source_ids,
        )

    def _parse_and_validate_response(
        self,
        data: dict,
        question: str,
        source_refs: list,
        valid_source_ids: set,
    ) -> ResearchReport:
        try:
            choices = data.get("choices", [])
            if not choices:
                raise LLMResponseError("LLM response did not contain any choices.")
            raw_content = choices[0].get("message", {}).get("content", "")
            if not raw_content or not raw_content.strip():
                raise LLMResponseError("LLM returned an empty message content.")
            parsed_json = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"LLM returned malformed non-JSON output: {raw_content}"
            ) from exc
        except Exception as exc:
            if isinstance(exc, LLMResponseError):
                raise
            raise LLMResponseError(f"Failed to parse LLM response payload: {str(exc)}") from exc

        try:
            sections_data = parsed_json.get("sections", [])
            sections: List[ResearchSection] = []
            for s in sections_data:
                citations = s.get("citations", [])
                if not isinstance(citations, list):
                    citations = [str(citations)]
                sections.append(
                    ResearchSection(
                        heading=s.get("heading") or "Findings",
                        content=s.get("content") or "",
                        citations=[str(c).strip() for c in citations if str(c).strip()],
                    )
                )

            report = ResearchReport(
                title=parsed_json.get("title") or f"Research Report: {question}",
                summary=parsed_json.get("summary") or "",
                sections=sections,
                sources=source_refs,
            )
        except Exception as exc:
            raise LLMResponseError(
                f"LLM response failed schema validation: {str(exc)}"
            ) from exc

        # Strictly validate that all cited source IDs exist in valid_source_ids
        for section in report.sections:
            for citation in section.citations:
                if citation not in valid_source_ids:
                    raise LLMResponseError(
                        f"Invalid citation ID '{citation}' in section '{section.heading}'. "
                        f"Allowed source IDs are: {sorted(list(valid_source_ids))}"
                    )

        return report
