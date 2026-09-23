import json
import logging
from typing import Optional
import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.planner import ResearchPlan
from app.services.planner.base import BaseResearchPlanner
from app.services.planner.exceptions import (
    PlannerConfigError,
    PlannerError,
    PlannerNetworkError,
    PlannerResponseError,
    PlannerTimeoutError,
)

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a research planning component.

Your job is NOT to answer the research question.

Your job is to decompose the question into a small number of independent research tasks.

Return ONLY valid JSON matching the required schema.

Each sub-question must contribute distinct evidence needed for the final answer.
Avoid duplicate or overlapping queries.

You must generate at most {max_sub_questions} focused sub-questions.
For each sub-question:
- "id": A short identifier such as "q1", "q2".
- "question": A natural language question exploring a distinct semantic aspect of the research topic.
- "search_query": A targeted keyword query optimized for web search engines.
- "reason": A concise explanation of why this specific evidence is required to answer the original question.

Schema:
{{
  "original_question": "The exact original research question",
  "sub_questions": [
    {{
      "id": "q1",
      "question": "Focused question for semantic understanding",
      "search_query": "Keyword search query for web search",
      "reason": "Why this information is needed"
    }}
  ]
}}"""


class LLMResearchPlanner(BaseResearchPlanner):
    """Research planner that decomposes research questions into focused sub-questions via LLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_sub_questions: Optional[int] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.LLM_API_KEY
        self.base_url = base_url if base_url is not None else settings.LLM_BASE_URL
        self.model = model if model is not None else settings.LLM_MODEL
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.PLANNER_TIMEOUT_SECONDS
        )
        self.max_sub_questions = (
            max_sub_questions
            if max_sub_questions is not None
            else settings.PLANNER_MAX_SUB_QUESTIONS
        )

    async def create_plan(self, question: str) -> ResearchPlan:
        """Decompose a research question into a structured research plan."""
        clean_question = (question or "").strip()
        if not clean_question:
            raise PlannerError("Research question cannot be empty or whitespace only.")

        if not self.api_key or not self.api_key.strip():
            raise PlannerConfigError(
                "LLM API key is not configured for research planning. Please set LLM_API_KEY."
            )

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        system_content = PLANNER_SYSTEM_PROMPT.format(max_sub_questions=self.max_sub_questions)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Decompose this research question into a structured plan:\n{clean_question}"},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            logger.error("Planner request timed out after %.1fs: %s", self.timeout_seconds, exc)
            raise PlannerTimeoutError(
                f"Planner request timed out after {self.timeout_seconds}s."
            ) from exc
        except httpx.NetworkError as exc:
            logger.error("Network error during planner request: %s", exc)
            raise PlannerNetworkError(
                f"Network transport error during planner request: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error during planner request: %s", exc)
            raise PlannerError(f"Unexpected error calling planner API: {exc}") from exc

        if response.status_code == 401:
            logger.error("Planner authentication failed (HTTP 401).")
            raise PlannerConfigError("LLM API authentication failed. Please check LLM_API_KEY.")
        if response.status_code == 429:
            logger.error("Planner rate limit exceeded (HTTP 429).")
            raise PlannerNetworkError("LLM API rate limit exceeded. Please try again later.")
        if response.status_code >= 500:
            logger.error("Planner upstream server error HTTP %d: %s", response.status_code, response.text)
            raise PlannerNetworkError(
                f"Upstream LLM provider error HTTP {response.status_code}."
            )
        if response.status_code != 200:
            logger.error("Planner unexpected HTTP %d: %s", response.status_code, response.text)
            raise PlannerResponseError(
                f"Planner API returned unexpected status code {response.status_code}."
            )

        try:
            body = response.json()
            raw_content = body["choices"][0]["message"]["content"]
            parsed_json = json.loads(raw_content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse planner JSON response: %s", exc)
            raise PlannerResponseError(
                f"Invalid or unparseable JSON returned by planner LLM: {exc}"
            ) from exc

        try:
            plan = ResearchPlan.model_validate(parsed_json)
        except ValidationError as exc:
            logger.error("Planner response failed schema validation: %s", exc)
            raise PlannerResponseError(
                f"Planner response does not conform to ResearchPlan schema: {exc}"
            ) from exc

        # Strict validation: do NOT silently truncate. Raise error if exceeding max_sub_questions.
        if len(plan.sub_questions) > self.max_sub_questions:
            logger.warning(
                "Planner generated %d sub-questions exceeding limit of %d",
                len(plan.sub_questions),
                self.max_sub_questions,
            )
            raise PlannerResponseError(
                f"Planner generated {len(plan.sub_questions)} sub-questions, exceeding maximum of {self.max_sub_questions}."
            )

        return plan
