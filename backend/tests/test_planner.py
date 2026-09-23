import json
import pytest
import httpx
from pydantic import ValidationError

from app.schemas.planner import ResearchPlan, ResearchSubQuestion
from app.services.planner import (
    LLMResearchPlanner,
    PlannerConfigError,
    PlannerError,
    PlannerNetworkError,
    PlannerResponseError,
    PlannerTimeoutError,
)


def test_sub_question_validation():
    # Valid
    sub_q = ResearchSubQuestion(
        id="q1",
        question="What are recent fusion milestones?",
        search_query="fusion milestones 2026",
        reason="Identify recent achievements.",
    )
    assert sub_q.id == "q1"
    assert sub_q.question == "What are recent fusion milestones?"

    # Empty id
    with pytest.raises(ValidationError):
        ResearchSubQuestion(
            id="  ",
            question="What are recent fusion milestones?",
            search_query="fusion milestones 2026",
            reason="Identify recent achievements.",
        )

    # Empty question
    with pytest.raises(ValidationError):
        ResearchSubQuestion(
            id="q1",
            question="  ",
            search_query="fusion milestones 2026",
            reason="Identify recent achievements.",
        )

    # Empty search query
    with pytest.raises(ValidationError):
        ResearchSubQuestion(
            id="q1",
            question="What are recent fusion milestones?",
            search_query="",
            reason="Identify recent achievements.",
        )


def test_research_plan_validation():
    sub_q = ResearchSubQuestion(
        id="q1",
        question="What are recent fusion milestones?",
        search_query="fusion milestones 2026",
        reason="Identify recent achievements.",
    )

    # Valid
    plan = ResearchPlan(
        original_question="What are the latest developments in fusion energy?",
        sub_questions=[sub_q],
    )
    assert len(plan.sub_questions) == 1

    # Empty original question
    with pytest.raises(ValidationError):
        ResearchPlan(
            original_question="   ",
            sub_questions=[sub_q],
        )

    # Empty sub_questions list
    with pytest.raises(ValidationError):
        ResearchPlan(
            original_question="What is fusion energy?",
            sub_questions=[],
        )


@pytest.mark.anyio
async def test_planner_missing_api_key():
    planner = LLMResearchPlanner(api_key="")
    with pytest.raises(PlannerConfigError) as exc_info:
        await planner.create_plan("What is quantum computing?")
    assert "LLM API key is not configured" in str(exc_info.value)


@pytest.mark.anyio
async def test_planner_empty_question():
    planner = LLMResearchPlanner(api_key="test-key")
    with pytest.raises(PlannerError) as exc_info:
        await planner.create_plan("   ")
    assert "cannot be empty" in str(exc_info.value)


@pytest.mark.anyio
async def test_planner_successful_plan(monkeypatch):
    mock_plan_dict = {
        "original_question": "What is quantum computing?",
        "sub_questions": [
            {
                "id": "q1",
                "question": "What are the core physical principles of quantum computing?",
                "search_query": "quantum computing basic physical principles qubits superposition",
                "reason": "Explain foundational physics.",
            },
            {
                "id": "q2",
                "question": "What are the leading hardware architectures for quantum computing?",
                "search_query": "quantum computing architectures superconducting ion trap photonics",
                "reason": "Examine current engineering approaches.",
            },
        ],
    }

    mock_response_body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(mock_plan_dict),
                }
            }
        ]
    }

    async def mock_post(self, url, json=None, headers=None):
        return httpx.Response(200, json=mock_response_body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    planner = LLMResearchPlanner(api_key="test-key", max_sub_questions=5)
    plan = await planner.create_plan("What is quantum computing?")

    assert isinstance(plan, ResearchPlan)
    assert plan.original_question == "What is quantum computing?"
    assert len(plan.sub_questions) == 2
    assert plan.sub_questions[0].id == "q1"
    assert plan.sub_questions[1].id == "q2"


@pytest.mark.anyio
async def test_planner_exceeding_max_sub_questions_raises_error(monkeypatch):
    """Adjustment 1: Validate and raise PlannerResponseError if count exceeds configured max."""
    mock_plan_dict = {
        "original_question": "What is fusion energy?",
        "sub_questions": [
            {
                "id": f"q{i}",
                "question": f"Sub question {i}",
                "search_query": f"search query {i}",
                "reason": f"reason {i}",
            }
            for i in range(1, 5)  # 4 sub-questions
        ],
    }

    mock_response_body = {
        "choices": [{"message": {"content": json.dumps(mock_plan_dict)}}]
    }

    async def mock_post(self, url, json=None, headers=None):
        return httpx.Response(200, json=mock_response_body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # Set max_sub_questions to 2: 4 > 2 must raise PlannerResponseError
    planner = LLMResearchPlanner(api_key="test-key", max_sub_questions=2)
    with pytest.raises(PlannerResponseError) as exc_info:
        await planner.create_plan("What is fusion energy?")
    assert "exceeding maximum of 2" in str(exc_info.value)


@pytest.mark.anyio
async def test_planner_malformed_json(monkeypatch):
    mock_response_body = {
        "choices": [{"message": {"content": "Not valid JSON at all!"}}]
    }

    async def mock_post(self, url, json=None, headers=None):
        return httpx.Response(200, json=mock_response_body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    planner = LLMResearchPlanner(api_key="test-key")
    with pytest.raises(PlannerResponseError) as exc_info:
        await planner.create_plan("What is fusion energy?")
    assert "Invalid or unparseable JSON" in str(exc_info.value)


@pytest.mark.anyio
async def test_planner_schema_validation_failure(monkeypatch):
    mock_plan_dict = {
        "original_question": "What is fusion energy?",
        "sub_questions": [
            {
                "id": "q1",
                # missing search_query and reason
                "question": "What is a tokamak?",
            }
        ],
    }
    mock_response_body = {
        "choices": [{"message": {"content": json.dumps(mock_plan_dict)}}]
    }

    async def mock_post(self, url, json=None, headers=None):
        return httpx.Response(200, json=mock_response_body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    planner = LLMResearchPlanner(api_key="test-key")
    with pytest.raises(PlannerResponseError) as exc_info:
        await planner.create_plan("What is fusion energy?")
    assert "does not conform to ResearchPlan schema" in str(exc_info.value)


@pytest.mark.anyio
async def test_planner_timeout(monkeypatch):
    async def mock_post(self, url, json=None, headers=None):
        raise httpx.TimeoutException("Request timed out")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    planner = LLMResearchPlanner(api_key="test-key", timeout_seconds=10.0)
    with pytest.raises(PlannerTimeoutError) as exc_info:
        await planner.create_plan("What is fusion energy?")
    assert "timed out" in str(exc_info.value)


@pytest.mark.anyio
async def test_planner_network_error(monkeypatch):
    async def mock_post(self, url, json=None, headers=None):
        raise httpx.NetworkError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    planner = LLMResearchPlanner(api_key="test-key")
    with pytest.raises(PlannerNetworkError) as exc_info:
        await planner.create_plan("What is fusion energy?")
    assert "Network transport error" in str(exc_info.value)


@pytest.mark.anyio
@pytest.mark.parametrize("status_code,expected_exc", [
    (401, PlannerConfigError),
    (429, PlannerNetworkError),
    (500, PlannerNetworkError),
    (503, PlannerNetworkError),
    (418, PlannerResponseError),
])
async def test_planner_http_errors(monkeypatch, status_code, expected_exc):
    async def mock_post(self, url, json=None, headers=None):
        return httpx.Response(status_code, text="Error payload", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    planner = LLMResearchPlanner(api_key="test-key")
    with pytest.raises(expected_exc):
        await planner.create_plan("What is fusion energy?")
