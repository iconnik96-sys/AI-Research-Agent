from typing import List
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.research import SourceItem
from app.services.search.base import (
    BaseSearchProvider,
    SearchConfigError,
    SearchProviderError,
    SearchTimeoutError,
)
from app.services.search.factory import get_search_provider


class MockSearchProvider(BaseSearchProvider):
    """Mock search provider for API integration tests."""

    def __init__(self, sources: List[SourceItem] = None, error: Exception = None):
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


@pytest.fixture
def client():
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_endpoint_success(client):
    mock_provider = MockSearchProvider()
    app.dependency_overrides[get_search_provider] = lambda: mock_provider

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["question"] == payload["question"]
    assert data["status"] == "completed"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["title"] == "Mock Source 1"
    assert data["sources"][0]["url"] == "https://example.com/source1"


def test_research_endpoint_missing_api_key(client):
    mock_provider = MockSearchProvider(
        error=SearchConfigError("Tavily API key is not configured. Please set TAVILY_API_KEY.")
    )
    app.dependency_overrides[get_search_provider] = lambda: mock_provider

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 503
    assert "Tavily API key is not configured" in response.json()["detail"]


def test_research_endpoint_timeout(client):
    mock_provider = MockSearchProvider(
        error=SearchTimeoutError("Search request timed out after 10.0s")
    )
    app.dependency_overrides[get_search_provider] = lambda: mock_provider

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 504


def test_research_endpoint_provider_error(client):
    mock_provider = MockSearchProvider(
        error=SearchProviderError("Upstream API error HTTP 500")
    )
    app.dependency_overrides[get_search_provider] = lambda: mock_provider

    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 502


def test_research_endpoint_missing_question(client):
    response = client.post("/api/research", json={})
    assert response.status_code == 422


def test_research_endpoint_empty_question(client):
    response = client.post("/api/research", json={"question": ""})
    assert response.status_code == 422
