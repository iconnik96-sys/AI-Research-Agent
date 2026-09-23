import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_endpoint_success(client):
    payload = {"question": "What are the latest breakthroughs in fusion energy?"}
    response = client.post("/api/research", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == payload["question"]
    assert data["status"] == "received"
    assert "message" in data


def test_research_endpoint_missing_question(client):
    response = client.post("/api/research", json={})
    assert response.status_code == 422


def test_research_endpoint_empty_question(client):
    response = client.post("/api/research", json={"question": ""})
    assert response.status_code == 422
