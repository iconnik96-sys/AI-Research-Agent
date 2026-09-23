import pytest
import httpx

from app.core.config import settings
from app.services.embedding.exceptions import (
    EmbeddingConfigError,
    EmbeddingNetworkError,
    EmbeddingProviderError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
)
from app.services.embedding.provider import OpenAICompatibleEmbeddingProvider


@pytest.mark.anyio
async def test_embed_empty_texts():
    provider = OpenAICompatibleEmbeddingProvider(api_key="test-key")
    result = await provider.embed_texts([])
    assert result == []


@pytest.mark.anyio
async def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_API_KEY", "")
    provider = OpenAICompatibleEmbeddingProvider(
        api_key="",
        base_url="https://api.openai.com/v1",
    )
    with pytest.raises(EmbeddingConfigError, match="EMBEDDING_API_KEY"):
        await provider.embed_texts(["sample text"])


@pytest.mark.anyio
async def test_successful_embed_texts(monkeypatch):
    dummy_vec_1 = [0.1] * 1536
    dummy_vec_2 = [0.2] * 1536

    async def mock_post(self, url, json=None, headers=None):
        assert json["model"] == "text-embedding-3-small"
        assert json["input"] == ["first text", "second text"]
        assert headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": dummy_vec_1},
                    {"object": "embedding", "index": 1, "embedding": dummy_vec_2},
                ],
                "model": "text-embedding-3-small",
            },
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = OpenAICompatibleEmbeddingProvider(
        api_key="test-key",
        model="text-embedding-3-small",
        dimensions=1536,
    )
    results = await provider.embed_texts(["first text", "second text"])

    assert len(results) == 2
    assert results[0] == dummy_vec_1
    assert results[1] == dummy_vec_2


@pytest.mark.anyio
async def test_embed_texts_preserves_order_when_scrambled(monkeypatch):
    dummy_vec_0 = [0.1] * 4
    dummy_vec_1 = [0.2] * 4

    async def mock_post(self, url, json=None, headers=None):
        # Return out of order: index 1 before index 0
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": dummy_vec_1},
                    {"index": 0, "embedding": dummy_vec_0},
                ]
            },
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = OpenAICompatibleEmbeddingProvider(
        api_key="test-key",
        dimensions=4,
    )
    results = await provider.embed_texts(["zero", "one"])
    assert results[0] == dummy_vec_0
    assert results[1] == dummy_vec_1


@pytest.mark.anyio
async def test_embed_single_text(monkeypatch):
    dummy_vec = [0.42] * 1536

    async def mock_post(self, url, json=None, headers=None):
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": dummy_vec}]},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = OpenAICompatibleEmbeddingProvider(api_key="test-key", dimensions=1536)
    vec = await provider.embed_text("single text")
    assert vec == dummy_vec


@pytest.mark.anyio
async def test_dimension_mismatch_raises_error(monkeypatch):
    async def mock_post(self, url, json=None, headers=None):
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},  # 2 dims instead of 1536
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = OpenAICompatibleEmbeddingProvider(api_key="test-key", dimensions=1536)
    with pytest.raises(EmbeddingResponseError, match="dimension mismatch"):
        await provider.embed_texts(["text"])


@pytest.mark.anyio
async def test_malformed_response_payload(monkeypatch):
    async def mock_post(self, url, json=None, headers=None):
        return httpx.Response(
            200,
            json={"wrong_field": []},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = OpenAICompatibleEmbeddingProvider(api_key="test-key", dimensions=1536)
    with pytest.raises(EmbeddingResponseError, match="missing valid 'data' array"):
        await provider.embed_texts(["text"])


@pytest.mark.anyio
async def test_timeout_raises_embedding_timeout_error(monkeypatch):
    async def mock_post(self, url, json=None, headers=None):
        raise httpx.TimeoutException("Connection timed out")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = OpenAICompatibleEmbeddingProvider(api_key="test-key", timeout_seconds=5.0)
    with pytest.raises(EmbeddingTimeoutError, match="timed out"):
        await provider.embed_texts(["text"])


@pytest.mark.anyio
async def test_network_error_raises_embedding_network_error(monkeypatch):
    async def mock_post(self, url, json=None, headers=None):
        raise httpx.ConnectError("DNS resolution failed")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = OpenAICompatibleEmbeddingProvider(api_key="test-key")
    with pytest.raises(EmbeddingNetworkError, match="Network error"):
        await provider.embed_texts(["text"])


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [401, 429, 500])
async def test_provider_http_error(monkeypatch, status_code):
    async def mock_post(self, url, json=None, headers=None):
        return httpx.Response(
            status_code,
            text=f"Error {status_code}",
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = OpenAICompatibleEmbeddingProvider(api_key="test-key")
    with pytest.raises(EmbeddingProviderError) as exc_info:
        await provider.embed_texts(["text"])
    assert exc_info.value.status_code == status_code
