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
from app.services.embedding.provider import SupabaseEmbeddingProvider


@pytest.mark.anyio
async def test_embed_empty_texts():
    provider = SupabaseEmbeddingProvider(function_url="https://mock.supabase.co/functions/v1/embed")
    result = await provider.embed_texts([])
    assert result == []


@pytest.mark.anyio
async def test_missing_function_url_raises_config_error(monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_EMBEDDING_FUNCTION_URL", "")
    monkeypatch.setattr(settings, "SUPABASE_URL", "")
    provider = SupabaseEmbeddingProvider(function_url="", supabase_url="")
    with pytest.raises(EmbeddingConfigError, match="Supabase embedding function URL is not configured"):
        await provider.embed_texts(["sample text"])


@pytest.mark.anyio
async def test_successful_embed_texts(monkeypatch):
    dummy_vec_1 = [0.1] * 384
    dummy_vec_2 = [0.2] * 384

    async def mock_post(self, url, json=None, headers=None):
        assert json["input"] == ["first text", "second text"]
        assert headers["Authorization"] == "Bearer mock-anon-key"
        assert headers["apikey"] == "mock-anon-key"
        return httpx.Response(
            200,
            json={
                "embeddings": [dummy_vec_1, dummy_vec_2],
            },
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = SupabaseEmbeddingProvider(
        function_url="https://mock.supabase.co/functions/v1/embed",
        anon_key="mock-anon-key",
        dimensions=384,
    )
    results = await provider.embed_texts(["first text", "second text"])

    assert len(results) == 2
    assert len(results[0]) == 384
    assert len(results[1]) == 384
    assert results[0] == dummy_vec_1
    assert results[1] == dummy_vec_2


@pytest.mark.anyio
async def test_embed_single_text(monkeypatch):
    dummy_vec = [0.42] * 384

    async def mock_post(self, url, json=None, headers=None):
        return httpx.Response(
            200,
            json={"embeddings": [dummy_vec]},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = SupabaseEmbeddingProvider(
        function_url="https://mock.supabase.co/functions/v1/embed",
        dimensions=384,
    )
    vec = await provider.embed_text("single text")
    assert vec == dummy_vec
    assert len(vec) == 384


@pytest.mark.anyio
async def test_dimension_mismatch_raises_error(monkeypatch):
    async def mock_post(self, url, json=None, headers=None):
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2]]},  # 2 dims instead of 384
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = SupabaseEmbeddingProvider(
        function_url="https://mock.supabase.co/functions/v1/embed",
        dimensions=384,
    )
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

    provider = SupabaseEmbeddingProvider(
        function_url="https://mock.supabase.co/functions/v1/embed",
        dimensions=384,
    )
    with pytest.raises(EmbeddingResponseError, match="missing valid 'embeddings' array"):
        await provider.embed_texts(["text"])


@pytest.mark.anyio
async def test_timeout_raises_embedding_timeout_error(monkeypatch):
    async def mock_post(self, url, json=None, headers=None):
        raise httpx.TimeoutException("Connection timed out")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = SupabaseEmbeddingProvider(
        function_url="https://mock.supabase.co/functions/v1/embed",
        timeout_seconds=5.0,
    )
    with pytest.raises(EmbeddingTimeoutError, match="timed out"):
        await provider.embed_texts(["text"])


@pytest.mark.anyio
async def test_network_error_raises_embedding_network_error(monkeypatch):
    async def mock_post(self, url, json=None, headers=None):
        raise httpx.ConnectError("DNS resolution failed")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = SupabaseEmbeddingProvider(
        function_url="https://mock.supabase.co/functions/v1/embed"
    )
    with pytest.raises(EmbeddingNetworkError, match="Network error"):
        await provider.embed_texts(["text"])


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [401, 404, 500])
async def test_provider_http_error(monkeypatch, status_code):
    async def mock_post(self, url, json=None, headers=None):
        return httpx.Response(
            status_code,
            text=f"Error {status_code}",
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = SupabaseEmbeddingProvider(
        function_url="https://mock.supabase.co/functions/v1/embed"
    )
    with pytest.raises(EmbeddingProviderError) as exc_info:
        await provider.embed_texts(["text"])
    assert exc_info.value.status_code == status_code
