import pytest
import httpx
from unittest.mock import AsyncMock, patch

from app.services.search.base import (
    SearchConfigError,
    SearchProviderError,
    SearchTimeoutError,
)
from app.services.search.tavily import TavilySearchProvider


@pytest.mark.anyio
async def test_tavily_search_success():
    provider = TavilySearchProvider(api_key="tvly-mock-test-key", timeout_seconds=5.0)

    mock_response = httpx.Response(
        status_code=200,
        json={
            "query": "fusion breakthroughs",
            "results": [
                {
                    "title": "Major Fusion Breakthrough",
                    "url": "https://example.com/fusion",
                    "content": "Researchers achieved net energy gain.",
                    "score": 0.95,
                },
                {
                    "title": "Fusion Energy Explained",
                    "url": "https://example.com/explain",
                    "content": "A comprehensive guide to nuclear fusion.",
                    "score": 0.88,
                },
            ],
        },
        request=httpx.Request("POST", "https://api.tavily.com/search"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        sources = await provider.search("fusion breakthroughs", max_results=2)

        assert len(sources) == 2
        assert sources[0].title == "Major Fusion Breakthrough"
        assert sources[0].url == "https://example.com/fusion"
        assert sources[0].content == "Researchers achieved net energy gain."
        assert sources[0].score == 0.95

        assert sources[1].title == "Fusion Energy Explained"
        assert sources[1].url == "https://example.com/explain"

        mock_post.assert_awaited_once()


@pytest.mark.anyio
async def test_tavily_search_missing_api_key():
    provider = TavilySearchProvider(api_key="")
    with pytest.raises(SearchConfigError, match="Tavily API key is not configured"):
        await provider.search("fusion energy")


@pytest.mark.anyio
async def test_tavily_search_timeout():
    provider = TavilySearchProvider(api_key="tvly-mock-key", timeout_seconds=2.0)

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Connection timed out")
        with pytest.raises(SearchTimeoutError, match="timed out"):
            await provider.search("fusion energy")


@pytest.mark.anyio
async def test_tavily_search_http_error():
    provider = TavilySearchProvider(api_key="tvly-mock-key")

    mock_response = httpx.Response(
        status_code=401,
        text="Unauthorized - invalid API key",
        request=httpx.Request("POST", "https://api.tavily.com/search"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(SearchProviderError, match="HTTP 401"):
            await provider.search("fusion energy")


@pytest.mark.anyio
async def test_tavily_search_network_error():
    provider = TavilySearchProvider(api_key="tvly-mock-key")

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.RequestError("Network unreachable")
        with pytest.raises(SearchProviderError, match="Network error"):
            await provider.search("fusion energy")


@pytest.mark.anyio
async def test_tavily_search_empty_results():
    provider = TavilySearchProvider(api_key="tvly-mock-key")

    mock_response = httpx.Response(
        status_code=200,
        json={"query": "obscure question", "results": []},
        request=httpx.Request("POST", "https://api.tavily.com/search"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        sources = await provider.search("obscure question")
        assert sources == []
