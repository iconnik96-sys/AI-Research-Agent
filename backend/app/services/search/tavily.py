from typing import List, Optional
import httpx

from app.core.config import settings
from app.schemas.research import SourceItem
from app.services.search.base import (
    BaseSearchProvider,
    SearchConfigError,
    SearchProviderError,
    SearchTimeoutError,
)

TAVILY_API_URL = "https://api.tavily.com/search"


class TavilySearchProvider(BaseSearchProvider):
    """Tavily search provider implementation using httpx."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.TAVILY_API_KEY
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.SEARCH_TIMEOUT_SECONDS
        )

    async def search(self, query: str, max_results: int = 5) -> List[SourceItem]:
        if not self.api_key or not self.api_key.strip():
            raise SearchConfigError(
                "Tavily API key is not configured. Please set TAVILY_API_KEY in your environment."
            )

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    TAVILY_API_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise SearchTimeoutError(
                f"Tavily search request timed out after {self.timeout_seconds}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise SearchProviderError(
                f"Tavily API error HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise SearchProviderError(
                f"Network error connecting to Tavily: {str(exc)}"
            ) from exc
        except Exception as exc:
            raise SearchProviderError(
                f"Unexpected error processing Tavily response: {str(exc)}"
            ) from exc

        raw_results = data.get("results", [])
        sources: List[SourceItem] = []
        for item in raw_results:
            sources.append(
                SourceItem(
                    title=item.get("title") or "Untitled",
                    url=item.get("url") or "",
                    content=item.get("content") or item.get("snippet") or "",
                    score=item.get("score"),
                )
            )

        return sources
