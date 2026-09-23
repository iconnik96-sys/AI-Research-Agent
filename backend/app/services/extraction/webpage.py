import asyncio
import logging
import re
from typing import List, Optional, Set
from bs4 import BeautifulSoup
import httpx

from app.core.config import settings
from app.schemas.document import Document
from app.schemas.research import SourceItem
from app.services.extraction.base import BaseExtractor
from app.services.extraction.exceptions import (
    ExtractionContentError,
    FetchNetworkError,
    FetchTimeoutError,
    InvalidResponseError,
)

logger = logging.getLogger(__name__)

DISCARD_TAGS: Set[str] = {
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "noscript",
    "aside",
    "form",
    "svg",
}
MIN_TEXT_CHARS: int = 40


def normalize_text_whitespace(text: str) -> str:
    """Normalize whitespace by collapsing spaces and excess blank lines."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


class WebpageExtractor(BaseExtractor):
    """Asynchronous webpage text extractor using BeautifulSoup4."""

    def __init__(
        self,
        timeout_seconds: Optional[float] = None,
        max_chars: Optional[int] = None,
        user_agent: Optional[str] = None,
    ):
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.EXTRACTION_TIMEOUT_SECONDS
        )
        self.max_chars = (
            max_chars if max_chars is not None else settings.EXTRACTION_MAX_CHARS
        )
        self.user_agent = (
            user_agent if user_agent is not None else settings.EXTRACTION_USER_AGENT
        )

    async def extract(
        self,
        url: str,
        title: Optional[str] = None,
        score: Optional[float] = None,
    ) -> Document:
        """Fetch a webpage and extract clean readable visible text."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, follow_redirects=True
            ) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                html_content = response.text
        except httpx.TimeoutException as exc:
            raise FetchTimeoutError(
                f"Timed out fetching {url} after {self.timeout_seconds}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise InvalidResponseError(
                f"HTTP {exc.response.status_code} while fetching {url}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise FetchNetworkError(
                f"Network error while fetching {url}: {str(exc)}"
            ) from exc

        return self.parse_html(
            html_content=html_content,
            url=url,
            fallback_title=title,
            score=score,
        )

    def parse_html(
        self,
        html_content: str,
        url: str,
        fallback_title: Optional[str] = None,
        score: Optional[float] = None,
    ) -> Document:
        """Parse raw HTML, remove noise, and normalize text."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title from HTML if not provided or empty
        page_title = fallback_title
        if not page_title:
            if soup.title and soup.title.string:
                page_title = soup.title.string.strip()
            elif soup.h1:
                page_title = soup.h1.get_text().strip()
            else:
                page_title = "Untitled"

        # Remove non-content elements
        for tag in soup.find_all(DISCARD_TAGS):
            tag.decompose()

        # Extract visible text
        raw_text = soup.get_text(separator="\n")
        clean_text = normalize_text_whitespace(raw_text)

        if len(clean_text) < MIN_TEXT_CHARS:
            raise ExtractionContentError(
                f"Webpage at {url} does not contain sufficient usable text content."
            )

        # Enforce configurable maximum document size
        if len(clean_text) > self.max_chars:
            clean_text = clean_text[: self.max_chars].rstrip()

        return Document(
            url=url,
            title=page_title or "Untitled",
            text=clean_text,
            score=score,
            char_count=len(clean_text),
        )

    async def extract_many(
        self,
        sources: List[SourceItem],
    ) -> List[Document]:
        """Concurrently extract documents from multiple sources.

        Individual fetch or parsing failures are caught and logged, ensuring
        a single broken source does not fail the overall research request.
        """
        if not sources:
            return []

        async def _safe_extract(source: SourceItem) -> Optional[Document]:
            try:
                return await self.extract(
                    url=source.url,
                    title=source.title,
                    score=source.score,
                )
            except Exception as exc:
                logger.warning(
                    "Extraction failed for source %s: %s",
                    source.url,
                    exc,
                )
                return None

        tasks = [_safe_extract(s) for s in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents: List[Document] = []
        for res in results:
            if isinstance(res, Document):
                documents.append(res)
            elif isinstance(res, Exception):
                logger.warning("Extraction task raised unexpected exception: %s", res)

        return documents
