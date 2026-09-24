import asyncio
from io import BytesIO
import logging
import re
from typing import List, Optional, Set
from bs4 import BeautifulSoup
import httpx
from pypdf import PdfReader

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


def sanitize_document_text(text: Optional[str], url: Optional[str] = None) -> str:
    """Sanitize document text by removing NUL characters (\x00 / U+0000), handling None/empty, and stripping whitespace."""
    if not text:
        return ""
    if "\x00" in text:
        count = text.count("\x00")
        logger.warning(
            "Sanitization removed %d NUL (\\x00) character(s) from document text%s",
            count,
            f" for {url}" if url else "",
        )
        text = text.replace("\x00", "")
    return text.strip()


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract visible text from PDF bytes page by page."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages)


class WebpageExtractor(BaseExtractor):
    """Asynchronous webpage and PDF text extractor."""

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
        """Fetch a webpage or PDF document and extract clean readable visible text."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.9,*/*;q=0.8",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, follow_redirects=True
            ) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                content_bytes = response.content
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

        # Detect PDF responses via Content-Type header, magic bytes (%PDF-), or URL extension
        is_pdf = (
            "application/pdf" in content_type
            or "application/x-pdf" in content_type
            or content_bytes[:5] == b"%PDF-"
            or url.lower().split("?")[0].endswith(".pdf")
        )

        if is_pdf:
            logger.info(
                "Detected PDF content from %s (content-type: '%s', size: %d bytes)",
                url,
                content_type,
                len(content_bytes),
            )
            return self.parse_pdf(
                pdf_bytes=content_bytes,
                url=url,
                fallback_title=title,
                score=score,
            )

        html_content = response.text
        return self.parse_html(
            html_content=html_content,
            url=url,
            fallback_title=title,
            score=score,
        )

    def parse_pdf(
        self,
        pdf_bytes: bytes,
        url: str,
        fallback_title: Optional[str] = None,
        score: Optional[float] = None,
    ) -> Document:
        """Parse raw PDF bytes, extract page-by-page text, sanitize, and validate."""
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            pages_text: List[str] = []
            total_chars = 0
            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                    if page_text:
                        pages_text.append(page_text)
                        total_chars += len(page_text)
                        if total_chars >= self.max_chars:
                            logger.debug(
                                "Reached max_chars limit (%d) at page %d of PDF %s, stopping page extraction.",
                                self.max_chars,
                                page_idx,
                                url,
                            )
                            break
                except Exception as p_exc:
                    logger.warning(
                        "Error extracting text from page %d of PDF %s: %s",
                        page_idx,
                        url,
                        p_exc,
                    )
            raw_text = "\n\n".join(pages_text)
            total_pages = len(reader.pages)
        except Exception as exc:
            logger.error("Failed to parse PDF binary from %s: %s", url, exc)
            raise ExtractionContentError(
                f"Failed to parse PDF document from {url}: {exc}"
            ) from exc

        # Extract title from PDF metadata if available
        page_title = fallback_title
        if not page_title:
            try:
                if reader.metadata and reader.metadata.title:
                    meta_title = sanitize_document_text(str(reader.metadata.title), url=url)
                    if meta_title and len(meta_title) > 2:
                        page_title = meta_title
            except Exception:
                pass

        if not page_title:
            # Fall back to URL filename
            filename = url.split("?")[0].rstrip("/").split("/")[-1]
            if filename.lower().endswith(".pdf"):
                filename = filename[:-4].replace("-", " ").replace("_", " ").title()
            page_title = filename if filename else "Untitled PDF Document"

        clean_text = normalize_text_whitespace(raw_text)
        clean_text = sanitize_document_text(clean_text, url=url)
        page_title = sanitize_document_text(page_title, url=url) or "Untitled PDF Document"

        if len(clean_text) < MIN_TEXT_CHARS:
            logger.warning(
                "PDF at %s yielded insufficient text (%d chars < min %d)",
                url,
                len(clean_text),
                MIN_TEXT_CHARS,
            )
            raise ExtractionContentError(
                f"PDF at {url} does not contain sufficient usable text content."
            )

        if len(clean_text) > self.max_chars:
            clean_text = clean_text[: self.max_chars].rstrip()

        char_count = len(clean_text)
        logger.info(
            "PDF text extraction succeeded for %s: extracted %d characters across %d page(s)",
            url,
            char_count,
            total_pages,
        )

        return Document(
            url=url,
            title=page_title,
            text=clean_text,
            score=score,
            char_count=char_count,
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
        clean_text = sanitize_document_text(clean_text, url=url)
        page_title = sanitize_document_text(page_title, url=url) or "Untitled"

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
