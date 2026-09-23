import pytest
import httpx
from unittest.mock import AsyncMock, patch

from app.schemas.research import SourceItem
from app.services.extraction.exceptions import (
    ExtractionContentError,
    FetchNetworkError,
    FetchTimeoutError,
    InvalidResponseError,
)
from app.services.extraction.webpage import WebpageExtractor


@pytest.mark.anyio
async def test_successful_html_extraction():
    extractor = WebpageExtractor(timeout_seconds=5.0)

    html_content = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Breakthrough Fusion Experiment Results</title>
      </head>
      <body>
        <h1>Net Energy Gain Demonstrated</h1>
        <p>Scientists and engineers at the National Ignition Facility announced significant net energy gain from controlled inertial confinement fusion.</p>
        <p>The experiment generated 3.15 megajoules of energy from an input of 2.05 megajoules to the target capsule.</p>
      </body>
    </html>
    """

    mock_response = httpx.Response(
        status_code=200,
        text=html_content,
        request=httpx.Request("GET", "https://example.com/fusion"),
    )

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        doc = await extractor.extract(
            url="https://example.com/fusion",
            title=None,
            score=0.96,
        )

        assert doc.url == "https://example.com/fusion"
        assert doc.title == "Breakthrough Fusion Experiment Results"
        assert "Net Energy Gain Demonstrated" in doc.text
        assert "3.15 megajoules" in doc.text
        assert doc.score == 0.96
        assert doc.char_count == len(doc.text)
        assert doc.char_count > 50


@pytest.mark.anyio
async def test_script_style_nav_footer_removal():
    extractor = WebpageExtractor()

    html_content = """
    <html>
      <head>
        <style>body { background: #000; display: none; }</style>
        <script>console.log("tracking code");</script>
      </head>
      <body>
        <header><p>Site Header Navigation Banner</p></header>
        <nav><a href="/">Home</a><a href="/about">About</a></nav>
        <main>
          <h1>Real Core Content Here</h1>
          <p>This is the actual article text that researchers care about reading and extracting for RAG pipelines.</p>
        </main>
        <aside><p>Sidebar Advertisements</p></aside>
        <footer><p>Copyright 2026 Corporation. All rights reserved.</p></footer>
        <noscript><p>Please enable javascript to view this site.</p></noscript>
      </body>
    </html>
    """

    doc = extractor.parse_html(
        html_content=html_content,
        url="https://example.com/clean-test",
    )

    assert "Real Core Content Here" in doc.text
    assert "actual article text that researchers care about" in doc.text
    assert "background: #000" not in doc.text
    assert "tracking code" not in doc.text
    assert "Site Header Navigation Banner" not in doc.text
    assert "Home" not in doc.text
    assert "Sidebar Advertisements" not in doc.text
    assert "Copyright 2026" not in doc.text
    assert "Please enable javascript" not in doc.text


def test_whitespace_normalization():
    extractor = WebpageExtractor()

    html_content = """
    <html>
      <body>
        <h1>   Heading    With   Multiple     Spaces   </h1>
        <p>Paragraph     one   with      extra    spaces.</p>
        <br><br><br><br><br>
        <p>Paragraph     two    after    lots   of    blank     lines.</p>
      </body>
    </html>
    """

    doc = extractor.parse_html(
        html_content=html_content,
        url="https://example.com/whitespace",
    )

    assert "Heading With Multiple Spaces" in doc.text
    assert "Paragraph one with extra spaces." in doc.text
    assert "Paragraph two after lots of blank lines." in doc.text
    # Should not have 3 or more consecutive newlines
    assert "\n\n\n" not in doc.text


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [403, 404, 500])
async def test_http_error_handling(status_code):
    extractor = WebpageExtractor()

    mock_response = httpx.Response(
        status_code=status_code,
        text="Error page",
        request=httpx.Request("GET", "https://example.com/error"),
    )

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        with pytest.raises(InvalidResponseError) as exc_info:
            await extractor.extract("https://example.com/error")
        assert exc_info.value.status_code == status_code


@pytest.mark.anyio
async def test_fetch_timeout():
    extractor = WebpageExtractor(timeout_seconds=2.0)

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.TimeoutException("Connection timed out")
        with pytest.raises(FetchTimeoutError, match="Timed out"):
            await extractor.extract("https://example.com/slow")


@pytest.mark.anyio
async def test_fetch_network_error():
    extractor = WebpageExtractor()

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        with pytest.raises(FetchNetworkError, match="Network error"):
            await extractor.extract("https://example.com/down")


def test_empty_unusable_page():
    extractor = WebpageExtractor()

    html_content = "<html><body><div>     </div></body></html>"
    with pytest.raises(ExtractionContentError, match="does not contain sufficient usable text"):
        extractor.parse_html(html_content, "https://example.com/empty")


def test_oversized_document_handling():
    extractor = WebpageExtractor(max_chars=120)

    long_text = "This is a long sentence providing valuable scientific context for research. " * 10
    html_content = f"<html><body><h1>Title</h1><p>{long_text}</p></body></html>"

    doc = extractor.parse_html(html_content, "https://example.com/oversized")
    assert len(doc.text) == 120
    assert doc.char_count == 120


@pytest.mark.anyio
async def test_extract_many_fault_tolerance():
    extractor = WebpageExtractor()

    sources = [
        SourceItem(
            title="Good Source",
            url="https://example.com/good",
            content="Summary snippet",
            score=0.9,
        ),
        SourceItem(
            title="Dead Link",
            url="https://example.com/broken",
            content="Summary snippet",
            score=0.5,
        ),
    ]

    async def mock_extract(url, title=None, score=None):
        if "good" in url:
            return extractor.parse_html(
                "<html><body><h1>Good Page</h1><p>Meaningful valid content extracted cleanly from good page.</p></body></html>",
                url,
                title,
                score,
            )
        raise InvalidResponseError("HTTP 404", status_code=404)

    with patch.object(extractor, "extract", side_effect=mock_extract):
        docs = await extractor.extract_many(sources)
        # One failed, but the overall batch still succeeds with the good document
        assert len(docs) == 1
        assert docs[0].url == "https://example.com/good"
        assert "Good Page" in docs[0].text
