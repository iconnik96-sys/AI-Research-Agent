import pytest

from app.schemas.chunk import DocumentChunk
from app.schemas.document import Document
from app.services.chunking.text_chunker import TextChunker, get_text_chunker


def test_chunker_invalid_config():
    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        TextChunker(chunk_size=0, chunk_overlap=10)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        TextChunker(chunk_size=100, chunk_overlap=-1)

    with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=150)


def test_normalize_text():
    raw = "  Hello   \x00world!  \r\n\r\nThis is a   test.\n\n\n\nAnother paragraph.  "
    normalized = TextChunker.normalize_text(raw)
    assert "\x00" not in normalized
    assert "\r" not in normalized
    assert "   " not in normalized
    assert "\n\n\n" not in normalized
    assert normalized.startswith("Hello world!")
    assert normalized.endswith("Another paragraph.")


def test_empty_or_whitespace_document():
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    doc_empty = Document(url="https://example.com/1", title="Empty", text="", char_count=0)
    assert chunker.chunk_document(doc_empty) == []

    doc_ws = Document(url="https://example.com/2", title="Whitespace", text="   \n\t   ", char_count=7)
    assert chunker.chunk_document(doc_ws) == []


def test_short_document_single_chunk():
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    short_text = "This is a brief article about nuclear fusion."
    doc = Document(url="https://example.com/fusion", title="Fusion", text=short_text, char_count=len(short_text))

    chunks = chunker.chunk_document(doc, document_id="doc-123", session_id="sess-456")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.document_id == "doc-123"
    assert chunk.session_id == "sess-456"
    assert chunk.chunk_index == 0
    assert chunk.text == short_text
    assert chunk.char_count == len(short_text)


def test_long_document_deterministic_chunking():
    chunker = TextChunker(chunk_size=100, chunk_overlap=30)
    # 250 characters composed of sentences
    text = (
        "Alpha beta gamma delta epsilon. "
        "Zeta eta theta iota kappa. "
        "Lambda mu nu xi omicron. "
        "Pi rho sigma tau upsilon. "
        "Phi chi psi omega finished."
    )
    doc = Document(url="https://example.com/greek", title="Greek", text=text, char_count=len(text))

    chunks_1 = chunker.chunk_document(doc, document_id="doc-1", session_id="sess-1")
    chunks_2 = chunker.chunk_document(doc, document_id="doc-1", session_id="sess-1")

    # Strict determinism
    assert len(chunks_1) > 1
    assert [c.model_dump() for c in chunks_1] == [c.model_dump() for c in chunks_2]

    # Verify chunk indexing and properties
    for i, c in enumerate(chunks_1):
        assert c.chunk_index == i
        assert c.document_id == "doc-1"
        assert c.session_id == "sess-1"
        assert len(c.text) > 0
        assert c.char_count == len(c.text)


def test_chunk_documents_batch_mapping():
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    docs = [
        Document(url="https://example.com/a", title="A", text="Doc A content " * 10, char_count=140),
        Document(url="https://example.com/b", title="B", text="Doc B content " * 10, char_count=140),
    ]
    doc_id_map = {
        "https://example.com/a": "uuid-doc-a",
        "https://example.com/b": "uuid-doc-b",
    }
    session_id = "uuid-session-1"

    all_chunks = chunker.chunk_documents(docs, document_id_map=doc_id_map, session_id=session_id)
    assert len(all_chunks) >= 2

    chunks_a = [c for c in all_chunks if c.document_id == "uuid-doc-a"]
    chunks_b = [c for c in all_chunks if c.document_id == "uuid-doc-b"]
    assert len(chunks_a) >= 1
    assert len(chunks_b) >= 1

    for c in chunks_a:
        assert c.session_id == session_id
        assert c.document_id == "uuid-doc-a"

    for c in chunks_b:
        assert c.session_id == session_id
        assert c.document_id == "uuid-doc-b"


def test_get_text_chunker_dependency():
    chunker = get_text_chunker()
    assert isinstance(chunker, TextChunker)
    assert chunker.chunk_size > 0
    assert chunker.chunk_overlap >= 0
