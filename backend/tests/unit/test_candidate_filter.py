import pytest
from app.schemas.chunk import DocumentChunk
from app.schemas.document import Document
from app.services.retrieval.candidate_filter import (
    CandidateChunkFilter,
    filter_candidate_chunks,
    tokenize,
)


def _make_chunk(doc_id: str, idx: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        document_id=doc_id,
        session_id="sess-001",
        chunk_index=idx,
        text=text,
        char_count=len(text),
    )


def test_tokenize_preserves_compounds_and_subparts():
    tokens = tokenize("Modern solid-state batteries achieve 300 Wh/kg in 2027-2030.")
    # Check that compounds are present
    assert "solid-state" in tokens
    assert "solid" in tokens
    assert "state" in tokens
    assert "wh/kg" in tokens
    assert "wh" in tokens
    assert "kg" in tokens
    assert "batteries" in tokens


def test_filter_returns_all_when_under_limit():
    chunks = [
        _make_chunk("doc-1", 0, "First chunk text about batteries."),
        _make_chunk("doc-1", 1, "Second chunk text about charging."),
    ]
    filtered = filter_candidate_chunks(
        chunks=chunks,
        queries=["batteries"],
        max_candidates=60,
        max_per_doc=6,
    )
    assert len(filtered) == 2
    assert filtered[0].text == chunks[0].text
    assert filtered[1].text == chunks[1].text


def test_filter_ranks_relevant_chunks_higher():
    c1 = _make_chunk("doc-1", 0, "The quick brown fox jumps over the lazy dog.")
    c2 = _make_chunk("doc-2", 0, "Solid-state batteries offer higher energy density than lithium-ion cells.")
    c3 = _make_chunk("doc-3", 0, "Unrelated paragraph about baking sourdough bread in an oven.")
    c4 = _make_chunk("doc-4", 0, "Thermal runaway in lithium-ion occurs when liquid electrolyte combusts.")

    filtered = filter_candidate_chunks(
        chunks=[c1, c2, c3, c4],
        queries=["solid-state batteries energy density", "thermal runaway lithium-ion"],
        max_candidates=2,
        max_per_doc=6,
    )

    assert len(filtered) == 2
    # c2 and c4 should be the top 2
    selected_texts = [c.text for c in filtered]
    assert c2.text in selected_texts
    assert c4.text in selected_texts
    assert c3.text not in selected_texts


def test_filter_respects_max_per_doc_cap():
    # 10 chunks from doc-1 (all highly relevant)
    chunks_doc1 = [
        _make_chunk("doc-1", i, f"Solid-state battery research paragraph {i} discussing EV range.")
        for i in range(10)
    ]
    # 5 chunks from doc-2 (also relevant)
    chunks_doc2 = [
        _make_chunk("doc-2", i, f"Solid-state electrolyte safety paragraph {i}.")
        for i in range(5)
    ]

    all_chunks = chunks_doc1 + chunks_doc2
    filtered = filter_candidate_chunks(
        chunks=all_chunks,
        queries=["solid-state battery EV range safety"],
        max_candidates=8,
        max_per_doc=4,
    )

    assert len(filtered) == 8
    doc1_count = sum(1 for c in filtered if c.document_id == "doc-1")
    doc2_count = sum(1 for c in filtered if c.document_id == "doc-2")

    # In pass 1, doc-1 is capped at 4, so doc-2 fills the remaining slots
    assert doc1_count <= 4
    assert doc2_count == 4


def test_filter_multi_query_composite_scoring():
    # c1 matches sub-question 1 strongly
    c1 = _make_chunk("doc-1", 0, "Gravimetric energy density reaches 450 Wh/kg in advanced solid-state prototypes.")
    # c2 matches sub-question 2 strongly
    c2 = _make_chunk("doc-2", 0, "Thermal runaway safety is improved because non-flammable solid electrolyte is used.")
    # c3 matches both queries moderately
    c3 = _make_chunk("doc-3", 0, "Solid-state batteries improve both energy density and thermal safety.")
    # c4 is irrelevant
    c4 = _make_chunk("doc-4", 0, "General history of automotive assembly lines and Henry Ford.")

    queries = [
        "What is the energy density of solid-state batteries?",
        "How does thermal runaway safety compare in solid-state cells?",
    ]

    filtered = filter_candidate_chunks(
        chunks=[c1, c2, c3, c4],
        queries=queries,
        max_candidates=3,
        max_per_doc=6,
    )

    assert len(filtered) == 3
    selected_ids = [c.document_id for c in filtered]
    assert "doc-1" in selected_ids
    assert "doc-2" in selected_ids
    assert "doc-3" in selected_ids
    assert "doc-4" not in selected_ids


def test_filter_fallback_when_matches_are_scarce():
    # Only 1 chunk has keywords, but max_candidates is 3
    c1 = _make_chunk("doc-1", 0, "Solid-state batteries breakthrough announced.")
    c2 = _make_chunk("doc-2", 0, "Apples and oranges are common fruits.")
    c3 = _make_chunk("doc-3", 0, "Rainfall patterns in South America during monsoon season.")
    c4 = _make_chunk("doc-4", 0, "Standard office furniture procurement guidelines.")

    filtered = filter_candidate_chunks(
        chunks=[c1, c2, c3, c4],
        queries=["solid-state batteries"],
        max_candidates=3,
        max_per_doc=6,
    )

    # Must return exactly 3 chunks via fallback without error
    assert len(filtered) == 3
    # c1 must be first
    assert filtered[0].document_id == "doc-1"


def test_filter_deterministic_ordering_on_ties():
    # Create chunks with identical text so scores are equal
    c1 = _make_chunk("doc-B", 1, "Identical content string for testing deterministic ordering.")
    c2 = _make_chunk("doc-A", 0, "Identical content string for testing deterministic ordering.")
    c3 = _make_chunk("doc-B", 0, "Identical content string for testing deterministic ordering.")
    c4 = _make_chunk("doc-A", 1, "Identical content string for testing deterministic ordering.")

    f1 = filter_candidate_chunks(
        chunks=[c1, c2, c3, c4],
        queries=["Identical content string"],
        max_candidates=3,
        max_per_doc=6,
    )

    f2 = filter_candidate_chunks(
        chunks=[c3, c4, c1, c2],  # reversed input order
        queries=["Identical content string"],
        max_candidates=3,
        max_per_doc=6,
    )

    # Output order must be identical regardless of input order:
    # Sorted by (-score, doc_id, chunk_index) -> doc-A chunk 0, doc-A chunk 1, doc-B chunk 0
    assert [(c.document_id, c.chunk_index) for c in f1] == [(c.document_id, c.chunk_index) for c in f2]
    assert f1[0].document_id == "doc-A" and f1[0].chunk_index == 0
    assert f1[1].document_id == "doc-A" and f1[1].chunk_index == 1
    assert f1[2].document_id == "doc-B" and f1[2].chunk_index == 0


def test_filter_preserves_original_metadata_and_instances():
    orig_chunk = _make_chunk("doc-99", 5, "Unique paragraph about lithium recycling processes.")
    orig_chunk.embedding = None

    filtered = filter_candidate_chunks(
        chunks=[orig_chunk],
        queries=["lithium recycling"],
        max_candidates=10,
        max_per_doc=6,
    )

    assert len(filtered) == 1
    # Identity / instance preservation
    assert filtered[0] is orig_chunk
    assert filtered[0].document_id == "doc-99"
    assert filtered[0].chunk_index == 5
    assert filtered[0].char_count == len(orig_chunk.text)
