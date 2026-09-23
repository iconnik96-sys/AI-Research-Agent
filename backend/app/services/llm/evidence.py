from typing import List, Tuple

from app.schemas.document import Document
from app.schemas.report import SourceReference
from app.schemas.retrieval import RetrievedChunk


def prepare_evidence(
    documents: List[Document],
    max_chars_per_doc: int = 4000,
) -> Tuple[List[SourceReference], str]:
    """Prepare and format extracted documents into deterministic evidence for LLM prompt.

    Assigns deterministic source IDs (S1, S2, ...), limits per-document character length,
    and returns both the list of SourceReferences and the formatted text context.

    Args:
        documents: List of extracted Document objects.
        max_chars_per_doc: Maximum character limit for each document text.

    Returns:
        Tuple containing:
        - List[SourceReference]: Normalized source metadata with assigned IDs.
        - str: Cleanly formatted evidence block to pass into LLM context.
    """
    if not documents:
        return [], ""

    source_references: List[SourceReference] = []
    evidence_blocks: List[str] = []

    for index, doc in enumerate(documents, start=1):
        source_id = f"S{index}"
        source_references.append(
            SourceReference(
                id=source_id,
                title=doc.title,
                url=doc.url,
            )
        )

        content = doc.text[:max_chars_per_doc].rstrip()
        block = (
            f"--- [Source ID: {source_id}] ---\n"
            f"Title: {doc.title}\n"
            f"URL: {doc.url}\n"
            f"Content:\n{content}"
        )
        evidence_blocks.append(block)

    formatted_context = "\n\n".join(evidence_blocks)
    return source_references, formatted_context


def prepare_evidence_from_chunks(
    chunks: List[RetrievedChunk],
    max_chars_per_chunk: int = 2000,
) -> Tuple[List[SourceReference], str]:
    """Prepare and format retrieved chunks into deterministic evidence for LLM prompt.

    Groups chunks by unique source (URL and title), assigns deterministic source IDs
    (S1, S2, ...), formats excerpts under each source, and avoids duplicate context.

    Args:
        chunks: List of RetrievedChunk objects from semantic retrieval.
        max_chars_per_chunk: Maximum characters allowed per retrieved chunk.

    Returns:
        Tuple containing:
        - List[SourceReference]: Normalized source metadata with assigned IDs.
        - str: Cleanly formatted evidence block to pass into LLM context.
    """
    if not chunks:
        return [], ""

    # Group chunks by unique source URL (preserving first-seen ordering based on best similarity)
    sources_dict: dict = {}
    for chunk in chunks:
        if chunk.url not in sources_dict:
            sources_dict[chunk.url] = {
                "title": chunk.title,
                "chunks": [],
            }
        sources_dict[chunk.url]["chunks"].append(chunk)

    source_references: List[SourceReference] = []
    evidence_blocks: List[str] = []

    for index, (url, info) in enumerate(sources_dict.items(), start=1):
        source_id = f"S{index}"
        source_references.append(
            SourceReference(
                id=source_id,
                title=info["title"],
                url=url,
            )
        )

        chunk_excerpts: List[str] = []
        for c in info["chunks"]:
            excerpt = c.text[:max_chars_per_chunk].rstrip()
            chunk_excerpts.append(
                f"[Excerpt (chunk {c.chunk_index}, similarity: {c.similarity:.2f})]:\n{excerpt}"
            )

        combined_excerpts = "\n\n".join(chunk_excerpts)
        block = (
            f"--- [Source ID: {source_id}] ---\n"
            f"Title: {info['title']}\n"
            f"URL: {url}\n"
            f"Relevant Evidence:\n{combined_excerpts}"
        )
        evidence_blocks.append(block)

    formatted_context = "\n\n".join(evidence_blocks)
    return source_references, formatted_context
