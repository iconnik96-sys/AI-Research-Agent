from typing import List, Tuple

from app.schemas.document import Document
from app.schemas.report import SourceReference


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
