from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """Normalized evidence item derived deterministically from a retrieved chunk."""

    evidence_id: str = Field(..., description="Deterministic session-scoped identifier (e.g. 'E1', 'E2')")
    chunk_id: str = Field(..., description="UUID string of the underlying document chunk")
    document_id: str = Field(..., description="UUID string of the parent document")
    session_id: str = Field(..., description="UUID string of the research session")
    url: str = Field(..., description="Canonical URL of the source document")
    title: str = Field(..., description="Title of the source document")
    text: str = Field(..., description="Text content of the retrieved chunk")
    similarity: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
