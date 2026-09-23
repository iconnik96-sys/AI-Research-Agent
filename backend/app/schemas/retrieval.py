from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class RetrievedChunk(BaseModel):
    """A document chunk retrieved via vector similarity search."""

    chunk_id: str = Field(..., description="UUID string of the chunk")
    document_id: str = Field(..., description="UUID string of the parent document")
    session_id: str = Field(..., description="UUID string of the research session")
    chunk_index: int = Field(..., description="Ordering index of the chunk within the document")
    text: str = Field(..., description="Text content of the retrieved chunk")
    similarity: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    url: str = Field(..., description="Canonical URL of the source document")
    title: str = Field(..., description="Title of the source document")


class RetrievalRequest(BaseModel):
    """Request payload for standalone semantic retrieval."""

    query: str = Field(..., min_length=1, description="Natural-language search query")
    session_id: Optional[str] = Field(default=None, description="Optional session UUID to filter chunks")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of chunks to return")
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class RetrievalResponse(BaseModel):
    """Response payload containing ranked retrieved chunks."""

    query: str = Field(..., description="The query string used for retrieval")
    results: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Ranked list of retrieved chunks ordered by similarity descending",
    )
    top_k: int = Field(..., description="The requested top_k limit")
