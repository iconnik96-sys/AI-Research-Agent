from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Deterministic chunk of an extracted document."""

    document_id: Optional[str] = Field(default=None, description="UUID of parent document in persistence layer")
    session_id: Optional[str] = Field(default=None, description="UUID of parent research session")
    chunk_index: int = Field(..., description="Zero-based ordering index of the chunk within the document")
    text: str = Field(..., description="Normalized text content of the chunk")
    char_count: int = Field(..., description="Length of the chunk text in characters")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding float values")
