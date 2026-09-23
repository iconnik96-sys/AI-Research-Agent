from typing import Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Normalized document extracted from a source webpage."""

    url: str = Field(..., description="Canonical URL of the extracted source")
    title: str = Field(..., description="Page or document title")
    text: str = Field(..., description="Cleaned, readable visible text extracted from the source")
    score: Optional[float] = Field(default=None, description="Search relevance score if available")
    char_count: int = Field(..., description="Total character count of the extracted text")
