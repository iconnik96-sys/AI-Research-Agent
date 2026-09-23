from typing import List
from pydantic import BaseModel, Field, field_validator


class ResearchSubQuestion(BaseModel):
    """A focused sub-question with dedicated web search query."""

    id: str = Field(..., description="Unique sub-question identifier (e.g. 'q1', 'q2')")
    question: str = Field(..., min_length=1, description="Focused research sub-question for semantic understanding")
    search_query: str = Field(..., min_length=1, description="Targeted keyword query for web search")
    reason: str = Field(..., min_length=1, description="Rationale for why this evidence is required")

    @field_validator("id", "question", "search_query", "reason")
    @classmethod
    def validate_non_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return v.strip()


class ResearchPlan(BaseModel):
    """Structured plan decomposing a complex research question into sub-questions."""

    original_question: str = Field(..., min_length=1, description="Original user research question")
    sub_questions: List[ResearchSubQuestion] = Field(
        ...,
        min_length=1,
        description="List of focused sub-questions to research",
    )

    @field_validator("original_question")
    @classmethod
    def validate_original_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Original question cannot be empty or whitespace only")
        return v.strip()

    @field_validator("sub_questions")
    @classmethod
    def validate_sub_questions_non_empty(cls, v: List[ResearchSubQuestion]) -> List[ResearchSubQuestion]:
        if not v:
            raise ValueError("Sub-questions list cannot be empty")
        return v
