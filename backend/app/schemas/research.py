from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The research question to investigate",
        json_schema_extra={"example": "What are the latest breakthroughs in fusion energy?"},
    )


class ResearchResponse(BaseModel):
    question: str = Field(
        ...,
        description="The research question that was submitted",
    )
    status: str = Field(
        default="received",
        description="Placeholder status for the research request",
    )
    message: str = Field(
        default="Research request received. Processing pipeline will be implemented in subsequent milestones.",
        description="Informational status message",
    )
