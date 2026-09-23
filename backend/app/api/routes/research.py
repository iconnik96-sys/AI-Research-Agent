from fastapi import APIRouter
from app.schemas.research import ResearchRequest, ResearchResponse

router = APIRouter(prefix="/research", tags=["research"])


@router.post("", response_model=ResearchResponse, summary="Submit a research question")
async def create_research(request: ResearchRequest) -> ResearchResponse:
    """Accept a research question and return received status."""
    return ResearchResponse(
        question=request.question,
        status="received",
        message="Research request received. Processing pipeline will be implemented in subsequent milestones.",
    )
