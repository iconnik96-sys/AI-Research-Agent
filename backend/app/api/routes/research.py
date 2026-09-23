import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.extraction import WebpageExtractor, get_webpage_extractor
from app.services.search import (
    BaseSearchProvider,
    SearchConfigError,
    SearchError,
    SearchProviderError,
    SearchTimeoutError,
    get_search_provider,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


@router.post("", response_model=ResearchResponse, summary="Submit a research question")
async def create_research(
    request: ResearchRequest,
    search_provider: BaseSearchProvider = Depends(get_search_provider),
    extractor: WebpageExtractor = Depends(get_webpage_extractor),
) -> ResearchResponse:
    """Accept a research question, search the web, and extract clean document text."""
    try:
        sources = await search_provider.search(
            query=request.question,
            max_results=settings.SEARCH_MAX_RESULTS,
        )
    except SearchConfigError as exc:
        logger.error("Search configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except SearchTimeoutError as exc:
        logger.error("Search timeout error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The search provider request timed out. Please try again.",
        ) from exc
    except SearchProviderError as exc:
        logger.error("Search provider error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream search provider returned an error.",
        ) from exc
    except SearchError as exc:
        logger.error("General search error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while executing the search.",
        ) from exc

    # Concurrently extract webpage contents from collected sources
    # Individual page failures are logged and do not fail the request
    documents = await extractor.extract_many(sources)

    return ResearchResponse(
        question=request.question,
        status="completed",
        sources=sources,
        documents=documents,
        message=f"Retrieved {len(sources)} source(s) and successfully extracted {len(documents)} document(s).",
    )
