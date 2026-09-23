import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.extraction import WebpageExtractor, get_webpage_extractor
from app.services.llm import (
    BaseLLMProvider,
    LLMConfigError,
    LLMError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
    get_llm_provider,
)
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
    llm_provider: BaseLLMProvider = Depends(get_llm_provider),
) -> ResearchResponse:
    """Execute end-to-end research: search web, extract documents, and synthesize cited report."""
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
    # Individual page failures are logged and do not fail the overall request
    documents = await extractor.extract_many(sources)

    # Synthesize research report via LLM provider
    try:
        report = await llm_provider.generate_report(
            question=request.question,
            documents=documents,
        )
    except LLMConfigError as exc:
        logger.error("LLM configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LLMTimeoutError as exc:
        logger.error("LLM timeout error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The LLM completion request timed out. Please try again.",
        ) from exc
    except LLMProviderError as exc:
        logger.error("LLM provider error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream LLM provider returned an error.",
        ) from exc
    except LLMResponseError as exc:
        logger.error("LLM response validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Invalid response from LLM synthesis: {str(exc)}",
        ) from exc
    except LLMError as exc:
        logger.error("General LLM error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the research report.",
        ) from exc

    return ResearchResponse(
        question=request.question,
        status="completed",
        sources=sources,
        documents=documents,
        report=report,
        message=f"Retrieved {len(sources)} source(s), extracted {len(documents)} document(s), and synthesized report.",
    )
