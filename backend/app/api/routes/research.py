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
from app.services.persistence import (
    BaseResearchRepository,
    DatabaseConfigError,
    DatabaseError,
    get_research_repository,
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
    repository: BaseResearchRepository = Depends(get_research_repository),
) -> ResearchResponse:
    """Execute end-to-end research: persist session, search web, extract documents, synthesize report."""
    # 1. Initialize research session in database
    try:
        session_id = await repository.create_session(question=request.question)
    except DatabaseConfigError as exc:
        logger.error("Database configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except DatabaseError as exc:
        logger.error("Database error creating session: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database persistence error.",
        ) from exc

    # 2. Search web for sources
    try:
        sources = await search_provider.search(
            query=request.question,
            max_results=settings.SEARCH_MAX_RESULTS,
        )
    except SearchConfigError as exc:
        logger.error("Search configuration error: %s", exc)
        await repository.fail_session(session_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except SearchTimeoutError as exc:
        logger.error("Search timeout error: %s", exc)
        await repository.fail_session(session_id, "Search request timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The search provider request timed out. Please try again.",
        ) from exc
    except SearchProviderError as exc:
        logger.error("Search provider error: %s", exc)
        await repository.fail_session(session_id, "Upstream search provider error")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream search provider returned an error.",
        ) from exc
    except SearchError as exc:
        logger.error("General search error: %s", exc)
        await repository.fail_session(session_id, "Search execution error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while executing the search.",
        ) from exc

    # 3. Persist search sources
    try:
        source_id_map = await repository.save_sources(session_id=session_id, sources=sources)
    except DatabaseError as exc:
        logger.error("Database error persisting sources: %s", exc)
        await repository.fail_session(session_id, "Failed to persist sources")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database persistence error.",
        ) from exc

    # 4. Concurrently extract webpage contents from collected sources
    # Individual page failures are logged and skipped
    documents = await extractor.extract_many(sources)

    # 5. Persist extracted documents
    try:
        await repository.save_documents(
            session_id=session_id,
            documents=documents,
            source_id_map=source_id_map,
        )
    except DatabaseError as exc:
        logger.error("Database error persisting documents: %s", exc)
        await repository.fail_session(session_id, "Failed to persist documents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database persistence error.",
        ) from exc

    # 6. Synthesize research report via LLM provider
    try:
        report = await llm_provider.generate_report(
            question=request.question,
            documents=documents,
        )
    except LLMConfigError as exc:
        logger.error("LLM configuration error: %s", exc)
        await repository.fail_session(session_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LLMTimeoutError as exc:
        logger.error("LLM timeout error: %s", exc)
        await repository.fail_session(session_id, "LLM request timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The LLM completion request timed out. Please try again.",
        ) from exc
    except LLMProviderError as exc:
        logger.error("LLM provider error: %s", exc)
        await repository.fail_session(session_id, "Upstream LLM provider error")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream LLM provider returned an error.",
        ) from exc
    except LLMResponseError as exc:
        logger.error("LLM response validation error: %s", exc)
        await repository.fail_session(session_id, f"Invalid LLM response: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Invalid response from LLM synthesis: {str(exc)}",
        ) from exc
    except LLMError as exc:
        logger.error("General LLM error: %s", exc)
        await repository.fail_session(session_id, "LLM synthesis error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the research report.",
        ) from exc

    # 7. Complete session with report JSONB
    try:
        await repository.complete_session(session_id=session_id, report=report)
    except DatabaseError as exc:
        logger.error("Database error completing session: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database persistence error.",
        ) from exc

    return ResearchResponse(
        session_id=session_id,
        question=request.question,
        status="completed",
        sources=sources,
        documents=documents,
        report=report,
        message=f"Retrieved {len(sources)} source(s), extracted {len(documents)} document(s), and synthesized report.",
    )
