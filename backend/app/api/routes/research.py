import asyncio
import logging
from typing import Dict, List
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.schemas.claim import (
    ExtractedClaim,
    VerifiedClaim,
    VerifyClaimsRequest,
    VerifyClaimsResponse,
)
from app.schemas.evidence import EvidenceItem
from app.schemas.planner import ResearchPlan
from app.schemas.research import ResearchRequest, ResearchResponse, SourceItem
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievedChunk
from app.services.chunking import TextChunker, get_text_chunker
from app.services.claim import (
    BaseClaimExtractor,
    BaseClaimVerifier,
    ClaimConfigError,
    ClaimError,
    ClaimExtractionError,
    ClaimResponseError,
    ClaimVerificationError,
    get_claim_extractor,
    get_claim_verifier,
)
from app.services.embedding import (
    BaseEmbeddingProvider,
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
    get_embedding_provider,
)
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
from app.services.pipeline import ResearchOrchestrator
from app.services.planner import (
    BaseResearchPlanner,
    PlannerConfigError,
    PlannerError,
    PlannerNetworkError,
    PlannerResponseError,
    PlannerTimeoutError,
    get_research_planner,
)
from app.services.retrieval import (
    BaseRetriever,
    RetrievalConfigError,
    RetrievalDatabaseError,
    RetrievalEmbeddingError,
    RetrievalError,
    get_retriever,
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


@router.post("/retrieve", response_model=RetrievalResponse, summary="Retrieve relevant document chunks via vector similarity")
async def retrieve_chunks(
    request: RetrievalRequest,
    retriever: BaseRetriever = Depends(get_retriever),
) -> RetrievalResponse:
    """Retrieve top-K similar document chunks using pgvector cosine similarity."""
    try:
        results = await retriever.retrieve(
            query=request.query,
            session_id=request.session_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
        )
    except RetrievalConfigError as exc:
        logger.error("Retrieval configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except RetrievalEmbeddingError as exc:
        logger.error("Retrieval query embedding error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except RetrievalDatabaseError as exc:
        logger.error("Retrieval database error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database retrieval error.",
        ) from exc
    except RetrievalError as exc:
        logger.error("General retrieval error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during evidence retrieval.",
        ) from exc

    return RetrievalResponse(
        query=request.query,
        results=results,
        top_k=request.top_k,
    )


@router.post("/plan", response_model=ResearchPlan, summary="Generate structured research plan")
async def plan_research(
    request: ResearchRequest,
    planner: BaseResearchPlanner = Depends(get_research_planner),
) -> ResearchPlan:
    """Decompose a research question into structured sub-questions and search queries."""
    try:
        return await planner.create_plan(request.question)
    except PlannerConfigError as exc:
        logger.error("Planner configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except PlannerTimeoutError as exc:
        logger.error("Planner timeout error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The research planning request timed out. Please try again.",
        ) from exc
    except (PlannerResponseError, PlannerNetworkError) as exc:
        logger.error("Planner response/network error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Research planner error: {str(exc)}",
        ) from exc
    except PlannerError as exc:
        logger.error("General planner error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the research plan.",
        ) from exc


@router.post("/verify", response_model=VerifyClaimsResponse, summary="Independently verify factual claims against evidence")
async def verify_claims_endpoint(
    request: VerifyClaimsRequest,
    verifier: BaseClaimVerifier = Depends(get_claim_verifier),
) -> VerifyClaimsResponse:
    """Independently verify candidate factual claims against provided evidence items."""
    try:
        verified = await verifier.verify_claims(
            question=request.question,
            claims=request.claims,
            evidence=request.evidence,
        )
    except ClaimConfigError as exc:
        logger.error("Claim verification configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ClaimVerificationError as exc:
        logger.error("Claim verification error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Claim verification error: {str(exc)}",
        ) from exc
    except ClaimError as exc:
        logger.error("General claim verification error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while verifying claims.",
        ) from exc

    return VerifyClaimsResponse(
        question=request.question,
        verified_claims=verified,
    )


@router.post("", response_model=ResearchResponse, summary="Submit a research question")
async def create_research(
    request: ResearchRequest,
    planner: BaseResearchPlanner = Depends(get_research_planner),
    search_provider: BaseSearchProvider = Depends(get_search_provider),
    extractor: WebpageExtractor = Depends(get_webpage_extractor),
    embedding_provider: BaseEmbeddingProvider = Depends(get_embedding_provider),
    chunker: TextChunker = Depends(get_text_chunker),
    retriever: BaseRetriever = Depends(get_retriever),
    claim_extractor: BaseClaimExtractor = Depends(get_claim_extractor),
    claim_verifier: BaseClaimVerifier = Depends(get_claim_verifier),
    llm_provider: BaseLLMProvider = Depends(get_llm_provider),
    repository: BaseResearchRepository = Depends(get_research_repository),
) -> ResearchResponse:
    """Execute end-to-end research: plan research, multi-query search, extract, embed, retrieve, synthesize report."""
    orchestrator = ResearchOrchestrator()
    try:
        result = await orchestrator.run(
            question=request.question,
            planner=planner,
            search_provider=search_provider,
            extractor=extractor,
            embedding_provider=embedding_provider,
            chunker=chunker,
            retriever=retriever,
            claim_extractor=claim_extractor,
            claim_verifier=claim_verifier,
            llm_provider=llm_provider,
            repository=repository,
            persist=True,
        )
    except DatabaseConfigError as exc:
        logger.error("Database configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except DatabaseError as exc:
        logger.error("Database persistence error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database persistence error.",
        ) from exc
    except PlannerConfigError as exc:
        logger.error("Planner configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except PlannerTimeoutError as exc:
        logger.error("Planner timeout error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The research planning request timed out. Please try again.",
        ) from exc
    except (PlannerResponseError, PlannerNetworkError) as exc:
        logger.error("Planner error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Research planner error: {str(exc)}",
        ) from exc
    except PlannerError as exc:
        logger.error("General planner error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the research plan.",
        ) from exc
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
    except EmbeddingConfigError as exc:
        logger.error("Embedding configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except EmbeddingTimeoutError as exc:
        logger.error("Embedding timeout error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The embedding request timed out. Please try again.",
        ) from exc
    except (EmbeddingProviderError, EmbeddingResponseError) as exc:
        logger.error("Embedding provider error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Invalid response from embedding provider: {str(exc)}" if isinstance(exc, EmbeddingResponseError) else "Upstream embedding provider returned an error.",
        ) from exc
    except EmbeddingError as exc:
        logger.error("General embedding error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating embeddings.",
        ) from exc
    except RetrievalConfigError as exc:
        logger.error("Retrieval configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except RetrievalEmbeddingError as exc:
        logger.error("Retrieval embedding error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except RetrievalDatabaseError as exc:
        logger.error("Retrieval database error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database retrieval error.",
        ) from exc
    except RetrievalError as exc:
        logger.error("General retrieval error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during evidence retrieval.",
        ) from exc
    except ClaimConfigError as exc:
        logger.error("Claim configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except (ClaimExtractionError, ClaimResponseError) as exc:
        logger.error("Claim extraction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Claim extraction error: {str(exc)}",
        ) from exc
    except ClaimVerificationError as exc:
        logger.error("Claim verification error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Claim verification error: {str(exc)}",
        ) from exc
    except ClaimError as exc:
        logger.error("General claim error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing factual claims.",
        ) from exc
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
        logger.error("LLM response error: %s", exc)
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
        session_id=result.session_id,
        question=result.question,
        status="completed",
        plan=result.plan,
        sources=result.sources,
        documents=result.documents,
        claims=result.verified_claims,
        report=result.report,
        message=f"Planned {len(result.plan.sub_questions) if result.plan else 0} sub-questions, retrieved {len(result.sources)} source(s), verified {len(result.verified_claims)} claim(s), and synthesized report.",
    )

