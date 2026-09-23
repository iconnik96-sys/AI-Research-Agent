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

    # 2. Decompose research question into structured plan
    try:
        plan = await planner.create_plan(request.question)
    except PlannerConfigError as exc:
        logger.error("Planner configuration error: %s", exc)
        await repository.fail_session(session_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except PlannerTimeoutError as exc:
        logger.error("Planner timeout error: %s", exc)
        await repository.fail_session(session_id, "Planner request timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The research planning request timed out. Please try again.",
        ) from exc
    except (PlannerResponseError, PlannerNetworkError) as exc:
        logger.error("Planner error: %s", exc)
        await repository.fail_session(session_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Research planner error: {str(exc)}",
        ) from exc
    except PlannerError as exc:
        logger.error("General planner error: %s", exc)
        await repository.fail_session(session_id, "Planning error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the research plan.",
        ) from exc

    # 3. Multi-query web search: search each sub-question's search_query concurrently with bounded concurrency
    concurrency_limit = min(len(plan.sub_questions), 5)
    semaphore = asyncio.Semaphore(concurrency_limit)
    search_errors: List[Exception] = []

    async def _search_sub_question(sub_q) -> List[SourceItem]:
        async with semaphore:
            try:
                return await search_provider.search(
                    query=sub_q.search_query,
                    max_results=settings.SEARCH_MAX_RESULTS,
                )
            except Exception as exc:
                logger.warning(
                    "Search failed for sub-question '%s' (query: '%s'): %s",
                    sub_q.id,
                    sub_q.search_query,
                    exc,
                )
                search_errors.append(exc)
                return []

    search_tasks = [_search_sub_question(sq) for sq in plan.sub_questions]
    search_results = await asyncio.gather(*search_tasks)

    all_sources: List[SourceItem] = []
    for res in search_results:
        all_sources.extend(res)

    # Deduplicate collected sources by canonical URL
    seen_urls = set()
    sources: List[SourceItem] = []
    for s in all_sources:
        if s.url not in seen_urls:
            seen_urls.add(s.url)
            sources.append(s)

    # If ALL searches failed and produced zero sources, inspect search_errors to raise appropriate exception
    if not sources and search_errors:
        config_err = next((e for e in search_errors if isinstance(e, SearchConfigError)), None)
        if config_err:
            logger.error("Search configuration error: %s", config_err)
            await repository.fail_session(session_id, str(config_err))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(config_err),
            ) from config_err

        timeout_err = next((e for e in search_errors if isinstance(e, SearchTimeoutError)), None)
        if timeout_err:
            logger.error("All searches failed with timeout: %s", timeout_err)
            await repository.fail_session(session_id, "Search request timed out")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="The search provider request timed out. Please try again.",
            ) from timeout_err

        provider_err = next((e for e in search_errors if isinstance(e, SearchProviderError)), None)
        if provider_err:
            logger.error("All searches failed with upstream error: %s", provider_err)
            await repository.fail_session(session_id, "Upstream search provider error")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Upstream search provider returned an error.",
            ) from provider_err

        general_err = search_errors[0]
        logger.error("All searches failed: %s", general_err)
        await repository.fail_session(session_id, "Search execution error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while executing the search.",
        ) from general_err

    # 4. Persist search sources
    try:
        source_id_map = await repository.save_sources(session_id=session_id, sources=sources)
    except DatabaseError as exc:
        logger.error("Database error persisting sources: %s", exc)
        await repository.fail_session(session_id, "Failed to persist sources")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database persistence error.",
        ) from exc

    # 5. Concurrently extract webpage contents from collected sources
    # Individual page failures are logged and skipped
    documents = await extractor.extract_many(sources)

    # 6. Persist extracted documents
    try:
        doc_id_map = await repository.save_documents(
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

    # 7. Chunk documents, compute embeddings, and persist chunks to pgvector
    if documents:
        chunks = chunker.chunk_documents(
            documents=documents,
            document_id_map=doc_id_map,
            session_id=session_id,
        )
        if chunks:
            try:
                chunk_texts = [c.text for c in chunks]
                embeddings = await embedding_provider.embed_texts(chunk_texts)
            except EmbeddingConfigError as exc:
                logger.error("Embedding configuration error: %s", exc)
                await repository.fail_session(session_id, str(exc))
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from exc
            except EmbeddingTimeoutError as exc:
                logger.error("Embedding timeout error: %s", exc)
                await repository.fail_session(session_id, "Embedding request timed out")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="The embedding request timed out. Please try again.",
                ) from exc
            except EmbeddingProviderError as exc:
                logger.error("Embedding provider error: %s", exc)
                await repository.fail_session(session_id, "Upstream embedding provider error")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Upstream embedding provider returned an error.",
                ) from exc
            except EmbeddingResponseError as exc:
                logger.error("Embedding response validation error: %s", exc)
                await repository.fail_session(session_id, f"Invalid embedding response: {str(exc)}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Invalid response from embedding provider: {str(exc)}",
                ) from exc
            except EmbeddingError as exc:
                logger.error("General embedding error: %s", exc)
                await repository.fail_session(session_id, "Embedding generation error")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An error occurred while generating embeddings.",
                ) from exc

            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

            try:
                await repository.save_chunks(session_id=session_id, chunks=chunks)
            except DatabaseError as exc:
                logger.error("Database error persisting document chunks: %s", exc)
                await repository.fail_session(session_id, "Failed to persist document chunks")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database persistence error.",
                ) from exc

    # 8. Multi-query semantic retrieval:
    # Use original question AND each sub-question.question for semantic retrieval
    retrieved_chunks: List[RetrievedChunk] = []
    if documents:
        retrieval_queries = [request.question] + [sq.question for sq in plan.sub_questions]
        chunk_map: Dict[str, RetrievedChunk] = {}

        for query_text in retrieval_queries:
            try:
                query_chunks = await retriever.retrieve(
                    query=query_text,
                    session_id=session_id,
                    top_k=settings.RETRIEVAL_TOP_K,
                    similarity_threshold=settings.RETRIEVAL_SIMILARITY_THRESHOLD,
                )
                for chunk in query_chunks:
                    if chunk.chunk_id not in chunk_map:
                        chunk_map[chunk.chunk_id] = chunk
                    else:
                        # Preserve highest similarity result when chunk retrieved multiple times
                        if chunk.similarity > chunk_map[chunk.chunk_id].similarity:
                            chunk_map[chunk.chunk_id] = chunk
            except RetrievalConfigError as exc:
                logger.error("Retrieval configuration error: %s", exc)
                await repository.fail_session(session_id, str(exc))
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from exc
            except RetrievalEmbeddingError as exc:
                logger.error("Retrieval query embedding error: %s", exc)
                await repository.fail_session(session_id, "Retrieval query embedding error")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=str(exc),
                ) from exc
            except RetrievalDatabaseError as exc:
                logger.error("Retrieval database error: %s", exc)
                await repository.fail_session(session_id, "Retrieval database error")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database retrieval error.",
                ) from exc
            except RetrievalError as exc:
                logger.error("General retrieval error: %s", exc)
                await repository.fail_session(session_id, "Retrieval error")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An error occurred during evidence retrieval.",
                ) from exc

        # Sort deduplicated chunks by similarity descending and cap at RETRIEVAL_MAX_TOTAL_CHUNKS
        sorted_chunks = sorted(
            chunk_map.values(),
            key=lambda c: c.similarity,
            reverse=True,
        )
        retrieved_chunks = sorted_chunks[:settings.RETRIEVAL_MAX_TOTAL_CHUNKS]

    # 9. Evidence Management & Claim Extraction & Verification
    evidence_items: List[EvidenceItem] = []
    evidence_map: Dict[str, EvidenceItem] = {}
    for index, chunk in enumerate(retrieved_chunks, start=1):
        eid = f"E{index}"
        item = EvidenceItem(
            evidence_id=eid,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            session_id=chunk.session_id or session_id,
            url=chunk.url,
            title=chunk.title,
            text=chunk.text,
            similarity=chunk.similarity,
        )
        evidence_items.append(item)
        evidence_map[eid] = item

    verified_claims: List[VerifiedClaim] = []
    if evidence_items:
        try:
            extracted_claims = await claim_extractor.extract_claims(
                question=request.question,
                evidence=evidence_items,
            )
        except ClaimConfigError as exc:
            logger.error("Claim extraction configuration error: %s", exc)
            await repository.fail_session(session_id, str(exc))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except (ClaimExtractionError, ClaimResponseError) as exc:
            logger.error("Claim extraction error: %s", exc)
            await repository.fail_session(session_id, str(exc))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Claim extraction error: {str(exc)}",
            ) from exc
        except ClaimError as exc:
            logger.error("General claim extraction error: %s", exc)
            await repository.fail_session(session_id, "Claim extraction error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while extracting factual claims.",
            ) from exc

        if extracted_claims:
            try:
                verified_claims = await claim_verifier.verify_claims(
                    question=request.question,
                    claims=extracted_claims,
                    evidence=evidence_items,
                )
            except ClaimConfigError as exc:
                logger.error("Claim verification configuration error: %s", exc)
                await repository.fail_session(session_id, str(exc))
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from exc
            except ClaimVerificationError as exc:
                logger.error("Claim verification error: %s", exc)
                await repository.fail_session(session_id, str(exc))
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Claim verification error: {str(exc)}",
                ) from exc
            except ClaimError as exc:
                logger.error("General claim verification error: %s", exc)
                await repository.fail_session(session_id, "Claim verification error")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An error occurred while verifying factual claims.",
                ) from exc

            # Persist verified claims and claim-evidence links
            try:
                await repository.save_claims(
                    session_id=session_id,
                    claims=verified_claims,
                    evidence_map=evidence_map,
                )
            except DatabaseError as exc:
                logger.error("Database error persisting claims: %s", exc)
                await repository.fail_session(session_id, "Failed to persist claims")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database persistence error.",
                ) from exc

    # 10. Synthesize grounded research report via LLM provider using ONLY retrieved chunks and verified claims
    try:
        report = await llm_provider.generate_report(
            question=request.question,
            chunks=retrieved_chunks,
            claims=verified_claims,
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

    # 11. Complete session with report JSONB
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
        plan=plan,
        sources=sources,
        documents=documents,
        claims=verified_claims,
        report=report,
        message=f"Planned {len(plan.sub_questions)} sub-questions, retrieved {len(sources)} source(s), verified {len(verified_claims)} claim(s), and synthesized report.",
    )
