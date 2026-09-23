import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.chunk import DocumentChunk
from app.schemas.claim import ExtractedClaim, VerifiedClaim
from app.schemas.document import Document
from app.schemas.evidence import EvidenceItem
from app.schemas.planner import ResearchPlan
from app.schemas.report import ResearchReport
from app.schemas.research import SourceItem
from app.schemas.retrieval import RetrievedChunk
from app.services.chunking import TextChunker
from app.services.claim import BaseClaimExtractor, BaseClaimVerifier
from app.services.embedding import BaseEmbeddingProvider
from app.services.extraction import WebpageExtractor
from app.services.llm import BaseLLMProvider
from app.services.persistence import BaseResearchRepository
from app.services.planner import BaseResearchPlanner
from app.services.retrieval import BaseRetriever
from app.services.search import (
    BaseSearchProvider,
    SearchConfigError,
    SearchError,
    SearchProviderError,
    SearchTimeoutError,
)

logger = logging.getLogger(__name__)


class QueryRetrievalResult(BaseModel):
    """Retrieval outcome preserved for a single query before deduplication."""

    query: str = Field(..., description="The query string used for retrieval")
    query_type: str = Field(..., description="'original' or 'sub_question'")
    sub_question_id: Optional[str] = Field(default=None, description="Sub-question ID if applicable")
    chunks: List[RetrievedChunk] = Field(default_factory=list, description="Retrieved chunks for this query")
    hit: bool = Field(default=False, description="True if at least one chunk was retrieved")


class PipelineExecutionResult(BaseModel):
    """Complete container of intermediate objects, metrics, and report from a research run."""

    session_id: str = Field(..., description="Research session UUID string")
    question: str = Field(..., description="Original research question")
    plan: Optional[ResearchPlan] = Field(default=None, description="Generated research plan")
    sources: List[SourceItem] = Field(default_factory=list, description="Collected web sources")
    documents: List[Document] = Field(default_factory=list, description="Extracted readable documents")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Chunked text with embeddings")
    query_retrieval_results: List[QueryRetrievalResult] = Field(
        default_factory=list,
        description="Preserved retrieval results per query before deduplication",
    )
    retrieved_chunks: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Deduplicated top-K chunks passed to synthesis",
    )
    evidence_items: List[EvidenceItem] = Field(default_factory=list, description="Normalized evidence items")
    extracted_claims: List[ExtractedClaim] = Field(default_factory=list, description="Extracted candidate claims")
    verified_claims: List[VerifiedClaim] = Field(default_factory=list, description="Independently verified claims")
    report: Optional[ResearchReport] = Field(default=None, description="Synthesized research report")
    stage_latencies: Dict[str, float] = Field(default_factory=dict, description="Elapsed seconds per stage")
    total_latency_seconds: float = Field(default=0.0, description="Total wall-clock duration in seconds")
    stage_errors: List[str] = Field(default_factory=list, description="Non-fatal warnings or stage errors")
    search_failure_count: int = Field(default=0, description="Count of searches that failed or produced 0 sources")
    empty_retrieval_count: int = Field(default=0, description="Count of queries that yielded 0 chunks")
    is_success: bool = Field(default=False, description="True if run completed without unhandled fatal error")


class ResearchOrchestrator:
    """Reusable research pipeline orchestrator executing research stages and measuring quality data."""

    async def run(
        self,
        question: str,
        planner: BaseResearchPlanner,
        search_provider: BaseSearchProvider,
        extractor: WebpageExtractor,
        embedding_provider: BaseEmbeddingProvider,
        chunker: TextChunker,
        retriever: BaseRetriever,
        claim_extractor: BaseClaimExtractor,
        claim_verifier: BaseClaimVerifier,
        llm_provider: BaseLLMProvider,
        repository: BaseResearchRepository,
        persist: bool = True,
    ) -> PipelineExecutionResult:
        """Execute end-to-end research pipeline.

        Args:
            question: The user research question.
            persist: If True (production), keeps records in database.
                     If False (evaluation), cleans up temporary session and chunks in finally block.

        Returns:
            PipelineExecutionResult containing all intermediate objects, latencies, and output report.
        """
        start_total = time.perf_counter()
        session_id: Optional[str] = None
        stage_latencies: Dict[str, float] = {}
        stage_errors: List[str] = []
        search_failure_count = 0
        empty_retrieval_count = 0

        # Result container initialized early so data is retained even on cleanup
        result = PipelineExecutionResult(
            session_id="",
            question=question,
        )

        try:
            # 1. Initialize session in database
            t0 = time.perf_counter()
            session_id = await repository.create_session(question=question)
            stage_latencies["session_init"] = time.perf_counter() - t0
            result.session_id = session_id

            # 2. Planning stage
            t0 = time.perf_counter()
            try:
                plan = await planner.create_plan(question)
                result.plan = plan
            except Exception as exc:
                await repository.fail_session(session_id, str(exc))
                raise
            stage_latencies["planning"] = time.perf_counter() - t0

            # 3. Multi-query web search
            t0 = time.perf_counter()
            concurrency_limit = min(len(plan.sub_questions), 5)
            semaphore = asyncio.Semaphore(concurrency_limit)
            search_errors: List[Exception] = []

            async def _search_sub_question(sub_q) -> List[SourceItem]:
                nonlocal search_failure_count
                async with semaphore:
                    try:
                        res = await search_provider.search(
                            query=sub_q.search_query,
                            max_results=settings.SEARCH_MAX_RESULTS,
                        )
                        if not res:
                            search_failure_count += 1
                        return res
                    except Exception as exc:
                        logger.warning(
                            "Search failed for sub-question '%s' (query: '%s'): %s",
                            sub_q.id,
                            sub_q.search_query,
                            exc,
                        )
                        search_errors.append(exc)
                        search_failure_count += 1
                        return []

            search_tasks = [_search_sub_question(sq) for sq in plan.sub_questions]
            search_results = await asyncio.gather(*search_tasks)

            all_sources: List[SourceItem] = []
            for res in search_results:
                all_sources.extend(res)

            # Deduplicate collected sources by URL
            seen_urls = set()
            sources: List[SourceItem] = []
            for s in all_sources:
                if s.url not in seen_urls:
                    seen_urls.add(s.url)
                    sources.append(s)
            result.sources = sources
            stage_latencies["search"] = time.perf_counter() - t0

            # If all searches failed
            if not sources and search_errors:
                config_err = next((e for e in search_errors if isinstance(e, SearchConfigError)), None)
                if config_err:
                    await repository.fail_session(session_id, str(config_err))
                    raise config_err
                timeout_err = next((e for e in search_errors if isinstance(e, SearchTimeoutError)), None)
                if timeout_err:
                    await repository.fail_session(session_id, "Search request timed out")
                    raise timeout_err
                provider_err = next((e for e in search_errors if isinstance(e, SearchProviderError)), None)
                if provider_err:
                    await repository.fail_session(session_id, "Upstream search provider error")
                    raise provider_err
                gen_err = search_errors[0]
                await repository.fail_session(session_id, "Search execution error")
                raise gen_err

            # 4. Save sources
            t0 = time.perf_counter()
            source_id_map = await repository.save_sources(session_id=session_id, sources=sources)
            stage_latencies["save_sources"] = time.perf_counter() - t0

            # 5. Extraction
            t0 = time.perf_counter()
            documents = await extractor.extract_many(sources)
            result.documents = documents
            stage_latencies["extraction"] = time.perf_counter() - t0

            # 6. Save documents
            t0 = time.perf_counter()
            doc_id_map = await repository.save_documents(
                session_id=session_id,
                documents=documents,
                source_id_map=source_id_map,
            )
            stage_latencies["save_documents"] = time.perf_counter() - t0

            # 7. Chunking & Embeddings
            t0 = time.perf_counter()
            chunks: List[DocumentChunk] = []
            if documents:
                chunks = chunker.chunk_documents(
                    documents=documents,
                    document_id_map=doc_id_map,
                    session_id=session_id,
                )
                if chunks:
                    chunk_texts = [c.text for c in chunks]
                    try:
                        embeddings = await embedding_provider.embed_texts(chunk_texts)
                    except Exception as exc:
                        await repository.fail_session(session_id, str(exc))
                        raise

                    for c, emb in zip(chunks, embeddings):
                        c.embedding = emb

                    try:
                        await repository.save_chunks(session_id=session_id, chunks=chunks)
                    except Exception as exc:
                        await repository.fail_session(session_id, "Failed to persist document chunks")
                        raise
            result.chunks = chunks
            stage_latencies["chunking_embedding"] = time.perf_counter() - t0

            # 8. Semantic Retrieval - Preserving per-query results
            t0 = time.perf_counter()
            query_retrieval_results: List[QueryRetrievalResult] = []
            chunk_map: Dict[str, RetrievedChunk] = {}

            if documents:
                retrieval_specs = [
                    (question, "original", None)
                ] + [
                    (sq.question, "sub_question", sq.id) for sq in plan.sub_questions
                ]

                for query_text, q_type, sq_id in retrieval_specs:
                    try:
                        query_chunks = await retriever.retrieve(
                            query=query_text,
                            session_id=session_id,
                            top_k=settings.RETRIEVAL_TOP_K,
                            similarity_threshold=settings.RETRIEVAL_SIMILARITY_THRESHOLD,
                        )
                        hit = len(query_chunks) > 0
                        if not hit:
                            empty_retrieval_count += 1

                        query_retrieval_results.append(
                            QueryRetrievalResult(
                                query=query_text,
                                query_type=q_type,
                                sub_question_id=sq_id,
                                chunks=query_chunks,
                                hit=hit,
                            )
                        )

                        for chunk in query_chunks:
                            if chunk.chunk_id not in chunk_map:
                                chunk_map[chunk.chunk_id] = chunk
                            elif chunk.similarity > chunk_map[chunk.chunk_id].similarity:
                                chunk_map[chunk.chunk_id] = chunk
                    except Exception as exc:
                        await repository.fail_session(session_id, str(exc))
                        raise

            result.query_retrieval_results = query_retrieval_results
            sorted_chunks = sorted(
                chunk_map.values(),
                key=lambda c: c.similarity,
                reverse=True,
            )
            retrieved_chunks = sorted_chunks[:settings.RETRIEVAL_MAX_TOTAL_CHUNKS]
            result.retrieved_chunks = retrieved_chunks
            stage_latencies["retrieval"] = time.perf_counter() - t0

            # 9. Evidence Management & Claim Verification
            t0 = time.perf_counter()
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
            result.evidence_items = evidence_items

            verified_claims: List[VerifiedClaim] = []
            if evidence_items:
                t_extract = time.perf_counter()
                try:
                    extracted_claims = await claim_extractor.extract_claims(
                        question=question,
                        evidence=evidence_items,
                    )
                    result.extracted_claims = extracted_claims
                except Exception as exc:
                    await repository.fail_session(session_id, str(exc))
                    raise
                stage_latencies["claim_extraction"] = time.perf_counter() - t_extract

                if extracted_claims:
                    t_verify = time.perf_counter()
                    try:
                        verified_claims = await claim_verifier.verify_claims(
                            question=question,
                            claims=extracted_claims,
                            evidence=evidence_items,
                        )
                        result.verified_claims = verified_claims
                    except Exception as exc:
                        await repository.fail_session(session_id, str(exc))
                        raise
                    stage_latencies["claim_verification"] = time.perf_counter() - t_verify

                    try:
                        await repository.save_claims(
                            session_id=session_id,
                            claims=verified_claims,
                            evidence_map=evidence_map,
                        )
                    except Exception as exc:
                        await repository.fail_session(session_id, "Failed to persist claims")
                        raise

            # 10. Grounded Report Synthesis
            t0 = time.perf_counter()
            try:
                report = await llm_provider.generate_report(
                    question=question,
                    chunks=retrieved_chunks,
                    claims=verified_claims,
                )
                result.report = report
            except Exception as exc:
                await repository.fail_session(session_id, str(exc))
                raise
            stage_latencies["report_synthesis"] = time.perf_counter() - t0

            # 11. Complete session if persisting
            t0 = time.perf_counter()
            if persist:
                try:
                    await repository.complete_session(session_id=session_id, report=report)
                except Exception as exc:
                    logger.error("Failed to complete session %s: %s", session_id, exc)
                    raise
            stage_latencies["persistence"] = time.perf_counter() - t0

            result.is_success = True
            return result

        finally:
            # Capture total time and metrics before cleanup (Correction 2)
            result.total_latency_seconds = time.perf_counter() - start_total
            result.stage_latencies = stage_latencies
            result.stage_errors = stage_errors
            result.search_failure_count = search_failure_count
            result.empty_retrieval_count = empty_retrieval_count

            # Session cleanup when persist=False (Evaluation mode)
            if not persist and session_id:
                try:
                    await repository.delete_session(session_id)
                    logger.debug("Successfully cleaned up evaluation session %s", session_id)
                except Exception as exc:
                    logger.warning("Could not delete evaluation session %s: %s", session_id, exc)
