import re
from typing import List, Optional, Set

from app.evaluation.schemas import (
    CaseEvaluationResult,
    CitationMetrics,
    ClaimMetrics,
    DimensionSpec,
    EvaluationCase,
    EvaluationRunSummary,
    PlannerMetrics,
    RetrievalMetrics,
)
from app.schemas.claim import ClaimVerificationStatus, VerifiedClaim
from app.schemas.evidence import EvidenceItem
from app.schemas.planner import ResearchPlan
from app.schemas.report import ResearchReport
from app.schemas.retrieval import RetrievedChunk
from app.services.pipeline.orchestrator import QueryRetrievalResult

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")


def calculate_planner_metrics(
    case: EvaluationCase,
    plan: Optional[ResearchPlan],
) -> PlannerMetrics:
    """Calculate deterministic quality metrics for research planning.

    Args:
        case: Benchmark evaluation case containing expected dimensions.
        plan: Generated ResearchPlan (or None if planner failed).

    Returns:
        PlannerMetrics with dimension coverage and query distinctness.
    """
    if plan is None or not plan.sub_questions:
        return PlannerMetrics(
            sub_question_count=0 if plan is not None else None,
            dimension_coverage=None,
            matched_dimensions=[],
            unmatched_dimensions=[d.name for d in case.expected_dimensions],
            distinctness=None,
        )

    sub_questions = plan.sub_questions
    sub_q_count = len(sub_questions)

    # 1. Distinctness of search queries
    search_queries = [sq.search_query.strip().lower() for sq in sub_questions if sq.search_query]
    unique_queries = set(search_queries)
    distinctness = (len(unique_queries) / sub_q_count) if sub_q_count > 0 else None

    # 2. Deterministic keyword matching for expected dimensions
    # A dimension is matched if ANY match term appears in the combined text of sub-questions
    combined_text = " ".join(
        [f"{sq.question} {sq.search_query}".lower() for sq in sub_questions]
    )

    matched: List[str] = []
    unmatched: List[str] = []

    for dim in case.expected_dimensions:
        is_matched = any(term.lower() in combined_text for term in dim.match_terms)
        if is_matched:
            matched.append(dim.name)
        else:
            unmatched.append(dim.name)

    total_expected = len(case.expected_dimensions)
    dimension_coverage = (len(matched) / total_expected) if total_expected > 0 else None

    return PlannerMetrics(
        sub_question_count=sub_q_count,
        dimension_coverage=dimension_coverage,
        matched_dimensions=matched,
        unmatched_dimensions=unmatched,
        distinctness=distinctness,
    )


def calculate_retrieval_metrics(
    query_retrieval_results: List[QueryRetrievalResult],
    retrieved_chunks: List[RetrievedChunk],
    total_sources: int,
) -> RetrievalMetrics:
    """Calculate deterministic quality metrics for semantic retrieval.

    Args:
        query_retrieval_results: Preserved per-query retrieval results before deduplication.
        retrieved_chunks: Deduplicated top-K chunks passed downstream.
        total_sources: Total unique source documents collected from search.

    Returns:
        RetrievalMetrics with hit rate, similarity, and source coverage.
    """
    query_count = len(query_retrieval_results)
    queries_with_hits = sum(1 for qr in query_retrieval_results if qr.hit)

    # Hit rate: strictly from preserved individual query results
    retrieval_hit_rate = (queries_with_hits / query_count) if query_count > 0 else None

    # Total chunks returned across all individual queries BEFORE final cross-query deduplication
    retrieval_count = sum(len(qr.chunks) for qr in query_retrieval_results)

    # Descriptive average cosine similarity across final deduplicated retrieved chunks
    # (Descriptive vector-proximity metric, not an answer-quality metric)
    dedup_chunk_count = len(retrieved_chunks)
    if dedup_chunk_count > 0:
        average_similarity = sum(c.similarity for c in retrieved_chunks) / dedup_chunk_count
    else:
        average_similarity = None

    # Evidence coverage ratio: unique retrieved sources / total source documents
    if total_sources > 0:
        unique_retrieved_urls = len({c.url for c in retrieved_chunks if c.url})
        evidence_coverage_ratio = min(1.0, unique_retrieved_urls / total_sources)
    else:
        evidence_coverage_ratio = None

    return RetrievalMetrics(
        retrieval_count=retrieval_count,
        retrieval_hit_rate=retrieval_hit_rate,
        average_similarity=average_similarity,
        evidence_coverage_ratio=evidence_coverage_ratio,
        query_count=query_count,
        queries_with_hits=queries_with_hits,
    )


def calculate_claim_metrics(
    claims: List[VerifiedClaim],
    evidence_items: List[EvidenceItem],
) -> ClaimMetrics:
    """Calculate deterministic quality metrics for claim extraction and verification.

    Args:
        claims: Verified claims evaluated by the claim verifier.
        evidence_items: All normalized evidence items provided to verification.

    Returns:
        ClaimMetrics with verification status rates and traceability integrity.
    """
    claim_count = len(claims)
    if claim_count == 0:
        return ClaimMetrics(
            claim_count=0,
            supported_claim_rate=None,
            partially_supported_claim_rate=None,
            unsupported_claim_rate=None,
            insufficient_evidence_claim_rate=None,
            evidence_support_ratio=None,
            traceability_integrity=True,
        )

    supported_count = sum(1 for c in claims if c.status == ClaimVerificationStatus.SUPPORTED)
    partially_count = sum(1 for c in claims if c.status == ClaimVerificationStatus.PARTIALLY_SUPPORTED)
    unsupported_count = sum(1 for c in claims if c.status == ClaimVerificationStatus.UNSUPPORTED)
    insufficient_count = sum(1 for c in claims if c.status == ClaimVerificationStatus.INSUFFICIENT_EVIDENCE)
    with_support_evidence_count = sum(1 for c in claims if len(c.supporting_evidence_ids) > 0)

    # Traceability integrity: assert every cited evidence ID resolves to a valid evidence item with chunk and document IDs
    evidence_map = {e.evidence_id: e for e in evidence_items}
    traceability_ok = True

    for c in claims:
        all_referenced_ids = set(c.evidence_ids) | set(c.supporting_evidence_ids)
        for eid in all_referenced_ids:
            item = evidence_map.get(eid)
            if not item or not item.chunk_id or not item.document_id:
                traceability_ok = False
                break
        if not traceability_ok:
            break

    return ClaimMetrics(
        claim_count=claim_count,
        supported_claim_rate=supported_count / claim_count,
        partially_supported_claim_rate=partially_count / claim_count,
        unsupported_claim_rate=unsupported_count / claim_count,
        insufficient_evidence_claim_rate=insufficient_count / claim_count,
        evidence_support_ratio=with_support_evidence_count / claim_count,
        traceability_integrity=traceability_ok,
    )


def extract_report_citations(report: Optional[ResearchReport]) -> List[str]:
    """Extract all citation references cited in a research report.

    Inspects both section.citations and in-text markdown markers [S1].

    Args:
        report: Synthesized ResearchReport.

    Returns:
        List of cited source identifier strings in order of occurrence.
    """
    if report is None:
        return []

    citations: List[str] = []

    # Check summary for inline citations
    if report.summary:
        citations.extend(CITATION_PATTERN.findall(report.summary))

    for section in report.sections:
        # Check inline citations in section text
        inline_matches = CITATION_PATTERN.findall(section.content or "")
        if inline_matches:
            citations.extend(inline_matches)
        elif section.citations:
            # Fall back to section.citations if no inline bracket citations were found
            citations.extend([str(c).strip() for c in section.citations if str(c).strip()])

    return citations


def calculate_citation_metrics(
    report: Optional[ResearchReport],
    total_sources: int,
) -> CitationMetrics:
    """Calculate deterministic quality metrics for report citations.

    Args:
        report: Synthesized ResearchReport.
        total_sources: Total source documents available for citation.

    Returns:
        CitationMetrics with precision, coverage, and invalid citation counts.
    """
    if report is None:
        return CitationMetrics(
            total_citations=0,
            valid_citation_count=0,
            invalid_citation_count=0,
            citation_precision=None,
            source_citation_coverage=None if total_sources == 0 else 0.0,
        )

    # Allowed valid source IDs from the report's source table
    valid_source_ids = {s.id for s in report.sources}
    cited_ids = extract_report_citations(report)
    total_citations = len(cited_ids)

    valid_count = 0
    invalid_count = 0
    unique_valid_cited: Set[str] = set()

    for cid in cited_ids:
        if cid in valid_source_ids:
            valid_count += 1
            unique_valid_cited.add(cid)
        else:
            invalid_count += 1

    # Precision: valid citations / total citations (None if 0 citations)
    citation_precision = (valid_count / total_citations) if total_citations > 0 else None

    # Coverage: unique valid sources cited / total sources (None if 0 sources)
    source_citation_coverage = (
        (len(unique_valid_cited) / total_sources) if total_sources > 0 else None
    )

    return CitationMetrics(
        total_citations=total_citations,
        valid_citation_count=valid_count,
        invalid_citation_count=invalid_count,
        citation_precision=citation_precision,
        source_citation_coverage=source_citation_coverage,
    )


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    """Calculate arithmetic mean of non-None values, or return None if empty."""
    valid = [v for v in values if v is not None]
    return (sum(valid) / len(valid)) if valid else None


def calculate_evaluation_run_summary(
    case_results: List[CaseEvaluationResult],
) -> EvaluationRunSummary:
    """Aggregate per-case evaluation results into a high-level summary.

    Args:
        case_results: List of CaseEvaluationResult for all evaluated cases.

    Returns:
        EvaluationRunSummary with success rate and mean metric scores.
    """
    total_cases = len(case_results)
    successful_cases = sum(1 for r in case_results if r.is_success)
    failed_cases = total_cases - successful_cases
    success_rate = (successful_cases / total_cases) if total_cases > 0 else 0.0

    latencies = [r.total_latency_seconds for r in case_results if r.is_success]
    hit_rates = [r.retrieval_metrics.retrieval_hit_rate for r in case_results if r.is_success]
    similarities = [r.retrieval_metrics.average_similarity for r in case_results if r.is_success]
    supported_rates = [r.claim_metrics.supported_claim_rate for r in case_results if r.is_success]
    unsupported_rates = [r.claim_metrics.unsupported_claim_rate for r in case_results if r.is_success]
    precisions = [r.citation_metrics.citation_precision for r in case_results if r.is_success]
    coverages = [r.planner_metrics.dimension_coverage for r in case_results if r.is_success]

    total_invalid_citations = sum(r.citation_metrics.invalid_citation_count for r in case_results)

    return EvaluationRunSummary(
        total_cases=total_cases,
        successful_cases=successful_cases,
        failed_cases=failed_cases,
        success_rate=success_rate,
        mean_total_latency_seconds=_safe_mean(latencies),
        mean_retrieval_hit_rate=_safe_mean(hit_rates),
        mean_retrieval_similarity=_safe_mean(similarities),
        mean_supported_claim_rate=_safe_mean(supported_rates),
        mean_unsupported_claim_rate=_safe_mean(unsupported_rates),
        mean_citation_precision=_safe_mean(precisions),
        total_invalid_citations=total_invalid_citations,
        mean_planner_dimension_coverage=_safe_mean(coverages),
    )
