from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DimensionSpec(BaseModel):
    """Specification of an expected research dimension with explicit match terms."""

    name: str = Field(..., min_length=1, description="Name or label of the research dimension")
    match_terms: List[str] = Field(
        ...,
        min_length=1,
        description="Deterministic keyword terms to match against planner sub-questions and search queries",
    )


class EvaluationCase(BaseModel):
    """A benchmark evaluation case representing a realistic research question."""

    id: str = Field(..., min_length=1, description="Unique case identifier (e.g. 'case-001')")
    question: str = Field(..., min_length=1, description="The research question to investigate")
    category: str = Field(
        ...,
        min_length=1,
        description="Domain category (e.g. 'ai_ml', 'software_engineering', 'technology', 'science', 'general_factual')",
    )
    expected_dimensions: List[DimensionSpec] = Field(
        default_factory=list,
        description="Key semantic subtopics or research dimensions expected from the planner",
    )
    expected_source_types: Optional[List[str]] = Field(
        default=None,
        description="Optional expected source characteristics",
    )
    reference_facts: Optional[List[str]] = Field(
        default=None,
        description="Optional reference anchor facts or expected findings",
    )


class EvaluationDataset(BaseModel):
    """Versioned container of benchmark evaluation cases."""

    version: str = Field(..., description="Dataset schema/semantic version string (e.g. '1.0.0')")
    name: str = Field(..., description="Name of the benchmark suite")
    description: str = Field(..., description="Description of the benchmark scope and objectives")
    cases: List[EvaluationCase] = Field(..., min_length=1, description="List of benchmark cases")


class PlannerMetrics(BaseModel):
    """Deterministic quality metrics for the research planner."""

    sub_question_count: Optional[int] = Field(
        default=None,
        description="Total sub-questions produced by planner (null if planner failed)",
    )
    dimension_coverage: Optional[float] = Field(
        default=None,
        description="Matched expected dimensions / total expected dimensions (null if no expected dimensions)",
    )
    matched_dimensions: List[str] = Field(
        default_factory=list,
        description="Names of expected dimensions that matched planner output",
    )
    unmatched_dimensions: List[str] = Field(
        default_factory=list,
        description="Names of expected dimensions not covered by planner output",
    )
    distinctness: Optional[float] = Field(
        default=None,
        description="Unique search queries / total sub-questions (null if 0 sub-questions)",
    )


class RetrievalMetrics(BaseModel):
    """Deterministic quality metrics for semantic retrieval."""

    retrieval_count: int = Field(
        default=0,
        description="Total chunks returned across all retrieval queries before cross-query deduplication",
    )
    retrieval_hit_rate: Optional[float] = Field(
        default=None,
        description="Queries with >= 1 chunk / total queries (null if 0 queries)",
    )
    average_similarity: Optional[float] = Field(
        default=None,
        description="Descriptive mean cosine similarity of deduplicated chunks (vector proximity metric, not answer correctness)",
    )
    evidence_coverage_ratio: Optional[float] = Field(
        default=None,
        description="Unique documents in retrieved chunks / total source documents (null if 0 sources)",
    )
    query_count: int = Field(default=0, description="Total queries executed for retrieval")
    queries_with_hits: int = Field(default=0, description="Queries that yielded at least one chunk")


class ClaimMetrics(BaseModel):
    """Deterministic quality metrics for evidence grounding and claim verification."""

    claim_count: int = Field(default=0, description="Total candidate factual claims extracted")
    supported_claim_rate: Optional[float] = Field(
        default=None,
        description="Proportion of claims verified as SUPPORTED (null if 0 claims)",
    )
    partially_supported_claim_rate: Optional[float] = Field(
        default=None,
        description="Proportion of claims verified as PARTIALLY_SUPPORTED (null if 0 claims)",
    )
    unsupported_claim_rate: Optional[float] = Field(
        default=None,
        description="Proportion of claims verified as UNSUPPORTED (null if 0 claims)",
    )
    insufficient_evidence_claim_rate: Optional[float] = Field(
        default=None,
        description="Proportion of claims classified as INSUFFICIENT_EVIDENCE (null if 0 claims)",
    )
    evidence_support_ratio: Optional[float] = Field(
        default=None,
        description="Proportion of claims with at least one verified supporting evidence ID (null if 0 claims)",
    )
    traceability_integrity: bool = Field(
        default=True,
        description="True if all cited evidence IDs resolve without broken links to chunks and documents",
    )


class CitationMetrics(BaseModel):
    """Deterministic quality metrics for report citations."""

    total_citations: int = Field(default=0, description="Total citation references in report text")
    valid_citation_count: int = Field(default=0, description="Citations matching a declared source ID")
    invalid_citation_count: int = Field(default=0, description="Hallucinated or unknown citation markers")
    citation_precision: Optional[float] = Field(
        default=None,
        description="Valid citations / total citations (null if 0 citations)",
    )
    source_citation_coverage: Optional[float] = Field(
        default=None,
        description="Unique valid sources cited / total sources supplied in report (null if 0 sources)",
    )


class CaseEvaluationResult(BaseModel):
    """Complete evaluation result for a single benchmark case."""

    case_id: str = Field(..., description="Benchmark case ID")
    question: str = Field(..., description="Research question evaluated")
    category: str = Field(..., description="Domain category")
    is_success: bool = Field(..., description="True if research pipeline completed cleanly")
    total_latency_seconds: float = Field(..., description="Total wall-clock execution time")
    stage_latencies: Dict[str, float] = Field(default_factory=dict, description="Elapsed seconds per stage")
    planner_metrics: PlannerMetrics = Field(..., description="Planner quality metrics")
    retrieval_metrics: RetrievalMetrics = Field(..., description="Retrieval quality metrics")
    claim_metrics: ClaimMetrics = Field(..., description="Claim and evidence grounding metrics")
    citation_metrics: CitationMetrics = Field(..., description="Citation integrity metrics")
    errors: List[str] = Field(default_factory=list, description="Any warnings or errors recorded")


class EvaluationRunSummary(BaseModel):
    """Aggregated summary statistics across an entire benchmark run."""

    total_cases: int = Field(..., description="Total cases evaluated")
    successful_cases: int = Field(..., description="Number of successfully completed cases")
    failed_cases: int = Field(..., description="Number of failed cases")
    success_rate: float = Field(..., description="Successful cases / total cases")
    mean_total_latency_seconds: Optional[float] = Field(default=None, description="Average latency across cases")
    mean_retrieval_hit_rate: Optional[float] = Field(default=None, description="Average query retrieval hit rate")
    mean_retrieval_similarity: Optional[float] = Field(default=None, description="Average retrieved chunk similarity")
    mean_supported_claim_rate: Optional[float] = Field(default=None, description="Average supported claim rate")
    mean_unsupported_claim_rate: Optional[float] = Field(default=None, description="Average unsupported claim rate")
    mean_citation_precision: Optional[float] = Field(default=None, description="Average citation precision")
    total_invalid_citations: int = Field(default=0, description="Total invalid citations across all reports")
    mean_planner_dimension_coverage: Optional[float] = Field(default=None, description="Average planner coverage")


class EvaluationReport(BaseModel):
    """Full machine-readable evaluation report persisted as JSON."""

    metadata: Dict[str, Any] = Field(..., description="Run metadata including models and configuration")
    summary: EvaluationRunSummary = Field(..., description="High-level aggregate summary")
    case_results: List[CaseEvaluationResult] = Field(..., description="Per-case detailed results")
