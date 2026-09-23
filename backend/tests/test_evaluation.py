import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    DatasetError,
    filter_cases,
    load_evaluation_dataset,
)
from app.evaluation.metrics import (
    calculate_citation_metrics,
    calculate_claim_metrics,
    calculate_evaluation_run_summary,
    calculate_planner_metrics,
    calculate_retrieval_metrics,
    extract_report_citations,
)
from app.evaluation.runner import EvaluationRunner, save_report
from app.evaluation.schemas import (
    CaseEvaluationResult,
    CitationMetrics,
    ClaimMetrics,
    DimensionSpec,
    EvaluationCase,
    EvaluationDataset,
    PlannerMetrics,
    RetrievalMetrics,
)
from app.schemas.chunk import DocumentChunk
from app.schemas.claim import ClaimVerificationStatus, VerifiedClaim
from app.schemas.document import Document
from app.schemas.evidence import EvidenceItem
from app.schemas.planner import ResearchPlan, ResearchSubQuestion
from app.schemas.report import ResearchReport, ResearchSection, SourceReference
from app.schemas.research import SourceItem
from app.schemas.retrieval import RetrievedChunk
from app.services.pipeline.orchestrator import (
    PipelineExecutionResult,
    QueryRetrievalResult,
    ResearchOrchestrator,
)


# =====================================================================
# 1. Dataset Loading & Validation Tests
# =====================================================================

def test_load_default_dataset():
    """Verify that the default benchmark dataset loads cleanly and has 15 valid cases."""
    dataset = load_evaluation_dataset()
    assert isinstance(dataset, EvaluationDataset)
    assert dataset.version == "1.0.0"
    assert len(dataset.cases) == 15

    categories = {c.category for c in dataset.cases}
    expected_categories = {"science", "ai_ml", "software_engineering", "technology", "general_factual"}
    assert categories == expected_categories

    for case in dataset.cases:
        assert case.id.startswith("case-")
        assert len(case.question) > 10
        assert len(case.expected_dimensions) >= 3
        for dim in case.expected_dimensions:
            assert len(dim.name) > 0
            assert len(dim.match_terms) >= 1
            for term in dim.match_terms:
                assert term == term.lower(), f"Term '{term}' should be lowercase"


def test_filter_cases():
    """Verify case filtering by limit and case_id."""
    dataset = load_evaluation_dataset()

    # Limit
    limited = filter_cases(dataset, limit=3)
    assert len(limited) == 3

    # Case ID
    single = filter_cases(dataset, case_id="case-004")
    assert len(single) == 1
    assert single[0].id == "case-004"

    # Invalid ID
    with pytest.raises(ValueError, match="not found in dataset"):
        filter_cases(dataset, case_id="invalid-case-id")


def test_dataset_error_handling(tmp_path):
    """Verify dataset error handling for missing or malformed files."""
    # Missing file
    with pytest.raises(DatasetError, match="not found"):
        load_evaluation_dataset(str(tmp_path / "nonexistent.json"))

    # Malformed JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{broken json", encoding="utf-8")
    with pytest.raises(DatasetError, match="Failed to parse dataset JSON"):
        load_evaluation_dataset(str(bad_json))

    # Schema validation failure
    invalid_schema = tmp_path / "invalid.json"
    invalid_schema.write_text(json.dumps({"version": "1.0", "name": "bad"}), encoding="utf-8")
    with pytest.raises(DatasetError, match="Dataset validation failed"):
        load_evaluation_dataset(str(invalid_schema))


# =====================================================================
# 2. Planner Metrics Tests
# =====================================================================

def test_planner_metrics_perfect_coverage():
    """Verify planner metrics when all dimensions match."""
    case = EvaluationCase(
        id="case-test",
        question="What are fusion bottlenecks?",
        category="science",
        expected_dimensions=[
            DimensionSpec(name="Tritium", match_terms=["tritium", "breeding"]),
            DimensionSpec(name="Materials", match_terms=["neutron", "flux"]),
        ],
    )
    plan = ResearchPlan(
        original_question="What are fusion bottlenecks?",
        sub_questions=[
            ResearchSubQuestion(id="SQ1", question="How is tritium breeding achieved?", search_query="fusion tritium breeding", reason="Breeding requirement"),
            ResearchSubQuestion(id="SQ2", question="What is neutron flux impact?", search_query="neutron flux first wall damage", reason="Materials requirement"),
        ],
    )
    metrics = calculate_planner_metrics(case, plan)
    assert metrics.sub_question_count == 2
    assert metrics.dimension_coverage == 1.0
    assert metrics.matched_dimensions == ["Tritium", "Materials"]
    assert metrics.unmatched_dimensions == []
    assert metrics.distinctness == 1.0


def test_planner_metrics_partial_coverage():
    """Verify planner metrics when only some dimensions match."""
    case = EvaluationCase(
        id="case-test",
        question="What are fusion bottlenecks?",
        category="science",
        expected_dimensions=[
            DimensionSpec(name="Tritium", match_terms=["tritium"]),
            DimensionSpec(name="Materials", match_terms=["neutron"]),
        ],
    )
    plan = ResearchPlan(
        original_question="What are fusion bottlenecks?",
        sub_questions=[
            ResearchSubQuestion(id="SQ1", question="How is tritium managed?", search_query="tritium supply", reason="Supply"),
            ResearchSubQuestion(id="SQ2", question="What is plasma confinement?", search_query="tokamak plasma confinement", reason="Stability"),
        ],
    )
    metrics = calculate_planner_metrics(case, plan)
    assert metrics.dimension_coverage == 0.5
    assert metrics.matched_dimensions == ["Tritium"]
    assert metrics.unmatched_dimensions == ["Materials"]


def test_planner_metrics_null_handling():
    """Verify null handling when plan is missing or empty."""
    case = EvaluationCase(
        id="case-test",
        question="Q?",
        category="science",
        expected_dimensions=[DimensionSpec(name="Dim1", match_terms=["term"])],
    )
    # None plan
    m_none = calculate_planner_metrics(case, None)
    assert m_none.sub_question_count is None
    assert m_none.dimension_coverage is None
    assert m_none.distinctness is None
    assert m_none.unmatched_dimensions == ["Dim1"]

    # Empty expected dimensions
    case_empty = EvaluationCase(id="case-e", question="Q?", category="science", expected_dimensions=[])
    plan = ResearchPlan(original_question="Q?", sub_questions=[ResearchSubQuestion(id="SQ1", question="Q?", search_query="Q", reason="R")])
    m_empty = calculate_planner_metrics(case_empty, plan)
    assert m_empty.dimension_coverage is None


# =====================================================================
# 3. Retrieval Metrics Tests
# =====================================================================

def test_retrieval_metrics_normal():
    """Verify retrieval hit rate and similarity calculations."""
    # Query 1 returns c1, c2
    # Query 2 returns c2 (overlap), c3
    # Query 3 returns nothing
    c1 = RetrievedChunk(chunk_id="c1", document_id="d1", session_id="s1", chunk_index=0, title="T1", url="https://a.com", text="chunk 1", similarity=0.88)
    c2 = RetrievedChunk(chunk_id="c2", document_id="d1", session_id="s1", chunk_index=1, title="T1", url="https://a.com", text="chunk 2", similarity=0.92)
    c3 = RetrievedChunk(chunk_id="c3", document_id="d2", session_id="s1", chunk_index=0, title="T2", url="https://b.com", text="chunk 3", similarity=0.90)

    query_results = [
        QueryRetrievalResult(
            query="fusion power",
            query_type="original",
            chunks=[c1, c2],
            hit=True,
        ),
        QueryRetrievalResult(
            query="tritium breeding",
            query_type="sub_question",
            sub_question_id="SQ1",
            chunks=[c2, c3],
            hit=True,
        ),
        QueryRetrievalResult(
            query="neutron damage",
            query_type="sub_question",
            sub_question_id="SQ2",
            chunks=[],
            hit=False,
        ),
    ]
    # Deduplicated chunks: c1, c2, c3 (3 unique chunks)
    retrieved_chunks = [c1, c2, c3]

    metrics = calculate_retrieval_metrics(
        query_retrieval_results=query_results,
        retrieved_chunks=retrieved_chunks,
        total_sources=2,
    )

    assert metrics.query_count == 3
    assert metrics.queries_with_hits == 2
    assert metrics.retrieval_hit_rate == pytest.approx(2 / 3)
    # retrieval_count MUST be the total across individual queries before deduplication (2 + 2 + 0 = 4)
    assert metrics.retrieval_count == 4
    # average_similarity uses deduplicated chunks ( (0.88 + 0.92 + 0.90) / 3 = 0.90 )
    assert metrics.average_similarity == pytest.approx(0.90)
    assert metrics.evidence_coverage_ratio == pytest.approx(1.0)  # 2 unique urls out of 2 total sources


def test_retrieval_metrics_zero_denominators():
    """Verify strict null handling for zero queries, chunks, and sources."""
    metrics = calculate_retrieval_metrics(
        query_retrieval_results=[],
        retrieved_chunks=[],
        total_sources=0,
    )
    assert metrics.retrieval_hit_rate is None
    assert metrics.average_similarity is None
    assert metrics.evidence_coverage_ratio is None
    assert metrics.retrieval_count == 0
    assert metrics.query_count == 0


# =====================================================================
# 4. Claim Metrics Tests
# =====================================================================

def test_claim_metrics_distribution_and_traceability():
    """Verify claim verification rate distribution and traceability check."""
    evidence = [
        EvidenceItem(evidence_id="E1", chunk_id="c1", document_id="d1", session_id="s1", url="https://a.com", title="T", text="txt", similarity=0.9),
        EvidenceItem(evidence_id="E2", chunk_id="c2", document_id="d2", session_id="s1", url="https://b.com", title="T", text="txt", similarity=0.8),
    ]
    claims = [
        VerifiedClaim(id="C1", claim="Claim 1", status=ClaimVerificationStatus.SUPPORTED, evidence_ids=["E1"], supporting_evidence_ids=["E1"], reason="ok"),
        VerifiedClaim(id="C2", claim="Claim 2", status=ClaimVerificationStatus.PARTIALLY_SUPPORTED, evidence_ids=["E2"], supporting_evidence_ids=["E2"], reason="ok"),
        VerifiedClaim(id="C3", claim="Claim 3", status=ClaimVerificationStatus.UNSUPPORTED, evidence_ids=["E1"], supporting_evidence_ids=[], reason="no"),
        VerifiedClaim(id="C4", claim="Claim 4", status=ClaimVerificationStatus.INSUFFICIENT_EVIDENCE, evidence_ids=["E2"], supporting_evidence_ids=[], reason="unclear"),
    ]

    metrics = calculate_claim_metrics(claims, evidence)
    assert metrics.claim_count == 4
    assert metrics.supported_claim_rate == 0.25
    assert metrics.partially_supported_claim_rate == 0.25
    assert metrics.unsupported_claim_rate == 0.25
    assert metrics.insufficient_evidence_claim_rate == 0.25
    assert metrics.evidence_support_ratio == 0.50
    assert metrics.traceability_integrity is True


def test_claim_metrics_broken_traceability():
    """Verify that referencing an unknown evidence ID fails traceability integrity."""
    evidence = [EvidenceItem(evidence_id="E1", chunk_id="c1", document_id="d1", session_id="s1", url="u", title="t", text="t", similarity=0.9)]
    claims = [
        VerifiedClaim(id="C1", claim="Claim", status=ClaimVerificationStatus.SUPPORTED, evidence_ids=["E999"], supporting_evidence_ids=["E999"], reason="ok")
    ]
    metrics = calculate_claim_metrics(claims, evidence)
    assert metrics.traceability_integrity is False


def test_claim_metrics_zero_claims():
    """Verify null handling when claim count is zero."""
    metrics = calculate_claim_metrics([], [])
    assert metrics.claim_count == 0
    assert metrics.supported_claim_rate is None
    assert metrics.partially_supported_claim_rate is None
    assert metrics.unsupported_claim_rate is None
    assert metrics.insufficient_evidence_claim_rate is None
    assert metrics.evidence_support_ratio is None
    assert metrics.traceability_integrity is True


# =====================================================================
# 5. Citation Metrics Tests
# =====================================================================

def test_citation_metrics_valid_and_invalid():
    """Verify citation precision and coverage with valid and invalid IDs."""
    sources = [
        SourceReference(id="S1", title="Source 1", url="https://a.com"),
        SourceReference(id="S2", title="Source 2", url="https://b.com"),
    ]
    report = ResearchReport(
        title="Title",
        summary="Summary citing [S1].",
        sections=[
            ResearchSection(heading="H1", content="Content with [S2] and invalid [S99].", citations=["S2", "S99"]),
        ],
        sources=sources,
    )

    metrics = calculate_citation_metrics(report, total_sources=2)
    assert metrics.total_citations == 3
    assert metrics.valid_citation_count == 2
    assert metrics.invalid_citation_count == 1
    assert metrics.citation_precision == pytest.approx(2 / 3)
    assert metrics.source_citation_coverage == 1.0  # Both S1 and S2 were cited


def test_citation_metrics_zero_citations():
    """Verify citation precision is None when 0 citations are present."""
    report = ResearchReport(
        title="Title",
        summary="No citations here.",
        sections=[ResearchSection(heading="H", content="No citations here either.", citations=[])],
        sources=[SourceReference(id="S1", title="S1", url="https://a.com")],
    )
    metrics = calculate_citation_metrics(report, total_sources=1)
    assert metrics.total_citations == 0
    assert metrics.valid_citation_count == 0
    assert metrics.invalid_citation_count == 0
    assert metrics.citation_precision is None
    assert metrics.source_citation_coverage == 0.0


def test_citation_metrics_zero_sources_null_coverage():
    """Verify user correction 1: source_citation_coverage must be None (null), not 0.0, when total_sources=0."""
    report = ResearchReport(
        title="Title",
        summary="Text",
        sections=[],
        sources=[],
    )
    metrics = calculate_citation_metrics(report, total_sources=0)
    assert metrics.total_citations == 0
    assert metrics.citation_precision is None
    assert metrics.source_citation_coverage is None, "source_citation_coverage must be None when total_sources == 0"


# =====================================================================
# 6. Summary Aggregation Tests
# =====================================================================

def test_calculate_evaluation_run_summary():
    """Verify aggregation across multiple case results."""
    case1 = CaseEvaluationResult(
        case_id="case-001",
        question="Q1",
        category="science",
        is_success=True,
        total_latency_seconds=10.0,
        stage_latencies={"planning": 2.0},
        planner_metrics=PlannerMetrics(sub_question_count=2, dimension_coverage=1.0, distinctness=1.0),
        retrieval_metrics=RetrievalMetrics(retrieval_count=5, retrieval_hit_rate=1.0, average_similarity=0.9, evidence_coverage_ratio=0.8, query_count=2, queries_with_hits=2),
        claim_metrics=ClaimMetrics(claim_count=2, supported_claim_rate=1.0, unsupported_claim_rate=0.0, evidence_support_ratio=1.0, traceability_integrity=True),
        citation_metrics=CitationMetrics(total_citations=2, valid_citation_count=2, invalid_citation_count=0, citation_precision=1.0, source_citation_coverage=1.0),
        errors=[],
    )
    case2 = CaseEvaluationResult(
        case_id="case-002",
        question="Q2",
        category="ai_ml",
        is_success=True,
        total_latency_seconds=20.0,
        stage_latencies={"planning": 3.0},
        planner_metrics=PlannerMetrics(sub_question_count=2, dimension_coverage=0.5, distinctness=1.0),
        retrieval_metrics=RetrievalMetrics(retrieval_count=3, retrieval_hit_rate=0.5, average_similarity=0.8, evidence_coverage_ratio=0.6, query_count=2, queries_with_hits=1),
        claim_metrics=ClaimMetrics(claim_count=2, supported_claim_rate=0.5, unsupported_claim_rate=0.5, evidence_support_ratio=0.5, traceability_integrity=True),
        citation_metrics=CitationMetrics(total_citations=2, valid_citation_count=1, invalid_citation_count=1, citation_precision=0.5, source_citation_coverage=0.5),
        errors=[],
    )
    case3 = CaseEvaluationResult(
        case_id="case-003",
        question="Q3",
        category="technology",
        is_success=False,
        total_latency_seconds=5.0,
        stage_latencies={},
        planner_metrics=PlannerMetrics(sub_question_count=None, dimension_coverage=None, distinctness=None),
        retrieval_metrics=RetrievalMetrics(retrieval_count=0, retrieval_hit_rate=None, average_similarity=None, evidence_coverage_ratio=None, query_count=0, queries_with_hits=0),
        claim_metrics=ClaimMetrics(claim_count=0, supported_claim_rate=None, unsupported_claim_rate=None, evidence_support_ratio=None, traceability_integrity=True),
        citation_metrics=CitationMetrics(total_citations=0, valid_citation_count=0, invalid_citation_count=0, citation_precision=None, source_citation_coverage=None),
        errors=["Search error"],
    )

    summary = calculate_evaluation_run_summary([case1, case2, case3])

    assert summary.total_cases == 3
    assert summary.successful_cases == 2
    assert summary.failed_cases == 1
    assert summary.success_rate == pytest.approx(2 / 3)
    assert summary.mean_total_latency_seconds == pytest.approx(15.0)
    assert summary.mean_retrieval_hit_rate == pytest.approx(0.75)
    assert summary.mean_retrieval_similarity == pytest.approx(0.85)
    assert summary.mean_supported_claim_rate == pytest.approx(0.75)
    assert summary.mean_unsupported_claim_rate == pytest.approx(0.25)
    assert summary.mean_citation_precision == pytest.approx(0.75)
    assert summary.total_invalid_citations == 1
    assert summary.mean_planner_dimension_coverage == pytest.approx(0.75)


# =====================================================================
# 7. Orchestrator Session Isolation & Capture Tests (Correction 2)
# =====================================================================

@pytest.mark.anyio
async def test_orchestrator_persist_false_cleans_up_and_captures():
    """Verify that persist=False captures intermediate data before calling delete_session in finally."""
    orchestrator = ResearchOrchestrator()

    # Mocks
    mock_planner = AsyncMock()
    mock_planner.create_plan.return_value = ResearchPlan(
        original_question="What is fusion?",
        sub_questions=[ResearchSubQuestion(id="SQ1", question="Sub Q?", search_query="query 1", reason="r1")],
    )

    mock_search = AsyncMock()
    mock_search.search.return_value = [SourceItem(title="Src", url="https://example.com", content="snp")]

    mock_extractor = AsyncMock()
    mock_extractor.extract_many.return_value = [Document(url="https://example.com", title="Src", text="Doc content", char_count=11)]

    mock_chunker = MagicMock()
    chunk = DocumentChunk(
        document_id="d1",
        session_id="sess-123",
        chunk_index=0,
        text="chunk content",
        char_count=13,
    )
    mock_chunker.chunk_documents.return_value = [chunk]

    mock_embedding = AsyncMock()
    mock_embedding.embed_texts.return_value = [[0.1] * 1536]

    retrieved_chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        session_id="sess-123",
        chunk_index=0,
        title="Src",
        url="https://example.com",
        text="chunk content",
        similarity=0.92,
    )
    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = [retrieved_chunk]

    mock_claim_extractor = AsyncMock()
    mock_claim_extractor.extract_claims.return_value = []

    mock_claim_verifier = AsyncMock()
    mock_claim_verifier.verify_claims.return_value = []

    report = ResearchReport(
        title="Test Report",
        summary="Summary [S1]",
        sections=[ResearchSection(heading="H", content="Content", citations=["S1"])],
        sources=[SourceReference(id="S1", title="Src", url="https://example.com")],
    )
    mock_llm = AsyncMock()
    mock_llm.generate_report.return_value = report

    mock_repo = AsyncMock()
    mock_repo.create_session.return_value = "sess-123"
    mock_repo.save_sources.return_value = {"https://example.com": "s1"}
    mock_repo.save_documents.return_value = {"https://example.com": "d1"}
    mock_repo.save_chunks.return_value = ["c1"]
    mock_repo.save_claims.return_value = []

    result = await orchestrator.run(
        question="What is fusion?",
        planner=mock_planner,
        search_provider=mock_search,
        extractor=mock_extractor,
        embedding_provider=mock_embedding,
        chunker=mock_chunker,
        retriever=mock_retriever,
        claim_extractor=mock_claim_extractor,
        claim_verifier=mock_claim_verifier,
        llm_provider=mock_llm,
        repository=mock_repo,
        persist=False,
    )

    # 1. Assert result captured all data
    assert result.is_success is True
    assert result.session_id == "sess-123"
    assert result.plan is not None
    assert len(result.sources) == 1
    assert len(result.documents) == 1
    assert len(result.chunks) == 1
    assert len(result.retrieved_chunks) == 1
    assert len(result.query_retrieval_results) == 2  # original question + 1 sub-question
    assert result.report == report
    assert result.total_latency_seconds > 0.0
    assert "planning" in result.stage_latencies
    assert "retrieval" in result.stage_latencies

    # 2. Assert delete_session was called, and complete_session was NOT called
    mock_repo.delete_session.assert_awaited_once_with("sess-123")
    mock_repo.complete_session.assert_not_awaited()


@pytest.mark.anyio
async def test_orchestrator_persist_true_does_not_delete():
    """Verify that persist=True calls complete_session and never calls delete_session."""
    orchestrator = ResearchOrchestrator()

    mock_planner = AsyncMock()
    mock_planner.create_plan.return_value = ResearchPlan(
        original_question="What is fusion?",
        sub_questions=[ResearchSubQuestion(id="SQ1", question="What is fusion?", search_query="fusion", reason="core")],
    )

    mock_search = AsyncMock()
    mock_search.search.return_value = []

    mock_extractor = AsyncMock()
    mock_extractor.extract_many.return_value = []

    mock_chunker = MagicMock()
    mock_chunker.chunk_documents.return_value = []

    mock_embedding = AsyncMock()
    mock_retriever = AsyncMock()
    mock_claim_extractor = AsyncMock()
    mock_claim_verifier = AsyncMock()

    report = ResearchReport(title="T", summary="S", sections=[], sources=[])
    mock_llm = AsyncMock()
    mock_llm.generate_report.return_value = report

    mock_repo = AsyncMock()
    mock_repo.create_session.return_value = "sess-prod"

    result = await orchestrator.run(
        question="What is fusion?",
        planner=mock_planner,
        search_provider=mock_search,
        extractor=mock_extractor,
        embedding_provider=mock_embedding,
        chunker=mock_chunker,
        retriever=mock_retriever,
        claim_extractor=mock_claim_extractor,
        claim_verifier=mock_claim_verifier,
        llm_provider=mock_llm,
        repository=mock_repo,
        persist=True,
    )

    assert result.is_success is True
    mock_repo.complete_session.assert_awaited_once_with(session_id="sess-prod", report=report)
    mock_repo.delete_session.assert_not_awaited()


# =====================================================================
# 8. Runner & Report Saving Tests
# =====================================================================

@pytest.mark.anyio
async def test_evaluation_runner_run(tmp_path):
    """Verify EvaluationRunner executes cases with persist=False and generates saved report."""
    mock_orchestrator = AsyncMock()
    mock_orchestrator.run.return_value = PipelineExecutionResult(
        session_id="sess-eval-1",
        question="How does CRISPR base editing work?",
        plan=ResearchPlan(
            original_question="How does CRISPR base editing work?",
            sub_questions=[
                ResearchSubQuestion(id="SQ1", question="What is base editing?", search_query="crispr base edit deaminase", reason="Mechanism"),
            ],
        ),
        sources=[SourceItem(title="S1", url="https://crispr.org", content="snp")],
        documents=[Document(url="https://crispr.org", title="S1", text="text", char_count=4)],
        chunks=[DocumentChunk(document_id="d1", session_id="s1", chunk_index=0, text="t", char_count=1)],
        query_retrieval_results=[
            QueryRetrievalResult(
                query="What is base editing?",
                query_type="sub_question",
                sub_question_id="SQ1",
                chunks=[RetrievedChunk(chunk_id="c1", document_id="d1", session_id="s1", chunk_index=0, title="S1", url="https://crispr.org", text="t", similarity=0.91)],
                hit=True,
            )
        ],
        retrieved_chunks=[RetrievedChunk(chunk_id="c1", document_id="d1", session_id="s1", chunk_index=0, title="S1", url="https://crispr.org", text="t", similarity=0.91)],
        evidence_items=[EvidenceItem(evidence_id="E1", chunk_id="c1", document_id="d1", session_id="s1", url="u", title="t", text="t", similarity=0.91)],
        extracted_claims=[],
        verified_claims=[
            VerifiedClaim(id="C1", claim="Claim", status=ClaimVerificationStatus.SUPPORTED, evidence_ids=["E1"], supporting_evidence_ids=["E1"], reason="ok")
        ],
        report=ResearchReport(
            title="CRISPR Report",
            summary="Summary citing [S1].",
            sections=[ResearchSection(heading="H", content="Content [S1]", citations=["S1"])],
            sources=[SourceReference(id="S1", title="S1", url="https://crispr.org")],
        ),
        stage_latencies={"planning": 1.2, "retrieval": 0.5},
        total_latency_seconds=3.5,
        is_success=True,
    )

    dataset = EvaluationDataset(
        version="1.0.0",
        name="test_dataset",
        description="Dataset for testing",
        cases=[
            EvaluationCase(
                id="case-crispr",
                question="How does CRISPR base editing work?",
                category="science",
                expected_dimensions=[
                    DimensionSpec(name="Base editing mechanism", match_terms=["base edit", "deaminase"]),
                ],
            )
        ],
    )

    runner = EvaluationRunner(
        orchestrator=mock_orchestrator,
        planner=AsyncMock(),
        search_provider=AsyncMock(),
        extractor=AsyncMock(),
        embedding_provider=AsyncMock(),
        chunker=MagicMock(),
        retriever=AsyncMock(),
        claim_extractor=AsyncMock(),
        claim_verifier=AsyncMock(),
        llm_provider=AsyncMock(),
        repository=AsyncMock(),
    )

    progress_calls = []

    def on_progress(idx, total, res):
        progress_calls.append((idx, total, res.case_id))

    report = await runner.run(dataset, on_case_progress=on_progress)

    # Check progress callback was invoked
    assert len(progress_calls) == 1
    assert progress_calls[0] == (1, 1, "case-crispr")

    # Check orchestrator was invoked with persist=False
    mock_orchestrator.run.assert_awaited_once()
    assert mock_orchestrator.run.call_args.kwargs["persist"] is False

    # Check report contents
    assert report.summary.total_cases == 1
    assert report.summary.successful_cases == 1
    assert report.summary.failed_cases == 0
    assert report.summary.success_rate == 1.0
    assert report.case_results[0].planner_metrics.dimension_coverage == 1.0
    assert report.case_results[0].citation_metrics.citation_precision == 1.0

    # Save report
    out_file = tmp_path / "out_report.json"
    saved_path = save_report(report, str(out_file))
    assert saved_path.exists()
    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved_data["summary"]["successful_cases"] == 1
    assert saved_data["case_results"][0]["case_id"] == "case-crispr"
