import datetime
import logging
from pathlib import Path
from typing import Callable, List, Optional

from app.core.config import settings
from app.evaluation.dataset import filter_cases
from app.evaluation.metrics import (
    calculate_citation_metrics,
    calculate_claim_metrics,
    calculate_evaluation_run_summary,
    calculate_planner_metrics,
    calculate_retrieval_metrics,
)
from app.evaluation.schemas import (
    CaseEvaluationResult,
    EvaluationCase,
    EvaluationDataset,
    EvaluationReport,
)
from app.services.chunking import TextChunker, get_text_chunker
from app.services.claim import (
    BaseClaimExtractor,
    BaseClaimVerifier,
    get_claim_extractor,
    get_claim_verifier,
)
from app.services.embedding import BaseEmbeddingProvider, get_embedding_provider
from app.services.extraction import WebpageExtractor, get_webpage_extractor
from app.services.llm import BaseLLMProvider, get_llm_provider
from app.services.persistence import BaseResearchRepository, get_research_repository
from app.services.pipeline import ResearchOrchestrator
from app.services.planner import BaseResearchPlanner, get_research_planner
from app.services.retrieval import BaseRetriever, get_retriever
from app.services.search import BaseSearchProvider, get_search_provider

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Evaluation runner that orchestrates benchmark execution and metric calculation."""

    def __init__(
        self,
        orchestrator: Optional[ResearchOrchestrator] = None,
        planner: Optional[BaseResearchPlanner] = None,
        search_provider: Optional[BaseSearchProvider] = None,
        extractor: Optional[WebpageExtractor] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        chunker: Optional[TextChunker] = None,
        retriever: Optional[BaseRetriever] = None,
        claim_extractor: Optional[BaseClaimExtractor] = None,
        claim_verifier: Optional[BaseClaimVerifier] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        repository: Optional[BaseResearchRepository] = None,
    ):
        self.orchestrator = orchestrator or ResearchOrchestrator()
        self.planner = planner or get_research_planner()
        self.search_provider = search_provider or get_search_provider()
        self.extractor = extractor or get_webpage_extractor()
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.chunker = chunker or get_text_chunker()
        self.retriever = retriever or get_retriever()
        self.claim_extractor = claim_extractor or get_claim_extractor()
        self.claim_verifier = claim_verifier or get_claim_verifier()
        self.llm_provider = llm_provider or get_llm_provider()
        self.repository = repository or get_research_repository()

    async def evaluate_case(self, case: EvaluationCase) -> CaseEvaluationResult:
        """Run the research pipeline on a single evaluation case with persist=False.

        Args:
            case: Benchmark case to evaluate.

        Returns:
            CaseEvaluationResult with calculated metrics and status.
        """
        try:
            pipeline_result = await self.orchestrator.run(
                question=case.question,
                planner=self.planner,
                search_provider=self.search_provider,
                extractor=self.extractor,
                embedding_provider=self.embedding_provider,
                chunker=self.chunker,
                retriever=self.retriever,
                claim_extractor=self.claim_extractor,
                claim_verifier=self.claim_verifier,
                llm_provider=self.llm_provider,
                repository=self.repository,
                persist=False,
            )

            planner_metrics = calculate_planner_metrics(case, pipeline_result.plan)
            retrieval_metrics = calculate_retrieval_metrics(
                pipeline_result.query_retrieval_results,
                pipeline_result.retrieved_chunks,
                total_sources=len(pipeline_result.sources),
            )
            claim_metrics = calculate_claim_metrics(
                pipeline_result.verified_claims,
                pipeline_result.evidence_items,
            )
            citation_metrics = calculate_citation_metrics(
                pipeline_result.report,
                total_sources=len(pipeline_result.sources),
            )

            errors = list(pipeline_result.stage_errors)
            if not pipeline_result.is_success:
                errors.append("Pipeline completed with failure status")

            return CaseEvaluationResult(
                case_id=case.id,
                question=case.question,
                category=case.category,
                is_success=pipeline_result.is_success,
                total_latency_seconds=pipeline_result.total_latency_seconds,
                stage_latencies=pipeline_result.stage_latencies,
                planner_metrics=planner_metrics,
                retrieval_metrics=retrieval_metrics,
                claim_metrics=claim_metrics,
                citation_metrics=citation_metrics,
                errors=errors,
            )
        except Exception as exc:
            logger.error("Evaluation execution failed for case %s: %s", case.id, exc, exc_info=True)
            return CaseEvaluationResult(
                case_id=case.id,
                question=case.question,
                category=case.category,
                is_success=False,
                total_latency_seconds=0.0,
                stage_latencies={},
                planner_metrics=calculate_planner_metrics(case, None),
                retrieval_metrics=calculate_retrieval_metrics([], [], 0),
                claim_metrics=calculate_claim_metrics([], []),
                citation_metrics=calculate_citation_metrics(None, 0),
                errors=[str(exc)],
            )

    async def run(
        self,
        dataset: EvaluationDataset,
        limit: Optional[int] = None,
        case_id: Optional[str] = None,
        on_case_progress: Optional[Callable[[int, int, CaseEvaluationResult], None]] = None,
    ) -> EvaluationReport:
        """Run the full benchmark evaluation across specified cases.

        Args:
            dataset: The loaded EvaluationDataset.
            limit: Optional maximum number of cases to run.
            case_id: Optional specific case ID to run.
            on_case_progress: Optional callback invoked after each case completes.

        Returns:
            EvaluationReport containing aggregate summary and per-case results.
        """
        cases = filter_cases(dataset, limit=limit, case_id=case_id)
        total_cases = len(cases)
        results: List[CaseEvaluationResult] = []

        for index, case in enumerate(cases, start=1):
            result = await self.evaluate_case(case)
            results.append(result)
            if on_case_progress:
                try:
                    on_case_progress(index, total_cases, result)
                except Exception as exc:
                    logger.warning("Error in progress callback: %s", exc)

        summary = calculate_evaluation_run_summary(results)

        model_name = getattr(self.llm_provider, "model", None) or getattr(settings, "LLM_MODEL", "unknown")
        embedding_model = getattr(self.embedding_provider, "model", None) or getattr(
            settings, "EMBEDDING_MODEL", "unknown"
        )

        metadata = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset_name": dataset.name,
            "dataset_version": dataset.version,
            "model": model_name,
            "embedding_model": embedding_model,
            "total_cases_evaluated": len(results),
        }

        return EvaluationReport(
            metadata=metadata,
            summary=summary,
            case_results=results,
        )


def save_report(report: EvaluationReport, output_path: str) -> Path:
    """Save an EvaluationReport to disk as formatted JSON.

    Args:
        report: EvaluationReport to save.
        output_path: Destination file path.

    Returns:
        Resolved Path of written file.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    return path
