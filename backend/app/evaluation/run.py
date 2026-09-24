import argparse
import asyncio
import datetime
import os
import sys
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.evaluation.dataset import DEFAULT_DATASET_PATH, load_evaluation_dataset
from app.evaluation.runner import EvaluationRunner, save_report
from app.evaluation.schemas import CaseEvaluationResult, EvaluationReport


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation benchmark."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation benchmark for AI Research Agent",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to evaluation dataset JSON file (default: research_benchmark_v1.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation report JSON (default: ../eval_results/eval_run_<timestamp>.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation cases to execute",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Specific case ID to evaluate (e.g. 'case-001')",
    )
    return parser.parse_args()


def format_pct(value: Optional[float]) -> str:
    """Format float as percentage or 'N/A' if None."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def format_val(value: Optional[float], decimals: int = 3) -> str:
    """Format float with fixed decimals or 'N/A' if None."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def print_banner(dataset_path: str, case_count: int, model: str, embedding_model: str) -> None:
    print("=" * 64)
    print("              AI RESEARCH AGENT BENCHMARK V1")
    print("=" * 64)
    print(f"Dataset:         {Path(dataset_path).name} ({case_count} cases)")
    print(f"Timestamp:       {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"Model:           {model} | Embeddings: {embedding_model}")
    print()
    print("Running evaluation...")


def print_case_progress(index: int, total: int, result: CaseEvaluationResult) -> None:
    status_str = "[OK]" if result.is_success else "[FAILED]"
    q_snippet = (result.question[:42] + "...") if len(result.question) > 45 else result.question
    lat_str = f"({result.total_latency_seconds:.1f}s)"
    print(f"  [{index}/{total}] {result.case_id}: {q_snippet:<45} {status_str} {lat_str}")
    if not result.is_success and result.errors:
        print(f"      Error: {result.errors[0]}")


def print_summary(report: EvaluationReport, output_path: str) -> None:
    s = report.summary
    print()
    print("=" * 64)
    print("                     EVALUATION SUMMARY")
    print("=" * 64)
    pct_pass = f"{s.success_rate * 100:.1f}%"
    print(f"Cases:                {s.total_cases} total | {s.successful_cases} passed | {s.failed_cases} failed ({pct_pass})")
    lat_avg = f" (avg: {s.mean_total_latency_seconds:.1f}s/case)" if s.mean_total_latency_seconds else ""
    total_time = sum(r.total_latency_seconds for r in report.case_results)
    print(f"Total Latency:        {total_time:.1f}s{lat_avg}")
    print()
    print("Planner Quality:")
    print(f"  - Dimension Coverage: {format_pct(s.mean_planner_dimension_coverage)}  [Keyword alignment with expected dimensions]")
    print()
    print("Retrieval Quality:")
    print(f"  - Hit Rate:           {format_pct(s.mean_retrieval_hit_rate)}")
    print(f"  - Avg Similarity:     {format_val(s.mean_retrieval_similarity)}  [Descriptive embedding proximity]")
    print()
    print("Evidence Grounding:")
    print(f"  - Supported Claims:   {format_pct(s.mean_supported_claim_rate)}")
    print(f"  - Unsupported Claims: {format_pct(s.mean_unsupported_claim_rate)}")
    print()
    print("Citation Integrity:")
    print(f"  - Citation Precision: {format_pct(s.mean_citation_precision)}")
    print(f"  - Invalid Citations:  {s.total_invalid_citations}")
    print("=" * 64)
    print(f"Report saved to: {output_path}")
    print("=" * 64)


async def main_async() -> int:
    args = parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"../eval_results/eval_run_{ts}.json"

    dataset = load_evaluation_dataset(args.dataset)
    runner = EvaluationRunner()

    print_banner(
        dataset_path=args.dataset,
        case_count=len(dataset.cases),
        model=getattr(runner.llm_provider, "model", settings.LLM_MODEL),
        embedding_model=getattr(runner.embedding_provider, "model", "gte-small"),
    )

    report = await runner.run(
        dataset=dataset,
        limit=args.limit,
        case_id=args.case_id,
        on_case_progress=print_case_progress,
    )

    saved_file = save_report(report, output_path)
    print_summary(report, str(saved_file))

    return 0 if report.summary.failed_cases == 0 else 1


def main() -> None:
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
