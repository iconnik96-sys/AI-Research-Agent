import json
import logging
from pathlib import Path
from typing import List, Optional

from app.evaluation.schemas import EvaluationCase, EvaluationDataset

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "data" / "research_benchmark_v1.json"


class DatasetError(Exception):
    """Exception raised when an evaluation dataset cannot be loaded or validated."""
    pass


def load_evaluation_dataset(path: Optional[str] = None) -> EvaluationDataset:
    """Load and validate an evaluation benchmark dataset from a JSON file.

    Args:
        path: Optional file path string. If None, defaults to research_benchmark_v1.json.

    Returns:
        Validated EvaluationDataset instance.

    Raises:
        DatasetError: If the file does not exist, cannot be decoded, or fails schema validation.
    """
    file_path = Path(path).resolve() if path else DEFAULT_DATASET_PATH

    if not file_path.exists() or not file_path.is_file():
        raise DatasetError(f"Evaluation dataset file not found at: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Failed to parse dataset JSON at {file_path}: {exc}") from exc
    except Exception as exc:
        raise DatasetError(f"Failed to read dataset file at {file_path}: {exc}") from exc

    try:
        return EvaluationDataset.model_validate(data)
    except Exception as exc:
        raise DatasetError(f"Dataset validation failed for {file_path}: {exc}") from exc


def filter_cases(
    dataset: EvaluationDataset,
    limit: Optional[int] = None,
    case_id: Optional[str] = None,
) -> List[EvaluationCase]:
    """Filter benchmark cases by case ID and/or limit.

    Args:
        dataset: The loaded EvaluationDataset.
        limit: Optional maximum number of cases to return.
        case_id: Optional specific case ID to evaluate.

    Returns:
        Filtered list of EvaluationCase objects.

    Raises:
        ValueError: If case_id is specified but not found in the dataset.
    """
    cases = dataset.cases

    if case_id:
        matching = [c for c in cases if c.id == case_id]
        if not matching:
            available_ids = [c.id for c in cases]
            raise ValueError(f"Case '{case_id}' not found in dataset. Available IDs: {available_ids}")
        cases = matching

    if limit is not None and limit > 0:
        cases = cases[:limit]

    return cases
