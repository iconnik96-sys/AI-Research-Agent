class PlannerError(Exception):
    """Base exception for all research planner errors."""
    pass


class PlannerConfigError(PlannerError):
    """Raised when planner configuration or API keys are missing/invalid."""
    pass


class PlannerTimeoutError(PlannerError):
    """Raised when a planning request times out."""
    pass


class PlannerNetworkError(PlannerError):
    """Raised when network transport issues occur during planning."""
    pass


class PlannerResponseError(PlannerError):
    """Raised when the LLM planner returns an unparseable, invalid, or oversized response."""
    pass
