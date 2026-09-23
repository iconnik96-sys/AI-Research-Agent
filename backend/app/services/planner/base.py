from abc import ABC, abstractmethod

from app.schemas.planner import ResearchPlan


class BaseResearchPlanner(ABC):
    """Abstract interface for research planning components."""

    @abstractmethod
    async def create_plan(self, question: str) -> ResearchPlan:
        """Decompose a research question into a structured research plan.

        Args:
            question: The user's original research question.

        Returns:
            A validated ResearchPlan containing focused sub-questions and search queries.

        Raises:
            PlannerConfigError: If required configuration or API keys are missing.
            PlannerTimeoutError: If the planning request times out.
            PlannerNetworkError: If network connectivity fails.
            PlannerResponseError: If the model returns invalid, unparseable, or excessive sub-questions.
            PlannerError: For general unexpected planning failures.
        """
        pass
