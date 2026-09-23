from app.services.planner.base import BaseResearchPlanner
from app.services.planner.llm import LLMResearchPlanner


def get_research_planner() -> BaseResearchPlanner:
    """FastAPI dependency factory returning configured research planner instance."""
    return LLMResearchPlanner()
