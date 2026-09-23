from app.core.config import settings
from app.services.persistence.base import BaseResearchRepository
from app.services.persistence.exceptions import DatabaseConfigError
from app.services.persistence.repository import SQLAlchemyResearchRepository


def get_research_repository() -> BaseResearchRepository:
    """Dependency provider factory returning the configured research repository.

    Raises:
        DatabaseConfigError: If DATABASE_URL is not set or empty.
    """
    if not settings.DATABASE_URL or not settings.DATABASE_URL.strip():
        raise DatabaseConfigError(
            "DATABASE_URL is not configured. Please set the DATABASE_URL environment variable."
        )
    return SQLAlchemyResearchRepository()
