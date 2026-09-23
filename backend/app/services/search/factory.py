from app.core.config import settings
from app.services.search.base import BaseSearchProvider, SearchConfigError
from app.services.search.tavily import TavilySearchProvider


def get_search_provider() -> BaseSearchProvider:
    """Dependency provider factory that returns the configured search provider."""
    provider_name = settings.SEARCH_PROVIDER.lower().strip()

    if provider_name == "tavily":
        return TavilySearchProvider()

    raise SearchConfigError(f"Unsupported search provider '{settings.SEARCH_PROVIDER}'")
