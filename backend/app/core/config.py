from pathlib import Path
from typing import List, Optional, Union
import json
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root: AI-Research-Agent/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT_ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(ROOT_ENV_FILE), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENV: str = "development"
    PROJECT_NAME: str = "AI Research Agent"
    API_V1_PREFIX: str = "/api"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: Union[List[str], str] = ["*"]

    # Search Service Settings
    TAVILY_API_KEY: str = ""
    SEARCH_PROVIDER: str = "tavily"
    SEARCH_TIMEOUT_SECONDS: float = 10.0
    SEARCH_MAX_RESULTS: int = 5

    # Extraction Service Settings
    EXTRACTION_TIMEOUT_SECONDS: float = 10.0
    EXTRACTION_MAX_CHARS: int = 50000
    EXTRACTION_USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    # LLM Service Settings
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_CHARS_PER_DOC: int = 4000

    # Embedding Service Settings
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://api.openai.com/v1"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    EMBEDDING_TIMEOUT_SECONDS: float = 30.0

    # Chunking Service Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Planner Service Settings
    PLANNER_MAX_SUB_QUESTIONS: int = 5
    PLANNER_TIMEOUT_SECONDS: float = 30.0

    # Retrieval Service Settings
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_SIMILARITY_THRESHOLD: Optional[float] = None
    RETRIEVAL_MAX_TOTAL_CHUNKS: int = 15

    # Claim & Verification Settings
    CLAIM_MAX_COUNT: int = 15
    CLAIM_EXTRACTION_TIMEOUT_SECONDS: float = 30.0
    CLAIM_VERIFICATION_TIMEOUT_SECONDS: float = 30.0
    CLAIM_VERIFICATION_CONCURRENCY: int = 5

    # Database Settings (Supabase PostgreSQL via asyncpg)
    DATABASE_URL: str = ""
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_RECYCLE: int = 300
    DATABASE_TIMEOUT_SECONDS: float = 10.0

    # Rate Limiting Settings (Lightweight in-memory per-worker abuse protection)
    # Default disabled in dev/testing so automated test suites are not throttled;
    # enable in production by setting RATE_LIMIT_ENABLED=true in the environment.
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    def validate_production_settings(self) -> None:
        """Validate required configuration when running in production environment.

        Raises:
            ValueError: If required production credentials, database URL, or explicit CORS origins are missing.
        """
        if self.ENV.lower() != "production":
            return

        errors: List[str] = []

        if not self.DATABASE_URL or not self.DATABASE_URL.strip():
            errors.append("DATABASE_URL must be configured in production.")
        elif "[YOUR-PASSWORD]" in self.DATABASE_URL or "[YOUR-PROJECT-REF]" in self.DATABASE_URL:
            errors.append("DATABASE_URL contains placeholder credentials.")

        if not self.TAVILY_API_KEY or not self.TAVILY_API_KEY.strip():
            errors.append("TAVILY_API_KEY must be configured in production.")

        if not self.LLM_API_KEY or not self.LLM_API_KEY.strip():
            errors.append("LLM_API_KEY must be configured in production.")

        if not self.EMBEDDING_API_KEY or not self.EMBEDDING_API_KEY.strip():
            errors.append("EMBEDDING_API_KEY must be configured in production.")

        origins = self.CORS_ORIGINS if isinstance(self.CORS_ORIGINS, list) else [self.CORS_ORIGINS]
        if "*" in origins:
            errors.append("Wildcard '*' in CORS_ORIGINS is forbidden in production. Explicit allowed origins must be configured.")

        if errors:
            raise ValueError(f"Production configuration validation failed: {'; '.join(errors)}")


settings = Settings()
