from typing import List, Union
import json
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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


settings = Settings()
