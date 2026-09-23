import logging
from unittest.mock import AsyncMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings, settings
from app.core.logging import SensitiveDataFilter, setup_logging
from app.core.security import (
    InMemoryRateLimiterMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    get_current_request_id,
)
from app.db.session import get_db_session
from app.main import create_application


# =====================================================================
# 1. Production Settings Validation Tests
# =====================================================================

def test_production_validation_success():
    """Verify that valid production settings pass validation without error."""
    prod_settings = Settings(
        ENV="production",
        DATABASE_URL="postgresql+asyncpg://user:realpassword@host:5432/db",
        TAVILY_API_KEY="tvly-secretkey",
        LLM_API_KEY="sk-llmsecret",
        EMBEDDING_API_KEY="sk-embsecret",
        CORS_ORIGINS=["https://app.example.com"],
    )
    # Should not raise
    prod_settings.validate_production_settings()


def test_production_validation_missing_keys():
    """Verify that missing credentials raise ValueError in production."""
    prod_settings = Settings(
        ENV="production",
        DATABASE_URL="",
        TAVILY_API_KEY="",
        LLM_API_KEY="",
        EMBEDDING_API_KEY="",
        CORS_ORIGINS=["https://app.example.com"],
    )
    with pytest.raises(ValueError, match="Production configuration validation failed") as exc_info:
        prod_settings.validate_production_settings()

    err_msg = str(exc_info.value)
    assert "DATABASE_URL must be configured" in err_msg
    assert "TAVILY_API_KEY must be configured" in err_msg
    assert "LLM_API_KEY must be configured" in err_msg
    assert "EMBEDDING_API_KEY must be configured" in err_msg


def test_production_validation_placeholder_database_url():
    """Verify that placeholder DATABASE_URL is rejected in production."""
    prod_settings = Settings(
        ENV="production",
        DATABASE_URL="postgresql+asyncpg://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-REF].supabase.co:5432/postgres",
        TAVILY_API_KEY="tvly-valid",
        LLM_API_KEY="sk-valid",
        EMBEDDING_API_KEY="sk-valid",
        CORS_ORIGINS=["https://app.example.com"],
    )
    with pytest.raises(ValueError, match="placeholder credentials"):
        prod_settings.validate_production_settings()


def test_production_validation_wildcard_cors_rejected():
    """Verify that wildcard CORS origins are rejected in production."""
    prod_settings = Settings(
        ENV="production",
        DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/db",
        TAVILY_API_KEY="tvly-valid",
        LLM_API_KEY="sk-valid",
        EMBEDDING_API_KEY="sk-valid",
        CORS_ORIGINS=["*"],
    )
    with pytest.raises(ValueError, match="Wildcard '\\*' in CORS_ORIGINS is forbidden in production"):
        prod_settings.validate_production_settings()


def test_development_mode_allows_blank_keys():
    """Verify that development mode does not fail validation when keys are blank."""
    dev_settings = Settings(
        ENV="development",
        DATABASE_URL="",
        TAVILY_API_KEY="",
        LLM_API_KEY="",
        EMBEDDING_API_KEY="",
        CORS_ORIGINS=["*"],
    )
    # Should not raise in development
    dev_settings.validate_production_settings()


# =====================================================================
# 2. Security Headers & Request ID Middleware Tests
# =====================================================================

def test_security_headers_present():
    """Verify standard production security headers are set on responses."""
    app = create_application()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def test_request_id_propagation():
    """Verify that a valid incoming X-Request-ID is propagated through the response."""
    app = create_application()
    client = TestClient(app)

    custom_id = "trace-test-uuid-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id


# =====================================================================
# 3. In-Memory Rate Limiter Middleware Tests
# =====================================================================

def test_rate_limiter_blocks_excessive_requests():
    """Verify in-memory rate limiter returns 429 when threshold exceeded."""
    app = FastAPI()
    app.add_middleware(InMemoryRateLimiterMiddleware, max_requests=2, window_seconds=60, enabled=True)

    @app.get("/test")
    def dummy_route():
        return {"ok": True}

    client = TestClient(app)

    # First 2 requests succeed
    res1 = client.get("/test")
    assert res1.status_code == 200
    res2 = client.get("/test")
    assert res2.status_code == 200

    # 3rd request is blocked with 429
    res3 = client.get("/test")
    assert res3.status_code == 429
    assert res3.json() == {"detail": "Too many requests. Please slow down."}
    assert "Retry-After" in res3.headers


def test_rate_limiter_exempts_health_and_ready():
    """Verify that /health and /ready are never rate limited."""
    app = FastAPI()
    app.add_middleware(InMemoryRateLimiterMiddleware, max_requests=1, window_seconds=60, enabled=True)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"status": "ready"}

    client = TestClient(app)

    # Multiple calls to /health should all succeed
    for _ in range(5):
        res = client.get("/health")
        assert res.status_code == 200

    # Multiple calls to /ready should all succeed
    for _ in range(5):
        res = client.get("/ready")
        assert res.status_code == 200


def test_rate_limiter_exempts_options_preflight():
    """Verify that CORS preflight OPTIONS requests are not rate-limited."""
    app = FastAPI()
    app.add_middleware(InMemoryRateLimiterMiddleware, max_requests=1, window_seconds=60, enabled=True)

    @app.options("/api/research")
    def options_route():
        return {}

    client = TestClient(app)

    for _ in range(5):
        res = client.options("/api/research")
        assert res.status_code == 200


def test_application_rate_limiting_configured_via_settings(monkeypatch):
    """Verify that create_application honors RATE_LIMIT_ENABLED and RATE_LIMIT_REQUESTS_PER_MINUTE."""
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 2)
    monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 60)

    app = create_application()
    client = TestClient(app)

    # Health check is always exempt even when rate limiter is active
    for _ in range(3):
        assert client.get("/health").status_code == 200

    # Non-exempt endpoint (POST /api/research/plan with invalid payload -> 422, but still counted)
    res1 = client.post("/api/research/plan", json={"question": ""})
    assert res1.status_code == 422
    res2 = client.post("/api/research/plan", json={"question": ""})
    assert res2.status_code == 422

    # 3rd request must be blocked by rate limiter with 429
    res3 = client.post("/api/research/plan", json={"question": ""})
    assert res3.status_code == 429
    assert res3.json() == {"detail": "Too many requests. Please slow down."}


# =====================================================================
# 4. Global Unhandled Exception Masking Tests
# =====================================================================

def test_unhandled_exception_sanitized_500():
    """Verify unexpected internal exception returns sanitized 500 without stack trace leak."""
    app = create_application()

    # Add a route that deliberately raises an unhandled unexpected error
    @app.get("/crash-me-internal")
    def crash_route():
        raise RuntimeError("Secret internal database password connection crash!")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/crash-me-internal")

    assert response.status_code == 500
    data = response.json()
    assert data == {"detail": "Internal server error"}
    assert "Secret internal database password" not in response.text
    assert "RuntimeError" not in response.text
    assert "X-Request-ID" in response.headers


# =====================================================================
# 5. Health & Readiness Probe Tests
# =====================================================================

def test_health_liveness_probe():
    """Verify /health returns 200 without database access."""
    app = create_application()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_ready_readiness_probe_success():
    """Verify /ready returns 200 when database ping succeeds."""
    app = create_application()

    # Mock get_db_session dependency to return an active mock session
    mock_session = AsyncMock()
    mock_session.execute.return_value = None

    async def override_get_db_session():
        yield mock_session

    app.dependency_overrides[get_db_session] = override_get_db_session
    client = TestClient(app)

    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "connected"}


@pytest.mark.anyio
async def test_ready_readiness_probe_failure():
    """Verify /ready returns 503 when database ping fails."""
    app = create_application()

    mock_session = AsyncMock()
    mock_session.execute.side_effect = ConnectionRefusedError("Database unreachable")

    async def override_get_db_session():
        yield mock_session

    app.dependency_overrides[get_db_session] = override_get_db_session
    client = TestClient(app)

    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "unhealthy", "database": "disconnected"}


# =====================================================================
# 6. Sensitive Data Redaction Filter Tests
# =====================================================================

def test_sensitive_data_filter_masks_credentials():
    """Verify SensitiveDataFilter redacts DB passwords, tokens, and API keys."""
    log_filter = SensitiveDataFilter()

    # Database URL
    record1 = logging.LogRecord("test", logging.INFO, "path", 1, "Connecting to postgresql://admin:super_secret_password@db.supabase.co:5432/db", (), None)
    log_filter.filter(record1)
    assert "super_secret_password" not in record1.msg
    assert "postgresql://admin:***@db.supabase.co" in record1.msg

    # Bearer token
    record2 = logging.LogRecord("test", logging.INFO, "path", 1, "Header Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret", (), None)
    log_filter.filter(record2)
    assert "Bearer ***" in record2.msg

    # OpenAI & Tavily API Keys
    record3 = logging.LogRecord("test", logging.INFO, "path", 1, "Using sk-1234567890abcdefghijklmn and tvly-1234567890abcdefghijklmn", (), None)
    log_filter.filter(record3)
    assert "sk-1234567890abcdefghijklmn" not in record3.msg
    assert "tvly-1234567890abcdefghijklmn" not in record3.msg
    assert "sk-***" in record3.msg
    assert "tvly-***" in record3.msg
