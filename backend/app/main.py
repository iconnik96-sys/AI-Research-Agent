import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.api.routes.research import router as research_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.security import (
    InMemoryRateLimiterMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    get_current_request_id,
)

logger = logging.getLogger(__name__)


def create_application() -> FastAPI:
    # 1. Initialize structured logging
    setup_logging()

    # 2. Validate configuration in production
    settings.validate_production_settings()

    application = FastAPI(
        title=settings.PROJECT_NAME,
        version="0.1.0",
        description="FastAPI backend for AI Research Agent",
    )

    # 3. Middlewares (execution order: outermost RequestID -> SecurityHeaders -> CORS -> RateLimiter -> Router)
    # Rate limiter (in-memory per-instance application-level abuse protection)
    application.add_middleware(
        InMemoryRateLimiterMiddleware,
        max_requests=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        enabled=settings.RATE_LIMIT_ENABLED,
    )

    # CORS configuration
    cors_origins = (
        settings.CORS_ORIGINS
        if isinstance(settings.CORS_ORIGINS, list)
        else [settings.CORS_ORIGINS]
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "Accept",
            "Origin",
            "X-Requested-With",
            "X-Request-ID",
        ],
        expose_headers=["X-Request-ID"],
    )

    # Request ID correlation
    application.add_middleware(RequestIDMiddleware)

    # Security response headers (outermost so all responses including errors get headers)
    application.add_middleware(SecurityHeadersMiddleware)

    # 4. Global unhandled exception handler (avoids leaking stack traces or internal DB info)
    @application.exception_handler(Exception)
    async def global_unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = get_current_request_id()
        logger.error(
            "Unhandled server exception during %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        headers = {}
        if req_id:
            headers["X-Request-ID"] = req_id
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
            headers=headers,
        )

    # 5. Mount health and readiness routes at root (/health, /ready)
    application.include_router(health_router)

    # 6. Mount core research API routes under /api
    application.include_router(research_router, prefix=settings.API_V1_PREFIX)

    return application


app = create_application()
