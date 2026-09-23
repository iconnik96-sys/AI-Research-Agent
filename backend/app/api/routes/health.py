import asyncio
import logging
from typing import Dict
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Dict[str, str], summary="Liveness check")
async def health_check() -> Dict[str, str]:
    """Return liveness status of the application process.

    This check is intentionally independent of all external services.
    """
    return {"status": "ok"}


@router.get("/ready", summary="Readiness check")
async def readiness_check(
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """Return readiness status verifying required database infrastructure.

    Executes a bounded SELECT 1 ping against PostgreSQL. Does not depend on
    external search or LLM APIs to avoid burning rate limits or adding latency.
    """
    try:
        # Bounded 3.0s timeout to prevent hanging connections
        await asyncio.wait_for(db.execute(text("SELECT 1")), timeout=3.0)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ready", "database": "connected"},
        )
    except Exception as exc:
        logger.warning("Readiness probe database check failed: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "database": "disconnected"},
        )
