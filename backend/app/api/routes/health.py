from typing import Dict
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Dict[str, str], summary="Health check")
async def health_check() -> Dict[str, str]:
    """Return health status of the application."""
    return {"status": "ok"}
