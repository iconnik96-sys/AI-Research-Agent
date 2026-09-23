import collections
import contextvars
import logging
import re
import time
import uuid
from typing import Callable, Dict, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Context variable for request correlation ID across async execution flow
request_id_ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

# Safe alphanumeric/hyphen pattern for incoming request IDs
SAFE_REQUEST_ID_REGEX = re.compile(r"^[A-Za-z0-9_\-\.]{1,64}$")


def get_current_request_id() -> str:
    """Return the active correlation request ID from context, or empty string."""
    return request_id_ctx_var.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate or propagate an X-Request-ID for request correlation."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming_id = request.headers.get("X-Request-ID", "").strip()
        if incoming_id and SAFE_REQUEST_ID_REGEX.match(incoming_id):
            req_id = incoming_id
        else:
            req_id = uuid.uuid4().hex

        token = request_id_ctx_var.set(req_id)
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "Unhandled server exception during %s %s: %s",
                request.method,
                request.url.path,
                exc,
                exc_info=True,
            )
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )
        finally:
            request_id_ctx_var.reset(token)

        response.headers["X-Request-ID"] = req_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to inject standard production security response headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


class InMemoryRateLimiterMiddleware(BaseHTTPMiddleware):
    """Lightweight in-memory per-instance application-level abuse protection.

    IMPORTANT ARCHITECTURAL NOTES:
    - This is NOT DDoS protection. Infrastructure-level DDoS mitigation must be
      handled by upstream reverse proxies, CDNs (e.g. Cloudflare), or cloud platforms.
    - Limits are tracked per process/worker in-memory, and therefore are NOT
      globally shared across multiple workers or horizontal instances.
    - Exempt paths: /health, /ready.
    - Exempt HTTP methods: OPTIONS (CORS preflight requests).
    - Entirely dependency-free: uses Python standard library collections.defaultdict.
    """

    def __init__(self, app, max_requests: int = 10, window_seconds: int = 60, enabled: bool = True):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.enabled = enabled
        self._history: Dict[str, List[float]] = collections.defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
            if client_ip:
                return client_ip
        return request.client.host if request.client else "127.0.0.1"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Exempt CORS preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # Exempt liveness and readiness health checks
        path = request.url.path.rstrip("/")
        if path in ("/health", "/ready"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.time()
        cutoff = now - self.window_seconds

        timestamps = self._history[client_ip]
        # Prune expired timestamps
        valid_timestamps = [t for t in timestamps if t > cutoff]
        self._history[client_ip] = valid_timestamps

        if len(valid_timestamps) >= self.max_requests:
            req_id = get_current_request_id()
            headers = {
                "Retry-After": str(self.window_seconds),
            }
            if req_id:
                headers["X-Request-ID"] = req_id
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down."},
                headers=headers,
            )

        self._history[client_ip].append(now)
        return await call_next(request)
