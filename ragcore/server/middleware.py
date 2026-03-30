"""FastAPI middleware for ragcore.

Provides:
- RateLimitMiddleware: sliding-window 60 req/min per client IP
- ObservabilityMiddleware: X-Request-ID echo/generation, X-Latency-Ms, structured logging
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


# ---------------------------------------------------------------------------
# Rate limiting — sliding window, 60 req/min per IP
# ---------------------------------------------------------------------------

_RATE_LIMIT = 60          # max requests
_WINDOW_SECONDS = 60.0    # per this many seconds


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter: 60 requests / 60 s per client IP."""

    def __init__(self, app: ASGIApp, limit: int = _RATE_LIMIT, window: float = _WINDOW_SECONDS) -> None:
        super().__init__(app)
        self._limit = limit
        self._window = window
        # ip -> deque of request timestamps (monotonic)
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        ip = self._get_client_ip(request)
        now = time.monotonic()
        window_start = now - self._window

        timestamps = self._requests[ip]
        # Evict old timestamps outside the sliding window
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= self._limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
            )

        timestamps.append(now)
        return await call_next(request)


# ---------------------------------------------------------------------------
# Observability — X-Request-ID + X-Latency-Ms + structured logging
# ---------------------------------------------------------------------------


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Add X-Request-ID and X-Latency-Ms headers; emit a structured log line per request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        t0 = time.perf_counter()

        response: Response = await call_next(request)

        latency_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"

        logger.info(
            "method={} path={} status={} request_id={} latency_ms={:.2f}",
            request.method,
            request.url.path,
            response.status_code,
            request_id,
            latency_ms,
        )

        return response
