"""Rate limiting middleware for the MemoryForge API."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Awaitable

import structlog
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Requests per window
    requests_per_minute: int = 60
    requests_per_hour: int = 1000

    # Burst allowance
    burst_size: int = 10

    # Sliding window size in seconds
    window_size: int = 60

    # Enable rate limiting
    enabled: bool = True

    # Exempt paths (no rate limiting)
    exempt_paths: list[str] = field(default_factory=lambda: ["/health", "/docs", "/openapi.json"])

    # Custom limits per path prefix
    path_limits: dict[str, int] = field(default_factory=dict)


@dataclass
class ClientState:
    """State for a rate-limited client."""

    requests: list[datetime] = field(default_factory=list)
    tokens: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    blocked_until: datetime | None = None

    def cleanup_old_requests(self, window_seconds: int) -> None:
        """Remove requests older than the window."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        self.requests = [r for r in self.requests if r > cutoff]


class TokenBucketLimiter:
    """Token bucket rate limiter.

    Provides smooth rate limiting with burst support.
    """

    def __init__(
        self,
        rate: float,  # Tokens per second
        capacity: float,  # Maximum burst size
    ):
        """Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum tokens (burst capacity)
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens: dict[str, float] = defaultdict(lambda: capacity)
        self._last_update: dict[str, datetime] = {}

    def allow(self, client_id: str, tokens: float = 1.0) -> bool:
        """Check if request is allowed.

        Args:
            client_id: Client identifier
            tokens: Tokens required for this request

        Returns:
            True if allowed
        """
        now = datetime.now(timezone.utc)

        # Refill tokens
        if client_id in self._last_update:
            elapsed = (now - self._last_update[client_id]).total_seconds()
            self._tokens[client_id] = min(
                self._capacity,
                self._tokens[client_id] + elapsed * self._rate,
            )

        self._last_update[client_id] = now

        # Check and consume
        if self._tokens[client_id] >= tokens:
            self._tokens[client_id] -= tokens
            return True

        return False

    def get_wait_time(self, client_id: str, tokens: float = 1.0) -> float:
        """Get time to wait for tokens to be available.

        Args:
            client_id: Client identifier
            tokens: Tokens needed

        Returns:
            Seconds to wait
        """
        needed = tokens - self._tokens.get(client_id, self._capacity)
        if needed <= 0:
            return 0.0
        return needed / self._rate

    def reset(self, client_id: str) -> None:
        """Reset a client's tokens to full capacity."""
        self._tokens[client_id] = self._capacity
        self._last_update[client_id] = datetime.now(timezone.utc)


class SlidingWindowLimiter:
    """Sliding window rate limiter.

    More accurate than fixed window but uses more memory.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
    ):
        """Initialize sliding window limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
        """
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: dict[str, list[datetime]] = defaultdict(list)

    def allow(self, client_id: str) -> bool:
        """Check if request is allowed.

        Args:
            client_id: Client identifier

        Returns:
            True if allowed
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._window_seconds)

        # Remove old requests
        self._requests[client_id] = [
            r for r in self._requests[client_id] if r > cutoff
        ]

        # Check limit
        if len(self._requests[client_id]) >= self._max_requests:
            return False

        # Record this request
        self._requests[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests in current window.

        Args:
            client_id: Client identifier

        Returns:
            Remaining request count
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._window_seconds)

        current = len([
            r for r in self._requests.get(client_id, []) if r > cutoff
        ])

        return max(0, self._max_requests - current)

    def get_reset_time(self, client_id: str) -> datetime:
        """Get when the rate limit resets.

        Args:
            client_id: Client identifier

        Returns:
            Reset timestamp
        """
        requests = self._requests.get(client_id, [])
        if not requests:
            return datetime.now(timezone.utc)

        oldest = min(requests)
        return oldest + timedelta(seconds=self._window_seconds)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        app,
        config: RateLimitConfig | None = None,
    ):
        """Initialize middleware.

        Args:
            app: FastAPI application
            config: Rate limit configuration
        """
        super().__init__(app)
        self._config = config or RateLimitConfig()

        # Initialize limiters
        self._minute_limiter = SlidingWindowLimiter(
            max_requests=self._config.requests_per_minute,
            window_seconds=60,
        )
        self._hour_limiter = SlidingWindowLimiter(
            max_requests=self._config.requests_per_hour,
            window_seconds=3600,
        )
        self._burst_limiter = TokenBucketLimiter(
            rate=self._config.requests_per_minute / 60,
            capacity=self._config.burst_size,
        )

        # Statistics
        self._total_requests = 0
        self._blocked_requests = 0

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process request with rate limiting.

        Args:
            request: Incoming request
            call_next: Next handler

        Returns:
            Response
        """
        if not self._config.enabled:
            return await call_next(request)

        # Check exempt paths
        path = request.url.path
        if path in self._config.exempt_paths:
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Check custom path limits
        path_limit = self._get_path_limit(path)
        if path_limit:
            if not self._check_path_limit(client_id, path, path_limit):
                return self._rate_limit_response(client_id)

        # Check rate limits
        self._total_requests += 1

        # Burst check (token bucket)
        if not self._burst_limiter.allow(client_id):
            self._blocked_requests += 1
            return self._rate_limit_response(client_id, "Burst limit exceeded")

        # Minute limit check
        if not self._minute_limiter.allow(client_id):
            self._blocked_requests += 1
            return self._rate_limit_response(client_id, "Minute limit exceeded")

        # Hour limit check
        if not self._hour_limiter.allow(client_id):
            self._blocked_requests += 1
            return self._rate_limit_response(client_id, "Hour limit exceeded")

        # Add rate limit headers
        response = await call_next(request)
        self._add_rate_limit_headers(response, client_id)

        return response

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Try API key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:16]}"

        # Try authorization header
        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            return f"token:{auth[7:23]}"

        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        client = request.client
        if client:
            return f"ip:{client.host}"

        return "unknown"

    def _get_path_limit(self, path: str) -> int | None:
        """Get custom limit for path."""
        for prefix, limit in self._config.path_limits.items():
            if path.startswith(prefix):
                return limit
        return None

    def _check_path_limit(
        self,
        client_id: str,
        path: str,
        limit: int,
    ) -> bool:
        """Check path-specific rate limit."""
        # Simple implementation - could be expanded
        return True

    def _rate_limit_response(
        self,
        client_id: str,
        message: str = "Rate limit exceeded",
    ) -> Response:
        """Create rate limit exceeded response."""
        reset_time = self._minute_limiter.get_reset_time(client_id)
        retry_after = max(
            0,
            int((reset_time - datetime.now(timezone.utc)).total_seconds()),
        )

        logger.warning(
            "Rate limit exceeded",
            client_id=client_id,
            retry_after=retry_after,
        )

        return Response(
            content=f'{{"detail": "{message}", "retry_after": {retry_after}}}',
            status_code=429,
            headers={
                "Content-Type": "application/json",
                "Retry-After": str(retry_after),
                "X-RateLimit-Remaining": "0",
            },
        )

    def _add_rate_limit_headers(
        self,
        response: Response,
        client_id: str,
    ) -> None:
        """Add rate limit headers to response."""
        remaining = self._minute_limiter.get_remaining(client_id)
        reset_time = self._minute_limiter.get_reset_time(client_id)

        response.headers["X-RateLimit-Limit"] = str(self._config.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = reset_time.isoformat()

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "enabled": self._config.enabled,
            "total_requests": self._total_requests,
            "blocked_requests": self._blocked_requests,
            "block_rate": (
                self._blocked_requests / self._total_requests
                if self._total_requests > 0 else 0
            ),
            "limits": {
                "per_minute": self._config.requests_per_minute,
                "per_hour": self._config.requests_per_hour,
                "burst_size": self._config.burst_size,
            },
        }

    def reset_client(self, client_id: str) -> None:
        """Reset rate limits for a client."""
        self._burst_limiter.reset(client_id)
        # Note: sliding window limiter doesn't support reset


def create_rate_limiter(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    burst_size: int = 10,
) -> RateLimitConfig:
    """Create a rate limit configuration.

    Args:
        requests_per_minute: Max requests per minute
        requests_per_hour: Max requests per hour
        burst_size: Burst allowance

    Returns:
        RateLimitConfig instance
    """
    return RateLimitConfig(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        burst_size=burst_size,
    )
