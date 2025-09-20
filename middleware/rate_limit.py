"""
Rate limiting middleware
"""

import time
from typing import Dict, Tuple
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
from collections import defaultdict, deque

try:
    from ..core.logging import get_logger
except ImportError:
    from core.logging import get_logger

logger = get_logger("rate_limit_middleware")

class InMemoryRateLimiter:
    """Simple in-memory rate limiter using sliding window"""

    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = asyncio.Lock()

    async def is_allowed(self, key: str, limit: int, window: int) -> Tuple[bool, Dict[str, int]]:
        """Check if request is allowed under rate limit"""
        async with self.lock:
            now = time.time()
            window_start = now - window

            # Clean old requests
            request_times = self.requests[key]
            while request_times and request_times[0] < window_start:
                request_times.popleft()

            # Check if under limit
            current_count = len(request_times)
            if current_count >= limit:
                reset_time = int(request_times[0] + window)
                return False, {
                    "limit": limit,
                    "remaining": 0,
                    "reset": reset_time,
                    "retry_after": reset_time - int(now)
                }

            # Add current request
            request_times.append(now)

            return True, {
                "limit": limit,
                "remaining": limit - current_count - 1,
                "reset": int(now + window),
                "retry_after": 0
            }

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with different limits per endpoint"""

    def __init__(self, app):
        super().__init__(app)
        self.limiter = InMemoryRateLimiter()

        # Define rate limits per endpoint pattern
        self.rate_limits = {
            "/api/v1/generate/app": (5, 60),      # 5 requests per minute
            "/api/v1/generate/code": (10, 60),    # 10 requests per minute
            "/api/v1/auth/login": (10, 300),      # 10 requests per 5 minutes
            "default": (100, 60)                  # 100 requests per minute default
        }

    def get_rate_limit(self, path: str) -> Tuple[int, int]:
        """Get rate limit for a specific path"""
        for pattern, (limit, window) in self.rate_limits.items():
            if pattern != "default" and path.startswith(pattern):
                return limit, window
        return self.rate_limits["default"]

    def get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Use user_id if authenticated, otherwise use IP
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        # Get real IP from headers (considering proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return f"ip:{real_ip}"

        return f"ip:{request.client.host}"

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path.startswith("/api/v1/health"):
            return await call_next(request)

        # Get rate limit configuration
        limit, window = self.get_rate_limit(request.url.path)
        client_id = self.get_client_id(request)

        # Create unique key for this client and endpoint
        rate_key = f"{client_id}:{request.url.path}:{request.method}"

        # Check rate limit
        allowed, rate_info = await self.limiter.is_allowed(rate_key, limit, window)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {client_id} on {request.url.path}: "
                f"{rate_info['retry_after']}s until reset"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_info["remaining"]),
                    "X-RateLimit-Reset": str(rate_info["reset"]),
                    "Retry-After": str(rate_info["retry_after"])
                }
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])

        return response