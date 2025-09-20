"""
Context middleware for request tracking
"""

import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class ContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add request context"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())

        # Add to request state
        request.state.request_id = request_id
        request.state.user_id = None  # Will be set by auth middleware

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response