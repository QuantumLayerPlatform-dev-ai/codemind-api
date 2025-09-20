"""
CodeMind API - Main FastAPI Application

The entry point for the CodeMind Cognitive Software Factory API.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

try:
    # Try relative imports first (when run as module)
    from .core.config import get_settings
    from .core.logging import setup_logging
    from .core.database import init_db
    from .api.v1.router import api_router
    from .middleware.auth import AuthMiddleware
    from .middleware.rate_limit import RateLimitMiddleware
    from .middleware.context import ContextMiddleware
except ImportError:
    # Fallback to absolute imports (when run directly)
    from core.config import get_settings
    from core.logging import setup_logging
    from core.database import init_db
    from api.v1.router import api_router
    from middleware.auth import AuthMiddleware
    from middleware.rate_limit import RateLimitMiddleware
    from middleware.context import ContextMiddleware


# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting CodeMind API")

    # Initialize database
    await init_db()
    logger.info("✅ Database initialized")

    # Initialize AI services
    try:
        from .services.llm_router import LLMRouter
    except ImportError:
        from services.llm_router import LLMRouter

    app.state.llm_router = LLMRouter()
    await app.state.llm_router.initialize()
    logger.info("✅ LLM Router initialized")

    # Skip other services for now to get API running
    logger.info("⚠️ Skipping Temporal, NATS, and Context Manager for initial startup")

    logger.info("🎉 CodeMind API started successfully")

    yield

    # Cleanup
    logger.info("🛑 Shutting down CodeMind API")

    if hasattr(app.state, 'temporal_client'):
        await app.state.temporal_client.disconnect()
        logger.info("✅ Temporal client disconnected")

    if hasattr(app.state, 'nats_client'):
        await app.state.nats_client.disconnect()
        logger.info("✅ NATS client disconnected")

    logger.info("👋 CodeMind API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="CodeMind API",
    description="The World's First Cognitive Software Factory",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(ContextMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path),
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    if settings.debug:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "path": str(request.url.path),
                }
            },
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "status_code": 500,
                    "path": str(request.url.path),
                }
            },
        )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "codemind-api",
        "version": "0.1.0",
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check(request: Request) -> Dict[str, Any]:
    """Detailed health check with service dependencies"""
    health_status = {
        "status": "healthy",
        "service": "codemind-api",
        "version": "0.1.0",
        "checks": {},
    }

    # Check database
    try:
        from .core.database import get_db_health
        db_healthy = await get_db_health()
        health_status["checks"]["database"] = "healthy" if db_healthy else "unhealthy"
    except Exception as e:
        health_status["checks"]["database"] = f"error: {str(e)}"

    # Check Redis
    try:
        from .core.redis import get_redis_health
        redis_healthy = await get_redis_health()
        health_status["checks"]["redis"] = "healthy" if redis_healthy else "unhealthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"error: {str(e)}"

    # Check Temporal
    try:
        if hasattr(request.app.state, 'temporal_client'):
            temporal_healthy = await request.app.state.temporal_client.health_check()
            health_status["checks"]["temporal"] = "healthy" if temporal_healthy else "unhealthy"
        else:
            health_status["checks"]["temporal"] = "not_initialized"
    except Exception as e:
        health_status["checks"]["temporal"] = f"error: {str(e)}"

    # Check NATS
    try:
        if hasattr(request.app.state, 'nats_client'):
            nats_healthy = await request.app.state.nats_client.health_check()
            health_status["checks"]["nats"] = "healthy" if nats_healthy else "unhealthy"
        else:
            health_status["checks"]["nats"] = "not_initialized"
    except Exception as e:
        health_status["checks"]["nats"] = f"error: {str(e)}"

    # Check LLM services
    try:
        if hasattr(request.app.state, 'llm_router'):
            llm_healthy = await request.app.state.llm_router.health_check()
            health_status["checks"]["llm_services"] = "healthy" if llm_healthy else "unhealthy"
        else:
            health_status["checks"]["llm_services"] = "not_initialized"
    except Exception as e:
        health_status["checks"]["llm_services"] = f"error: {str(e)}"

    # Overall status
    unhealthy_checks = [
        check for check in health_status["checks"].values()
        if check not in ["healthy", "not_initialized"]
    ]

    if unhealthy_checks:
        health_status["status"] = "unhealthy"

    return health_status


# Include API router
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )