#!/usr/bin/env python3
"""
Production-ready CodeMind API Application
Enterprise-grade FastAPI application with proper module structure
"""

import sys
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

# Set up the Python path for proper module imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set up environment
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import our modules
from core.config import get_settings
from core.logging import setup_logging, get_logger
from core.database import init_db, close_db
from api.v1.router import api_router
from services.llm_router import LLMRouter
from middleware.context import ContextMiddleware
from middleware.auth import AuthMiddleware
from middleware.rate_limit import RateLimitMiddleware

# Initialize logging
setup_logging()
logger = get_logger(__name__)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting CodeMind API")

    # Initialize LLM Router (critical)
    try:
        app.state.llm_router = LLMRouter()
        await app.state.llm_router.initialize()
        logger.info("✅ LLM Router initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM Router: {e}")
        raise

    # Initialize database (optional for initial testing)
    try:
        await init_db()
        logger.info("✅ Database initialized")
        app.state.database_available = True
    except Exception as e:
        logger.warning(f"⚠️ Database initialization failed: {e}")
        logger.info("🔄 Starting API without database - limited functionality")
        app.state.database_available = False

    logger.info("🎉 CodeMind API started successfully")

    yield

    # Cleanup
    logger.info("🛑 Shutting down CodeMind API")

    try:
        if hasattr(app.state, 'database_available') and app.state.database_available:
            await close_db()
            logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

    logger.info("👋 CodeMind API shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="CodeMind API",
    description="The World's First Cognitive Software Factory - Enterprise Grade",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# Add middleware in correct order (last added is executed first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(ContextMiddleware)

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
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled exception for request {request_id}: {exc}", exc_info=True)

    if settings.debug:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "path": str(request.url.path),
                    "request_id": request_id,
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
                    "request_id": request_id,
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
        "environment": "development" if settings.debug else "production",
    }

@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check(request: Request) -> Dict[str, Any]:
    """Detailed health check with service dependencies"""
    health_status = {
        "status": "healthy",
        "service": "codemind-api",
        "version": "0.1.0",
        "environment": "development" if settings.debug else "production",
        "checks": {},
    }

    # Check LLM services
    try:
        if hasattr(request.app.state, 'llm_router'):
            llm_healthy = await request.app.state.llm_router.health_check()
            health_status["checks"]["llm_services"] = "healthy" if llm_healthy else "unhealthy"
        else:
            health_status["checks"]["llm_services"] = "not_initialized"
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
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

# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "message": "Welcome to CodeMind API - The World's First Cognitive Software Factory",
        "version": "0.1.0",
        "docs": "/docs" if settings.debug else "Not available in production",
        "health": "/health",
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
        access_log=True,
    )