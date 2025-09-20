"""
API v1 Router for CodeMind
"""

from fastapi import APIRouter

from .endpoints import generation, health, auth

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(generation.router, prefix="/generate", tags=["Generation"])