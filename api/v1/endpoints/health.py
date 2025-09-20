"""
Health check endpoints
"""

from typing import Dict, Any
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "codemind-api",
        "version": "0.1.0"
    }


@router.get("/ping")
async def ping() -> Dict[str, str]:
    """Simple ping endpoint"""
    return {"message": "pong"}