"""
Configuration management for CodeMind API
"""

import os
from typing import List, Optional
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""

    # App
    app_name: str = "CodeMind API"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # CORS
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="ALLOWED_ORIGINS"
    )

    # Database
    database_url: str = Field(
        default="postgresql://postgres:codemind-dev-password@localhost:5432/codemind",
        env="DATABASE_URL"
    )

    async_database_url: str = Field(
        default="postgresql+asyncpg://postgres:codemind-dev-password@localhost:5432/codemind",
        env="ASYNC_DATABASE_URL"
    )

    # Redis
    redis_url: str = Field(
        default="redis://:codemind-dev-password@localhost:6379",
        env="REDIS_URL"
    )

    # Vector Database
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_url: str = Field(
        default="http://localhost:6333",
        env="QDRANT_URL"
    )

    # Temporal
    temporal_host: str = Field(default="localhost", env="TEMPORAL_HOST")
    temporal_port: int = Field(default=7233, env="TEMPORAL_PORT")
    temporal_namespace: str = Field(default="default", env="TEMPORAL_NAMESPACE")

    # NATS
    nats_url: str = Field(default="nats://localhost:4222", env="NATS_URL")

    # MinIO
    minio_endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="codemind-dev-password", env="MINIO_SECRET_KEY")
    minio_secure: bool = Field(default=False, env="MINIO_SECURE")

    # AWS
    aws_region: str = Field(default="eu-west-2", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")

    # Azure OpenAI
    azure_region: str = Field(default="uksouth", env="AZURE_REGION")
    azure_openai_endpoint: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(default="2024-02-01", env="AZURE_OPENAI_API_VERSION")

    # LLM Configuration
    default_model: str = Field(default="claude-3-7-sonnet", env="DEFAULT_MODEL")
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")

    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds

    # File Upload
    max_upload_size: int = Field(default=10 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 10MB

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()