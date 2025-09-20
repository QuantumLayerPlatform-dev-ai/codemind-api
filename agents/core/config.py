"""
Enterprise Configuration Management for CodeMind Agents
======================================================

Comprehensive configuration system with:
- Environment-based configuration
- Validation and type checking
- Secret management integration
- Configuration hot-reloading
- Health checks and monitoring
"""

import os
import json
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from ...core.logging import get_logger
except ImportError:
    from core.logging import get_logger

logger = get_logger("agent_config")


class Environment(str, Enum):
    """Supported environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


class CacheType(str, Enum):
    """Supported cache types"""
    REDIS = "redis"
    MEMORY = "memory"
    MEMCACHED = "memcached"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "codemind"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    ssl: bool = False
    pool_size: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5


@dataclass
class LLMConfig:
    """LLM service configuration"""
    aws_region: str = "eu-west-2"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_api_version: str = "2024-02-01"
    default_model: str = "claude-3-7-sonnet"
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout_seconds: int = 30
    max_retries: int = 3


@dataclass
class TemporalConfig:
    """Temporal configuration"""
    host: str = "localhost"
    port: int = 7233
    namespace: str = "default"
    task_queue: str = "codemind-agents"
    workflow_timeout_seconds: int = 3600
    activity_timeout_seconds: int = 300


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    metrics_port: int = 8080
    jaeger_endpoint: str = "http://localhost:14268"
    prometheus_endpoint: str = "http://localhost:9090"
    log_level: LogLevel = LogLevel.INFO
    structured_logging: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    api_key_length: int = 32
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 100
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])


class AgentSettings(BaseSettings):
    """
    Comprehensive agent configuration using Pydantic BaseSettings.

    Supports:
    - Environment variable loading
    - Configuration file loading
    - Type validation
    - Secret management
    """

    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        env="ENVIRONMENT",
        description="Runtime environment"
    )

    # Application
    app_name: str = Field(
        default="CodeMind Agents",
        env="APP_NAME",
        description="Application name"
    )
    app_version: str = Field(
        default="1.0.0",
        env="APP_VERSION",
        description="Application version"
    )
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )

    # Database
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/codemind",
        env="DATABASE_URL",
        description="Database connection URL"
    )
    database_pool_size: int = Field(
        default=10,
        env="DATABASE_POOL_SIZE",
        description="Database connection pool size"
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL",
        description="Redis connection URL"
    )
    redis_password: Optional[SecretStr] = Field(
        default=None,
        env="REDIS_PASSWORD",
        description="Redis password"
    )

    # LLM Services
    aws_region: str = Field(
        default="eu-west-2",
        env="AWS_REGION",
        description="AWS region for Bedrock"
    )
    aws_access_key_id: SecretStr = Field(
        default="",
        env="AWS_ACCESS_KEY_ID",
        description="AWS access key ID"
    )
    aws_secret_access_key: SecretStr = Field(
        default="",
        env="AWS_SECRET_ACCESS_KEY",
        description="AWS secret access key"
    )
    azure_openai_endpoint: str = Field(
        default="",
        env="AZURE_OPENAI_ENDPOINT",
        description="Azure OpenAI endpoint"
    )
    azure_openai_api_key: SecretStr = Field(
        default="",
        env="AZURE_OPENAI_API_KEY",
        description="Azure OpenAI API key"
    )

    # Temporal
    temporal_host: str = Field(
        default="localhost",
        env="TEMPORAL_HOST",
        description="Temporal server host"
    )
    temporal_port: int = Field(
        default=7233,
        env="TEMPORAL_PORT",
        description="Temporal server port"
    )
    temporal_namespace: str = Field(
        default="default",
        env="TEMPORAL_NAMESPACE",
        description="Temporal namespace"
    )

    # Agent Configuration
    agent_timeout_seconds: int = Field(
        default=300,
        env="AGENT_TIMEOUT_SECONDS",
        description="Default agent timeout"
    )
    agent_max_retries: int = Field(
        default=3,
        env="AGENT_MAX_RETRIES",
        description="Default agent retry count"
    )
    agent_max_concurrency: int = Field(
        default=10,
        env="AGENT_MAX_CONCURRENCY",
        description="Maximum concurrent agents"
    )

    # Context Management
    context_ttl_seconds: int = Field(
        default=86400,
        env="CONTEXT_TTL_SECONDS",
        description="Context TTL in seconds"
    )
    context_lock_timeout_seconds: int = Field(
        default=30,
        env="CONTEXT_LOCK_TIMEOUT_SECONDS",
        description="Context lock timeout"
    )

    # Monitoring
    enable_metrics: bool = Field(
        default=True,
        env="ENABLE_METRICS",
        description="Enable metrics collection"
    )
    enable_tracing: bool = Field(
        default=True,
        env="ENABLE_TRACING",
        description="Enable distributed tracing"
    )
    jaeger_endpoint: str = Field(
        default="http://localhost:14268",
        env="JAEGER_ENDPOINT",
        description="Jaeger collector endpoint"
    )

    # Security
    jwt_secret_key: SecretStr = Field(
        default="",
        env="JWT_SECRET_KEY",
        description="JWT secret key"
    )
    api_rate_limit: int = Field(
        default=100,
        env="API_RATE_LIMIT",
        description="API rate limit per minute"
    )

    # Performance
    max_workers: int = Field(
        default=4,
        env="MAX_WORKERS",
        description="Maximum worker threads"
    )
    request_timeout_seconds: int = Field(
        default=60,
        env="REQUEST_TIMEOUT_SECONDS",
        description="Request timeout"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        frozen=True,
        extra="allow"
    )

    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting"""
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                raise ValueError(f"Invalid environment: {v}")
        return v

    @field_validator('aws_region')
    @classmethod
    def validate_aws_region(cls, v):
        """Validate AWS region format"""
        if v and not v.startswith(('us-', 'eu-', 'ap-', 'ca-', 'sa-')):
            logger.warning(f"Unusual AWS region format: {v}")
        return v

    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format"""
        if not v.startswith(('postgresql://', 'postgresql+asyncpg://', 'mysql://', 'sqlite://')):
            raise ValueError(f"Unsupported database URL format: {v}")
        return v

    @field_validator('redis_url')
    @classmethod
    def validate_redis_url(cls, v):
        """Validate Redis URL format"""
        if not v.startswith('redis://'):
            raise ValueError(f"Invalid Redis URL format: {v}")
        return v

    def get_database_config(self) -> DatabaseConfig:
        """Get parsed database configuration"""
        # Parse database URL
        # This is a simplified parser - in production, use a proper URL parser
        from urllib.parse import urlparse

        parsed = urlparse(self.database_url)

        return DatabaseConfig(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip('/') if parsed.path else "codemind",
            username=parsed.username or "postgres",
            password=parsed.password or "",
            pool_size=self.database_pool_size
        )

    def get_redis_config(self) -> RedisConfig:
        """Get parsed Redis configuration"""
        from urllib.parse import urlparse

        parsed = urlparse(self.redis_url)

        return RedisConfig(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            database=int(parsed.path.lstrip('/')) if parsed.path else 0,
            password=self.redis_password.get_secret_value() if self.redis_password else None
        )

    def get_llm_config(self) -> LLMConfig:
        """Get LLM service configuration"""
        return LLMConfig(
            aws_region=self.aws_region,
            aws_access_key_id=self.aws_access_key_id.get_secret_value(),
            aws_secret_access_key=self.aws_secret_access_key.get_secret_value(),
            azure_endpoint=self.azure_openai_endpoint,
            azure_api_key=self.azure_openai_api_key.get_secret_value()
        )

    def get_temporal_config(self) -> TemporalConfig:
        """Get Temporal configuration"""
        return TemporalConfig(
            host=self.temporal_host,
            port=self.temporal_port,
            namespace=self.temporal_namespace
        )

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return MonitoringConfig(
            enable_metrics=self.enable_metrics,
            enable_tracing=self.enable_tracing,
            jaeger_endpoint=self.jaeger_endpoint
        )

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return SecurityConfig(
            jwt_secret_key=self.jwt_secret_key.get_secret_value(),
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=self.api_rate_limit
        )

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT

    def is_testing(self) -> bool:
        """Check if running in testing"""
        return self.environment == Environment.TESTING

    def validate_configuration(self) -> List[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of validation errors/warnings
        """
        issues = []

        # Check required secrets in production
        if self.is_production():
            if not self.jwt_secret_key.get_secret_value():
                issues.append("JWT secret key is required in production")

            if not self.aws_access_key_id.get_secret_value():
                issues.append("AWS credentials are required in production")

        # Check database connectivity requirements
        if self.database_url.startswith('sqlite://') and self.is_production():
            issues.append("SQLite is not recommended for production")

        # Check Redis configuration
        if not self.redis_password and self.is_production():
            issues.append("Redis password should be set in production")

        # Check timeout values
        if self.agent_timeout_seconds < 10:
            issues.append("Agent timeout seems too low (< 10 seconds)")

        if self.context_ttl_seconds < 3600:
            issues.append("Context TTL seems too low (< 1 hour)")

        return issues

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging (excluding secrets)"""
        return {
            "environment": self.environment.value,
            "app_name": self.app_name,
            "app_version": self.app_version,
            "debug": self.debug,
            "database_type": self.database_url.split("://")[0],
            "redis_configured": bool(self.redis_url),
            "aws_region": self.aws_region,
            "temporal_host": self.temporal_host,
            "monitoring_enabled": {
                "metrics": self.enable_metrics,
                "tracing": self.enable_tracing
            },
            "agent_config": {
                "timeout_seconds": self.agent_timeout_seconds,
                "max_retries": self.agent_max_retries,
                "max_concurrency": self.agent_max_concurrency
            }
        }


class ConfigurationManager:
    """
    Configuration manager with hot-reloading and validation.
    """

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._settings: Optional[AgentSettings] = None
        self._file_mtime: Optional[float] = None

    def get_settings(self, reload: bool = False) -> AgentSettings:
        """
        Get configuration settings with optional hot-reloading.

        Args:
            reload: Force reload from file

        Returns:
            AgentSettings instance
        """
        if self._settings is None or reload or self._should_reload():
            self._load_settings()

        return self._settings

    def _load_settings(self) -> None:
        """Load settings from environment and files"""
        try:
            # Create settings with environment variable override
            self._settings = AgentSettings()

            # Validate configuration
            issues = self._settings.validate_configuration()
            if issues:
                for issue in issues:
                    if self._settings.is_production():
                        logger.error(f"Configuration issue: {issue}")
                    else:
                        logger.warning(f"Configuration issue: {issue}")

            # Update file modification time
            if self.config_file and os.path.exists(self.config_file):
                self._file_mtime = os.path.getmtime(self.config_file)

            logger.info("Configuration loaded successfully")
            logger.debug(f"Configuration summary: {self._settings.get_config_summary()}")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _should_reload(self) -> bool:
        """Check if configuration file has been modified"""
        if not self.config_file or not os.path.exists(self.config_file):
            return False

        if self._file_mtime is None:
            return True

        current_mtime = os.path.getmtime(self.config_file)
        return current_mtime > self._file_mtime

    def reload_if_changed(self) -> bool:
        """
        Reload configuration if file has changed.

        Returns:
            True if configuration was reloaded
        """
        if self._should_reload():
            logger.info("Configuration file changed, reloading...")
            self._load_settings()
            return True
        return False


# Global configuration manager
_config_manager = ConfigurationManager()


def get_settings() -> AgentSettings:
    """Get global configuration settings"""
    return _config_manager.get_settings()


def reload_config() -> AgentSettings:
    """Force reload configuration"""
    return _config_manager.get_settings(reload=True)


def validate_environment() -> None:
    """
    Validate environment setup and configuration.

    Raises:
        RuntimeError: If critical configuration issues are found
    """
    settings = get_settings()
    issues = settings.validate_configuration()

    critical_issues = []
    warnings = []

    for issue in issues:
        if any(word in issue.lower() for word in ['required', 'missing', 'invalid']):
            critical_issues.append(issue)
        else:
            warnings.append(issue)

    # Log warnings
    for warning in warnings:
        logger.warning(f"Configuration warning: {warning}")

    # Fail on critical issues in production
    if critical_issues and settings.is_production():
        error_msg = f"Critical configuration issues: {critical_issues}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info("Environment validation completed successfully")