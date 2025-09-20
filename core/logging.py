"""
Logging configuration for CodeMind API
"""

import logging
import sys
from typing import Dict, Any

from .config import get_settings

settings = get_settings()


def setup_logging() -> None:
    """Set up application logging"""

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    root_logger.addHandler(console_handler)

    # Configure specific loggers
    loggers_config = {
        "codemind": settings.log_level.upper(),
        "fastapi": "INFO",
        "uvicorn": "INFO",
        "sqlalchemy.engine": "WARNING",  # Reduce SQL query noise
        "boto3": "WARNING",
        "botocore": "WARNING",
        "openai": "WARNING",
    }

    for logger_name, level in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level))

    # Silence noisy loggers in production
    if not settings.debug:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(f"codemind.{name}")