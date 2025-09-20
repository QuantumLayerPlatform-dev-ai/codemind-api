"""
Database configuration and connection
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from typing import AsyncGenerator

try:
    from .config import get_settings
    from .logging import get_logger
    from ..models.base import Base
except ImportError:
    from core.config import get_settings
    from core.logging import get_logger
    from models.base import Base

logger = get_logger("database")

# Get database settings
settings = get_settings()

# Create async engine
async_engine = create_async_engine(
    settings.async_database_url,
    echo=settings.debug,
    pool_pre_ping=True
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create sync engine for migrations
sync_engine = create_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True
)

# Create sync session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

def get_sync_db() -> Session:
    """Get synchronous database session"""
    return SessionLocal()

async def init_db():
    """Initialize database tables"""
    try:
        async with async_engine.begin() as conn:
            # Import all models to ensure they're registered
            try:
                from ..models import User, GenerationRequestModel, GenerationResponseModel
            except ImportError:
                from models import User, GenerationRequestModel, GenerationResponseModel

            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ Database tables created successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise

async def close_db():
    """Close database connections"""
    await async_engine.dispose()
    logger.info("🔒 Database connections closed")