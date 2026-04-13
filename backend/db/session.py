"""
OMNIVIS — Database Session Management
Async SQLAlchemy session factory. Uses SQLite for dev, PostgreSQL for production.
"""
import os
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import asynccontextmanager

# Default to SQLite for easy development — no external DB needed
_default_db = f"sqlite+aiosqlite:///{Path(__file__).parent.parent / 'omnivis_dev.db'}"
DATABASE_URL = os.getenv("DATABASE_URL", _default_db)

# Auto-fix common URL issues
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

_is_sqlite = DATABASE_URL.startswith("sqlite")

engine_kwargs: dict = {
    "echo": False,
}
if not _is_sqlite:
    engine_kwargs.update(pool_size=20, max_overflow=10, pool_pre_ping=True, pool_recycle=3600)

engine = create_async_engine(DATABASE_URL, **engine_kwargs)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_session():
    """Provide a transactional scope around a series of operations."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db():
    """FastAPI dependency for database sessions."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Create all tables on startup."""
    from .models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Dispose engine on shutdown."""
    await engine.dispose()
