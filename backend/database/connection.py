import logging
import os
import re
from typing import AsyncGenerator
from urllib.parse import quote_plus, unquote

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.database.models import Base

logger = logging.getLogger(__name__)

# Load environment variables
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE_DIR, ".env"), override=False)
load_dotenv(override=False)

DEFAULT_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/astro_chat"


def normalize_database_url(url: str) -> str:
    """
    Ensure the database URL has a valid asyncpg dialect and properly URL-encodes
    special characters (such as '@') in the password component.
    """
    if not url:
        return url

    # Ensure async driver prefix
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)

    # Safely match: scheme://user:password@host[:port]/dbname[?params]
    pattern = re.compile(
        r"^(?P<scheme>[a-zA-Z0-9_+]+://)(?P<user>[^:]+):(?P<password>.+)@(?P<hostport>[^@/]+)/(?P<rest>.*)$"
    )
    match = pattern.match(url)
    if match:
        scheme = match.group("scheme")
        user = match.group("user")
        raw_password = match.group("password")
        hostport = match.group("hostport")
        rest = match.group("rest")

        # Decode if already partially encoded, then quote properly
        clean_password = unquote(raw_password)
        encoded_password = quote_plus(clean_password)
        return f"{scheme}{user}:{encoded_password}@{hostport}/{rest}"

    return url


RAW_DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
DATABASE_URL = normalize_database_url(RAW_DATABASE_URL)

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency session generator that yields an AsyncSession.
    Commits on successful completion and automatically rolls back if an exception occurs.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Creates all tables defined in models if they do not already exist.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("[DATABASE] PostgreSQL tables initialized successfully via async engine.")
    except Exception as exc:
        logger.error("[DATABASE ERROR] Failed to initialize PostgreSQL tables: %s", exc, exc_info=True)
        raise exc


async def close_db() -> None:
    """
    Closes the connection pool and disposes of the engine.
    """
    try:
        await engine.dispose()
        logger.info("[DATABASE] PostgreSQL async engine disposed.")
    except Exception as exc:
        logger.warning("[DATABASE WARNING] Error disposing PostgreSQL engine: %s", exc)
