import logging
import os
import re
import traceback
from typing import Any, AsyncGenerator, Dict, Tuple
from urllib.parse import quote_plus, unquote

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

try:
    from backend.database.models import Base
except ImportError:
    from database.models import Base

logger = logging.getLogger(__name__)

# Load environment variables
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE_DIR, ".env"), override=False)
load_dotenv(override=False)

DEFAULT_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/astro_chat"


def normalize_database_url(url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Ensure the database URL has a valid asyncpg dialect, properly URL-encodes
    special characters (such as '@' or '#') in the password component, and
    configures SSL parameters appropriately for Neon and PostgreSQL.

    Returns:
        (clean_url: str, connect_args: dict)
    """
    if not url:
        return DEFAULT_DATABASE_URL, {}

    normalized = url.strip()

    # 1. Ensure asyncpg driver prefix
    if normalized.startswith("postgresql://"):
        normalized = normalized.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif normalized.startswith("postgres://"):
        normalized = normalized.replace("postgres://", "postgresql+asyncpg://", 1)

    # 2. Safely URL-encode credentials if present
    pattern = re.compile(
        r"^(?P<scheme>[a-zA-Z0-9_+]+://)(?P<user>[^:]+):(?P<password>.+)@(?P<hostport>[^@/]+)/(?P<rest>.*)$"
    )
    match = pattern.match(normalized)
    if match:
        scheme = match.group("scheme")
        user = match.group("user")
        raw_password = match.group("password")
        hostport = match.group("hostport")
        rest = match.group("rest")

        clean_password = unquote(raw_password)
        encoded_password = quote_plus(clean_password)
        normalized = f"{scheme}{user}:{encoded_password}@{hostport}/{rest}"

    # 3. Detect Neon / SSL requirements
    connect_args: Dict[str, Any] = {}
    is_neon = "neon.tech" in normalized.lower()
    requires_ssl = (
        is_neon
        or "sslmode=require" in normalized.lower()
        or "ssl=require" in normalized.lower()
        or os.getenv("RENDER") == "true"
        or os.getenv("ENVIRONMENT") == "production"
    )

    # asyncpg dialect does not accept 'sslmode' query param directly in URL
    if "sslmode=" in normalized:
        normalized = re.sub(r"[\?&]sslmode=[^&]+", "", normalized)
        if "?" not in normalized and "&" in normalized:
            normalized = normalized.replace("&", "?", 1)

    if requires_ssl:
        connect_args["ssl"] = "require"

    return normalized, connect_args


RAW_DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
DATABASE_URL, CONNECT_ARGS = normalize_database_url(RAW_DATABASE_URL)

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,  # Crucial for Neon pooler to detect terminated idle connections
    pool_recycle=300,    # Recycle connections every 5 minutes for Neon PgBouncer stability
    pool_size=10,
    max_overflow=20,
    connect_args=CONNECT_ARGS,
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
        except Exception as exc:
            await session.rollback()
            logger.error("[DATABASE ERROR] Session rollback due to exception: %s", exc)
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initializes PostgreSQL tables in Neon/PostgreSQL if they do not exist.
    Logs progress directly to stdout for clear visibility in Render dashboard logs.
    """
    try:
        print("[DATABASE] Initializing PostgreSQL database tables on Neon/PostgreSQL...", flush=True)
        logger.info("[DATABASE] Initializing PostgreSQL database tables on Neon/PostgreSQL...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("[DATABASE] PostgreSQL tables initialized successfully via async engine.", flush=True)
        logger.info("[DATABASE] PostgreSQL tables initialized successfully via async engine.")
    except Exception as exc:
        error_msg = f"[DATABASE ERROR] Failed to initialize PostgreSQL tables: {exc}"
        print(error_msg, flush=True)
        traceback.print_exc()
        logger.error(error_msg, exc_info=True)
        raise exc


async def close_db() -> None:
    """
    Closes the connection pool and disposes of the engine on application shutdown.
    """
    try:
        await engine.dispose()
        print("[DATABASE] PostgreSQL async engine disposed.", flush=True)
        logger.info("[DATABASE] PostgreSQL async engine disposed.")
    except Exception as exc:
        print(f"[DATABASE WARNING] Error disposing PostgreSQL engine: {exc}", flush=True)
        logger.warning("[DATABASE WARNING] Error disposing PostgreSQL engine: %s", exc)
