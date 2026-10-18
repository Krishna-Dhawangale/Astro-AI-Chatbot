r"""
Standalone Supabase PostgreSQL Connection & Table Creation Test Script.
Verifies asynchronous connection via SQLAlchemy + asyncpg and creates all schema tables.

Run with:
    .\Scripts\python.exe test_supabase.py
or (if inside backend/):
    ..\Scripts\python.exe ..\test_supabase.py
"""

import asyncio
import os
import re
import sys
from urllib.parse import quote_plus, unquote

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

# 1. Load environment variables from .env
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
load_dotenv(os.path.join(BASE_DIR, "backend", ".env"), override=False)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from backend.database.models import Base


def normalize_database_url(url: str) -> str:
    """
    Ensure the database URL has a valid asyncpg dialect (postgresql+asyncpg://)
    and properly URL-encodes special characters in the password.
    """
    if not url:
        return ""

    clean_url = url.strip()

    # Ensure async driver prefix
    if clean_url.startswith("postgresql://"):
        clean_url = clean_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif clean_url.startswith("postgres://"):
        clean_url = clean_url.replace("postgres://", "postgresql+asyncpg://", 1)

    # Safely encode special characters (such as '@', '%', '#') in password
    pattern = re.compile(
        r"^(?P<scheme>[a-zA-Z0-9_+]+://)(?P<user>[^:]+):(?P<password>.+)@(?P<hostport>[^@/]+)/(?P<rest>.*)$"
    )
    match = pattern.match(clean_url)
    if match:
        scheme = match.group("scheme")
        user = match.group("user")
        raw_password = match.group("password")
        hostport = match.group("hostport")
        rest = match.group("rest")

        clean_password = unquote(raw_password)
        encoded_password = quote_plus(clean_password)
        return f"{scheme}{user}:{encoded_password}@{hostport}/{rest}"

    return clean_url


def mask_url_for_display(url: str) -> str:
    """Masks database password for safe terminal logging."""
    pattern = re.compile(r":([^:@/]+)@")
    return pattern.sub(r":****@", url)


async def test_supabase_connection():
    print("=" * 80)
    print("           SUPABASE POSTGRESQL ASYNC CONNECTION & TABLE VERIFIER")
    print("=" * 80)

    # 2. Read DATABASE_URL from .env
    raw_db_url = os.getenv("DATABASE_URL")
    if not raw_db_url:
        print("[ERROR] DATABASE_URL is not set in your .env file!")
        print("Please add DATABASE_URL=postgresql+asyncpg://postgres:[PASSWORD]@[HOST]:[PORT]/postgres to .env")
        sys.exit(1)

    db_url = normalize_database_url(raw_db_url)
    print(f"\n[INFO] Target Database: {mask_url_for_display(db_url)}")

    # 3. Create async SQLAlchemy engine
    print("[INFO] Creating asynchronous SQLAlchemy engine (asyncpg)...")
    engine = create_async_engine(
        db_url,
        echo=False,
        pool_pre_ping=True,
    )

    try:
        # 4. Connect to Supabase and execute test query
        print("[INFO] Connecting to Supabase...")
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version();"))
            pg_version = result.scalar()
            print("\n" + "=" * 80)
            print("[SUCCESS] Connected to Supabase PostgreSQL successfully!")
            print(f"[VERSION] {pg_version}")
            print("=" * 80)

            # 5. Automatically create all tables (users, chat_sessions, chat_messages)
            print("\n[INFO] Running Base.metadata.create_all to ensure tables exist...")
            await conn.run_sync(Base.metadata.create_all)
            await conn.commit()
            print("[SUCCESS] Base.metadata.create_all executed successfully.")

            # 6. Verify existing tables in the database
            tables_query = text(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
                """
            )
            tables_result = await conn.execute(tables_query)
            tables = [row[0] for row in tables_result.fetchall()]

            print("\n[VERIFICATION] Public tables found in Supabase:")
            expected_tables = {"users", "chat_sessions", "chat_messages"}
            for t in tables:
                marker = " [TARGET TABLE]" if t in expected_tables else ""
                print(f"  - {t}{marker}")

            missing = expected_tables - set(tables)
            if not missing:
                print("\n[ALL CONFIRMED] All required tables (users, chat_sessions, chat_messages) are present!")
            else:
                print(f"\n[WARNING] Missing expected tables: {missing}")

    except Exception as exc:
        print("\n" + "!" * 80)
        print("[FAILURE] Failed to connect or initialize Supabase database:")
        print(f"Error Type   : {type(exc).__name__}")
        print(f"Error Message: {exc}")
        print("!" * 80)
        print("\nTroubleshooting Tips:")
        print("1. If using Supabase Connection Pooler (port 6543 or 5432), verify password.")
        print("2. Ensure your IP is allowed or Supabase project is active (not paused).")
        print("3. Check that the connection string uses asyncpg (e.g. postgresql+asyncpg://...).")
    finally:
        await engine.dispose()
        print("\n[INFO] Engine connection pool disposed cleanly.")


if __name__ == "__main__":
    asyncio.run(test_supabase_connection())
