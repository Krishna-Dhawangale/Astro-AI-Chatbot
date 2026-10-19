"""
Convenient CLI utility to view all users, chat sessions, and messages in PostgreSQL.
Run anytime with:
    python view_db.py
"""
import asyncio
from backend.database.connection import AsyncSessionLocal
from backend.database.models import User, ChatSession, ChatMessage
from sqlalchemy import select, desc


async def view_database():
    print("=" * 80)
    print("                ASTRO CHAT DATABASE VIEWER (POSTGRESQL)")
    print("=" * 80)

    async with AsyncSessionLocal() as session:
        # 1. Users
        users = (await session.execute(select(User).order_by(desc(User.created_at)))).scalars().all()
        print(f"\n[USERS] Total: {len(users)}")
        print("-" * 80)
        if not users:
            print("  No users found.")
        for u in users:
            print(f"  ID: {u.id}")
            print(f"  Name: {u.name or 'N/A'} | Email: {u.email or 'N/A'}")
            print(f"  Lagna: {u.lagna or 'N/A'} | Moon Sign: {u.moon_sign or 'N/A'}")
            print(f"  Created At: {u.created_at}")
            print("  " + "-" * 40)

        # 2. Chat Sessions
        sessions = (await session.execute(select(ChatSession).order_by(desc(ChatSession.created_at)))).scalars().all()
        print(f"\n[CHAT SESSIONS] Total: {len(sessions)}")
        print("-" * 80)
        if not sessions:
            print("  No sessions found.")
        for s in sessions:
            print(f"  Session ID: {s.id}")
            print(f"  User ID:    {s.user_id}")
            print(f"  Title:      {s.title}")
            print(f"  Created At: {s.created_at}")
            print("  " + "-" * 40)

        # 3. Chat Messages
        messages = (await session.execute(select(ChatMessage).order_by(desc(ChatMessage.timestamp)))).scalars().all()
        print(f"\n[CHAT MESSAGES] Total: {len(messages)}")
        print("-" * 80)
        if not messages:
            print("  No messages recorded yet.")
        for m in messages:
            status_tag = "[FAILED/UNRESOLVED]" if m.is_unresolved else "[RESOLVED]"
            print(f"  {status_tag} Message ID: {m.id}")
            print(f"  Session:    {m.session_id}")
            print(f"  Timestamp:  {m.timestamp}")
            print(f"  Mode:       {m.execution_mode} | Tokens: In={m.input_tokens}, Out={m.output_tokens}")
            print(f"  User Query: {m.user_query}")
            preview = m.bot_response.strip().replace('\n', ' ')
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print(f"  Bot Answer: {preview}")
            if m.calculation_footer:
                print(f"  Footer:     {m.calculation_footer}")
            print("  " + "-" * 40)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(view_database())
