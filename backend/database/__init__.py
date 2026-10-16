from backend.database.models import Base, User, ChatSession, ChatMessage
from backend.database.connection import (
    DATABASE_URL,
    engine,
    AsyncSessionLocal,
    get_db,
    init_db,
    close_db,
)
from backend.database.services import (
    log_chat_interaction,
    get_unresolved_queries,
    mark_query_resolved,
    get_or_create_user,
    get_or_create_session,
    coerce_uuid,
)

__all__ = [
    "Base",
    "User",
    "ChatSession",
    "ChatMessage",
    "DATABASE_URL",
    "engine",
    "AsyncSessionLocal",
    "get_db",
    "init_db",
    "close_db",
    "log_chat_interaction",
    "get_unresolved_queries",
    "mark_query_resolved",
    "get_or_create_user",
    "get_or_create_session",
    "coerce_uuid",
]
