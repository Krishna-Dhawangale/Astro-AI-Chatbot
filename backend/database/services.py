import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import AsyncSessionLocal
from backend.database.models import ChatMessage, ChatSession, User

logger = logging.getLogger(__name__)


def coerce_uuid(val: Any) -> uuid.UUID:
    """
    Safely convert various string representations (UUID string, 32-char hex, or arbitrary id)
    into a valid Python uuid.UUID instance.
    """
    if isinstance(val, uuid.UUID):
        return val
    if isinstance(val, str) and val.strip():
        s = val.strip()
        try:
            return uuid.UUID(s)
        except (ValueError, AttributeError):
            try:
                # Try hex
                return uuid.UUID(hex=s)
            except (ValueError, AttributeError):
                # Fallback to deterministic UUID5 based on the string
                return uuid.uuid5(uuid.NAMESPACE_DNS, s)
    return uuid.uuid4()


async def get_or_create_user(
    db: AsyncSession,
    user_id: Optional[Union[str, uuid.UUID]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
) -> Optional[User]:
    """
    Finds or creates a User row based on user_id and profile attributes.
    """
    if not user_id and not user_profile:
        return None

    user_uuid = coerce_uuid(user_id) if user_id else uuid.uuid4()
    profile = user_profile or {}

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalars().first()

    name = profile.get("name")
    email = profile.get("email")
    lagna = profile.get("lagna") or profile.get("ascendant")
    moon_sign = profile.get("moon_sign") or profile.get("rashi")

    birth_date = None
    by = profile.get("birth_year")
    bm = profile.get("birth_month")
    bd = profile.get("birth_day")
    bh = profile.get("birth_hour", 0)
    if by and bm and bd:
        try:
            hour_int = int(bh) if bh is not None else 0
            birth_date = datetime.datetime(int(by), int(bm), int(bd), hour_int, 0, 0)
        except Exception:
            birth_date = None

    if not user:
        user = User(
            id=user_uuid,
            name=name,
            email=email,
            birth_date=birth_date,
            lagna=lagna,
            moon_sign=moon_sign,
        )
        db.add(user)
        await db.flush()
    else:
        # Update existing profile fields if new data was provided
        updated = False
        if name and user.name != name:
            user.name = name
            updated = True
        if email and user.email != email:
            user.email = email
            updated = True
        if lagna and user.lagna != lagna:
            user.lagna = lagna
            updated = True
        if moon_sign and user.moon_sign != moon_sign:
            user.moon_sign = moon_sign
            updated = True
        if birth_date and user.birth_date != birth_date:
            user.birth_date = birth_date
            updated = True
        if updated:
            await db.flush()

    return user


async def get_or_create_session(
    db: AsyncSession,
    session_id: Union[str, uuid.UUID],
    user_id: Optional[Union[str, uuid.UUID]] = None,
    title: Optional[str] = None,
) -> ChatSession:
    """
    Finds or creates a ChatSession row for the given session_id.
    """
    session_uuid = coerce_uuid(session_id)
    user_uuid = coerce_uuid(user_id) if user_id else None

    result = await db.execute(select(ChatSession).where(ChatSession.id == session_uuid))
    chat_session = result.scalars().first()

    if not chat_session:
        chat_session = ChatSession(
            id=session_uuid,
            user_id=user_uuid,
            title=(title[:250] if title else "Astrology Consultation"),
        )
        db.add(chat_session)
        await db.flush()
    elif user_uuid and chat_session.user_id != user_uuid:
        chat_session.user_id = user_uuid
        await db.flush()

    return chat_session


async def log_chat_interaction(
    session_id: Union[str, uuid.UUID],
    user_query: str,
    bot_response: str,
    execution_mode: str,
    detected_intents: Optional[Any] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    calculation_footer: Optional[str] = None,
    is_unresolved: bool = False,
    user_id: Optional[Union[str, uuid.UUID]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    db: Optional[AsyncSession] = None,
) -> Optional[ChatMessage]:
    """
    Persists a single chat interaction turn into PostgreSQL:
    - multi-user chat sessions
    - user profiles
    - individual messages
    - execution mode (LOCAL, HYBRID, LLM, UNRESOLVED)
    - token metrics
    - calculation footer
    - is_unresolved boolean flag
    """
    async def _execute_logging(session: AsyncSession) -> ChatMessage:
        # 1. Ensure user exists if profile / user_id provided
        user_record = await get_or_create_user(
            db=session,
            user_id=user_id or session_id,
            user_profile=user_profile,
        )

        resolved_user_id = user_record.id if user_record else (coerce_uuid(user_id) if user_id else None)

        # 2. Ensure chat session exists
        clean_title = (user_query.strip()[:60] + "...") if len(user_query.strip()) > 60 else user_query.strip()
        chat_sess = await get_or_create_session(
            db=session,
            session_id=session_id,
            user_id=resolved_user_id,
            title=clean_title or "Astrology Chat",
        )

        # 3. Normalize execution mode
        valid_modes = {"LOCAL", "HYBRID", "LLM", "UNRESOLVED"}
        norm_mode = (execution_mode or "").upper().strip()
        if is_unresolved:
            norm_mode = "UNRESOLVED"
        elif norm_mode not in valid_modes:
            if any(k in norm_mode for k in ["HYBRID", "SYNTHESIS"]):
                norm_mode = "HYBRID"
            elif any(k in norm_mode for k in ["LLM", "GEMINI", "GROQ"]):
                norm_mode = "LLM"
            elif any(k in norm_mode for k in ["LOCAL", "RULE", "DETERMINISTIC", "FAQ", "FACT"]):
                norm_mode = "LOCAL"
            else:
                norm_mode = "LOCAL"

        # 4. Normalize detected_intents to valid JSON structure
        intents_payload = detected_intents
        if isinstance(detected_intents, str):
            intents_payload = [detected_intents]
        elif detected_intents is None:
            intents_payload = []

        # 5. Create ChatMessage
        message = ChatMessage(
            id=uuid.uuid4(),
            session_id=chat_sess.id,
            user_query=user_query,
            bot_response=bot_response,
            execution_mode=norm_mode,
            detected_intents=intents_payload,
            input_tokens=max(0, int(input_tokens or 0)),
            output_tokens=max(0, int(output_tokens or 0)),
            calculation_footer=calculation_footer,
            is_unresolved=bool(is_unresolved),
        )
        session.add(message)
        await session.commit()
        await session.refresh(message)
        logger.info(
            "[DATABASE] Chat message persisted. Session=%s, Mode=%s, Unresolved=%s, InTokens=%d, OutTokens=%d",
            chat_sess.id,
            norm_mode,
            is_unresolved,
            input_tokens,
            output_tokens,
        )
        return message

    try:
        if db is not None:
            return await _execute_logging(db)
        else:
            async with AsyncSessionLocal() as session:
                return await _execute_logging(session)
    except Exception as exc:
        logger.error("[DATABASE ERROR] Failed to log chat interaction to PostgreSQL: %s", exc, exc_info=True)
        return None


async def get_unresolved_queries(
    db: AsyncSession,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[ChatMessage], int]:
    """
    Retrieves all queries where is_unresolved == True, ordered by timestamp descending,
    along with the total count for admin review.
    """
    count_stmt = select(func.count(ChatMessage.id)).where(ChatMessage.is_unresolved.is_(True))
    total_count = (await db.execute(count_stmt)).scalar() or 0

    query_stmt = (
        select(ChatMessage)
        .where(ChatMessage.is_unresolved.is_(True))
        .order_by(desc(ChatMessage.timestamp))
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(query_stmt)
    records = list(result.scalars().all())

    return records, total_count


async def mark_query_resolved(
    db: AsyncSession,
    message_id: Union[str, uuid.UUID],
) -> Optional[ChatMessage]:
    """
    Marks an unresolved query as resolved (e.g. after review or retraining).
    """
    msg_uuid = coerce_uuid(message_id)
    result = await db.execute(select(ChatMessage).where(ChatMessage.id == msg_uuid))
    msg = result.scalars().first()
    if msg:
        msg.is_unresolved = False
        await db.commit()
        await db.refresh(msg)
    return msg
