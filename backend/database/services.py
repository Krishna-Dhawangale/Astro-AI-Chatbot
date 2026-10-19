import datetime
import logging
import math
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from backend.database.connection import AsyncSessionLocal
    from backend.database.models import ChatMessage, ChatSession, User
except ImportError:
    from database.connection import AsyncSessionLocal
    from database.models import ChatMessage, ChatSession, User

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
    Finds or creates a User row based on user_id and profile attributes,
    preserving full birth timing and geographic coordinates.
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
    birth_place = profile.get("birth_place") or profile.get("place")

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

    lat = profile.get("latitude")
    lon = profile.get("longitude")
    tz = profile.get("timezone")

    lat_val = float(lat) if (lat is not None and math.isfinite(float(lat))) else None
    lon_val = float(lon) if (lon is not None and math.isfinite(float(lon))) else None
    tz_val = float(tz) if (tz is not None and math.isfinite(float(tz))) else None
    bh_val = float(bh) if (bh is not None and math.isfinite(float(bh))) else None

    if not user:
        user = User(
            id=user_uuid,
            name=name,
            email=email,
            birth_date=birth_date,
            birth_hour=bh_val,
            birth_place=birth_place,
            latitude=lat_val or 28.6139,
            longitude=lon_val or 77.2090,
            timezone=tz_val or 5.5,
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
        if bh_val is not None and user.birth_hour != bh_val:
            user.birth_hour = bh_val
            updated = True
        if birth_place and getattr(user, "birth_place", None) != birth_place:
            user.birth_place = birth_place
            updated = True
        if lat_val is not None and getattr(user, "latitude", None) != lat_val:
            user.latitude = lat_val
            updated = True
        if lon_val is not None and getattr(user, "longitude", None) != lon_val:
            user.longitude = lon_val
            updated = True
        if tz_val is not None and getattr(user, "timezone", None) != tz_val:
            user.timezone = tz_val
            updated = True
        if updated:
            await db.flush()

    return user


def is_valid_uuid(val: Any) -> bool:
    if isinstance(val, uuid.UUID):
        return True
    if not isinstance(val, str) or not val.strip():
        return False
    try:
        uuid.UUID(val.strip())
        return True
    except (ValueError, AttributeError):
        return False


async def get_user_birth_details_from_db(
    user_or_session_id: Optional[Union[str, uuid.UUID]] = None,
    session_id: Optional[Union[str, uuid.UUID]] = None,
    fallback_name: Optional[str] = None,
    db: Optional[AsyncSession] = None,
) -> Optional[Dict[str, Any]]:
    """
    Authoritatively fetches the user's saved birth details (date, time, location, lagna, moon_sign)
    from Neon / PostgreSQL using user_id, session_id, or fallback name.
    Guarantees robust fallback for single-user personal installations and handles
    session-isolated and username-based lookups gracefully.
    """
    async def _query(session: AsyncSession) -> Optional[Dict[str, Any]]:
        user: Optional[User] = None
        chat_sess: Optional[ChatSession] = None

        # 1. Direct User lookup by valid UUID
        if user_or_session_id and is_valid_uuid(user_or_session_id):
            target_uuid = coerce_uuid(user_or_session_id)
            res = await session.execute(select(User).where(User.id == target_uuid))
            user = res.scalars().first()

        # 2. Direct User lookup by username or email if string identifier provided
        if (not user or not user.birth_date) and isinstance(user_or_session_id, str) and not is_valid_uuid(user_or_session_id):
            clean_str = user_or_session_id.strip().lower()
            if clean_str not in ("guest", "seeker", "there", "anonymous", ""):
                u_str_res = await session.execute(
                    select(User)
                    .where(
                        (func.lower(User.name) == clean_str) | (func.lower(User.email) == clean_str),
                        User.birth_date.isnot(None),
                    )
                    .order_by(desc(User.created_at))
                )
                user = u_str_res.scalars().first()

        # 3. Lookup via ChatSession -> User (by user_or_session_id)
        if (not user or not user.birth_date) and user_or_session_id:
            target_uuid = coerce_uuid(user_or_session_id)
            s_res = await session.execute(select(ChatSession).where(ChatSession.id == target_uuid))
            chat_sess = s_res.scalars().first()
            if chat_sess and chat_sess.user_id:
                u_res = await session.execute(select(User).where(User.id == chat_sess.user_id))
                user = u_res.scalars().first()

        # 4. Lookup via active session_id (if distinct from user_or_session_id)
        if (not user or not user.birth_date) and session_id:
            sess_uuid = coerce_uuid(session_id)
            if not chat_sess or chat_sess.id != sess_uuid:
                s_res = await session.execute(select(ChatSession).where(ChatSession.id == sess_uuid))
                chat_sess = s_res.scalars().first()
            if chat_sess and chat_sess.user_id:
                u_res = await session.execute(select(User).where(User.id == chat_sess.user_id))
                user = u_res.scalars().first()
            if not user:
                u_direct_sess = await session.execute(select(User).where(User.id == sess_uuid))
                user = u_direct_sess.scalars().first()

        # 5. Lookup by fallback_name if user not found or has no birth_date
        if (not user or not user.birth_date) and fallback_name:
            clean_name = fallback_name.strip().lower()
            if clean_name not in ("guest", "seeker", "there", "anonymous", ""):
                name_res = await session.execute(
                    select(User)
                    .where(func.lower(User.name) == clean_name, User.birth_date.isnot(None))
                    .order_by(desc(User.created_at))
                )
                user = name_res.scalars().first()

        # 6. Check ChatSession.birth_profile JSONB if User record is absent but session has stored profile
        if (not user or not user.birth_date) and chat_sess and chat_sess.birth_profile and isinstance(chat_sess.birth_profile, dict):
            bp = chat_sess.birth_profile
            b_yr = int(bp.get("birth_year") or 2000)
            if b_yr != 2000 or bp.get("birth_month") is not None:
                return {
                    "found": True,
                    "user_id": str(chat_sess.user_id) if chat_sess.user_id else str(chat_sess.id),
                    "name": bp.get("name") or fallback_name or "Seeker",
                    "birth_year": b_yr,
                    "birth_month": int(bp.get("birth_month") or 1),
                    "birth_day": int(bp.get("birth_day") or 1),
                    "birth_hour": float(bp.get("birth_hour") or 12.0),
                    "birth_place": bp.get("birth_place") or "New Delhi, India",
                    "latitude": float(bp.get("latitude") if bp.get("latitude") is not None else 28.6139),
                    "longitude": float(bp.get("longitude") if bp.get("longitude") is not None else 77.2090),
                    "timezone": float(bp.get("timezone") if bp.get("timezone") is not None else 5.5),
                    "lagna": bp.get("lagna"),
                    "moon_sign": bp.get("moon_sign"),
                }

        # 7. Authoritative fallback for personal chatbot / single-user installations:
        # If user birth profile is stored in DB but session didn't link user_id, pick the active registered user
        if not user or not user.birth_date:
            active_user_res = await session.execute(
                select(User)
                .where(
                    User.birth_date.isnot(None),
                    User.name.isnot(None),
                    User.name.notin_(["Guest", "IntegrationTester"]),
                )
                .order_by(desc(User.created_at))
            )
            user = active_user_res.scalars().first()

        if not user or not user.birth_date:
            return None

        # Extract birth parameters safely
        b_date = user.birth_date
        b_hour_val = getattr(user, "birth_hour", None)
        if b_hour_val is not None and math.isfinite(b_hour_val):
            calc_hour = float(b_hour_val)
        else:
            calc_hour = float(b_date.hour) + (float(b_date.minute) / 60.0) + (float(b_date.second) / 3600.0)

        lat = getattr(user, "latitude", None)
        lon = getattr(user, "longitude", None)
        tz = getattr(user, "timezone", None)

        return {
            "found": True,
            "user_id": str(user.id),
            "name": user.name or fallback_name or "Seeker",
            "email": user.email,
            "birth_date": b_date,
            "birth_year": int(b_date.year),
            "birth_month": int(b_date.month),
            "birth_day": int(b_date.day),
            "birth_hour": float(calc_hour),
            "birth_place": getattr(user, "birth_place", None) or "New Delhi, India",
            "latitude": float(lat if lat is not None and math.isfinite(lat) else 28.6139),
            "longitude": float(lon if lon is not None and math.isfinite(lon) else 77.2090),
            "timezone": float(tz if tz is not None and math.isfinite(tz) else 5.5),
            "lagna": user.lagna,
            "moon_sign": user.moon_sign,
        }

    try:
        if db is not None:
            return await _query(db)
        else:
            async with AsyncSessionLocal() as s:
                return await _query(s)
    except Exception as exc:
        err_trace = traceback.format_exc()
        print(f"[RENDER DATABASE ERROR] Failed to fetch user birth details: {exc}\n{err_trace}", flush=True)
        logger.error("[RENDER DATABASE ERROR] Failed to fetch user birth details: %s", exc, exc_info=True)
        return None


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
    execution_mode: str = "LOCAL",
    is_unresolved: bool = False,
    detected_intents: Optional[Any] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    calculation_footer: Optional[str] = None,
    user_id: Optional[Union[str, uuid.UUID]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    db: Optional[AsyncSession] = None,
) -> Optional[ChatMessage]:
    """
    Persists a single chat interaction turn into Neon / PostgreSQL database:
    - multi-user chat sessions & user profile updates
    - individual chat message record
    - execution_mode ('LOCAL', 'HYBRID', 'LLM', 'UNRESOLVED', 'CRASH_FALLBACK')
    - is_unresolved boolean flag for failure tracking and telemetry
    - input/output token usage
    - calculation footers / provenance notes

    Outputs detailed stack traces to stdout for immediate visibility in Render logs
    if any database connection or write exception occurs.
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

        # 3. Normalize execution mode & error locking flags
        raw_mode = (execution_mode or "").upper().strip()
        valid_modes = {"LOCAL", "HYBRID", "LLM", "UNRESOLVED", "CRASH_FALLBACK", "UNRESOLVED_INTENT"}

        unresolved_flag = bool(is_unresolved)

        if raw_mode == "UNRESOLVED_INTENT":
            norm_mode = "UNRESOLVED_INTENT"
            unresolved_flag = True
        elif raw_mode == "CRASH_FALLBACK":
            norm_mode = "CRASH_FALLBACK"
            unresolved_flag = True
        elif raw_mode == "UNRESOLVED":
            norm_mode = "UNRESOLVED"
            unresolved_flag = True
        elif raw_mode in valid_modes:
            norm_mode = raw_mode
        elif unresolved_flag:
            norm_mode = "UNRESOLVED"
        else:
            if any(k in raw_mode for k in ["HYBRID", "SYNTHESIS"]):
                norm_mode = "HYBRID"
            elif any(k in raw_mode for k in ["LLM", "GEMINI", "GROQ"]):
                norm_mode = "LLM"
            elif any(k in raw_mode for k in ["CRASH", "FAIL"]):
                norm_mode = "CRASH_FALLBACK"
                unresolved_flag = True
            elif any(k in raw_mode for k in ["INTENT", "OUT_OF_SCOPE"]):
                norm_mode = "UNRESOLVED_INTENT"
                unresolved_flag = True
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
            is_unresolved=unresolved_flag,
            detected_intents=intents_payload,
            input_tokens=max(0, int(input_tokens or 0)),
            output_tokens=max(0, int(output_tokens or 0)),
            calculation_footer=calculation_footer,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )
        session.add(message)
        await session.commit()
        await session.refresh(message)
        logger.info(
            "[DATABASE] Chat message persisted to Neon. Session=%s, Mode=%s, Unresolved=%s, InTokens=%d, OutTokens=%d",
            chat_sess.id,
            norm_mode,
            unresolved_flag,
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
        err_trace = traceback.format_exc()
        print(f"[RENDER DATABASE ERROR] Failed to log interaction to Neon DB: {exc}\n{err_trace}", flush=True)
        logger.error("[RENDER DATABASE ERROR] Failed to log interaction to Neon DB: %s", exc, exc_info=True)
        return None


async def log_unresolved_intent(
    session_id: Union[str, uuid.UUID],
    user_query: str,
    bot_response: str = "Please share an appropriate question so I can assist you better with your astrology query.",
    db: Optional[AsyncSession] = None,
) -> Optional[ChatMessage]:
    """
    High-visibility logging to stdout and Neon PostgreSQL persistence for unhandled,
    low-confidence, or out-of-scope intents that trigger a fallback response.

    Invariants:
    1. High-visibility warning to Render stdout via Python logger.
    2. Inserts into `chat_messages` table with execution_mode='UNRESOLVED_INTENT',
       is_unresolved=True, and timestamp=UTC now.
    3. Non-blocking error handling: guarantees the API response to the client is never broken.
    """
    logger.warning(
        "\n%s\n[UNRESOLVED INTENT WARNING]\n"
        "Session ID   : %s\n"
        "User Query   : %s\n"
        "Bot Response : %s\n"
        "Execution Mode: UNRESOLVED_INTENT\n%s",
        "=" * 65,
        session_id,
        user_query,
        bot_response,
        "=" * 65,
    )
    print(
        f"[UNRESOLVED INTENT WARNING] Session={session_id} | Query='{user_query}' | ExecutionMode=UNRESOLVED_INTENT",
        flush=True,
    )

    try:
        return await log_chat_interaction(
            session_id=session_id,
            user_query=user_query,
            bot_response=bot_response,
            execution_mode="UNRESOLVED_INTENT",
            is_unresolved=True,
            detected_intents=["unresolved_intent"],
            db=db,
        )
    except Exception as exc:
        err_trace = traceback.format_exc()
        logger.error("[DATABASE] Failed to log unresolved intent to Neon: %s\n%s", exc, err_trace)
        print(f"[RENDER DATABASE ERROR] Failed to log unresolved intent: {exc}", flush=True)
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
