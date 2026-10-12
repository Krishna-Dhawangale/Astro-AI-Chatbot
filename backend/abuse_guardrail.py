"""
Safety & Abuse Guardrail Engine: Multi-Lingual Toxicity Detection with 3-Strike Session Termination.

Rule 2 (Safety & Profanity Guardrail):
- Multi-lingual Toxicity / Vulgarity Detection (English, Hindi, Hinglish).
- In-memory `abuse_strikes` tracking per session ID.
- Strike 1 & 2: Return warning: "⚠️ Warning ({count}/3): Inappropriate or vulgar language is not permitted."
- Strike 3: Terminate chat session immediately and lock further inputs:
  "🚫 Session Terminated: You have exceeded the limit for inappropriate language."

Rule 1 Integration (Fast Domain Restriction):
- Integrates domain validation in middleware and route checks.
"""

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from router import OUT_OF_SCOPE_RESPONSE, validate_astrology_domain
except ImportError:
    from backend.router import OUT_OF_SCOPE_RESPONSE, validate_astrology_domain

logger = logging.getLogger(__name__)

# Standard Guardrail Templates mandated by Rule Book
WARNING_TEMPLATE = "⚠️ Warning ({count}/3): Inappropriate or vulgar language is not permitted."
TERMINATED_MESSAGE = "🚫 Session Terminated: You have exceeded the limit for inappropriate language."

# Character substitution mapping for leetspeak / masked profanity
LEET_SUBS = {
    "@": "a",
    "4": "a",
    "$": "s",
    "5": "s",
    "1": "i",
    "!": "i",
    "0": "o",
    "3": "e",
}

# ---------------------------------------------------------------------------
# Multi-Lingual Abuse & Toxicity Regex Rules (English, Hindi, Hinglish)
# ---------------------------------------------------------------------------
ABUSE_REGEX_PATTERNS = [
    # --- English Profanity, Vulgarity & Sexually Explicit Terms ---
    r"\bf+u*c+k+(?:er|ing|ed|s|off)?\b",
    r"\bf+c+k+\b",
    r"\bmother\s*f+u*c+k+(?:er)?\b",
    r"\bb+i+t+c+h+(?:es|ing)?\b",
    r"\bs+h+i+t+(?:ty|s|ting)?\b",
    r"\bbullshit\b",
    r"\ba+s+s+h+o+l+e+(?:s)?\b",
    r"\ba+r+s+e+h+o+l+e+(?:s)?\b",
    r"\b(?:dumb|jack)ass\b",
    r"\bb+a+s+t+a+r+d+(?:s)?\b",
    r"\bd+i+c+k+(?:head|s)?\b",
    r"\bp+u+s+s+y+(?:es)?\b",
    r"\bc+u+n+t+(?:s)?\b",
    r"\bs+l+u+t+(?:s)?\b",
    r"\bw+h+o+r+e+(?:s)?\b",
    r"\bn+i+g+g+e+r+(?:s)?\b",
    r"\bf+a+g+g+o+t+(?:s)?\b",
    r"\br+e+t+a+r+d+(?:s|ed)?\b",
    r"\bp+o+r+n+(?:o|ography)?\b",
    r"\bblowjob\b",
    r"\bhandjob\b",
    r"\bmasturbat\w*\b",
    r"\b(?:sex|sexy|nude|nudes|boobs|tits|penis|vagina|incest|xxx|xvideos|porno)\b",

    # --- Hindi / Hinglish Profanity (Latin Script) ---
    r"\b(?:behen|bhen|bhn)\s*c+h+o+d+\b",
    r"\bb+c+\b",
    r"\bb+s+d+k+\b",
    r"\bb+h+o+s+[a-z]*d+i+k+e+\b",
    r"\bb+h+o+s+[a-z]*d+i+(?:wala|wali|wale)?\b",
    r"\bb+h+o+s+d+a+\b",
    r"\b(?:madar|maadar|mader)\s*c+h+o+d+\b",
    r"\bm+c+\b",
    r"\bmadarja+t+\b",
    r"\bc+h+u+t+i+y+a+(?:s|n|pan)?\b",
    r"\bc+h+u+t+i+y+e+\b",
    r"\bc+h+u+t+i+y+a+p+a*\b",
    r"\bc+h+o+o+t+i+y+a+\b",
    r"\bg+a+n+d+u+(?:s)?\b",
    r"\bg+a+a+n+d+\s*(?:marna|mara|mari|faad|fat|masti|chhed)?\b",
    r"\bg+a+n+d+\s*(?:marna|mara|mari|faad|fat)\b",
    r"\bh+a+r+a+m+i+(?:yo|pan)?\b",
    r"\bh+a+r+a+m+k+h+o+r+\b",
    r"\bk+a+m+i+n+e+\b",
    r"\bk+a+m+i+n+a+\b",
    r"\bl+a+u+d+[aeiu]+\b",
    r"\bl+o+d+[aeiu]+\b",
    r"\bl+a+v+d+[aeiu]+\b",
    r"\bl+u+n+d+(?:s)?\b",
    r"\br+a+n+d+i+(?:ya|baaz)?\b",
    r"\br+a+n+d+w+a+\b",
    r"\bc+h+i+n+a+l+\b",
    r"\bt+a+t+t+e+\b",
    r"\bj+h+a+a*t+\b",
    r"\bteri\s+(?:maa|ma|behen|bhen)\s+ki\b",
    r"\bmaa\s+ki\s+choot\b",
    r"\bmaa\s+ka\s+bhosda\b",
    r"\bsaale\s+kutte\b",
    r"\b(?:chudai|chodna|chodo|chode|muth|muthal)\b",

    # --- Hindi Devanagari Script ---
    r"मादरचोद",
    r"बहनचोद",
    r"भोसड़ीके",
    r"भोसड़ी",
    r"चूतिया",
    r"चूतिये",
    r"गांडू",
    r"हरामी",
    r"कमीने",
    r"लौड़े",
    r"लौड़ा",
    r"लंड",
    r"रंडी",
    r"साले\s*कुत्ते",
    r"चोदना",
    r"चुदाई",
    r"मुठ",
    r"सेक्स",
]

COMPILED_PATTERNS = [re.compile(pat, re.IGNORECASE) for pat in ABUSE_REGEX_PATTERNS]


def normalize_text_for_detection(text: str) -> str:
    """Normalize leetspeak and letter elongations while preserving word structure."""
    if not text:
        return ""
    t = text.lower()
    for symbol, rep in LEET_SUBS.items():
        t = t.replace(symbol, rep)
    # Replace wildcard '*' with 'u' if inside words (e.g. f*ck -> fuck)
    t = re.sub(r'(?<=\w)\*(?=\w)', 'u', t)
    # Collapse 3+ repeating characters (e.g., fuuuuck -> fuuck, bheeencoood -> bhenchod)
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    return t


def is_abusive_language(text: str) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether input text contains offensive, vulgar, sexual, or abusive language.
    Returns (is_abusive: bool, matched_reason: Optional[str]).
    """
    if not text or not text.strip():
        return False, None

    normalized = normalize_text_for_detection(text)

    for pattern in COMPILED_PATTERNS:
        match = pattern.search(normalized) or pattern.search(text)
        if match:
            reason = match.group(0)
            logger.warning("Abuse guardrail flagged term: '%s' in query", reason)
            return True, reason

    return False, None


# ---------------------------------------------------------------------------
# Session Tracking Store (In-Memory with Optional Redis)
# ---------------------------------------------------------------------------
class AbuseSessionTracker:
    """
    Manages session-level abuse strike tracking with thread safety.
    Maintains an in-memory `abuse_strikes` counter per session ID.
    Enforces the 3-Strike policy:
      - Strikes 1 & 2: Warning
      - Strike 3: Permanent session termination and lockout of all further inputs.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self._lock = threading.RLock()
        self._memory_sessions: Dict[str, Dict[str, Any]] = {}
        self.redis_client = None

        url = redis_url or os.getenv("REDIS_URL")
        if url:
            try:
                import redis
                client = redis.from_url(url, decode_responses=True, socket_connect_timeout=1.0)
                client.ping()
                self.redis_client = client
                logger.info("AbuseSessionTracker connected to Redis at %s", url)
            except Exception as e:
                logger.info("AbuseSessionTracker Redis unavailable (%s); using thread-safe in-memory store", e)
                self.redis_client = None

    def _get_raw_session(self, session_id: str) -> Dict[str, Any]:
        """Fetch raw session record from Redis or memory."""
        if self.redis_client:
            try:
                raw = self.redis_client.get(f"abuse_guardrail:{session_id}")
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.warning("Error reading session %s from Redis: %s", session_id, e)

        with self._lock:
            if session_id not in self._memory_sessions:
                self._memory_sessions[session_id] = {
                    "abuse_strikes": 0,
                    "is_terminated": False,
                    "last_updated": time.time(),
                }
            return dict(self._memory_sessions[session_id])

    def _save_raw_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Persist session record to memory and Redis with 24h TTL."""
        data["last_updated"] = time.time()
        with self._lock:
            self._memory_sessions[session_id] = dict(data)

        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"abuse_guardrail:{session_id}",
                    86400,  # 24 hours TTL
                    json.dumps(data),
                )
            except Exception as e:
                logger.warning("Error saving session %s to Redis: %s", session_id, e)

    def check_message(self, session_id: str, message: str) -> Optional[Dict[str, Any]]:
        """
        Process a message through the 3-Strike guardrail.
        Returns:
          - None: Allowed to proceed.
          - Dict: Intercepted payload (Warning or Terminated).
        """
        session = self._get_raw_session(session_id)

        # 1. If session is already terminated, lock further inputs immediately
        if session.get("is_terminated"):
            return {
                "blocked": True,
                "terminated": True,
                "strikes": session.get("abuse_strikes", 3),
                "message": TERMINATED_MESSAGE,
            }

        # 2. Check for abusive language
        is_abusive, _ = is_abusive_language(message)

        if is_abusive:
            session["abuse_strikes"] = session.get("abuse_strikes", 0) + 1
            strikes = session["abuse_strikes"]

            if strikes >= 3:
                session["is_terminated"] = True
                self._save_raw_session(session_id, session)
                logger.warning("Session %s reached Strike 3 and is TERMINATED.", session_id)
                return {
                    "blocked": True,
                    "terminated": True,
                    "strikes": strikes,
                    "message": TERMINATED_MESSAGE,
                }
            else:
                self._save_raw_session(session_id, session)
                warning_msg = WARNING_TEMPLATE.format(count=strikes)
                logger.warning("Session %s received Strike %d/3 warning.", session_id, strikes)
                return {
                    "blocked": True,
                    "terminated": False,
                    "strikes": strikes,
                    "message": warning_msg,
                }

        # 3. Message is clean; persist and allow downstream processing
        self._save_raw_session(session_id, session)
        return None

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        """Admin reset of a session's strikes and termination status."""
        session = {
            "abuse_strikes": 0,
            "is_terminated": False,
            "last_updated": time.time(),
        }
        self._save_raw_session(session_id, session)
        return session

    def get_status(self, session_id: str) -> Dict[str, Any]:
        """Inspect current state of a session."""
        return self._get_raw_session(session_id)


# Global Singleton Guardrail Tracker
_GUARDRAIL_TRACKER = AbuseSessionTracker()


def check_abuse_guardrail(session_id: str, query: str) -> Optional[Dict[str, Any]]:
    """
    Public safety guardrail function.
    Returns:
      - None: Query is clean and allowed to proceed.
      - Dict with {blocked: True, terminated: bool, strikes: int, message: str}: Intercepted.
    """
    if not session_id:
        session_id = "default_user"
    return _GUARDRAIL_TRACKER.check_message(session_id, query)


def reset_abuse_session(session_id: str) -> Dict[str, Any]:
    """Admin function to reset strikes and unblock a session."""
    return _GUARDRAIL_TRACKER.reset_session(session_id)


def get_abuse_session_status(session_id: str) -> Dict[str, Any]:
    """Admin function to inspect session guardrail stats."""
    return _GUARDRAIL_TRACKER.get_status(session_id)


# ---------------------------------------------------------------------------
# FastAPI Middleware: Enforces Rule 1 (Domain) & Rule 2 (Safety/3-Strike)
# ---------------------------------------------------------------------------
class AbuseGuardrailMiddleware(BaseHTTPMiddleware):
    """
    FastAPI Middleware inspecting chat requests at the HTTP layer.
    Enforces:
      1. Safety & Profanity Guardrail (3-strike vulgarity rule & lockout)
      2. Domain Restriction Guardrail (Vedic astrology & life-path only)
    """

    def __init__(self, app, protected_paths: Optional[List[str]] = None):
        super().__init__(app)
        self.protected_paths = protected_paths or ["/chat", "/api/chat"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.method == "POST" and any(request.url.path.startswith(p) for p in self.protected_paths):
            try:
                body_bytes = await request.body()

                # Re-assign receive so downstream route handlers can read the body again
                async def receive():
                    return {"type": "http.request", "body": body_bytes}
                request._receive = receive

                if body_bytes:
                    data = json.loads(body_bytes.decode("utf-8"))
                    query = (
                        data.get("query")
                        or data.get("prompt")
                        or data.get("user_query")
                        or ""
                    )
                    user_id = (
                        data.get("user_id")
                        or (data.get("user_profile") or {}).get("user_id")
                        or (data.get("user_profile") or {}).get("id")
                        or "default_user"
                    )

                    # Step 1: Safety & Profanity Guardrail check (3-strike rule)
                    abuse_result = check_abuse_guardrail(str(user_id), str(query))
                    if abuse_result:
                        status_str = "terminated" if abuse_result.get("terminated") else "warning"
                        return JSONResponse(
                            status_code=200,
                            content={
                                "status": status_str,
                                "answer": abuse_result["message"],
                                "response": abuse_result["message"],
                                "abuse_strikes": abuse_result["strikes"],
                                "terminated": abuse_result["terminated"],
                                "related_questions": [],
                            },
                        )

                    # Step 2: Domain Restriction Guardrail check (Astrology Only)
                    is_valid, out_of_scope_msg = validate_astrology_domain(str(query))
                    if not is_valid:
                        return JSONResponse(
                            status_code=200,
                            content={
                                "status": "out_of_scope",
                                "answer": out_of_scope_msg,
                                "response": out_of_scope_msg,
                                "abuse_strikes": 0,
                                "terminated": False,
                                "related_questions": [
                                    "How does my birth chart influence my career?",
                                    "When is a favorable period for marriage in my chart?",
                                    "What do my planetary transits say about my health?",
                                ],
                            },
                        )
            except Exception as exc:
                logger.warning("AbuseGuardrailMiddleware error: %s", exc)

        return await call_next(request)


# Backwards compatible alias
AstroGuardrailMiddleware = AbuseGuardrailMiddleware


# ---------------------------------------------------------------------------
# Administrative APIRouter for Guardrail Status & Testing
# ---------------------------------------------------------------------------
guardrail_router = APIRouter(prefix="/api/guardrail", tags=["Abuse Guardrail"])


@guardrail_router.get("/status/{session_id}")
def guardrail_status(session_id: str):
    """Fetch active strike count and status for a session."""
    return {"session_id": session_id, "guardrail": get_abuse_session_status(session_id)}


@guardrail_router.post("/reset/{session_id}")
def guardrail_reset(session_id: str):
    """Manually reset strikes and unblock a session."""
    data = reset_abuse_session(session_id)
    return {"session_id": session_id, "status": "reset_successful", "guardrail": data}
