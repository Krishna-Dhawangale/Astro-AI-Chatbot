"""
backend/abuse_guardrail.py
==========================
Calibrated Multilingual Abuse Guardrail & Isolated Session Tracking
(Proposal 1 Integration)

Features & Design Invariants:
1. Multilingual profanity detection across English, Hinglish, leetspeak, and Devanagari.
2. Contextual disambiguation for historical 'BC', astrological 'MC', biological 'sex' (progeny),
   and proper names (e.g. 'Randi Zuckerberg').
3. Cryptographically signed, server-controlled session tokens (HMAC-SHA256).
   - Secret loaded from SESSION_SECRET_KEY environment variable.
   - If unset (development mode), an ephemeral 32-byte secret is generated dynamically.
     NOTE: In development, restarting the server invalidates existing tokens.
     In production, configure a stable SESSION_SECRET_KEY to preserve sessions across restarts.
   - Automatic token rotation after 12 hours (TOKEN_ROTATION_INTERVAL).
   - Hard cryptographic token expiration after 24 hours (TOKEN_MAX_AGE).
4. Thread-safe, bounded session strike tracker (CalibratedAbuseSessionTracker).
   - 2-hour inactivity strike decay.
   - 24-hour lockout expiration.
   - Bounded cache (10,000 sessions) with lazy pruning to prevent memory exhaustion.
5. Strict safety precedence support (record_safety_query tracks strikes without suppressing
   unsupported clinical medical/death limitation responses).
6. Response cookie management (attach_session_cookie) with HttpOnly, SameSite=Lax, and Secure flags.
"""

import hashlib
import hmac
import os
import re
import secrets
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from starlette.responses import Response

# ---------------------------------------------------------------------------
# Standard Guardrail Templates
# ---------------------------------------------------------------------------
WARNING_TEMPLATE = "⚠️ Warning ({count}/3): Inappropriate or vulgar language is not permitted."
TERMINATED_MESSAGE = "🚫 Session Terminated: You have exceeded the limit for inappropriate language."
PROFANITY_REFUSAL_MESSAGE = "⚠️ Warning (1/3): Inappropriate or vulgar language is not permitted."

SESSION_COOKIE_NAME = "astro_session_id"
TOKEN_MAX_AGE = 86400  # 24 hours in seconds
TOKEN_ROTATION_INTERVAL = 43200  # 12 hours in seconds

# ---------------------------------------------------------------------------
# Leetspeak & Normalization Substitutions
# ---------------------------------------------------------------------------
LEET_SUBS = {
    "@": "a", "4": "a", "$": "s", "5": "s", "1": "i", "!": "i", "0": "o", "3": "e"
}

# ---------------------------------------------------------------------------
# Contextual Exemption Dictionaries
# ---------------------------------------------------------------------------
ASTROLOGY_CONTEXT_TOKENS = {
    "midheaven", "natal", "chart", "sign", "degree", "house", "conjunct", "trine",
    "square", "aspect", "ic", "ascendant", "zenith", "astrology", "vedic", "horoscope",
    "lagna", "rashi", "planet", "graha", "transit"
}

PROGENY_DEMOGRAPHIC_TOKENS = {
    "child", "children", "baby", "progeny", "birth", "gender", "male", "female",
    "biological", "son", "daughter", "pregnancy", "demographic", "foetus", "fetus",
    "santan", "santaan", "family", "born", "unborn"
}

HISTORICAL_DATE_TOKENS = {
    "era", "century", "ad", "bce", "ce", "dating", "dates", "date", "calendar",
    "ancient", "history", "established", "founded", "period", "written", "compiled"
}

# ---------------------------------------------------------------------------
# Calibrated Regular Expressions
# ---------------------------------------------------------------------------
RAW_ABUSE_PATTERNS = [
    # English Profanity
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
    r"\b(?:sexy|nude|nudes|boobs|tits|penis|vagina|incest|xxx|xvideos|porno)\b",

    # Hindi / Hinglish Profanity
    r"\b(?:behen|bhen|bhn)\s*c+h+o+d+\b",
    r"\bb+s+d+k+\b",
    r"\bb+k+l+\b",
    r"\bb+h+o+s+[a-z]*d+i+k+e+\b",
    r"\bb+h+o+s+[a-z]*d+i+(?:wala|wali|wale)?\b",
    r"\bb+h+o+s+d+a+\b",
    r"\b(?:madar|maadar|mader)\s*c+h+o+d+\b",
    r"\bmadarja+t+\b",
    r"\bc+h+u+t+i+y+a+(?:s|n|pan)?\b",
    r"\bc+h+u+t+i+y+e+\b",
    r"\bc+h+u+t+y+a+\b",
    r"\bc+h+u+t+i+y+a+p+a*\b",
    r"\bc+h+o+o+t+i+y+a+\b",
    r"\bc+h+o+o+t+\b",
    r"\bc+h+u+t+\b",
    r"\bg+a+n+d+u+(?:s)?\b",
    r"\bg+a+a+n+d+u+(?:s)?\b",
    r"\bg+a+n+d+\b",
    r"\bg+a+a+n+d+\b",
    r"\bg+a+a+n+d+\s*(?:marna|mara|mari|faad|fat|masti|chhed)?\b",
    r"\bh+a+r+a+m+i+(?:yo|pan)?\b",
    r"\bh+a+r+a+m+k+h+o+r+\b",
    r"\bh+a+r+a+m+z+a+d+[ae]\b",
    r"\bk+a+m+i+n+e+\b",
    r"\bk+a+m+i+n+a+\b",
    r"\bk+a+m+i+n+i+\b",
    r"\bl+a+u+d+[aeiu]+\b",
    r"\bl+o+d+[aeiu]+\b",
    r"\bl+a+v+d+[aeiu]+\b",
    r"\bl+u+n+d+(?:s)?\b",
    r"\bc+h+i+n+a+l+\b",
    r"\bc+h+i+n+a+r+\b",
    r"\bt+a+t+t+e+\b",
    r"\bt+a+t+t+a+\b",
    r"\bj+h+a+a*t+\b",
    r"\bj+h+a+a*t+u+\b",
    r"\br+a+n+d+w+a+\b",
    r"\br+a+a+n+d+\b",
    r"\bteri\s+(?:maa|ma|behen|bhen)\s+ki\b",
    r"\bmaa\s+ki\s+choot\b",
    r"\bmaa\s+ka\s+bhosda\b",
    r"\bteri\s+maa\s+ka\b",
    r"\bsaale\s+kutte\b",
    r"\b(?:chudai|chodna|chodo|chode|chuda|chudwa|muthal|mutthal)\b",

    # Devanagari Hindi
    r"मादरचोद", r"बहनचोद", r"भोसड़ीके", r"भोसड़ी", r"भोसड़ा", r"चूतिया", r"चूतिये",
    r"गांडू", r"गांड", r"हरामी", r"कमीने", r"लौड़े", r"लौड़ा", r"लंड", r"रंडी",
    r"साले\s*कुत्ते", r"चोदना", r"चुदाई", r"मुठ", r"सेक्स"
]

COMPILED_PATTERNS = [re.compile(pat, re.IGNORECASE) for pat in RAW_ABUSE_PATTERNS]

# Context-Sensitive Patterns that require negative disambiguation checks
PATTERN_BC = re.compile(r"\bb+c+\b", re.IGNORECASE)
PATTERN_MC = re.compile(r"\bm+c+\b", re.IGNORECASE)
PATTERN_SEX = re.compile(r"\bsex\b", re.IGNORECASE)
PATTERN_RANDI = re.compile(r"\br+a+n+d+i+(?:ya|baaz)?\b", re.IGNORECASE)


def normalize_text_for_detection(text: str) -> str:
    """Normalize leetspeak, wildcards, and elongated characters for detection."""
    if not text:
        return ""
    t = text.lower()
    for symbol, rep in LEET_SUBS.items():
        t = t.replace(symbol, rep)
    t = re.sub(r'(?<=\w)\*(?=\w)', 'u', t)
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    return t


def is_contextual_abusive_language(query: str) -> Tuple[bool, Optional[str]]:
    """
    Context-aware abuse evaluation.
    Enforces genuine profanity detection while exempting:
    1. Historical 'BC' (e.g. 1500 BC, BC era).
    2. Astrological 'MC' (Midheaven / Medium Coeli).
    3. Biological 'sex' in child/gender/progeny context.
    4. Capitalized proper nouns containing 'Randi' (e.g. Randi Zuckerberg).
    """
    if not query or not query.strip():
        return False, None

    normalized = normalize_text_for_detection(query)
    q_low = query.lower()
    words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", q_low))

    # 1. Check Standard Unambiguous Abuse Patterns
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(normalized) or pattern.search(query)
        if match:
            return True, match.group(0)

    # 2. Contextual Check: 'BC' (Behenchod vs Before Christ)
    match_bc = PATTERN_BC.search(query) or PATTERN_BC.search(normalized)
    if match_bc:
        is_date_bc = bool(re.search(r"\b\d+\s*bc\b", q_low))
        has_historical_context = any(t in words for t in HISTORICAL_DATE_TOKENS)
        if not (is_date_bc or has_historical_context):
            return True, match_bc.group(0)

    # 3. Contextual Check: 'MC' (Madarchod vs Midheaven)
    match_mc = PATTERN_MC.search(query) or PATTERN_MC.search(normalized)
    if match_mc:
        has_astro_context = any(t in words for t in ASTROLOGY_CONTEXT_TOKENS)
        is_mc_axis = bool(re.search(r"\b(?:my|natal|the|sign\s+of)\s+mc\b", q_low)) or "midheaven" in q_low
        if not (has_astro_context or is_mc_axis):
            return True, match_mc.group(0)

    # 4. Contextual Check: 'sex' (Vulgarity vs Child's biological gender/sex)
    match_sex = PATTERN_SEX.search(query)
    if match_sex:
        has_progeny_context = any(t in words for t in PROGENY_DEMOGRAPHIC_TOKENS)
        if not has_progeny_context:
            return True, match_sex.group(0)

    # 5. Contextual Check: 'Randi' (Slur vs Proper Name)
    match_randi = PATTERN_RANDI.search(query) or PATTERN_RANDI.search(normalized)
    if match_randi:
        is_proper_name = bool(re.search(r"\bRandi\s+[A-Z][a-z]+", query))
        has_author_title = any(t in words for t in {"author", "book", "writer", "person", "mr", "ms", "dr"})
        if not (is_proper_name and has_author_title):
            return True, match_randi.group(0)

    # 6. Dot-separated swear acronyms (b.s.d.k, b.k.l)
    dot_clean = re.sub(r"\.", "", q_low)
    for bad_acronym in ["bsdk", "bkl"]:
        if re.search(r"\b" + bad_acronym + r"\b", dot_clean):
            return True, bad_acronym

    return False, None


# ---------------------------------------------------------------------------
# Cryptographic Session Token Management
# ---------------------------------------------------------------------------
_MODULE_EPHEMERAL_SECRET: Optional[bytes] = None


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def is_production_mode() -> bool:
    """Robustly detect production mode across standard environment variables."""
    for key in ("ENVIRONMENT", "ENV", "APP_ENV"):
        val = (os.getenv(key) or "").strip().lower()
        if val in ("production", "prod"):
            return True
    return False


def get_session_secret() -> bytes:
    """
    Retrieve configured SESSION_SECRET_KEY or generate an ephemeral 32-byte secret.
    In development, an ephemeral secret suffices, but server restarts invalidate tokens.
    In production, define SESSION_SECRET_KEY in the environment for stable multi-process operation.
    """
    global _MODULE_EPHEMERAL_SECRET
    env_secret = (os.getenv("SESSION_SECRET_KEY") or "").strip()
    if env_secret:
        return env_secret.encode("utf-8")

    # Fail fast in production if secret is missing or empty
    if is_production_mode():
        raise RuntimeError(
            "CRITICAL SECURITY CONFIGURATION ERROR: SESSION_SECRET_KEY environment variable is missing or empty in production mode. "
            "A stable, cryptographically secure key must be configured in production."
        )

    if _MODULE_EPHEMERAL_SECRET is None:
        _MODULE_EPHEMERAL_SECRET = secrets.token_bytes(32)
    return _MODULE_EPHEMERAL_SECRET


def issue_secure_session_token(session_uuid: Optional[str] = None, current_time: Optional[float] = None) -> str:
    """
    Generate an HMAC-signed session token containing session_uuid and issued_timestamp:
    token = f"{session_uuid}.{ts}.{sig}"
    """
    now_ts = int(current_time or time.time())
    sid = session_uuid or str(uuid.uuid4())
    payload = f"{sid}.{now_ts}"
    secret = get_session_secret()
    sig = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
    return f"{payload}.{sig}"


def validate_secure_session_token(
    token: Optional[str],
    current_time: Optional[float] = None
) -> Tuple[bool, Optional[str], bool]:
    """
    Validate server-issued session token signature and age.

    Returns:
        (is_valid: bool, session_uuid: Optional[str], needs_rotation: bool)
    """
    if not token or not isinstance(token, str):
        return False, None, False

    parts = token.split(".")
    if len(parts) != 3:
        return False, None, False

    sid, ts_str, sig = parts
    try:
        ts = int(ts_str)
    except ValueError:
        return False, None, False

    now = int(current_time or time.time())

    # Verify signature
    payload = f"{sid}.{ts_str}"
    secret = get_session_secret()
    expected_sig = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
    if not hmac.compare_digest(sig, expected_sig):
        return False, None, False

    # Check expiration (cannot be older than TOKEN_MAX_AGE or more than 60s in the future)
    token_age = now - ts
    if token_age > TOKEN_MAX_AGE or token_age < -60:
        return False, None, False

    # Check soft rotation interval
    needs_rotation = (token_age >= TOKEN_ROTATION_INTERVAL)
    return True, sid, needs_rotation


# ---------------------------------------------------------------------------
# Thread-Safe Bounded Session Strike Tracker
# ---------------------------------------------------------------------------
class CalibratedAbuseSessionTracker:
    """
    Thread-safe, bounded session tracker with:
    - Inactivity strike decay (default 2 hours / 7200s).
    - Lockout expiration (default 24 hours / 86400s).
    - Lazy pruning of expired sessions to cap memory usage.
    """

    def __init__(
        self,
        decay_seconds: float = 7200.0,
        lockout_seconds: float = 86400.0,
        max_sessions: int = 10000
    ):
        self._lock = threading.RLock()
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self.decay_seconds = decay_seconds
        self.lockout_seconds = lockout_seconds
        self.max_sessions = max_sessions

    def _prune_expired_sessions_locked(self, current_time: float) -> None:
        """Internal helper to remove expired sessions and prevent memory growth."""
        to_delete = []
        for sid, s in self._sessions.items():
            if s.get("is_terminated"):
                if current_time - s.get("locked_at", 0) >= self.lockout_seconds:
                    to_delete.append(sid)
            else:
                if current_time - s.get("last_activity", 0) >= self.decay_seconds:
                    to_delete.append(sid)
        for sid in to_delete:
            del self._sessions[sid]

    def get_session_state(self, session_id: str, current_time: Optional[float] = None) -> Dict[str, Any]:
        """Fetch and evaluate session status applying decay and lockout expiration."""
        now = current_time or time.time()
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return {"abuse_strikes": 0, "is_terminated": False, "last_activity": now}

            # Check Lockout Expiration
            if session.get("is_terminated"):
                locked_at = session.get("locked_at", 0)
                if now - locked_at >= self.lockout_seconds:
                    session["abuse_strikes"] = 0
                    session["is_terminated"] = False
                    session["locked_at"] = None

            # Check Strike Inactivity Decay (if not currently locked)
            elif session.get("abuse_strikes", 0) > 0:
                last_act = session.get("last_activity", 0)
                if now - last_act >= self.decay_seconds:
                    session["abuse_strikes"] = 0

            return dict(session)

    def record_safety_query(self, session_id: str, query: str, current_time: Optional[float] = None) -> None:
        """
        Record and audit queries that trigger clinical medical or death boundaries.
        Tracks strikes internally if vulgar, but NEVER returns a blocking response,
        preserving unbreakable safety boundary response delivery.
        """
        now = current_time or time.time()
        with self._lock:
            state = self.get_session_state(session_id, current_time=now)
            is_abusive, _ = is_contextual_abusive_language(query)
            if is_abusive and not state.get("is_terminated"):
                state["abuse_strikes"] = min(3, state.get("abuse_strikes", 0) + 1)
                state["last_activity"] = now
                if state["abuse_strikes"] >= 3:
                    state["is_terminated"] = True
                    state["locked_at"] = now
                self._sessions[session_id] = state

    def check_message(
        self,
        session_id: str,
        query: str,
        current_time: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process non-safety queries through the 3-strike policy.

        Returns:
            None: Message is clean, proceed downstream.
            Dict: Intercepted response (Warning or Terminated).
        """
        now = current_time or time.time()

        with self._lock:
            # Enforce max sessions bound via lazy pruning
            if len(self._sessions) > self.max_sessions:
                self._prune_expired_sessions_locked(now)

            state = self.get_session_state(session_id, current_time=now)
            is_terminated = state.get("is_terminated", False)

            # Locked session enforcement for all non-safety queries
            if is_terminated:
                return {
                    "blocked": True,
                    "terminated": True,
                    "strikes": 3,
                    "message": TERMINATED_MESSAGE
                }

            # Evaluate contextual abuse
            is_abusive, matched_term = is_contextual_abusive_language(query)

            if is_abusive:
                strikes = state.get("abuse_strikes", 0) + 1
                state["abuse_strikes"] = strikes
                state["last_activity"] = now

                if strikes >= 3:
                    state["is_terminated"] = True
                    state["locked_at"] = now
                    self._sessions[session_id] = state
                    return {
                        "blocked": True,
                        "terminated": True,
                        "strikes": strikes,
                        "message": TERMINATED_MESSAGE,
                        "flagged_term": matched_term
                    }
                else:
                    self._sessions[session_id] = state
                    warning_msg = WARNING_TEMPLATE.format(count=strikes)
                    return {
                        "blocked": True,
                        "terminated": False,
                        "strikes": strikes,
                        "message": warning_msg,
                        "flagged_term": matched_term
                    }

            # Clean message: update activity
            state["last_activity"] = now
            self._sessions[session_id] = state
            return None

    def reset_session(self, session_id: str) -> None:
        """Admin or test reset of session strikes."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]


# Global tracker instance
abuse_tracker = CalibratedAbuseSessionTracker()


# ---------------------------------------------------------------------------
# Response Cookie Utility
# ---------------------------------------------------------------------------
def attach_session_cookie(
    response: Response,
    token: str,
    is_secure: Optional[bool] = None
) -> Response:
    """
    Attach the signed session token as an HttpOnly, SameSite=Lax cookie on the response.
    Applies to all Starlette/FastAPI responses (JSONResponse, StreamingResponse, etc.).
    """
    if is_secure is None:
        is_secure = os.getenv("COOKIE_SECURE", "false").lower() == "true"

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=TOKEN_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=is_secure,
        path="/"
    )
    return response
