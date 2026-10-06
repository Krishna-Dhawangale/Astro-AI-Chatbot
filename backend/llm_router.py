"""
Token-Optimized Cascading LLM Router:
High-concurrency, performance-engineered AI streaming router with:
1. Complete Single-Paragraph Constraints:
    - SYSTEM_PROMPT: concise Vedic astrology guidance for a single 40-50-word paragraph without formatting.
   - Cap: max_tokens=160 across all completion payloads to guarantee complete sentences.
2. Unambiguous User Profile Format (~18 Tokens):
   - compress_user_profile(): "Ascendant: Cancer | Moon: Leo | Dasha: Rahu"
   - Eliminates model confusion and prevents wasted completion tokens.
3. Smart Sentence End Truncation (trim_to_last_sentence) & Output Sanitization:
   - Strips <think>...</think> and <thought>...</thought> blocks via Regex.
   - Strips leaked headers ('Career Outlook:', 'Remedy:'), bold tags, and bullet points.
   - Smartly trims back to the last complete sentence punctuation mark ('.', '!', '?') if cut off.
   - Appends UI suffix '(LLM: model_name)' strictly after sanitization.
4. Single-Model Execution Guarantee:
   - Explicit return immediately after the active model streams content.
   - Cascades through verified active Groq models, OpenRouter tier, and Gemini safety net.
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import groq
from google import genai
from google.genai import types
import httpx
import openai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
load_dotenv(override=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multi-Account Groq Configuration
# ---------------------------------------------------------------------------
GROQ_ACCOUNTS: List[Dict[str, Optional[str]]] = [
    {
        "account_id": "Account 1",
        "api_key": (
            os.getenv("GROQ_API_KEY_1")
            or os.getenv("GROQ_API_KEY_ACCOUNT1")
            or os.getenv("GROQ_API_KEY_ACC1")
            or os.getenv("GROQ_API_KEY")
        ),
    },
    {
        "account_id": "Account 2",
        "api_key": (
            os.getenv("GROQ_API_KEY_2")
            or os.getenv("GROQ_API_KEY_ACCOUNT2")
            or os.getenv("GROQ_API_KEY_ACC2")
        ),
    },
]

# ---------------------------------------------------------------------------
# Tier 1: Active Production Groq Models
# Cascade: llama-3.3-70b-versatile -> llama-3.1-8b-instant -> gemma2-9b-it
# ---------------------------------------------------------------------------
MODEL_CASCADE: List[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]
GROQ_MODELS: List[str] = MODEL_CASCADE


# ---------------------------------------------------------------------------
# Tier 2: OpenRouter Fallback Tier (Dedicated base URL & client instance)
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODELS: List[str] = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
    "qwen/qwen3.8-27b",
]

# ---------------------------------------------------------------------------
# Tier 3: Secondary Provider Safety Net (Google Gemini Flash)
# ---------------------------------------------------------------------------
GEMINI_SAFETY_MODELS: List[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]

# Output Token Cap: max_tokens=130 ensures the model has enough output budget to finish the 6th line completely
MAX_TOKENS: int = 130
DEFAULT_TEMPERATURE: float = 0.4
STRICT_HISTORY_TURNS: int = 2  # 1 previous user turn + 1 previous assistant response

# ---------------------------------------------------------------------------
# Step 1: Ground-Truth Natal Context & House Rulership Engine
# ---------------------------------------------------------------------------
ZODIAC_SIGNS: List[str] = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

SIGN_LORDS: Dict[str, str] = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": "Mars",
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}

HOUSE_NAMES: List[str] = [
    "1st_house", "2nd_house", "3rd_house", "4th_house",
    "5th_house", "6th_house", "7th_house", "8th_house",
    "9th_house", "10th_house", "11th_house", "12th_house"
]

RASHI_ALIASES: Dict[str, str] = {
    "mesha": "Aries", "mesh": "Aries",
    "vrishabha": "Taurus", "vrishabh": "Taurus", "vrisabh": "Taurus",
    "mithuna": "Gemini", "mithun": "Gemini",
    "karka": "Cancer", "kark": "Cancer", "karkat": "Cancer",
    "simha": "Leo", "singh": "Leo", "sinh": "Leo",
    "kanya": "Virgo",
    "tula": "Libra",
    "vrishchika": "Scorpio", "vrischika": "Scorpio", "vrischik": "Scorpio",
    "dhanu": "Sagittarius", "dhanus": "Sagittarius",
    "makara": "Capricorn", "makar": "Capricorn",
    "kumbha": "Aquarius", "kumbh": "Aquarius",
    "meena": "Pisces", "meen": "Pisces"
}


def calculate_house_rulerships(ascendant: str) -> Dict[str, str]:
    """Calculate deterministic house rulerships (1st to 12th house) for any given Ascendant."""
    raw = (ascendant or "Libra").strip().lower()
    clean_asc = RASHI_ALIASES.get(raw, ascendant.strip().capitalize() if ascendant else "Libra")

    match_idx = 6  # default Libra
    for idx, sign in enumerate(ZODIAC_SIGNS):
        if sign.lower() == clean_asc.lower():
            match_idx = idx
            break

    rulerships: Dict[str, str] = {}
    for i, house_name in enumerate(HOUSE_NAMES):
        sign = ZODIAC_SIGNS[(match_idx + i) % 12]
        lord = SIGN_LORDS[sign]
        rulerships[house_name] = f"{lord} ({sign})"
    return rulerships


def get_user_natal_context(
    user_profile: Optional[Any] = None,
    ascendant: Optional[str] = None,
    moon_sign: Optional[str] = None,
    active_dasha: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Step 1: Ground-Truth Injection
    Pre-calculates explicit birth facts into an immutable user_natal_context dictionary.
    """
    def _extract_val(candidates: List[str], fallback: str) -> str:
        if not user_profile:
            return fallback
        if isinstance(user_profile, dict):
            for k in candidates:
                v = user_profile.get(k)
                if v:
                    return str(v)
            return fallback
        for k in candidates:
            v = getattr(user_profile, k, None)
            if v:
                return str(v)
        return fallback

    final_asc = ascendant or _extract_val(["ascendant", "lagna", "asc_sign"], "Libra")
    final_moon = moon_sign or _extract_val(["rashi", "moon_sign", "moon"], "Gemini")
    raw_dasha = active_dasha or _extract_val(["dasha", "active_dasha", "current_dasha"], "Rahu-Mars")
    if isinstance(raw_dasha, dict):
        raw_dasha = raw_dasha.get("current_dasha") or raw_dasha.get("major") or "Rahu-Mars"
    final_dasha = str(raw_dasha).replace("/", "-").strip()

    rulerships = calculate_house_rulerships(final_asc)
    return {
        "ascendant": final_asc,
        "moon_sign": final_moon,
        "active_dasha": final_dasha,
        "house_rulerships": rulerships,
    }


# ---------------------------------------------------------------------------
# Step 2: Strict System Prompt Guardrails & Length Constraints
# ---------------------------------------------------------------------------
DEFAULT_NATAL_CONTEXT: Dict[str, Any] = {
    "ascendant": "Libra",
    "moon_sign": "Gemini",
    "active_dasha": "Rahu-Mars",
    "house_rulerships": calculate_house_rulerships("Libra"),
}

SYSTEM_PROMPT = """You are a direct, empathetic, and accurate Vedic Astrology assistant.

STRICT OUTPUT RULES:
1. NO CHEESY ANALOGIES: Avoid fluffy metaphors like "love is like a fruit tree/garden/ocean". Keep the tone natural, grounded, and practical.
2. LENGTH & COMPLETENESS: Respond in strictly 4 to 6 lines total. Complete every sentence properly—NEVER stop mid-sentence or cut off mid-thought.
3. LANGUAGE REGISTER: Use natural Hinglish (Roman script) or clear English depending on user input.
4. STRUCTURE (Max 4-5 bullet points):
   - Line 1: Direct answer to the user's specific question.
   - Line 2-3: Core astrological reason (using user's Lagna/Dasha context cleanly).
   - Line 4-5: Favorable timing window or practical guidance.
"""

LENGTH_GUARDRAIL_PROMPT = SYSTEM_PROMPT
STRUCTURED_SYSTEM_PROMPT = SYSTEM_PROMPT
ULTRA_COMPACT_SYSTEM_PROMPT = SYSTEM_PROMPT

# Fluff recognition set for filtering low-information conversational turns
FLUFF_PATTERNS = {
    "hi", "hello", "hey", "namaste", "greetings", "thanks", "thank you",
    "thx", "okay", "ok", "k", "bye", "goodbye", "good morning",
    "good evening", "good night", "yes", "no", "sure", "cool", "alright",
}

# Meta preamble phrases emitted by models during reasoning leakage
META_PREAMBLE_PHRASES = (
    "we need to respond",
    "user gave",
    "the user gave",
    "the user says",
    "the user asks",
    "let's craft",
    "let's respond",
    "internal thinking",
    "planning:",
    "as an ai",
)

# Proactive Switching Configuration
TOKEN_SAFETY_THRESHOLD_PCT: float = 10.0   # <10% remaining tokens triggers proactive switch
TOKEN_SAFETY_THRESHOLD_ABS: int = 300      # <=300 tokens remaining triggers proactive switch
RATE_LIMIT_COOLDOWN_SECONDS: float = 60.0  # Groq rate-limits reset on a 60-second window
TOTAL_MODEL_QUOTA: int = 8000
SESSION_TOKEN_USAGE_TTL_SECONDS: float = RATE_LIMIT_COOLDOWN_SECONDS

# Real-time In-Memory Model Quota & Health Registry
MODEL_QUOTA_REGISTRY: Dict[str, Dict[str, Any]] = {}
ACCOUNT_MODEL_QUOTA_REGISTRY: Dict[str, Dict[str, Any]] = {}
SESSION_TOKEN_USAGE: Dict[str, Dict[str, Any]] = {}
SESSION_TOKEN_USAGE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Proactive Multi-Account Model-Specific Token Thresholds & Rolling Tracker
# ---------------------------------------------------------------------------
TOKEN_THRESHOLDS: Dict[str, int] = {
    "llama-3.3-70b-versatile": 1000,
    "llama-3.1-8b-instant": 1000,
    "gemma2-9b-it": 1000,
}

# In-Memory Account Token Tracker (cumulative tokens within 60-second rolling window)
# Format: { account_id: { model_id: total_used_tokens } }
account_token_usage: Dict[str, Dict[str, int]] = {
    "Account 1": {},
    "Account 2": {},
}
_ACCOUNT_TOKEN_LEDGER: Dict[str, Dict[str, List[Tuple[float, int]]]] = {
    "Account 1": {},
    "Account 2": {},
}
ACCOUNT_TOKEN_LOCK = threading.Lock()
ACCOUNT_TOKEN_WINDOW_SECONDS: float = 60.0


def get_account_model_tokens(account_id: str, model_id: str) -> int:
    """
    Returns the cumulative tokens used by account_id on model_id within the 60-second
    rolling window. Expired entries (> 60s) are automatically pruned.
    """
    now = time.monotonic()
    with ACCOUNT_TOKEN_LOCK:
        acc_ledger = _ACCOUNT_TOKEN_LEDGER.setdefault(account_id, {})
        records = acc_ledger.get(model_id, [])
        valid_records = [
            (ts, tok) for ts, tok in records
            if now - ts <= ACCOUNT_TOKEN_WINDOW_SECONDS
        ]
        acc_ledger[model_id] = valid_records
        total_used = sum(tok for _, tok in valid_records)
        account_token_usage.setdefault(account_id, {})[model_id] = total_used
        return total_used


def record_account_token_usage(account_id: Optional[str], model_id: str, tokens: int) -> int:
    """
    Records token usage for account_id on model_id within the 60-second rolling window.
    """
    if not account_id or tokens <= 0:
        return 0
    now = time.monotonic()
    with ACCOUNT_TOKEN_LOCK:
        acc_ledger = _ACCOUNT_TOKEN_LEDGER.setdefault(account_id, {})
        records = acc_ledger.get(model_id, [])
        valid_records = [
            (ts, tok) for ts, tok in records
            if now - ts <= ACCOUNT_TOKEN_WINDOW_SECONDS
        ]
        valid_records.append((now, tokens))
        acc_ledger[model_id] = valid_records
        total_used = sum(tok for _, tok in valid_records)
        account_token_usage.setdefault(account_id, {})[model_id] = total_used
        return total_used


def reset_account_token_usage(account_id: Optional[str] = None, model_id: Optional[str] = None) -> None:
    """Resets the rolling token usage tracker (useful for testing or session reset)."""
    with ACCOUNT_TOKEN_LOCK:
        if account_id and model_id:
            _ACCOUNT_TOKEN_LEDGER.get(account_id, {}).pop(model_id, None)
            account_token_usage.get(account_id, {}).pop(model_id, None)
        elif account_id:
            _ACCOUNT_TOKEN_LEDGER[account_id] = {}
            account_token_usage[account_id] = {}
        else:
            for acc in list(_ACCOUNT_TOKEN_LEDGER.keys()):
                _ACCOUNT_TOKEN_LEDGER[acc] = {}
                account_token_usage[acc] = {}


# ---------------------------------------------------------------------------
# Local Q&A Database Cache & Exact-Match Lookup (0 Tokens for short FAQs)
# ---------------------------------------------------------------------------
QA_DB_PATH = os.path.join(BASE_DIR, "qa_database.json")
qa_database_index: Dict[str, Any] = {}
qa_database_raw: Dict[str, dict] = {}


def _init_qa_database_index() -> Dict[str, Any]:
    global qa_database_index, qa_database_raw
    candidate_paths = [
        QA_DB_PATH,
        os.path.join(BASE_DIR, "..", "qa_database.json"),
        "qa_database.json",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        clean_k = k.lower().strip("?.! ")
                        if isinstance(v, dict):
                            qa_database_raw[clean_k] = v
                            qa_database_index[clean_k] = v
                            qa_database_index[f"what is {clean_k}"] = v
                            qa_database_index[f"what is a {clean_k}"] = v
                            qa_database_index[f"what is an {clean_k}"] = v
                            qa_database_index[f"tell me about {clean_k}"] = v
                            qa_database_index[f"explain {clean_k}"] = v
                            # Index all explicit keywords
                            for kw in v.get("keywords", []):
                                clean_kw = str(kw).lower().strip("?.! ")
                                if clean_kw:
                                    qa_database_raw[clean_kw] = v
                                    qa_database_index[clean_kw] = v
                                    qa_database_index[f"what is {clean_kw}"] = v
                                    qa_database_index[f"tell me about {clean_kw}"] = v
                                    qa_database_index[f"explain {clean_kw}"] = v
                        elif isinstance(v, str):
                            clean_k = k.lower().strip("?.! ")
                            qa_database_index[clean_k] = v
                            qa_database_index[f"what is {clean_k}"] = v
                            qa_database_index[f"what is a {clean_k}"] = v
                            qa_database_index[f"what is an {clean_k}"] = v
                            qa_database_index[f"tell me about {clean_k}"] = v
                            qa_database_index[f"explain {clean_k}"] = v
                    break
            except Exception as exc:
                logger.warning("Error loading qa_database_index from %s: %s", path, exc)
    return qa_database_index


_init_qa_database_index()

# ---------------------------------------------------------------------------
# Strict Query Classification Middleware & Personal Intent Guard
# ---------------------------------------------------------------------------
PERSONAL_INTENT_PATTERNS = (
    r"\bmy\b", r"\bmine\b", r"\bi\s+have\b", r"\bi\s+am\b", r"\bam\s+i\b",
    r"\bplaced\s+in\b", r"\bsitting\s+in\b", r"\baspecting\b", r"\baspected\b",
    r"\bconjunction\b", r"\bcombust\b", r"\bexalted\b", r"\bdebilitated\b",
    r"\bhouse\s+lord\b", r"\blord\s+in\b", r"\bruler\s+in\b",
    r"\bmahadasha\b", r"\bantardasha\b", r"\bpratyantardasha\b", r"\bdasha\b",
    r"\bmaha\s+dasha\b", r"\bantar\s+dasha\b",
    r"\bnakshatra\b", r"\bpada\b", r"\bnavamsha\b", r"\bnavamsa\b", r"\bd9\b", r"\bd10\b",
    r"\bnext\s+\d+\s+(?:months?|years?|weeks?|days?)\b", r"\bnext\s+(?:month|year|week)\b",
    r"\bupcoming\b", r"\bwhen\s+will\b", r"\bwill\s+i\b", r"\bcan\s+i\b", r"\bshould\s+i\b",
    r"\bhow\s+will\b", r"\bwhat\s+will\s+happen\b",
    r"\bremedy\b", r"\bremedies\b", r"\bmantras?\b", r"\bgemstones?\b",
    r"\bmarriage\s+remedies\b", r"\bcareer\s+remedies\b",
    r"\bobstacles?\b", r"\bchallenges?\b", r"\bfacing\b", r"\bdelays?\b",
    r"\bprospects?\b", r"\bopportunities\b", r"\btransits?\b",
    r"\bsade\s+sati\s+period\b", r"\bshani\s+ki\s+sade\s+sati\b",
    r"\bin\s+the\s+\d+(?:st|nd|rd|th)?\s+house\b",
    r"\bin\s+\d+(?:st|nd|rd|th)?\s+house\b",
)


def should_bypass_local_qa(query: str) -> bool:
    """
    Strict Length & Intent Guard for Local QA:
    Returns True (bypassing qa_database.json completely and sending to Groq LLMs) if:
    1. The user query is 10 words or more (only queries under 10 words can be local FAQs).
    2. Any query containing personal natal positions, Mahadasha/Antardasha details,
       Nakshatras, or specific temporal advice ('next 6 months', 'marriage remedies').
    """
    if not query:
        return True

    words = query.strip().split()
    # 1. Under 10 words check (< 10 words allowed; >= 10 words MUST bypass)
    if len(words) >= 10:
        return True

    # 2. Check for personal natal positions, dashas, nakshatras, remedies, or temporal advice
    lower_query = query.lower()
    for pattern in PERSONAL_INTENT_PATTERNS:
        if re.search(pattern, lower_query):
            return True

    return False


def format_local_qa_response(qa_data: dict, user_context: dict) -> str:
    """
    Step 4: Dynamic Local Response Formatter
    When a local database hit occurs, dynamically populates the structured JSON fields
    with the user's active birth facts.
    """
    ascendant = user_context.get("ascendant", "Libra")
    moon_sign = user_context.get("moon_sign", "Gemini")
    active_dasha = user_context.get("active_dasha", "Rahu-Mars")
    house_rulerships = user_context.get("house_rulerships") or calculate_house_rulerships(ascendant)

    # Substitution parameters for template markers
    subs: Dict[str, Any] = {
        "ascendant": ascendant,
        "moon_sign": moon_sign,
        "active_dasha": active_dasha,
    }
    for h_name, h_val in house_rulerships.items():
        subs[f"{h_name}_full"] = h_val
        m = re.match(r"^(\w+)\s*\(([^)]+)\)$", str(h_val).strip())
        if m:
            lord, sign = m.group(1), m.group(2)
            subs[f"{h_name}_lord"] = lord
            subs[f"{h_name}_sign"] = sign
            subs[h_name] = sign
        else:
            subs[f"{h_name}_lord"] = str(h_val)
            subs[h_name] = str(h_val)

    class SafeDict(dict):
        def __missing__(self, k: str) -> str:
            return f"{{{k}}}"

    safe_subs = SafeDict(**subs)

    analogy = str(qa_data.get("analogy", qa_data.get("simple_analogy", "")))
    core_analysis = str(qa_data.get("core_analysis", qa_data.get("core_meaning", "")))
    dasha_influence = str(qa_data.get("dasha_influence", qa_data.get("what_it_means", "")))
    recommendation = str(qa_data.get("recommendation", qa_data.get("actionable_takeaway", "")))

    try:
        analogy = analogy.format_map(safe_subs)
        core_analysis = core_analysis.format_map(safe_subs)
        dasha_influence = dasha_influence.format_map(safe_subs)
        recommendation = recommendation.format_map(safe_subs)
    except Exception:
        pass

    # If static text contains default Libra Lagna references but user has another Lagna, adapt accurately
    if ascendant and ascendant.lower() != "libra":
        h10_sign = subs.get("10th_house", "Cancer")
        h10_lord = subs.get("10th_house_lord", "Moon")
        h6_sign = subs.get("6th_house", "Pisces")
        h6_lord = subs.get("6th_house_lord", "Jupiter")
        core_analysis = core_analysis.replace("(Cancer)", f"({h10_sign})").replace("Moon-ruled", f"{h10_lord}-ruled")
        core_analysis = core_analysis.replace("(Pisces)", f"({h6_sign})").replace("Jupiter-ruled", f"{h6_lord}-ruled")
    if active_dasha and active_dasha.lower() != "rahu-mars":
        dasha_influence = dasha_influence.replace("Rahu-Mars", active_dasha)

    return (
        f"**Career Guidance ({user_context.get('ascendant', 'Libra')} Lagna | {user_context.get('moon_sign', 'Gemini')} Moon)**\n\n"
        f"**Everyday Analogy:** {analogy}\n\n"
        f"**Astrological Insight:** {core_analysis} {dasha_influence}\n\n"
        f"**Actionable Advice:** {recommendation}"
    )


def get_local_qa_response(
    user_message: str,
    user_profile: Optional[Any] = None,
    user_natal_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Strict Length & Intent Guard for Local QA:
    qa_database.json is ONLY queried if:
    - The user query is under 10 words AND matches an exact short intent
      (e.g., 'What is 10th house?', 'Define Sade Sati').
    - Any query containing personal natal positions, Mahadasha/Antardasha details,
      Nakshatras, or specific temporal advice ('next 6 months', 'marriage remedies')
      MUST BYPASS qa_database.json completely.
    """
    if not user_message:
        return None

    # Strict Length (<10 words) & Personal Intent Guard
    if should_bypass_local_qa(user_message):
        return None

    clean_msg = user_message.lower().strip("?.! ")
    template = qa_database_index.get(clean_msg, None)
    if not template:
        # Check standard short definition prefixes
        for prefix in ("what is a ", "what is an ", "what is ", "define ", "meaning of ", "explain "):
            if clean_msg.startswith(prefix):
                term = clean_msg[len(prefix):].strip("?.! ")
                template = qa_database_index.get(term, None)
                if template:
                    break

    if not template:
        clean_msg_norm = re.sub(r"[^\w\s]", "", clean_msg).strip()
        template = qa_database_index.get(clean_msg_norm, None)

    if not template:
        return None

    # Resolve active natal context
    if not user_natal_context:
        user_natal_context = get_user_natal_context(user_profile=user_profile)

    # If matched template is a structured dict from qa_database.json, use format_local_qa_response
    if isinstance(template, dict):
        return format_local_qa_response(template, user_natal_context)

    # Format template if user_profile or user_natal_context is provided
    if user_profile or user_natal_context:
        profile_params: Dict[str, Any] = {}
        if isinstance(user_profile, dict):
            profile_params = dict(user_profile)
        elif hasattr(user_profile, "model_dump"):
            profile_params = user_profile.model_dump()

        lagna = user_natal_context.get("ascendant") or profile_params.get("lagna") or "Libra"
        sun = profile_params.get("sun_sign") or profile_params.get("sun") or "Aries"
        moon = user_natal_context.get("moon_sign") or profile_params.get("moon_sign") or "Gemini"
        dasha = user_natal_context.get("active_dasha") or profile_params.get("current_dasha") or "Rahu-Mars"

        format_dict = {
            "name": profile_params.get("name", "Seeker"),
            "lagna": lagna,
            "ascendant": lagna,
            "moon_sign": moon,
            "moon": moon,
            "sun_sign": sun,
            "sun": sun,
            "current_dasha": str(dasha),
            "dasha": str(dasha),
            "active_dasha": str(dasha),
            "jupiter_house": profile_params.get("jupiter_house", "9th House"),
            "jupiter": profile_params.get("jupiter_house", "9th House"),
        }
        class SafeDict(dict):
            def __missing__(self, k: str) -> str:
                return f"{{{k}}}"
        try:
            return template.format_map(SafeDict(**format_dict))
        except Exception:
            return template

    return template


def begin_token_usage_session(session_id: Optional[str]) -> None:
    """Expire inactive session counters and mark this request as session activity."""
    if not session_id:
        return

    now = time.monotonic()
    with SESSION_TOKEN_USAGE_LOCK:
        expired_sessions = [
            key for key, state in SESSION_TOKEN_USAGE.items()
            if now - state["last_request_at"] > SESSION_TOKEN_USAGE_TTL_SECONDS
        ]
        for key in expired_sessions:
            del SESSION_TOKEN_USAGE[key]

        state = SESSION_TOKEN_USAGE.setdefault(
            session_id,
            {"last_request_at": now, "models": {}},
        )
        state["last_request_at"] = now


def record_session_token_usage(
    session_id: Optional[str],
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Tuple[int, int, float]:
    """Add this turn's tokens and return (turn, session total, remaining percent)."""
    used_this_turn = max(0, prompt_tokens) + max(0, completion_tokens)
    if not session_id:
        return used_this_turn, used_this_turn, max(
            0.0, (TOTAL_MODEL_QUOTA - used_this_turn) / TOTAL_MODEL_QUOTA * 100
        )

    now = time.monotonic()
    with SESSION_TOKEN_USAGE_LOCK:
        state = SESSION_TOKEN_USAGE.setdefault(
            session_id,
            {"last_request_at": now, "models": {}},
        )
        model_usage = state["models"].get(model_id, 0) + used_this_turn
        state["models"][model_id] = model_usage
        remaining_tokens = max(0, TOTAL_MODEL_QUOTA - model_usage)
        percentage_remaining = (remaining_tokens / TOTAL_MODEL_QUOTA) * 100
        return used_this_turn, model_usage, percentage_remaining


_GROQ_CLIENT_CACHE: Dict[str, groq.AsyncGroq] = {}
_SHARED_HTTPX_CLIENT: Optional[httpx.AsyncClient] = None


def get_shared_httpx_client() -> httpx.AsyncClient:
    """Returns a shared, pooled AsyncClient for external LLM API calls with expanded connection pools."""
    global _SHARED_HTTPX_CLIENT
    if _SHARED_HTTPX_CLIENT is None or _SHARED_HTTPX_CLIENT.is_closed:
        _SHARED_HTTPX_CLIENT = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=100, keepalive_expiry=60.0),
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
    return _SHARED_HTTPX_CLIENT


def get_groq_accounts() -> List[Dict[str, str]]:
    """Returns active Groq accounts with valid API keys using GROQ_API_KEY_1 and GROQ_API_KEY_2."""
    acc1_key = (
        os.getenv("GROQ_API_KEY_1")
        or os.getenv("GROQ_API_KEY_ACCOUNT1")
        or os.getenv("GROQ_API_KEY_ACC1")
        or os.getenv("GROQ_API_KEY")
    )
    acc2_key = (
        os.getenv("GROQ_API_KEY_2")
        or os.getenv("GROQ_API_KEY_ACCOUNT2")
        or os.getenv("GROQ_API_KEY_ACC2")
    )

    accounts: List[Dict[str, str]] = []
    if acc1_key and acc1_key.strip():
        accounts.append({"account_id": "Account 1", "api_key": acc1_key.strip()})
    if acc2_key and acc2_key.strip():
        accounts.append({"account_id": "Account 2", "api_key": acc2_key.strip()})

    for i in range(3, 10):
        ki = os.getenv(f"GROQ_API_KEY_{i}") or os.getenv(f"GROQ_API_KEY_ACCOUNT{i}") or os.getenv(f"GROQ_API_KEY_ACC{i}")
        if ki and ki.strip():
            accounts.append({"account_id": f"Account {i}", "api_key": ki.strip()})

    return accounts


def get_groq_client_for_account(api_key: str) -> Optional[groq.AsyncGroq]:
    """Get or create cached AsyncGroq client for a specific account API key with connection pooling."""
    if not api_key:
        return None
    if api_key not in _GROQ_CLIENT_CACHE:
        try:
            _GROQ_CLIENT_CACHE[api_key] = groq.AsyncGroq(
                api_key=api_key,
                http_client=get_shared_httpx_client(),
            )
        except Exception:
            _GROQ_CLIENT_CACHE[api_key] = groq.AsyncGroq(api_key=api_key)
    return _GROQ_CLIENT_CACHE[api_key]


def get_groq_client() -> Optional[groq.AsyncGroq]:
    """Backward-compatible helper returning the primary Account 1 client."""
    accounts = get_groq_accounts()
    if accounts:
        return get_groq_client_for_account(accounts[0]["api_key"])
    return None


def get_openrouter_client() -> Optional[openai.AsyncOpenAI]:
    """
    Initialize AsyncOpenAI client for the OpenRouter fallback tier with connection pooling.
    - If OPENROUTER_API_KEY is configured, points to openrouter.ai.
    - If OPENROUTER_API_KEY is not set, bridges using primary Groq account key
      against Groq's OpenAI-compatible base URL (https://api.groq.com/openai/v1)
      so models like openai/gpt-oss-120b run with full compatibility.
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        return openai.AsyncOpenAI(
            api_key=openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            http_client=get_shared_httpx_client(),
            default_headers={
                "HTTP-Referer": "http://127.0.0.1:8000",
                "X-Title": "Astro AI Chatbot",
            },
        )
    # Automatic fallback: Groq's OpenAI-compatible endpoint
    accounts = get_groq_accounts()
    if accounts:
        return openai.AsyncOpenAI(
            api_key=accounts[0]["api_key"],
            base_url="https://api.groq.com/openai/v1",
            http_client=get_shared_httpx_client(),
        )
    return None



def get_gemini_client() -> Optional[genai.Client]:
    """Initialize Google GenAI client with environment key."""
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        logger.warning("GEMINI_API_KEY is not configured in environment")
        return None
    return genai.Client(api_key=gemini_api_key)


# ---------------------------------------------------------------------------
# [SPECIFICATION 1: Unambiguous User Profile Format (~18 Tokens)]
# Clear, unambiguous key-value pairs requiring zero model guessing.
# Example: "Ascendant: Cancer | Moon: Leo | Dasha: Rahu"
# ---------------------------------------------------------------------------
def compress_user_profile(user_profile: Optional[Any] = None) -> str:
    """
    Refactor user profile into clear, short, unambiguous key-value pairs (~18 tokens).
    Format: "Ascendant: <lagna> | Moon: <moon> | Dasha: <dasha>"
    Eliminates model confusion and prevents reasoning tokens from being wasted guessing fields.
    """
    if not user_profile:
        return "Ascendant: Cancer | Moon: Leo | Dasha: Rahu"

    def _get(keys: List[str], default: str = "") -> str:
        if isinstance(user_profile, dict):
            for k in keys:
                val = user_profile.get(k)
                if val:
                    return str(val)
            return default
        for k in keys:
            val = getattr(user_profile, k, None)
            if val:
                return str(val)
        return default

    lagna = _get(["lagna", "asc_sign", "ascendant"], "Cancer")
    sun = _get(["sun_sign", "sun"], "Aries")
    moon = _get(["moon_sign", "rashi", "moon"], "Leo")
    raw_dasha = _get(["current_dasha", "dasha"], "Rahu")
    if isinstance(raw_dasha, dict):
        raw_dasha = raw_dasha.get("current_dasha") or raw_dasha.get("major") or "Rahu"

    dasha_str = str(raw_dasha).replace("/", "-").strip()

    return f"Ascendant: {lagna} | Sun: {sun} | Moon: {moon} | Dasha: {dasha_str}"



# Backwards-compatible alias
format_compact_user_profile = compress_user_profile


def build_system_prompt(
    user_profile: Optional[Any] = None,
    session_profile: Optional[Dict[str, Any]] = None,
    user_natal_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Step 2: Strict System Prompt Guardrails
    Binds the AI strictly to the provided natal context and enforces a direct, empathetic,
    and accurate Vedic Astrology tone adhering strictly to 4 to 6 lines without cheesy analogies.
    """
    if not user_natal_context:
        user_natal_context = get_user_natal_context(user_profile=user_profile)

    system_prompt_str = f"""You are a direct, empathetic, and accurate Vedic Astrology assistant.

### MANDATORY NATAL DATA FOR THIS USER:
- Ascendant (Lagna): {user_natal_context['ascendant']}
- Moon Sign: {user_natal_context['moon_sign']}
- Active Dasha: {user_natal_context['active_dasha']}
- Pre-Calculated House Lords: {user_natal_context['house_rulerships']}

### STRICT GUARDRAILS:
1. GROUND TRUTH ONLY: Never fabricate house lords. (e.g., For Libra Lagna, 10th house is ALWAYS Moon-ruled Cancer; 6th house is ALWAYS Jupiter-ruled Pisces. Do NOT claim otherwise.)
2. NO ABSOLUTE DATES/GUARANTEES: Express promotions, marriages, or timing in terms of "favorable planetary alignment windows" rather than absolute certainty.
3. NO HALLUCINATED TRANSITS: Do not make up future transit dates or planetary conjunctions.

STRICT OUTPUT RULES:
1. NO CHEESY ANALOGIES: Avoid fluffy metaphors like "love is like a fruit tree/garden/ocean". Keep the tone natural, grounded, and practical.
2. LENGTH & COMPLETENESS: Respond in strictly 4 to 6 lines total. Complete every sentence properly—NEVER stop mid-sentence or cut off mid-thought.
3. LANGUAGE REGISTER: Use natural Hinglish (Roman script) or clear English depending on user input.
4. STRUCTURE (Max 4-5 bullet points):
   - Line 1: Direct answer to the user's specific question.
   - Line 2-3: Core astrological reason (using user's Lagna/Dasha context cleanly).
   - Line 4-5: Favorable timing window or practical guidance.
"""

    preferred_language = (session_profile or {}).get("preferred_language", "english")
    if str(preferred_language).casefold() == "hinglish":
        system_prompt_str += "\nEnsure the explanation uses friendly, conversational Hinglish (Roman script) while strictly adhering to the 4 to 6 line response format and guardrails."

    return system_prompt_str


# ---------------------------------------------------------------------------
# [SPECIFICATION 3 & 4: Output Sanitization, Smart Sentence Truncation & Response Formatting]
# ---------------------------------------------------------------------------
def strip_server_metadata(text: str) -> str:
    """
    Strips attached server metadata strings like:
    - (OpenRouter | LLM: ...)
    - (Account 1 | LLM: ...)
    - (LLM: ...)
    - (Backend)
    - (Gemini | LLM: ...)
    from the response before returning the payload to the frontend.
    """
    if not text:
        return ""
    # Matches (OpenRouter | LLM: ...), (Account 1 | LLM: ...), (LLM: ...), (Backend), [LLM: ...], etc.
    cleaned = re.sub(
        r"\s*[\(\[]\s*(?:(?:Account\s*\d+|OpenRouter|Groq|Gemini)\s*\|\s*)?LLM:[^\)\]]+[\)\]]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*[\(\[]\s*Backend\s*[\)\]]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*[\(\[]\s*(?:Account\s*\d+|OpenRouter|Groq|Gemini)\s*[\)\]]", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def clean_llm_output(text: str) -> str:
    """
    Utility function to strip away internal reasoning, chain-of-thought blocks,
    meta-commentary planning preambles, decorative icons/emojis, and server metadata:
    1. Strips complete and unclosed <think> and <thought> blocks via Regex.
    2. Strips meta planning phrases if they appear in text.
    3. Strips decorative icons and emojis (✨, 💡, 📌, 🎯, etc.).
    4. Strips server metadata tags ((OpenRouter | LLM: ...), (Backend), etc.).
    5. Preserves 4 to 6 line structured formatting and structural breaks.
    """
    if not text:
        return ""

    text = text.replace("\u2011", "-").replace("\u2010", "-").replace("\u2013", "-").replace("\u2014", "--")
    text = text.replace("\u00a0", " ").replace("\u202f", " ")

    # Strip any attached server metadata strings
    text = strip_server_metadata(text)

    # Step 1: Strip complete think/thought blocks and any stray/unclosed tags
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?(?:think|thought)>", "", cleaned, flags=re.IGNORECASE)

    # Step 2: Strip meta planning phrases if they appear in text
    for phrase in META_PREAMBLE_PHRASES:
        pattern = re.compile(rf"^\s*{re.escape(phrase)}[^\n.]*[.:]?", re.IGNORECASE)
        cleaned = pattern.sub("", cleaned)

    # Step 3: Strip decorative icons and emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs (💡, 📌, 🎯, etc.)
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats (✨)
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE,
    )
    cleaned = emoji_pattern.sub("", cleaned)
    # Clean up leading whitespace on lines left by removed icons
    cleaned = re.sub(r"^[ \t]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\*\*[ \t]+", "**", cleaned)

    # Step 4: Collapse redundant excessive empty lines while preserving structural breaks
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # Step 5: Final strip of server metadata
    return strip_server_metadata(cleaned)


def trim_to_last_sentence(text: str) -> str:
    """
    If an LLM response gets cut off mid-sentence due to max_tokens,
    this function smartly trims back to the last complete punctuation mark.
    Also strips any trailing server metadata.
    """
    text = strip_server_metadata(text).strip()
    if not text:
        return ""

    if text and text[-1] not in [".", "!", "?"]:
        match = re.search(r"^.*[.!?]", text, re.DOTALL)
        if match:
            text = match.group(0).strip()
        else:
            text = f"{text}."

    return strip_server_metadata(text)


def format_llm_response(
    text: str,
    model_id: Optional[str] = None,
    account_id: Optional[str] = None,
    include_suffix: bool = False,
) -> str:
    """
    Response Formatting Sequence:
    Clean and sanitize the LLM response in this exact order before returning to the frontend:
    1. Run string cleaning / strip reasoning or meta-tags (clean_llm_output).
    2. Apply trim_to_last_sentence() so the output always ends cleanly with '.', '!', or '?'.
    3. Strip server metadata unless include_suffix=True is explicitly requested.
    """
    if not text:
        return ""

    # 1. Run string cleaning / strip reasoning or meta-tags
    clean_text = clean_llm_output(text)

    # 2. Apply trim_to_last_sentence() so output always ends cleanly with '.', '!', or '?'
    clean_text = trim_to_last_sentence(clean_text)

    # 3. Strip server metadata
    clean_text = strip_server_metadata(clean_text)

    # 4. Optional UI model and account tag (disabled by default)
    if include_suffix and model_id:
        if model_id == "Backend":
            return f"{clean_text} (Backend)"
        if account_id:
            return f"{clean_text} ({account_id} | LLM: {model_id})"
        if "LLM:" in model_id or "|" in model_id:
            return f"{clean_text} ({model_id})"
        return f"{clean_text} (LLM: {model_id})"

    return clean_text


async def sanitize_stream(stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """
    Real-time streaming sanitizer:
    Buffers initial tokens to detect and strip <think> blocks and meta-preambles.
    Enforces a single continuous paragraph by converting newlines to spaces
    and stripping bold tags/section headers on the fly.
    """
    buffer = ""
    in_think_block = False
    started_real_content = False

    async for chunk in stream:
        # Convert newlines to spaces to enforce a single continuous paragraph
        clean_chunk = chunk.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

        if started_real_content:
            # Yield real content tokens; strip bold formatting characters on the fly
            clean_chunk = clean_chunk.replace("**", "").replace("__", "")
            if clean_chunk:
                yield clean_chunk
            continue

        buffer += clean_chunk

        # Check if inside a <think> or <thought> block
        lower_buf = buffer.lower()
        if "<think>" in lower_buf or "<thought>" in lower_buf:
            in_think_block = True
            if "</think>" in lower_buf or "</thought>" in lower_buf:
                # Strip completed think block
                buffer = re.sub(r"<think>.*?</think>", "", buffer, flags=re.DOTALL | re.IGNORECASE)
                buffer = re.sub(r"<thought>.*?</thought>", "", buffer, flags=re.DOTALL | re.IGNORECASE)
                in_think_block = False
            else:
                continue

        if in_think_block:
            continue

        # Check if buffer starts with a known meta phrase like "We need to", "User gave"
        has_meta = any(phrase in buffer.lower() for phrase in META_PREAMBLE_PHRASES)
        if has_meta:
            # Hold in buffer waiting for actual astrology response
            continue

        # Strip header markers like "Career Outlook:" or "Remedy:" if model emitted them
        buffer = re.sub(r"\*{0,2}(?:Career Outlook|Remedy):?\*{0,2}", "", buffer, flags=re.IGNORECASE).lstrip()

        # Once buffer has non-empty text, flush buffer and start direct streaming
        if len(buffer) >= 20 or (" " in buffer and len(buffer) >= 10):
            started_real_content = True
            cleaned_buf = buffer.replace("**", "").replace("__", "")
            if cleaned_buf:
                yield cleaned_buf
            buffer = ""

    # If stream ended and real content never started, sanitize remaining buffer and yield
    if buffer:
        cleaned = clean_llm_output(buffer)
        if cleaned:
            yield cleaned


def is_conversational_fluff(text: str) -> bool:
    """Checks if a user or assistant message is conversational fluff."""
    if not text:
        return True
    cleaned = re.sub(r"[^\w\s]", "", text).strip().lower()
    return cleaned in FLUFF_PATTERNS or len(cleaned) == 0


def filter_chat_history(
    history: Optional[List[Dict[str, Any]]],
    max_messages: int = STRICT_HISTORY_TURNS,
) -> List[Dict[str, str]]:
    """
    1. Ignores conversational fluff ('Hi', 'Hello', 'Thanks', 'Okay').
    2. Restricts the returned history array strictly to the last 2 messages
       (1 previous user turn + 1 previous assistant response).
    """
    if not history:
        return []

    non_fluff: List[Dict[str, str]] = []
    for item in history:
        role = item.get("role", "user")
        content = " ".join(str(item.get("content", "")).split())
        if not content or is_conversational_fluff(content):
            continue
        if role in ("user", "assistant"):
            non_fluff.append({"role": role, "content": content})

    history_limit = max(0, min(max_messages, STRICT_HISTORY_TURNS))
    return non_fluff[-history_limit:] if history_limit else []


apply_sliding_window = filter_chat_history


def build_chat_messages(
    system_prompt: str,
    prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    max_history_messages: int = STRICT_HISTORY_TURNS,
) -> List[Dict[str, str]]:
    """
    Final payload uses a compact system prompt, at most one prior Q&A, and a normalized query.
    """
    clean_system_prompt = system_prompt.strip()
    clean_user_message = " ".join(prompt.split())
    messages: List[Dict[str, str]] = [{"role": "system", "content": clean_system_prompt}]
    messages.extend(
        filter_chat_history(
            history,
            max_messages=min(max_history_messages, STRICT_HISTORY_TURNS),
        )
    )
    messages.append({"role": "user", "content": clean_user_message})
    return messages


# ---------------------------------------------------------------------------
# Proactive Threshold Logic & Quota Tracking
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Proactive Threshold Logic & Quota Tracking
# ---------------------------------------------------------------------------

def inspect_and_update_model_quota(
    model_name: str,
    headers: Any,
    account_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inspects official Groq rate-limit response headers:
      - x-ratelimit-remaining-tokens (Remaining tokens in current minute window)
      - x-ratelimit-limit-tokens (Total minute token limit)
      - x-ratelimit-reset-tokens (Duration until token quota resets)

    Proactive Switching Threshold Logic:
      If remaining token balance drops to <= 300 tokens (or < 10% of total limit):
      1. Logs warning flag: [TOKEN ALERT] Account <account_id> on model <model_name> has <remaining> tokens left. Threshold reached.
      2. Dynamically marks (account, model) as temporarily exhausted (cooldown period)
         so the cascading router automatically fails over to the next account.
    """
    if not headers:
        return MODEL_QUOTA_REGISTRY.get(model_name, {})

    remaining_str = headers.get("x-ratelimit-remaining-tokens")
    limit_str = headers.get("x-ratelimit-limit-tokens")
    reset_str = headers.get("x-ratelimit-reset-tokens", "")

    if remaining_str is not None and limit_str is not None:
        try:
            remaining = int(remaining_str)
            limit = int(limit_str)
            remaining_pct = (remaining / limit) * 100.0 if limit > 0 else 0.0

            entry = MODEL_QUOTA_REGISTRY.setdefault(model_name, {})
            entry.update({
                "remaining_tokens": remaining,
                "limit_tokens": limit,
                "remaining_pct": remaining_pct,
                "reset_time_str": str(reset_str),
                "last_updated": time.time(),
            })

            # Also record in ACCOUNT_MODEL_QUOTA_REGISTRY if account_id is present
            acc_entry = None
            if account_id:
                acc_key = f"{account_id}::{model_name}"
                acc_entry = ACCOUNT_MODEL_QUOTA_REGISTRY.setdefault(acc_key, {})
                acc_entry.update({
                    "remaining_tokens": remaining,
                    "limit_tokens": limit,
                    "remaining_pct": remaining_pct,
                    "reset_time_str": str(reset_str),
                    "last_updated": time.time(),
                })

            # Evaluate proactive threshold (<= 300 tokens or < 10% total capacity)
            if remaining <= TOKEN_SAFETY_THRESHOLD_ABS or remaining_pct < TOKEN_SAFETY_THRESHOLD_PCT:
                entry["is_exhausted"] = True
                entry["cooldown_until"] = time.time() + RATE_LIMIT_COOLDOWN_SECONDS
                entry["bypass_reason"] = (
                    f"Proactive threshold reached: {remaining}/{limit} tokens left ({remaining_pct:.1f}% remaining)"
                )
                if acc_entry is not None:
                    acc_entry["is_exhausted"] = True
                    acc_entry["cooldown_until"] = time.time() + RATE_LIMIT_COOLDOWN_SECONDS
                    acc_entry["bypass_reason"] = entry["bypass_reason"]

                target_name = f"{account_id} on model {model_name}" if account_id else f"Model {model_name}"
                alert_msg = f"[TOKEN ALERT] {target_name} has {remaining} tokens left. Threshold reached."
                logger.warning(alert_msg)
                print(alert_msg)
            else:
                entry["is_exhausted"] = False
                entry["cooldown_until"] = 0.0
                entry["bypass_reason"] = None
                if acc_entry is not None:
                    acc_entry["is_exhausted"] = False
                    acc_entry["cooldown_until"] = 0.0
                    acc_entry["bypass_reason"] = None

            return entry

        except (ValueError, TypeError) as err:
            logger.debug("Failed parsing rate-limit headers for model %s: %s", model_name, err)

    return MODEL_QUOTA_REGISTRY.get(model_name, {})


# Backwards-compatible alias
update_model_quota_from_headers = inspect_and_update_model_quota


def mark_model_rate_limited(model_name: str, cooldown_seconds: float = RATE_LIMIT_COOLDOWN_SECONDS) -> None:
    """Marks a model as temporarily rate-limited when HTTP 429 is encountered."""
    entry = MODEL_QUOTA_REGISTRY.setdefault(model_name, {})
    entry["is_exhausted"] = True
    entry["cooldown_until"] = time.time() + cooldown_seconds
    entry["bypass_reason"] = f"HTTP 429 RateLimitError (cooling down for {cooldown_seconds:.0f}s)"
    logger.warning(
        "[RATE LIMIT] Model: %s rate-limited (HTTP 429); cooling down for %.0fs",
        model_name, cooldown_seconds
    )


rate_limited_keys: Dict[str, float] = {}
ACCOUNT_RATE_LIMIT_REGISTRY: Dict[str, float] = {}

SYSTEM_BUSY_MESSAGE: str = (
    "Astro AI is currently experiencing high traffic. Please wait 10 seconds and try again. (System Busy)"
)
SYSTEM_BUSY_PAYLOAD: Dict[str, Any] = {
    "status": "busy",
    "response": SYSTEM_BUSY_MESSAGE,
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
}


def get_system_busy_payload() -> Dict[str, Any]:
    """Returns clean HTTP 200 JSON payload for rate-limit or capacity exhaustion."""
    return dict(SYSTEM_BUSY_PAYLOAD)


def mark_key_rate_limited(api_key: str, cooldown_seconds: float = RATE_LIMIT_COOLDOWN_SECONDS) -> None:
    """Flags an API key for cooldown_seconds (default 60s) whenever it throws RateLimitError."""
    if not api_key:
        return
    rate_limited_keys[api_key] = time.time() + cooldown_seconds


def is_key_rate_limited(api_key: str) -> bool:
    """Checks if an API key is currently in rate-limit cooldown."""
    if not api_key:
        return False
    return time.time() < rate_limited_keys.get(api_key, 0.0)


def mark_account_rate_limited(
    account_id: str,
    cooldown_seconds: float = RATE_LIMIT_COOLDOWN_SECONDS,
    api_key: Optional[str] = None,
) -> None:
    """
    Marks an entire account and API key as rate-limited for cooldown_seconds (default 60s) on HTTP 429.
    Triggers immediate failover to the next available account.
    """
    if account_id:
        ACCOUNT_RATE_LIMIT_REGISTRY[account_id] = time.time() + cooldown_seconds
    if api_key:
        mark_key_rate_limited(api_key, cooldown_seconds)
    logger.warning(
        "[RATE LIMIT] %s hit HTTP 429; cooling down for %.0fs. Immediately failing over to next account.",
        account_id, cooldown_seconds
    )
    print(f"[RATE LIMIT] {account_id} hit HTTP 429; cooling down for {cooldown_seconds:.0f}s. Immediate failover.")


def is_account_rate_limited(account_id: str, api_key: Optional[str] = None) -> bool:
    """Checks if an account or API key is currently in rate-limit cooldown."""
    if api_key and is_key_rate_limited(api_key):
        return True
    if not account_id:
        return False
    cooldown = ACCOUNT_RATE_LIMIT_REGISTRY.get(account_id, 0.0)
    return time.time() < cooldown


def mark_account_model_rate_limited(
    account_id: str,
    model_name: str,
    cooldown_seconds: float = RATE_LIMIT_COOLDOWN_SECONDS,
) -> None:
    """Marks an account-specific model key as temporarily rate-limited when HTTP 429 is encountered."""
    key = f"{account_id}::{model_name}"
    entry = ACCOUNT_MODEL_QUOTA_REGISTRY.setdefault(key, {})
    entry["is_exhausted"] = True
    entry["cooldown_until"] = time.time() + cooldown_seconds
    entry["bypass_reason"] = f"HTTP 429 RateLimitError on {account_id} ({cooldown_seconds:.0f}s cooldown)"
    logger.warning(
        "[RATE LIMIT] %s on model %s rate-limited (HTTP 429); cooling down for %.0fs",
        account_id, model_name, cooldown_seconds
    )


DEPRECATED_MODELS: set = set()
BLACKLISTED_MODELS: set = DEPRECATED_MODELS


def is_decommissioned_or_404_error(status_code: Any, error: Any = None) -> bool:
    """
    Returns True if the response indicates an HTTP 404 (not found) or
    HTTP 400 (decommissioned or unsupported model).
    """
    try:
        sc = int(status_code)
    except (TypeError, ValueError):
        sc = 0

    if sc == 404:
        return True

    err_str = (str(error) if error else "").lower()
    if sc == 400:
        if any(k in err_str for k in ("decommission", "not_found", "not found", "not supported", "model")):
            return True
        # On Groq completions, 400 for a model endpoint indicates decommissioned / invalid model
        return True

    if any(k in err_str for k in ("decommission", "model_not_found", "not_found", "not found")):
        return True

    return False


def mark_model_deprecated(
    model_name: str,
    reason: str = "HTTP 400/404: decommissioned or model_not_found",
) -> None:
    """
    Flags and permanently blacklists a model in memory across all accounts and provider loops
    during backend runtime (e.g. for HTTP 400 decommissioned or HTTP 404).
    Once blacklisted, the router never wastes latency checking it on future API calls.
    """
    DEPRECATED_MODELS.add(model_name)
    entry = MODEL_QUOTA_REGISTRY.setdefault(model_name, {})
    entry["is_deprecated"] = True
    entry["is_blacklisted"] = True
    entry["status"] = "DEPRECATED"
    entry["is_exhausted"] = True
    entry["cooldown_until"] = float("inf")  # Permanent in-memory blacklist
    entry["bypass_reason"] = f"PERMANENTLY BLACKLISTED: {reason}"

    # Flag across all statically known account entries
    for acc in GROQ_ACCOUNTS:
        acc_id = acc.get("account_id")
        if acc_id:
            key = f"{acc_id}::{model_name}"
            acc_entry = ACCOUNT_MODEL_QUOTA_REGISTRY.setdefault(key, {})
            acc_entry["is_deprecated"] = True
            acc_entry["is_blacklisted"] = True
            acc_entry["status"] = "DEPRECATED"
            acc_entry["is_exhausted"] = True
            acc_entry["cooldown_until"] = float("inf")
            acc_entry["bypass_reason"] = f"PERMANENTLY BLACKLISTED: {reason}"

    # Also flag in dynamically resolved accounts
    try:
        for acc in get_groq_accounts():
            acc_id = acc.get("account_id")
            if acc_id:
                key = f"{acc_id}::{model_name}"
                acc_entry = ACCOUNT_MODEL_QUOTA_REGISTRY.setdefault(key, {})
                acc_entry["is_deprecated"] = True
                acc_entry["is_blacklisted"] = True
                acc_entry["status"] = "DEPRECATED"
                acc_entry["is_exhausted"] = True
                acc_entry["cooldown_until"] = float("inf")
                acc_entry["bypass_reason"] = f"PERMANENTLY BLACKLISTED: {reason}"
    except Exception:
        pass

    logger.warning(
        "[MODEL BLACKLISTED] Model %s permanently blacklisted in memory (%s). Skipping immediately across all accounts without retrying.",
        model_name, reason
    )
    print(f"[MODEL BLACKLISTED] Model {model_name} permanently blacklisted in memory ({reason}). Skipping immediately across all accounts.")


mark_model_blacklisted = mark_model_deprecated


def is_model_deprecated(model_name: str) -> bool:
    """Returns True if the model has been marked as DEPRECATED or permanently blacklisted."""
    if model_name in DEPRECATED_MODELS:
        return True
    entry = MODEL_QUOTA_REGISTRY.get(model_name)
    if entry and (entry.get("is_deprecated") or entry.get("is_blacklisted") or entry.get("status") == "DEPRECATED"):
        return True
    return False


is_model_blacklisted = is_model_deprecated


def mark_account_model_unavailable(
    account_id: str,
    model_name: str,
    reason: str = "model_not_found",
) -> None:
    """Marks a model as unavailable for a specific account (or permanently blacklisted if 400 decommissioned or 404)."""
    if "404" in str(reason) or "400" in str(reason) or "decommission" in str(reason).lower() or "not_found" in str(reason).lower():
        mark_model_deprecated(model_name, reason=reason)
        return

    key = f"{account_id}::{model_name}"
    entry = ACCOUNT_MODEL_QUOTA_REGISTRY.setdefault(key, {})
    entry["is_exhausted"] = True
    entry["cooldown_until"] = time.time() + 86400.0  # 24-hour cache
    entry["bypass_reason"] = f"Model unavailable on {account_id} ({reason})"
    logger.warning(
        "[MODEL STATUS] Model %s on %s marked unavailable: %s. Proactively bypassing...",
        model_name, account_id, reason
    )


def is_account_model_available(account_id: str, model_name: str, api_key: Optional[str] = None) -> Tuple[bool, str]:
    """
    Checks if an account-model pair is ready for traffic or should be proactively bypassed.
    Returns (True, reason) if healthy, or (False, reason) if exhausted / cooling down / blacklisted.
    """
    if is_model_deprecated(model_name):
        return False, f"Model {model_name} is permanently blacklisted in memory (decommissioned/404)"

    if is_account_rate_limited(account_id, api_key):
        cooldown_ts = max(ACCOUNT_RATE_LIMIT_REGISTRY.get(account_id, 0.0), rate_limited_keys.get(api_key or "", 0.0))
        rem = round(cooldown_ts - time.time(), 1)
        return False, f"{account_id} / API key rate-limited on HTTP 429 ({rem}s cooldown remaining)"

    key = f"{account_id}::{model_name}"
    quota_info = ACCOUNT_MODEL_QUOTA_REGISTRY.get(key)
    now = time.time()

    if quota_info:
        if quota_info.get("is_deprecated") or quota_info.get("is_blacklisted") or quota_info.get("status") == "DEPRECATED":
            return False, f"Model {model_name} is permanently blacklisted in memory on {account_id}"
        cooldown = quota_info.get("cooldown_until", 0.0)
        is_exhausted = quota_info.get("is_exhausted", False)
        if (is_exhausted or cooldown > now) and now < cooldown:
            remaining_wait = round(cooldown - now, 1)
            reason = quota_info.get("bypass_reason", "Quota exhausted / rate-limit cooldown")
            return False, f"{reason} ({remaining_wait}s remaining)"

        if is_exhausted and now >= cooldown:
            quota_info["is_exhausted"] = False
            quota_info["cooldown_until"] = 0.0
            quota_info["bypass_reason"] = None

    return True, "Quota healthy"


def mark_model_unavailable(model_name: str, reason: str = "model_not_found") -> None:
    """
    Marks a model as unavailable.
    If HTTP 404 or HTTP 400 decommissioned is encountered, permanently blacklists the model in memory.
    """
    if "404" in str(reason) or "400" in str(reason) or "decommission" in str(reason).lower() or "not_found" in str(reason).lower():
        mark_model_deprecated(model_name, reason=reason)
        return

    entry = MODEL_QUOTA_REGISTRY.setdefault(model_name, {})
    entry["is_exhausted"] = True
    entry["cooldown_until"] = time.time() + 86400.0  # 24 hour cooldown
    entry["bypass_reason"] = f"Model unavailable on active API key ({reason})"
    logger.warning("[MODEL STATUS] Model %s marked unavailable: %s. Proactively bypassing...", model_name, reason)


def is_model_proactively_available(model_name: str) -> Tuple[bool, str]:
    """
    Checks if a model is ready for traffic or should be proactively bypassed.
    Returns (True, reason) if healthy, or (False, reason) if exhausted / cooling down / blacklisted.
    """
    if is_model_deprecated(model_name):
        return False, f"Model {model_name} is permanently blacklisted in memory (decommissioned/404)"

    quota_info = MODEL_QUOTA_REGISTRY.get(model_name)
    if not quota_info:
        return True, "Ready (no prior usage recorded)"

    if quota_info.get("is_deprecated") or quota_info.get("is_blacklisted") or quota_info.get("status") == "DEPRECATED":
        return False, f"Model {model_name} is permanently blacklisted in memory"

    now = time.time()
    cooldown = quota_info.get("cooldown_until", 0.0)
    is_exhausted = quota_info.get("is_exhausted", False)

    if (is_exhausted or cooldown > now) and now < cooldown:
        remaining_wait = round(cooldown - now, 1)
        reason = quota_info.get("bypass_reason", "Quota exhausted / rate-limit cooldown")
        return False, f"{reason} ({remaining_wait}s remaining)"

    # Cooldown expired: reset exhausted flag
    if is_exhausted and now >= cooldown:
        quota_info["is_exhausted"] = False
        quota_info["cooldown_until"] = 0.0
        quota_info["bypass_reason"] = None

    return True, "Quota healthy"


def get_model_quota_status() -> Dict[str, Any]:
    """Returns a snapshot of the current model quota registry for monitoring."""
    return dict(MODEL_QUOTA_REGISTRY)


# ---------------------------------------------------------------------------
# Streaming Implementations with Output Sanitization & Token Logging
# ---------------------------------------------------------------------------

async def _stream_raw_groq(
    client: groq.AsyncGroq,
    model: str,
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
    account_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Raw generator streaming tokens from Groq API.
    Inspects response headers and logs token usage after stream completion
    without blocking or slowing down text generation.
    """
    messages = build_chat_messages(
        system_prompt=system_prompt,
        prompt=prompt,
        history=history,
        max_history_messages=STRICT_HISTORY_TURNS,
    )

    extra_body: Dict[str, Any] = {"stream_options": {"include_usage": True}}
    # Enable low reasoning effort for reasoning models (e.g. gpt-oss) to prevent token exhaustion
    if any(m_id in model.lower() for m_id in ("gpt-oss", "deepseek-r1", "reasoning")):
        extra_body["reasoning_effort"] = "low"

    call_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "stream": True,
        "extra_body": extra_body,
    }

    # Execute raw request to access HTTP rate limit headers
    try:
        raw_response = await client.chat.completions.with_raw_response.create(**call_kwargs)
    except groq.RateLimitError:
        if account_id:
            mark_account_rate_limited(account_id, 60.0)
            mark_account_model_rate_limited(account_id, model, 60.0)
        raise
    except groq.APIStatusError as ase:
        sc = getattr(ase, "status_code", None)
        if sc == 429:
            if account_id:
                mark_account_rate_limited(account_id, 60.0)
                mark_account_model_rate_limited(account_id, model, 60.0)
        elif is_decommissioned_or_404_error(sc, ase):
            mark_model_deprecated(model, reason=f"HTTP {sc}: {ase}")
        raise
    except Exception as exc:
        sc = getattr(exc, "status_code", None)
        if sc == 429 or "429" in str(exc) or "rate limit" in str(exc).lower():
            if account_id:
                mark_account_rate_limited(account_id, 60.0)
                mark_account_model_rate_limited(account_id, model, 60.0)
        elif is_decommissioned_or_404_error(sc, exc):
            mark_model_deprecated(model, reason=f"HTTP {sc or 'ERR'}: {exc}")
        raise
    response_headers = raw_response.headers

    stream = await raw_response.parse()
    usage_logged = False
    usage_data = None

    # Stream chunks immediately with zero blocking overhead
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            text = delta.content.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            if text:
                yield text

        # Capture usage metadata from the final chunk
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage_data = chunk_usage

    # After stream completes: inspect rate limit headers and log structured metrics
    quota_entry = inspect_and_update_model_quota(model, response_headers, account_id=account_id)

    if usage_data is not None and not usage_logged:
        usage_logged = True
        prompt_tokens = getattr(usage_data, "prompt_tokens", 0)
        completion_tokens = getattr(usage_data, "completion_tokens", 0)
        total_tokens = getattr(usage_data, "total_tokens", 0) or prompt_tokens + completion_tokens
        used_this_turn, session_total_used, percentage_remaining = record_session_token_usage(
            session_id, model, prompt_tokens, completion_tokens
        )
        remaining_tokens = max(0, TOTAL_MODEL_QUOTA - session_total_used)

        if account_id:
            record_account_token_usage(account_id, model, total_tokens)

        acc_tag = f" {account_id} |" if account_id else ""
        log_line = (
            f"[INPUT OPTIMIZATION]{acc_tag} Model: {model} | "
            f"Prompt Tokens: {prompt_tokens} | "
            f"Completion Tokens: {completion_tokens} | "
            f"Total: {total_tokens} | "
            f"Used This Turn: {used_this_turn} | "
            f"Session Total Used: {session_total_used} | "
            f"Remaining Quota: {remaining_tokens}/{TOTAL_MODEL_QUOTA} ({percentage_remaining:.1f}%)"
        )
        logger.info(log_line)
        print(log_line)


async def stream_groq_model(
    client: groq.AsyncGroq,
    model: str,
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
    account_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Streams sanitized response tokens from a single Groq model.
    Passes raw stream through sanitize_stream() to strip meta-commentary and <think> blocks.
    """
    raw_gen = _stream_raw_groq(
        client, model, prompt, system_prompt, history=history, session_id=session_id, account_id=account_id
    )
    async for sanitized_token in sanitize_stream(raw_gen):
        yield sanitized_token


async def _stream_raw_openrouter(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Raw generator streaming tokens from OpenRouter API or OpenAI-compatible endpoint."""
    messages = build_chat_messages(
        system_prompt=system_prompt,
        prompt=prompt,
        history=history,
        max_history_messages=STRICT_HISTORY_TURNS,
    )

    extra_body: Dict[str, Any] = {}
    if any(m_id in model.lower() for m_id in ("gpt-oss", "deepseek-r1", "reasoning")):
        extra_body["reasoning_effort"] = "low"

    call_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_body:
        call_kwargs["extra_body"] = extra_body

    raw_response = await client.chat.completions.with_raw_response.create(**call_kwargs)
    response_headers = raw_response.headers

    stream = raw_response.parse()
    usage_logged = False
    usage_data = None

    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            text = delta.content.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            if text:
                yield text

        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage_data = chunk_usage

    quota_entry = inspect_and_update_model_quota(model, response_headers)

    if usage_data is not None and not usage_logged:
        usage_logged = True
        prompt_tokens = getattr(usage_data, "prompt_tokens", 0)
        completion_tokens = getattr(usage_data, "completion_tokens", 0)
        total_tokens = getattr(usage_data, "total_tokens", 0) or prompt_tokens + completion_tokens
        used_this_turn, session_total_used, percentage_remaining = record_session_token_usage(
            session_id, model, prompt_tokens, completion_tokens
        )
        remaining_tokens = max(0, TOTAL_MODEL_QUOTA - session_total_used)

        log_line = (
            f"[INPUT OPTIMIZATION] Model: {model} | "
            f"Prompt Tokens: {prompt_tokens} | "
            f"Completion Tokens: {completion_tokens} | "
            f"Total: {total_tokens} | "
            f"Used This Turn: {used_this_turn} | "
            f"Session Total Used: {session_total_used} | "
            f"Remaining Quota: {remaining_tokens}/{TOTAL_MODEL_QUOTA} ({percentage_remaining:.1f}%)"
        )
        logger.info(log_line)
        print(log_line)


async def stream_openrouter_model(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Streams sanitized response tokens from OpenRouter fallback tier."""
    raw_gen = _stream_raw_openrouter(
        client, model, prompt, system_prompt, history=history, session_id=session_id
    )
    async for sanitized_token in sanitize_stream(raw_gen):
        yield sanitized_token


async def _stream_raw_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Raw generator streaming tokens from Gemini API."""
    trimmed_history = filter_chat_history(history, max_messages=STRICT_HISTORY_TURNS)
    history_turns = [
        f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content', '')}"
        for msg in trimmed_history
    ]

    full_contents = (
        f"Context:\n" + "\n".join(history_turns) + f"\n\nQuestion: {prompt}"
        if history_turns else prompt
    )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    )

    response_stream = await client.aio.models.generate_content_stream(
        model=model,
        contents=full_contents,
        config=config,
    )

    usage_logged = False
    async for chunk in response_stream:
        if chunk.text:
            yield chunk.text

        usage = getattr(chunk, "usage_metadata", None)
        if usage and not usage_logged:
            usage_logged = True
            prompt_tokens = getattr(usage, "prompt_token_count", 0)
            completion_tokens = getattr(usage, "candidates_token_count", 0)
            total_tokens = getattr(usage, "total_token_count", 0) or prompt_tokens + completion_tokens
            used_this_turn, session_total_used, percentage_remaining = record_session_token_usage(
                session_id, model, prompt_tokens, completion_tokens
            )
            remaining_tokens = max(0, TOTAL_MODEL_QUOTA - session_total_used)
            log_line = (
                f"[INPUT OPTIMIZATION] Model: {model} | "
                f"Prompt Tokens: {prompt_tokens} | "
                f"Completion Tokens: {completion_tokens} | "
                f"Total: {total_tokens} | "
                f"Used This Turn: {used_this_turn} | "
                f"Session Total Used: {session_total_used} | "
                f"Remaining Quota: {remaining_tokens}/{TOTAL_MODEL_QUOTA} ({percentage_remaining:.1f}%)"
            )
            logger.info(log_line)
            print(log_line)


async def stream_gemini_model(
    client: genai.Client,
    model: str,
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Streams sanitized response tokens from Gemini safety net.
    Passes raw stream through sanitize_stream() to strip meta-commentary and <think> blocks.
    """
    raw_gen = _stream_raw_gemini(
        client, model, prompt, system_prompt, history=history, session_id=session_id
    )
    async for sanitized_token in sanitize_stream(raw_gen):
        yield sanitized_token


# ---------------------------------------------------------------------------
# Cascading Router with Proactive Failover & Single-Model Execution Guarantee
# ---------------------------------------------------------------------------

async def stream_cascading_router(
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
    user_natal_context: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Tuple[str, str], None]:
    """
    Cascades sequentially through active Groq models and accounts, OpenRouter tier, and Gemini safety net.
    Multi-Account Rotation:
      For each model in GROQ_MODELS:
        Iterate through available Groq accounts (Account 1, Account 2).
        If Account 1 hits HTTP 429 or quota exhaustion, switch to Account 2 for that same model.
        If all accounts fail for that model, move to next model starting from Account 1.
    As soon as ANY model successfully generates/streams response content:
    - Yields the sanitized token chunks with the active account & model identifier (e.g. 'Account 1 | LLM: model').
    - Logs [SUCCESS] Response generated via <Account X> | Model: <model>
    - Explicitly terminates and returns immediately upon completion.
    """
    if not system_prompt and user_natal_context:
        system_prompt = build_system_prompt(user_natal_context=user_natal_context)
    begin_token_usage_session(session_id)
    accounts = get_groq_accounts()

    # Tier 1: Official Groq Models with Account Failover Cascade Loop
    if accounts:
        for model_idx, model_name in enumerate(MODEL_CASCADE):
            if is_model_deprecated(model_name):
                logger.warning(
                    "[MODEL DEPRECATED] Proactively skipping %s across all accounts without retrying.",
                    model_name
                )
                continue

            skip_model_across_accounts = False
            for acc_idx, acc_info in enumerate(accounts):
                if skip_model_across_accounts or is_model_deprecated(model_name):
                    break

                acc_id = acc_info["account_id"]
                api_key = acc_info["api_key"]
                groq_client = get_groq_client_for_account(api_key)
                if not groq_client:
                    continue

                # 1. Proactive multi-account token threshold check (60-second rolling window)
                current_usage = get_account_model_tokens(acc_id, model_name)
                threshold = TOKEN_THRESHOLDS.get(model_name, 1000)
                if current_usage >= threshold:
                    if acc_idx + 1 < len(accounts):
                        next_acc = accounts[acc_idx + 1]["account_id"]
                        switch_msg = (
                            f"[PROACTIVE SWITCH] Model {model_name} reached {threshold} tokens on {acc_id} -> Switching to {next_acc}"
                        )
                        logger.warning(switch_msg)
                        print(switch_msg)
                    else:
                        logger.warning(
                            "Model %s reached threshold (%d tokens) on %s across all accounts. Moving to next model...",
                            model_name, threshold, acc_id
                        )
                    continue

                # 2. Check provider rate limit or exhaustion cooldown
                is_avail, reason = is_account_model_available(acc_id, model_name, api_key=api_key)
                if not is_avail:
                    if acc_idx + 1 < len(accounts):
                        next_acc = accounts[acc_idx + 1]["account_id"]
                        switch_msg = f"[ACCOUNT SWITCH] {acc_id} exhausted on {model_name} -> Switching to {next_acc}"
                        logger.warning(switch_msg)
                        print(switch_msg)
                    else:
                        logger.warning(
                            "Proactively bypassing %s on %s: %s. Cascading to next model...",
                            acc_id, model_name, reason
                        )
                    continue

                t_start = time.perf_counter()
                tokens_yielded = 0
                active_label = f"{acc_id} | LLM: {model_name}"

                try:
                    async for token in stream_groq_model(
                        groq_client,
                        model_name,
                        prompt,
                        system_prompt,
                        history=history,
                        session_id=session_id,
                        account_id=acc_id,
                    ):
                        tokens_yielded += 1
                        yield token, active_label

                    # [Explicit Return on Success]
                    if tokens_yielded > 0:
                        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                        success_msg = f"[SUCCESS] Response generated via {acc_id} | Model: {model_name}"
                        logger.info(success_msg)
                        print(success_msg)
                        logger.info(
                            "Successfully completed response with Groq %s [%s] (%d chunks in %.1fms)",
                            acc_id, model_name, tokens_yielded, elapsed_ms
                        )
                        return

                except groq.RateLimitError as rle:
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    mark_account_rate_limited(acc_id, 60.0, api_key=api_key)
                    mark_account_model_rate_limited(acc_id, model_name, 60.0)
                    if acc_idx + 1 < len(accounts):
                        next_acc = accounts[acc_idx + 1]["account_id"]
                        switch_msg = f"[ACCOUNT SWITCH] {acc_id} hit HTTP 429 rate limit (cooling down 60s) -> Immediately failing over to {next_acc}"
                        logger.warning(switch_msg)
                        print(switch_msg)
                    else:
                        logger.warning(
                            "All accounts exhausted on Groq model %s (HTTP 429 in %.1fms). Moving to next model...",
                            model_name, elapsed_ms
                        )
                    if tokens_yielded > 0:
                        return
                    continue

                except groq.APIStatusError as ase:
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    status_code = getattr(ase, "status_code", "STATUS_ERR")
                    if status_code == 429:
                        mark_account_rate_limited(acc_id, 60.0, api_key=api_key)
                        mark_account_model_rate_limited(acc_id, model_name, 60.0)
                        if acc_idx + 1 < len(accounts):
                            next_acc = accounts[acc_idx + 1]["account_id"]
                            switch_msg = f"[ACCOUNT SWITCH] {acc_id} hit HTTP 429 rate limit (cooling down 60s) -> Immediately failing over to {next_acc}"
                            logger.warning(switch_msg)
                            print(switch_msg)
                        if tokens_yielded > 0:
                            return
                        continue
                    if is_decommissioned_or_404_error(status_code, ase):
                        mark_model_deprecated(model_name, reason=f"HTTP {status_code}: decommissioned/model_not_found")
                        skip_model_across_accounts = True
                        logger.warning(
                            "[MODEL BLACKLISTED] Groq model %s returned HTTP %s on %s (%.1fms). Permanently blacklisted in memory; skipping immediately across all accounts without retrying.",
                            model_name, status_code, acc_id, elapsed_ms
                        )
                        if tokens_yielded > 0:
                            return
                        break

                    if acc_idx + 1 < len(accounts):
                        next_acc = accounts[acc_idx + 1]["account_id"]
                        logger.warning(
                            "Groq model %s failed on %s (HTTP %s in %.1fms). Trying next account %s...",
                            model_name, acc_id, status_code, elapsed_ms, next_acc
                        )
                    else:
                        logger.warning(
                            "Groq model %s failed across all accounts (HTTP %s in %.1fms). Moving to next model...",
                            model_name, status_code, elapsed_ms
                        )
                    if tokens_yielded > 0:
                        return
                    continue

                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    sc = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                    if is_decommissioned_or_404_error(sc, exc):
                        mark_model_deprecated(model_name, reason=f"HTTP {sc or 'ERR'}: {exc}")
                        skip_model_across_accounts = True
                        logger.warning(
                            "[MODEL BLACKLISTED] Groq model %s failed with decommissioned/404 on %s (%.1fms). Permanently blacklisted in memory; skipping immediately across all accounts without retrying.",
                            model_name, acc_id, elapsed_ms
                        )
                        if tokens_yielded > 0:
                            return
                        break

                    if acc_idx + 1 < len(accounts):
                        next_acc = accounts[acc_idx + 1]["account_id"]
                        logger.warning(
                            "Groq model %s failed on %s in %.1fms (%s). Trying next account %s...",
                            model_name, acc_id, elapsed_ms, exc, next_acc
                        )
                    else:
                        logger.warning(
                            "Groq model %s failed across all accounts in %.1fms (%s). Moving to next model...",
                            model_name, elapsed_ms, exc
                        )
                    if tokens_yielded > 0:
                        return
                    continue

    # Tier 2: OpenRouter Partner Tier (Dedicated fallback tier for OSS models)
    openrouter_client = get_openrouter_client()
    if openrouter_client:
        for idx, model_name in enumerate(OPENROUTER_MODELS):
            is_available, reason = is_model_proactively_available(model_name)
            if not is_available:
                logger.warning(
                    "Proactively bypassing OpenRouter model [%d/%d] %s: %s. Cascading to next model...",
                    idx + 1, len(OPENROUTER_MODELS), model_name, reason
                )
                continue

            t_start = time.perf_counter()
            tokens_yielded = 0
            active_label = f"OpenRouter | LLM: {model_name}"
            try:
                async for token in stream_openrouter_model(
                    openrouter_client,
                    model_name,
                    prompt,
                    system_prompt,
                    history=history,
                    session_id=session_id,
                ):
                    tokens_yielded += 1
                    yield token, active_label

                if tokens_yielded > 0:
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    success_msg = f"[SUCCESS] Response generated via OpenRouter | Model: {model_name}"
                    logger.info(success_msg)
                    print(success_msg)
                    logger.info(
                        "Successfully completed response with OpenRouter [%d/%d]: %s (%d chunks in %.1fms)",
                        idx + 1, len(OPENROUTER_MODELS), model_name, tokens_yielded, elapsed_ms
                    )
                    return

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                logger.warning(
                    "OpenRouter model [%d/%d] %s failed in %.1fms: %s. Cascading...",
                    idx + 1, len(OPENROUTER_MODELS), model_name, elapsed_ms, exc
                )
                if tokens_yielded > 0:
                    return
                continue

    # Tier 3: Secondary provider safety net (Google Gemini)
    logger.warning("Groq and OpenRouter tiers exhausted. Cascading to Gemini safety net...")
    gemini_client = get_gemini_client()
    if gemini_client:
        for idx, gemini_model in enumerate(GEMINI_SAFETY_MODELS):
            t_start = time.perf_counter()
            tokens_yielded = 0
            active_label = f"Gemini | LLM: {gemini_model}"
            try:
                async for token in stream_gemini_model(
                    gemini_client,
                    gemini_model,
                    prompt,
                    system_prompt,
                    history=history,
                    session_id=session_id,
                ):
                    tokens_yielded += 1
                    yield token, active_label

                if tokens_yielded > 0:
                    success_msg = f"[SUCCESS] Response generated via Gemini | Model: {gemini_model}"
                    logger.info(success_msg)
                    print(success_msg)
                    return
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                logger.warning(
                    "Gemini safety model %s failed in %.1fms: %s. Cascading...",
                    gemini_model, elapsed_ms, exc
                )
                if tokens_yielded > 0:
                    return
                continue

    yield SYSTEM_BUSY_MESSAGE, "Backend"


async def stream_chat_response(
    prompt: str,
    user_profile: Optional[Any] = None,
    system_instruction: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    include_suffix: bool = False,
    session_id: Optional[str] = None,
    user_natal_context: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    High-level streaming generator yielding token strings for FastAPI StreamingResponse.
    Guarantees:
      1. Clean output sanitized of internal reasoning, <think> blocks, and meta-preambles.
      2. Exactly one active model completes the response per request.
      3. Dynamic model suffix injection: (Account X | LLM: <active_model>) appended strictly after
         the streaming sanitization completes, ensuring it is never cut off.
    """
    if not user_natal_context and user_profile:
        user_natal_context = get_user_natal_context(user_profile=user_profile)

    system_prompt = system_instruction or build_system_prompt(
        user_profile=user_profile,
        user_natal_context=user_natal_context,
    )
    active_model = None
    last_char = ""

    async for chunk, model_name in stream_cascading_router(
        prompt,
        system_prompt,
        history=history,
        session_id=session_id,
        user_natal_context=user_natal_context,
    ):
        active_model = model_name
        stripped_chunk = chunk.strip()
        if stripped_chunk:
            last_char = stripped_chunk[-1]
        yield chunk

    # Guarantee streamed text ends with a complete sentence
    if last_char and last_char not in (".", "!", "?"):
        yield "."

    if include_suffix and active_model:
        if active_model != "Backend":
            if "LLM:" in active_model or "|" in active_model:
                yield f" ({active_model})"
            else:
                yield f" (LLM: {active_model})"
        else:
            yield " (Backend)"


# ---------------------------------------------------------------------------
# [SPECIFICATION: Non-Streaming Support with Proactive Threshold Logic]
# ---------------------------------------------------------------------------

async def generate_groq_completion(
    client: groq.AsyncGroq,
    model: str,
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    include_suffix: bool = False,
    session_id: Optional[str] = None,
    account_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Non-streaming Groq execution with raw header inspection and proactive threshold evaluation.
    Logs token metrics after completion and updates the quota registry.
    Sanitizes raw response first, then appends UI suffix strictly after sanitization if requested.
    Returns (cleaned_text, usage_metadata).
    """
    begin_token_usage_session(session_id)
    messages = build_chat_messages(
        system_prompt=system_prompt,
        prompt=prompt,
        history=history,
        max_history_messages=STRICT_HISTORY_TURNS,
    )

    extra_body: Dict[str, Any] = {}
    if any(m_id in model.lower() for m_id in ("gpt-oss", "deepseek-r1", "reasoning")):
        extra_body["reasoning_effort"] = "low"

    call_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "stream": False,
    }
    if extra_body:
        call_kwargs["extra_body"] = extra_body

    try:
        raw_response = await client.chat.completions.with_raw_response.create(**call_kwargs)
    except groq.RateLimitError:
        if account_id:
            mark_account_rate_limited(account_id, 60.0)
            mark_account_model_rate_limited(account_id, model, 60.0)
        raise
    except groq.APIStatusError as ase:
        sc = getattr(ase, "status_code", None)
        if sc == 429:
            if account_id:
                mark_account_rate_limited(account_id, 60.0)
                mark_account_model_rate_limited(account_id, model, 60.0)
        elif is_decommissioned_or_404_error(sc, ase):
            mark_model_deprecated(model, reason=f"HTTP {sc}: {ase}")
        raise
    except Exception as exc:
        sc = getattr(exc, "status_code", None)
        if sc == 429 or "429" in str(exc) or "rate limit" in str(exc).lower():
            if account_id:
                mark_account_rate_limited(account_id, 60.0)
                mark_account_model_rate_limited(account_id, model, 60.0)
        elif is_decommissioned_or_404_error(sc, exc):
            mark_model_deprecated(model, reason=f"HTTP {sc or 'ERR'}: {exc}")
        raise
    quota_entry = inspect_and_update_model_quota(model, raw_response.headers, account_id=account_id)
    completion = await raw_response.parse()

    raw_content = ""
    if completion.choices:
        raw_content = completion.choices[0].message.content or ""

    # Response Formatting Sequence: 1. clean_llm_output -> 2. trim_to_last_sentence -> 3. UI suffix
    cleaned_text = format_llm_response(
        raw_content,
        model_id=model,
        account_id=account_id,
        include_suffix=include_suffix,
    )

    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    total_tokens = (getattr(usage, "total_tokens", 0) or prompt_tokens + completion_tokens) if usage else 0

    used_this_turn, session_total_used, percentage_remaining = record_session_token_usage(
        session_id, model, prompt_tokens, completion_tokens
    )
    rem_tokens = max(0, TOTAL_MODEL_QUOTA - session_total_used)

    if account_id:
        record_account_token_usage(account_id, model, total_tokens)

    acc_tag = f" {account_id} |" if account_id else ""
    log_line = (
        f"[INPUT OPTIMIZATION]{acc_tag} Model: {model} | "
        f"Prompt Tokens: {prompt_tokens} | "
        f"Completion Tokens: {completion_tokens} | "
        f"Total: {total_tokens} | "
        f"Used This Turn: {used_this_turn} | "
        f"Session Total Used: {session_total_used} | "
        f"Remaining Quota: {rem_tokens}/{TOTAL_MODEL_QUOTA} ({percentage_remaining:.1f}%)"
    )
    logger.info(log_line)
    print(log_line)

    meta = {
        "model": model,
        "account_id": account_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "remaining_tokens": rem_tokens,
        "limit_tokens": TOTAL_MODEL_QUOTA,
        "remaining_pct": percentage_remaining,
    }
    return cleaned_text, meta


async def generate_openrouter_completion(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    include_suffix: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Non-streaming OpenRouter execution with proactive threshold tracking.
    Sanitizes raw response first, then appends UI suffix strictly after sanitization if requested.
    """
    messages = build_chat_messages(
        system_prompt=system_prompt,
        prompt=prompt,
        history=history,
        max_history_messages=STRICT_HISTORY_TURNS,
    )

    extra_body: Dict[str, Any] = {}
    if any(m_id in model.lower() for m_id in ("gpt-oss", "deepseek-r1", "reasoning")):
        extra_body["reasoning_effort"] = "low"

    call_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "stream": False,
    }
    if extra_body:
        call_kwargs["extra_body"] = extra_body

    raw_response = await client.chat.completions.with_raw_response.create(**call_kwargs)
    quota_entry = inspect_and_update_model_quota(model, raw_response.headers)
    completion = raw_response.parse()

    raw_content = ""
    if completion.choices:
        raw_content = completion.choices[0].message.content or ""

    # Response Formatting Sequence: 1. clean_llm_output -> 2. trim_to_last_sentence -> 3. UI suffix
    cleaned_text = format_llm_response(raw_content, model_id=model, include_suffix=include_suffix)

    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    total_tokens = (getattr(usage, "total_tokens", 0) or prompt_tokens + completion_tokens) if usage else 0

    rem_tokens = quota_entry.get("remaining_tokens", "N/A")
    lim_tokens = quota_entry.get("limit_tokens", "N/A")
    pct_str = (
        f"{quota_entry.get('remaining_pct', 0.0):.1f}%"
        if "remaining_pct" in quota_entry else "N/A"
    )

    log_line = (
        f"[INPUT OPTIMIZATION] Model: {model} | "
        f"Prompt Tokens: {prompt_tokens} | "
        f"Completion Tokens: {completion_tokens} | "
        f"Total: {total_tokens} | "
        f"Provider Remaining Quota: {rem_tokens}/{lim_tokens} ({pct_str})"
    )
    logger.info(log_line)
    print(log_line)

    meta = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "remaining_tokens": rem_tokens,
        "limit_tokens": lim_tokens,
        "remaining_pct": quota_entry.get("remaining_pct"),
    }
    return cleaned_text, meta


async def generate_gemini_completion(
    client: genai.Client,
    model: str,
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    include_suffix: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Non-streaming Gemini execution safety net.
    Sanitizes raw response first, then appends UI suffix strictly after sanitization if requested.
    """
    trimmed_history = filter_chat_history(history, max_messages=STRICT_HISTORY_TURNS)
    history_turns = [
        f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content', '')}"
        for msg in trimmed_history
    ]
    full_contents = (
        f"Context:\n" + "\n".join(history_turns) + f"\n\nQuestion: {prompt}"
        if history_turns else prompt
    )
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    )
    response = await client.aio.models.generate_content(
        model=model,
        contents=full_contents,
        config=config,
    )

    # Response Formatting Sequence: 1. clean_llm_output -> 2. trim_to_last_sentence -> 3. UI suffix
    cleaned_text = format_llm_response(response.text or "", model_id=model, include_suffix=include_suffix)

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
    total_tokens = (getattr(usage, "total_token_count", 0) or prompt_tokens + completion_tokens) if usage else 0

    log_line = (
        f"[INPUT OPTIMIZATION] Model: {model} | "
        f"Prompt Tokens: {prompt_tokens} | "
        f"Completion Tokens: {completion_tokens} | "
        f"Total: {total_tokens}"
    )
    logger.info(log_line)
    print(log_line)

    meta = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    return cleaned_text, meta


async def cascading_chat_completion(
    prompt: str,
    user_profile: Optional[Any] = None,
    system_instruction: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    include_suffix: bool = False,
    session_id: Optional[str] = None,
    user_natal_context: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Non-streaming cascading router.
    Evaluates proactive threshold quota before each model call across:
      Tier 1: Official Groq Models (with multi-account failover loop)
      Tier 2: OpenRouter Models
      Tier 3: Gemini Safety Net
    Returns (response_text, active_model_label, metadata).
    """
    if not user_natal_context and user_profile:
        user_natal_context = get_user_natal_context(user_profile=user_profile)

    system_prompt = system_instruction or build_system_prompt(
        user_profile=user_profile,
        user_natal_context=user_natal_context,
    )

    # Tier 1: Official Groq Models with Account Failover Cascade Loop
    accounts = get_groq_accounts()
    if accounts:
        for idx, model_name in enumerate(MODEL_CASCADE):
            if is_model_deprecated(model_name):
                logger.warning(
                    "[MODEL DEPRECATED] Proactively skipping %s across all accounts without retrying.",
                    model_name
                )
                continue

            skip_model_across_accounts = False
            for acc_idx, acc_info in enumerate(accounts):
                if skip_model_across_accounts or is_model_deprecated(model_name):
                    break

                acc_id = acc_info["account_id"]
                api_key = acc_info["api_key"]
                groq_client = get_groq_client_for_account(api_key)
                if not groq_client:
                    continue

                # 1. Proactive multi-account token threshold check (60-second rolling window)
                current_usage = get_account_model_tokens(acc_id, model_name)
                threshold = TOKEN_THRESHOLDS.get(model_name, 1000)
                if current_usage >= threshold:
                    if acc_idx + 1 < len(accounts):
                        next_acc = accounts[acc_idx + 1]["account_id"]
                        switch_msg = (
                            f"[PROACTIVE SWITCH] Model {model_name} reached {threshold} tokens on {acc_id} -> Switching to {next_acc}"
                        )
                        logger.warning(switch_msg)
                        print(switch_msg)
                    continue

                # 2. Check provider rate limit or exhaustion cooldown
                is_avail, reason = is_account_model_available(acc_id, model_name, api_key=api_key)
                if not is_avail:
                    if acc_idx + 1 < len(accounts):
                        next_acc = accounts[acc_idx + 1]["account_id"]
                        switch_msg = f"[ACCOUNT SWITCH] {acc_id} exhausted on {model_name} -> Switching to {next_acc}"
                        logger.warning(switch_msg)
                        print(switch_msg)
                    continue

                t_start = time.perf_counter()
                try:
                    text, meta = await generate_groq_completion(
                        groq_client,
                        model_name,
                        prompt,
                        system_prompt,
                        history=history,
                        include_suffix=include_suffix,
                        account_id=acc_id,
                    )
                    if text:
                        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                        success_msg = f"[SUCCESS] Response generated via {acc_id} | Model: {model_name}"
                        logger.info(success_msg)
                        print(success_msg)
                        active_label = f"{acc_id} | LLM: {model_name}"
                        return text, active_label, meta
                except groq.RateLimitError:
                    mark_account_rate_limited(acc_id, 60.0, api_key=api_key)
                    mark_account_model_rate_limited(acc_id, model_name, 60.0)
                    if acc_idx + 1 < len(accounts):
                        next_acc = accounts[acc_idx + 1]["account_id"]
                        switch_msg = f"[ACCOUNT SWITCH] {acc_id} hit HTTP 429 rate limit (cooling down 60s) -> Immediately failing over to {next_acc}"
                        logger.warning(switch_msg)
                        print(switch_msg)
                    else:
                        logger.warning(
                            "All accounts exhausted on Groq model %s (HTTP 429). Moving to next model...",
                            model_name
                        )
                    continue
                except groq.APIStatusError as ase:
                    status_code = getattr(ase, "status_code", 400)
                    if status_code == 429:
                        mark_account_rate_limited(acc_id, 60.0, api_key=api_key)
                        mark_account_model_rate_limited(acc_id, model_name, 60.0)
                        if acc_idx + 1 < len(accounts):
                            next_acc = accounts[acc_idx + 1]["account_id"]
                            switch_msg = f"[ACCOUNT SWITCH] {acc_id} hit HTTP 429 rate limit (cooling down 60s) -> Immediately failing over to {next_acc}"
                            logger.warning(switch_msg)
                            print(switch_msg)
                        continue
                    if is_decommissioned_or_404_error(status_code, ase):
                        mark_model_deprecated(model_name, reason=f"HTTP {status_code}: decommissioned/model_not_found")
                        skip_model_across_accounts = True
                        logger.warning(
                            "[MODEL BLACKLISTED] Groq model %s returned HTTP %s on %s. Permanently blacklisted in memory; skipping immediately across all accounts without retrying.",
                            model_name, status_code, acc_id
                        )
                        break
                    continue
                except Exception as exc:
                    sc = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                    if is_decommissioned_or_404_error(sc, exc):
                        mark_model_deprecated(model_name, reason=f"HTTP {sc or 'ERR'}: {exc}")
                        skip_model_across_accounts = True
                        logger.warning(
                            "[MODEL BLACKLISTED] Groq model %s failed with decommissioned/404 on %s. Permanently blacklisted in memory; skipping immediately across all accounts without retrying.",
                            model_name, acc_id
                        )
                        break
                    logger.warning("Groq model %s on %s failed: %s. Cascading...", model_name, acc_id, exc)
                    continue

    # Tier 2: OpenRouter Models
    openrouter_client = get_openrouter_client()
    if openrouter_client:
        for idx, model_name in enumerate(OPENROUTER_MODELS):
            is_available, reason = is_model_proactively_available(model_name)
            if not is_available:
                logger.warning(
                    "Proactively bypassing OpenRouter model [%d/%d] %s: %s. Cascading to next model...",
                    idx + 1, len(OPENROUTER_MODELS), model_name, reason
                )
                continue

            t_start = time.perf_counter()
            try:
                text, meta = await generate_openrouter_completion(
                    openrouter_client,
                    model_name,
                    prompt,
                    system_prompt,
                    history=history,
                    include_suffix=include_suffix,
                )
                if text:
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    logger.info(
                        "Successfully completed non-streaming response with OpenRouter [%d/%d]: %s (in %.1fms)",
                        idx + 1, len(OPENROUTER_MODELS), model_name, elapsed_ms
                    )
                    return text, model_name, meta
            except Exception as exc:
                logger.warning("OpenRouter model %s non-streaming failed: %s. Cascading...", model_name, exc)
                continue

    # Tier 3: Fallback to Gemini safety net
    gemini_client = get_gemini_client()
    if gemini_client:
        for gemini_model in GEMINI_SAFETY_MODELS:
            try:
                text, meta = await generate_gemini_completion(
                    gemini_client,
                    gemini_model,
                    prompt,
                    system_prompt,
                    history=history,
                    include_suffix=include_suffix,
                )
                if text:
                    return text, gemini_model, meta
            except Exception as exc:
                logger.warning("Gemini safety model %s non-streaming failed: %s", gemini_model, exc)
                continue

    return SYSTEM_BUSY_MESSAGE, "Backend", dict(SYSTEM_BUSY_PAYLOAD)
