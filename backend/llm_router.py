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
# Base URL for Groq's OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_CASCADE: List[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
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
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]

# Output Token Cap: max_tokens=140 calibrates response length strictly for 5-7 lines, 50-65 words
MAX_TOKENS: int = 140
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
    final_sun = _extract_val(["sun_sign", "sun", "surya"], "")
    raw_dasha = active_dasha or _extract_val(["dasha", "active_dasha", "current_dasha"], "Rahu-Mars")
    if isinstance(raw_dasha, dict):
        raw_dasha = raw_dasha.get("current_dasha") or raw_dasha.get("major") or "Rahu-Mars"
    final_dasha = str(raw_dasha).replace("/", "-").strip()

    name = _extract_val(["name", "user_name", "first_name"], "").strip()
    if name.lower() in ("guest", "there", "friend"):
        name = ""

    birth_place = _extract_val(["birth_place", "place", "city", "birth_city", "location"], "").strip()
    birth_details = _extract_val(["birth_details", "dob", "birth_date_time", "birth_datetime"], "").strip()

    rulerships = calculate_house_rulerships(final_asc)
    return {
        "name": name,
        "birth_place": birth_place,
        "birth_details": birth_details,
        "ascendant": final_asc,
        "moon_sign": final_moon,
        "sun_sign": final_sun,
        "active_dasha": final_dasha,
        "house_rulerships": rulerships,
    }


def format_user_intro(user_natal_context: Dict[str, Any], hinglish: bool = False) -> str:
    """
    Builds personalized greeting and birth grounding prefix according to user name,
    birth details (date & time), and birth place.
    Example:
      English: 'Rahul, based on your birth chart (15 May 1995, 14:30 | Delhi),'
      Hinglish: 'Rahul ji, aapki Janam Kundali (15 May 1995, 14:30 | Delhi) ke anusaar'
    """
    if not isinstance(user_natal_context, dict):
        return "Aapki Janam Kundali ke anusaar" if hinglish else "Based on your sidereal birth chart calculations,"

    name = (user_natal_context.get("name") or "").strip()
    if name.lower() in ("guest", "there", "friend"):
        name = ""

    birth_details = (user_natal_context.get("birth_details") or "").strip()
    birth_place = (user_natal_context.get("birth_place") or "").strip()

    meta_parts = []
    if birth_details:
        meta_parts.append(birth_details)
    if birth_place:
        meta_parts.append(birth_place)
    meta_str = " | ".join(meta_parts)

    if hinglish:
        if name and meta_str:
            return f"{name} ji, aapki Janam Kundali ({meta_str}) ke anusaar"
        elif name:
            return f"{name} ji, aapki Janam Kundali ke anusaar"
        elif meta_str:
            return f"Aapki Janam Kundali ({meta_str}) ke anusaar"
        return "Aapki Janam Kundali ke shuddh sidereal ganit ke anusaar"
    else:
        if name and meta_str:
            return f"{name}, based on your birth chart ({meta_str}),"
        elif name:
            return f"{name}, based on your sidereal birth chart calculations,"
        elif meta_str:
            return f"Based on your birth chart ({meta_str}),"
        return "Based on your sidereal birth chart calculations,"


# ---------------------------------------------------------------------------
# Step 2: Strict System Prompt Guardrails & Length Constraints
# ---------------------------------------------------------------------------
DEFAULT_NATAL_CONTEXT: Dict[str, Any] = {
    "ascendant": "Libra",
    "moon_sign": "Gemini",
    "sun_sign": "Virgo",
    "active_dasha": "Rahu-Mars",
    "house_rulerships": calculate_house_rulerships("Libra"),
}

# ---------------------------------------------------------------------------
# Robust Hinglish & English Language Detection Engine
# ---------------------------------------------------------------------------
ENGLISH_STOPWORDS = {
    # Question words
    "what", "when", "where", "why", "how", "who", "which", "whose", "whom",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "myy", "mine", "your", "yours", "his", "hers", "its", "our", "ours", "their", "theirs",
    # Slang & Typos for English pronouns/words
    "que", "ques", "question", "ur", "u", "r", "aso", "also",
    # Auxiliaries & common verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "will", "shall", "would", "should", "can", "could", "may", "might", "must",
    "do", "does", "did", "done", "have", "has", "had",
    "tell", "give", "show", "explain", "check", "know", "want", "need", "please",
    "suit", "suits", "favor", "favors", "get", "help", "suggest", "predict",
    # Prepositions & Conjunctions
    "the", "a", "an", "and", "or", "but", "if", "then", "so",
    "in", "on", "at", "to", "for", "with", "about", "of", "from", "by", "as",
    "into", "like", "through", "after", "over", "between", "out", "against",
    "during", "without", "before", "under", "around", "among",
    "any", "some", "all", "both", "each", "every",
    # English astrological terms (distinct from Sanskrit)
    "sign", "sun", "moon", "chart", "horoscope", "ascendant"
}

# Strictly Hindi / Hinglish grammatical markers (auxiliary verbs, pronouns, question words, particles)
# Excludes Sanskrit astrological terms (dasha, lagna, rashi, kundali, etc.) that are used in English queries
TRUE_HINGLISH_MARKERS = {
    # Pronouns & Possessives
    "aap", "aapka", "aapki", "aapke", "tum", "tumhara", "tumhari", "tumhare",
    "tera", "teri", "tere", "mera", "meri", "mere", "mujhe", "mujhko",
    "hum", "hume", "humein", "humko", "apna", "apni", "apne",
    "kiska", "kiski", "kiske", "kisne", "iska", "iski", "iske", "unka", "unki", "unke",
    # Question words
    "kya", "kab", "kyun", "kyu", "kaise", "kaisa", "kaisi", "kitna", "kitni", "kitne",
    "kahan", "kaha", "kidhar", "kaun", "kaunsa", "kaunsi", "kaunse", "kisliye",
    # Postpositions & Particles
    "ka", "ki", "ke", "ko", "mein", "me", "se", "liye", "bhi", "par", "pe",
    "aur", "ya", "toh", "to", "wala", "wali", "wale",
    # Verbs, Auxiliaries & Tense markers
    "hai", "hain", "ho", "hoon", "hun", "tha", "thi", "thay",
    "hoga", "hogi", "hoge", "honge", "hona", "hoti", "hota", "hote", "hua", "hui", "hue",
    "raha", "rahi", "rahe", "rahega", "rahegi", "rahenge",
    "karo", "karna", "karne", "karta", "karti", "karte", "kar", "kijiye", "kiya", "kiye", "kare", "karega", "karegi", "karenge",
    "batao", "bataiye", "bataye", "batana", "batayein",
    "milega", "milegi", "milege", "milenge", "milna", "milne", "milta", "milti", "milte", "mila",
    "chahiye", "chahta", "chahti", "chahte",
    "sakta", "sakti", "sakte", "sakenge", "sakungi",
    "aayega", "aayegi", "aayenge", "aana", "aata", "aati", "aaya", "aayi", "aaye",
    "jayega", "jayegi", "jayenge", "jata", "jati", "jate", "gaya", "gayi", "gaye",
    "badhega", "badhegi", "badhenge", "lagta", "lagti", "lagte", "lagega", "lagegi",
    "chalega", "chalegi", "chalenge",
    # Particles, adverbs, and conversational fillers
    "nahi", "nahin", "bohot", "bahut", "zyada", "jyada", "thoda", "thodi", "thode",
    "achha", "accha", "achhi", "acchi", "theek", "thik", "bilkul", "pakka",
    "abhi", "aaj", "kal", "parso", "kabhi", "hamesha", "pehle", "baad",
    "lekin", "magar", "kyunki", "kyuki", "yadi", "agar",
    # Hindi-specific inquiry terms (distinct from Sanskrit chart parameters)
    "naukri", "nokri", "kaam", "dhandha", "vyapar", "paisa", "paise",
    "shaadi", "shadi", "vivah", "rishta", "pyaar", "prem", "sehat", "padhai"
}

# Alias for backward compatibility
HINGLISH_WORDS = TRUE_HINGLISH_MARKERS


def is_hinglish_query(text: str) -> bool:
    """
    Detects whether a query is written in Hinglish (Hindi in Roman script) or Hindi (Devanagari).
    Returns True if Hinglish/Hindi syntax is detected, False for English queries.
    Never misclassifies English questions that merely contain Sanskrit/astrology terms
    like 'dasha', 'lagna', 'rashi', 'kundali', etc.
    """
    if not text or not str(text).strip():
        return False

    raw_text = str(text)

    # 1. Any Devanagari Hindi characters -> definitely Hindi
    if re.search(r"[\u0900-\u097F]", raw_text):
        return True

    words = set(re.findall(r"\b[a-zA-Z']+\b", raw_text.lower()))
    if not words:
        return False

    eng_matches = len(words & ENGLISH_STOPWORDS)
    hing_matches = len(words & TRUE_HINGLISH_MARKERS)

    # If NO Hinglish grammar markers were found at all -> strictly English
    if hing_matches == 0:
        return False

    # If Hinglish markers are found:
    # Check balance between English grammatical structure and Hinglish markers
    if eng_matches >= hing_matches:
        strong_eng_grammar = words & {
            "what", "when", "where", "why", "how", "who", "which",
            "is", "am", "are", "was", "were", "will", "would", "should", "can", "could", "does", "did"
        }
        strong_hing_grammar = words & {
            "kya", "kab", "kaise", "kaisa", "kaisi", "kyun", "kyu", "kidhar", "kaun",
            "hoga", "hogi", "honge", "hai", "hain", "milega", "milegi", "batao", "bataiye"
        }
        if strong_eng_grammar and not strong_hing_grammar:
            return False
        if strong_hing_grammar and not strong_eng_grammar:
            return True
        if len(strong_eng_grammar) >= len(strong_hing_grammar):
            return False
        return True

    return True


SYSTEM_PROMPT = """You are a direct, empathetic, and accurate Vedic Astrology assistant.

### STRICT OPERATING RULES:
1. MANDATORY 2-PARAGRAPH LAYOUT:
   Your response must strictly consist of exactly two paragraphs with the following bold headings:
   **Astrological Insight:** (2–3 sentences analyzing the user's birth chart, Lagna, houses, planets, or active Dasha influences directly answering their query).

   **Actionable Advice:** (2 sentences providing concrete remedies, timing windows, or grounded guidance).

2. OUTPUT BUDGET CONSTRAINT:
   Keep total length strictly within 5–7 lines and 50–65 words. Ensure all thoughts finish completely within the max_tokens limit.
3. NO FILLER OR ANALOGIES:
   No analogies, metaphors, conversational filler ("Sure! Here is your reading"), or bullet lists.
4. EXACT SUB-QUESTION MANDATE:
   Address the user's exact query directly. Never provide generic, repetitive domain fluff.
5. COMPLETENESS & PUNCTUATION:
   Every sentence must be complete and end with terminal punctuation ('.', '!', '?'). Never cut off mid-sentence.
6. LANGUAGE REGISTER:
   If the user asked in Hinglish, respond ONLY in conversational Hinglish (Roman script) keeping the standard bold headers. If in English, respond STRICTLY in clear English.
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
# Per-Session Last-Call Token Stats Registry
# Populated after every LLM stream/completion so endpoints can surface
# token usage in the streaming response ([TOKEN_USAGE] JSON line).
# ---------------------------------------------------------------------------
_LAST_TOKEN_STATS: Dict[str, Dict[str, Any]] = {}
_LAST_TOKEN_STATS_LOCK = threading.Lock()

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

# ---------------------------------------------------------------------------
# State Tracking for Local DB Templates in Active Conversations
# ---------------------------------------------------------------------------
SERVED_LOCAL_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {}
SERVED_TEMPLATES_LOCK = threading.Lock()
RECENT_TEMPLATE_WINDOW: int = 5  # Template served within last 5 turns will force LLM fallback


def record_served_template(session_id: Optional[str], template_key: str) -> None:
    """Record that a specific local DB template was served in the active conversation."""
    if not session_id or not template_key:
        return
    with SERVED_TEMPLATES_LOCK:
        records = SERVED_LOCAL_TEMPLATES.setdefault(str(session_id), [])
        records.append({
            "template_key": template_key.lower().strip(),
            "timestamp": time.time(),
        })
        if len(records) > 30:
            SERVED_LOCAL_TEMPLATES[str(session_id)] = records[-30:]


def is_template_recently_served(
    session_id: Optional[str],
    template_key: str,
    recent_limit: int = 5,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Check if a specific local DB template was recently served in the active conversation.
    Checks both session-level tracking and active chat_history messages.
    If it was served, returning True forces fallback to the LLM pipeline instead of repeating.
    """
    if not template_key:
        return False

    clean_target = template_key.lower().strip()

    # 1. Check in-memory session tracking
    if session_id:
        with SERVED_TEMPLATES_LOCK:
            records = SERVED_LOCAL_TEMPLATES.get(str(session_id), [])
            window = records[-recent_limit:] if recent_limit > 0 else records
            for entry in window:
                if entry.get("template_key", "") == clean_target:
                    return True

    # 2. Check conversation history messages
    if chat_history:
        for msg in chat_history:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = str(msg.get("content", "")).lower()
                target_phrase = clean_target.replace("_", " ")
                if target_phrase in content or f"**{clean_target}" in content:
                    return True

    return False


def clear_served_templates(session_id: Optional[str] = None) -> None:
    """Clear served template tracking for a session or globally."""
    with SERVED_TEMPLATES_LOCK:
        if session_id:
            SERVED_LOCAL_TEMPLATES.pop(str(session_id), None)
        else:
            SERVED_LOCAL_TEMPLATES.clear()


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
                        clean_k = k.lower().strip("?.! ").replace("_", " ")
                        raw_k = k.lower().strip("?.! ")
                        if isinstance(v, dict):
                            # Filter out broad ambiguous keywords like "career", "job", "work"
                            raw_keywords = [
                                str(kw).lower().strip("?.! ") for kw in v.get("keywords", [])
                                if kw and str(kw).lower().strip("?.! ") not in ("career", "job", "work")
                            ]
                            targets = [clean_k, raw_k] + raw_keywords
                            for t in targets:
                                if not t or len(t) < 3:
                                    continue
                                qa_database_raw[t] = v
                                qa_database_index[t] = v
                                qa_database_index[f"what is {t}"] = v
                                qa_database_index[f"what is a {t}"] = v
                                qa_database_index[f"what is an {t}"] = v
                                qa_database_index[f"what is my {t}"] = v
                                qa_database_index[f"tell me about {t}"] = v
                                qa_database_index[f"tell me about my {t}"] = v
                                qa_database_index[f"how is my {t}"] = v
                                qa_database_index[f"explain {t}"] = v
                                qa_database_index[f"meaning of {t}"] = v
                                qa_database_index[f"{t} meaning"] = v
                                qa_database_index[f"{t} in astrology"] = v
                                qa_database_index[f"{t} guidance"] = v
                                qa_database_index[f"{t} prediction"] = v
                                qa_database_index[f"how does {t} work"] = v
                            # Always register the exact key in raw storage
                            qa_database_raw[k] = v
                        elif isinstance(v, str):
                            targets = [clean_k, raw_k]
                            for t in targets:
                                if not t or len(t) < 3:
                                    continue
                                qa_database_index[t] = v
                                qa_database_index[f"what is {t}"] = v
                                qa_database_index[f"what is a {t}"] = v
                                qa_database_index[f"what is an {t}"] = v
                                qa_database_index[f"tell me about {t}"] = v
                                qa_database_index[f"explain {t}"] = v
                            qa_database_raw[k] = {"template": v}
                    break
            except Exception as exc:
                logger.warning("Error loading qa_database_index from %s: %s", path, exc)
    return qa_database_index


_init_qa_database_index()

# ---------------------------------------------------------------------------
# Strict Query Classification Middleware & Granular Sub-Intent Matcher
# ---------------------------------------------------------------------------
PERSONAL_INTENT_PATTERNS = (
    r"\bi\s+have\b", r"\bi\s+am\b", r"\bam\s+i\b",
    r"\bplaced\s+in\b", r"\bsitting\s+in\b", r"\baspecting\b", r"\baspected\b",
    r"\bconjunction\b", r"\bcombust\b", r"\bexalted\b", r"\bdebilitated\b",
    r"\bhouse\s+lord\s+in\b", r"\blord\s+in\b", r"\bruler\s+in\b",
    r"\bnext\s+\d+\s+(?:months?|years?|weeks?|days?)\b", r"\bnext\s+(?:month|year|week)\b",
    r"\bwill\s+i\s+pass\b", r"\bwill\s+i\s+get\s+divorce\b", r"\bcourt\s+case\b",
    r"\bin\s+\d{4}\b",
)


def match_sub_intent(query: str, db: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, dict]]:
    """
    Evaluates queries against granular sub-intents (e.g. career_gemstones, career_timing_promotion,
    business_vs_job, career_general) using keyword and semantic markers.
    Prevents broad domain matching from hijacking specific sub-questions.
    Returns (template_key, template_dict) or None.
    """
    if not query or not query.strip():
        return None

    clean_lower = re.sub(r'[\'\".,?!;:()\[\]{}_+\-=/\\<>]', ' ', query.lower())
    clean_lower = " ".join(clean_lower.split())
    if not clean_lower:
        return None

    raw_db = db or qa_database_raw

    # 1. Career Gemstones (Gemstones/Ratna + Career/Job/Work context)
    gemstone_terms = (
        r"\bgemstones?\b", r"\bratna\b", r"\bstone\b", r"\bstones\b",
        r"\bruby\b", r"\bemerald\b", r"\bsapphire\b", r"\bpukhraj\b",
        r"\bneelam\b", r"\bmanik\b", r"\bpearl\b", r"\bmoti\b",
        r"\bdiamond\b", r"\bheera\b", r"\bgomed\b", r"\bcat'?s\s+eye\b"
    )
    has_gemstone = any(re.search(pat, clean_lower) for pat in gemstone_terms)
    career_context = any(re.search(pat, clean_lower) for pat in (
        r"\bcareer\b", r"\bjob\b", r"\bprofession\b", r"\bwork\b", r"\bbusiness\b",
        r"\bnaukri\b", r"\bgrowth\b", r"\bsuit\b", r"\bsuits\b", r"\blucky\b"
    ))
    if has_gemstone and career_context:
        if "career_gemstones" in raw_db and isinstance(raw_db["career_gemstones"], dict):
            return "career_gemstones", raw_db["career_gemstones"]

    # 2. Career Timing & Promotion (Promotion, Appraisal, Hike)
    promotion_terms = (
        r"\bpromotions?\b", r"\bpromoted\b", r"\bappraisals?\b",
        r"\bhike\b", r"\bincrements?\b", r"\belevation\b",
        r"\bwhen\s+will\s+i\s+get\s+a\s+promotion\b", r"\bpromotion\s+timing\b"
    )
    has_promotion = any(re.search(pat, clean_lower) for pat in promotion_terms)
    if has_promotion:
        if "career_timing_promotion" in raw_db and isinstance(raw_db["career_timing_promotion"], dict):
            return "career_timing_promotion", raw_db["career_timing_promotion"]

    # 3. Business vs Job Comparison
    business_job_patterns = (
        r"\bbusiness\s+(?:or|vs|ya)\s+job\b",
        r"\bjob\s+(?:or|vs|ya)\s+business\b",
        r"\bdoes\s+my\s+chart\s+favor\s+business\s+or\s+(?:a\s+)?job\b",
        r"\bfavor\s+business\s+or\s+(?:a\s+)?job\b",
        r"\bfavor\s+job\s+or\s+business\b",
        r"\bbusiness\s+or\s+(?:a\s+)?job\b",
        r"\bjob\s+or\s+business\b",
        r"\bshould\s+i\s+do\s+business\s+or\s+(?:a\s+)?job\b",
        r"\bstart\s+a\s+business\s+or\s+(?:do\s+a\s+)?job\b",
        r"\bentrepreneurship\s+vs\s+(?:employment|job)\b",
        r"\bstartup\s+vs\s+(?:corporate\s+)?job\b",
        r"\bjob\s+or\s+startup\b",
        r"\bservice\s+or\s+business\b"
    )
    if any(re.search(pat, clean_lower) for pat in business_job_patterns):
        if "business_vs_job" in raw_db and isinstance(raw_db["business_vs_job"], dict):
            return "business_vs_job", raw_db["business_vs_job"]

    # 4. Career General (Direction, suitability, choice - only when NOT gemstones, promotion, or business/job)
    career_general_patterns = (
        r"\bwhich\s+career\s+suits\s+me\s+best\b",
        r"\bwhich\s+career\s+suits\s+me\b",
        r"\bwhat\s+career\s+suits\s+me\b",
        r"\bbest\s+career\s+for\s+me\b",
        r"\bcareer\s+suitability\b",
        r"\bwhat\s+career\s+should\s+i\s+choose\b",
        r"\bideal\s+career\s+path\b",
        r"\bwhich\s+profession\s+suits\s+me\b",
        r"\bbest\s+profession\s+for\s+me\b",
        r"\bcareer\s+direction\b",
        r"\bcareer\s+choice\b"
    )
    if any(re.search(pat, clean_lower) for pat in career_general_patterns):
        if "career_general" in raw_db and isinstance(raw_db["career_general"], dict):
            return "career_general", raw_db["career_general"]

    # 5. Check Other Predefined Structural Entries in raw_db
    for key, entry in raw_db.items():
        if key in ("career_general", "career_gemstones", "career_timing_promotion", "business_vs_job"):
            continue
        if not isinstance(entry, dict):
            continue
        keywords = entry.get("keywords", [])
        for kw in keywords:
            kw_clean = str(kw).lower().strip()
            if not kw_clean or len(kw_clean) < 3 or kw_clean in ("career", "job", "work"):
                continue
            if re.search(rf"\b{re.escape(kw_clean)}\b", clean_lower):
                return key, entry

    return None


COMPOUND_CONCEPT_CATEGORIES = {
    "moon_sign": [r"\bmoon\s*sign\b", r"\brashi\b", r"\bjanma\s*rashi\b", r"\bchandra\s*rashi\b"],
    "sun_sign": [r"\bsun\s*sign\b", r"\bsurya\s*rashi\b", r"\bsun\s*in\s*astrology\b"],
    "lagna": [r"\blagna\b", r"\bascendant\b", r"\brising\s*sign\b"],
    "dasha": [r"\bdasha\b", r"\bmahadasha\b", r"\bantardasha\b", r"\bvimshottari\b"],
    "nakshatra": [r"\bnakshatra\b", r"\bbirth\s*star\b", r"\blunar\s*mansion\b"],
    "career": [r"\bcareer\b", r"\bjob\b", r"\bprofession\b", r"\bwork\b", r"\bpromotion\b", r"\bbusiness\b", r"\bnaukri\b", r"\bvyapar\b"],
    "marriage": [r"\bmarriage\b", r"\bshaadi\b", r"\bshadi\b", r"\bvivah\b", r"\bspouse\b", r"\blife\s*partner\b", r"\brelationship\b", r"\blove\s*life\b"],
    "wealth": [r"\bwealth\b", r"\bmoney\b", r"\bfinance\b", r"\bfinancial\b", r"\bsavings\b", r"\bpaisa\b", r"\bdhan\b"],
    "health": [r"\bhealth\b", r"\bsehat\b", r"\bdisease\b", r"\billness\b", r"\bfitness\b"],
    "education": [r"\beducation\b", r"\bstudies\b", r"\bstudy\b", r"\bexam\b", r"\bcollege\b", r"\bpadhai\b"],
    "travel": [r"\btravel\b", r"\bforeign\b", r"\bvidesh\b", r"\bvisa\b", r"\babroad\b"],
    "children": [r"\bchildren\b", r"\bbaby\b", r"\bchild\b", r"\bpregnancy\b", r"\bsantaan\b", r"\bbacche\b"],
    "gemstones": [r"\bgemstone\b", r"\bgemstones\b", r"\bratna\b"],
}


def is_compound_query(query: str) -> bool:
    """
    Detects whether a user prompt is a compound / multi-part question asking about
    multiple distinct topics or natal chart placements (e.g. 'moon sign, sun sign and aso dasha',
    'career and marriage', 'health and wealth', multiple questions separated by conjunctions/punctuation).
    Compound questions can NEVER be answered accurately by a single static template,
    and must always bypass local QA to be fully answered by the LLM pipeline.
    """
    if not query or not str(query).strip():
        return False

    lower = str(query).lower().strip()

    # 1. Check if 2 or more distinct concept categories are mentioned
    matched_cats = set()
    for cat_name, patterns in COMPOUND_CONCEPT_CATEGORIES.items():
        if any(re.search(pat, lower) for pat in patterns):
            matched_cats.add(cat_name)

    # Specific non-compound pairing: "which gemstones suit my career" -> single sub-intent
    if matched_cats == {"career", "gemstones"}:
        return False

    if len(matched_cats) >= 2:
        return True

    # 2. Coordinating conjunctions linking multiple inquiries
    multi_inquiry_patterns = (
        r"\b(?:and\s+(?:also|aso|too|as\s+well)|as\s+well\s+as|along\s+with|in\s+addition\s+to)\b",
        r"\b(?:what|kya)\s+is\s+my\b.*\b(?:and|aur|also|aso)\b",
        r"\b(?:tell\s+me\s+about|batao)\b.*\b(?:and|aur|also|aso)\b",
        r"\b(?:both|dono)\b",
        r"\?.*\?",
        r"\b(what|when|how|which|kya|kab|kaise)\b.*\b(and|aur)\b.*\b(what|when|how|which|kya|kab|kaise|also|aso)\b",
    )
    if any(re.search(pat, lower) for pat in multi_inquiry_patterns):
        return True

    return False


def should_bypass_local_qa(query: str, session_id: Optional[str] = None) -> bool:
    """
    Returns True if a query cannot or should not be answered by local predefined QA/FAQ
    because:
    1. It is a compound / multi-part question asking for multiple chart parameters/topics.
    2. It is a complex, multi-variable, predictive transit question.
    3. It matches a granular sub-intent that was ALREADY served in this active session (state tracking).
    4. It does not match any indexed concept/sub-intent in the local DB.
    """
    if not query or not query.strip():
        return False

    # 0. Direct calculation queries (moon sign, sun sign, lagna, dasha, planetary positions)
    # MUST NEVER bypass to LLM - they are computed directly from API/ephemeris data (0 token cost, <10ms).
    if is_direct_calculation_query(query):
        return False

    # 0b. Marriage queries (granular micro-intent check: partner_traits, marriage_remedies, marriage_timing)
    m_intent = classify_marriage_intent(query)
    if m_intent:
        if session_id and is_template_recently_served(session_id, m_intent):
            return True
        return False
    elif is_marriage_query(query):
        if session_id and is_template_recently_served(session_id, "marriage_and_relationships"):
            return True
        return False

    # 1. Compound / multi-part questions MUST ALWAYS bypass local QA to LLM
    if is_compound_query(query):
        return True

    lower = query.lower().strip()

    # 2. Complex transit or specific conjunction queries must always bypass to LLM
    complex_transit_markers = (
        r"\baspecting\b", r"\bcombust\b", r"\bexalted\b", r"\bdebilitated\b",
        r"\bconjunction\b", r"\bnavamsha\b", r"\bd9\b", r"\bd10\b",
        r"\bwill\s+i\s+pass\b", r"\bwill\s+i\s+get\s+divorce\b",
        r"\bcourt\s+case\b", r"\bin\s+\d{4}\b"
    )
    if any(re.search(pat, lower) for pat in complex_transit_markers):
        return True

    for pattern in PERSONAL_INTENT_PATTERNS:
        if re.search(pattern, lower):
            return True

    # 3. Check if this query matches a granular sub-intent
    match_result = match_sub_intent(query, qa_database_raw)
    if match_result:
        template_key, _ = match_result
        # If already served in this session, force bypass to LLM
        if session_id and is_template_recently_served(session_id, template_key):
            return True
        return False

    # 4. Predefined basic astrological terminology concepts (never bare 'career' or 'job')
    predefined_concept_keywords = (
        "dasha", "mahadasha", "antardasha", "sade sati", "lagna", "rashi",
        "1st house", "2nd house", "3rd house", "4th house", "5th house",
        "6th house", "7th house", "8th house", "9th house", "10th house",
        "11th house", "12th house", "sun", "surya", "moon", "chandra",
        "mars", "mangal", "mercury", "budh", "jupiter", "guru", "venus",
        "shukra", "saturn", "shani", "rahu", "ketu", "nakshatra", "manglik",
        "dosha", "hello", "hi", "namaste", "hey", "who are you"
    )
    for kw in predefined_concept_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", lower):
            return False

    return True


# ---------------------------------------------------------------------------
# Direct-API Calculation Path & Dynamic Source Badging
# ---------------------------------------------------------------------------
DIRECT_CALCULATION_PATTERNS = [
    r"\b(?:moon\s*sign|chandra\s*rashi|rashi)\b",
    r"\b(?:sun\s*sign|surya\s*rashi|surya)\b",
    r"\b(?:lagna|ascendant|rising\s*sign)\b",
    r"\b(?:current\s*dasha|active\s*dasha|my\s*dasha|which\s*dasha|dasha|mahadasha|antardasha|vimshottari)\b",
    r"\b(?:planetary\s*positions?|planet\s*positions?|planetary\s*placements?|graha\s*sthiti|planets?\s*(?:in|status|placement))\b",
]

DIRECT_CALCULATION_EXCLUSIONS = [
    r"\b(?:gemstone|ratna|stone|when\s+will|promotion|job\s*change|marriage\s*timing|career\s*suit|business\s*or\s*job)\b",
    r"\b(?:how\s+to|remedy|upay|puja|totke|mantra|delay|problem|conflict)\b",
]


def is_direct_calculation_query(query: str) -> bool:
    """
    Checks if query is asking directly about fundamental birth chart parameters:
    moon sign, sun sign, rashi, lagna, ascendant, dasha, mahadasha, or planetary positions.
    Such queries bypass LLM calls completely (0 token cost, <10ms response).
    """
    if not query or not str(query).strip():
        return False
    clean_q = str(query).lower().strip()
    if any(re.search(pat, clean_q) for pat in DIRECT_CALCULATION_EXCLUSIONS):
        return False
    return any(re.search(pat, clean_q) for pat in DIRECT_CALCULATION_PATTERNS)


MARRIAGE_PATTERNS = [
    r"\b(?:shadi|shaadi|vivah|marriage|married|wedding|spouse|partner|husband|wife|soulmate)\b",
    r"\b(?:meri\s+shadi\s+kab\s+hogi|shadi\s+kab\s+hogi|shaadi\s+kab\s+hogi|kab\s+hogi\s+meri\s+shadi)\b",
    r"\b(?:when\s+will\s+i\s+get\s+married|marriage\s+timing|marriage\s+kab\s+hogi)\b",
    r"\b(?:love\s+marriage|arranged\s+marriage|life\s+partner)\b",
    r"\b(?:7th\s+house|seventh\s+house|yuvati\s+bhava?)\b",
]


def is_marriage_query(query: str) -> bool:
    """
    Checks if query belongs to marriage & relationship domain (English or Hinglish).
    Maps queries like 'shadi', 'meri shadi kab hogi', 'vivah', 'marriage' to the marriage domain.
    """
    if not query or not str(query).strip():
        return False
    clean_q = str(query).lower().strip()
    return any(re.search(pat, clean_q) for pat in MARRIAGE_PATTERNS)


MARRIAGE_SUB_INTENT_PATTERNS: Dict[str, List[str]] = {
    "marriage_remedies": [
        r"\b(?:remed(?:y|ies)\s+for\s+(?:delay\s+in\s+)?marriage|marriage\s+(?:delay\s+)?remed(?:y|ies))\b",
        r"\b(?:remed(?:y|ies)\s+for\s+(?:delay|obstacles?|problems?|issues?|troubles?)\s+(?:in\s+)?(?:marriage|vivah|shadi|shaadi))\b",
        r"\b(?:shadi|shaadi|vivah)\s+m(?:e|ein)\s+(?:deri|rukawat|badha|problem|delay)\s+ke\s+(?:upay|upaye|totke)\b",
        r"\b(?:late\s+marriage\s+remed(?:y|ies)|delay\s+in\s+marriage\s+remed(?:y|ies))\b",
        r"\b(?:upay\s+for\s+(?:late\s+)?marriage|upay\s+for\s+shadi|shadi\s+ke\s+upay|vivah\s+ke\s+upay)\b",
        r"\b(?:mangal\s+dosha\s+remed(?:y|ies)|manglik\s+remed(?:y|ies)|shukra\s+upay|guru\s+upay)\b",
        r"\b(?:how\s+to\s+speed\s+up\s+marriage|solve\s+marriage\s+delay|remedies\s+for\s+marriage)\b",
    ],
    "partner_traits": [
        r"\b(?:partner(?:'s)?\s+(?:nature|personality|traits?|characteristics?|behavior|appearance|look|looks|profession|job|career))\b",
        r"\b(?:spouse(?:'s)?\s+(?:nature|personality|traits?|characteristics?|behavior|appearance|look|looks|profession|job|career))\b",
        r"\b(?:future\s+(?:husband|wife|spouse|partner)(?:'s)?\s+(?:nature|personality|traits?|characteristics?|behavior|appearance|looks?|profession|job))\b",
        r"\b(?:what\s+will\s+my\s+(?:future\s+)?(?:partner|spouse|husband|wife)\s+be\s+like)\b",
        r"\b(?:how\s+will\s+my\s+(?:partner|spouse|husband|wife)\s+be)\b",
        r"\b(?:nature\s+of\s+(?:life\s+)?partner|nature\s+of\s+spouse)\b",
        r"\b(?:traits?\s+of\s+(?:future\s+)?(?:partner|spouse|husband|wife))\b",
        r"\b(?:characteristics?\s+of\s+(?:my\s+)?(?:partner|spouse))\b",
        r"\b(?:partner\s+kaisa\s+hoga|jeevansathi\s+kaisa\s+hoga|patni\s+kaisi\s+hogi|pati\s+kaisa\s+hoga)\b",
        r"\b(?:partner\s+ka\s+swabhav|jeevansathi\s+ka\s+swabhav|partner\s+ki\s+personality)\b",
        r"\b(?:who\s+will\s+be\s+my\s+(?:life\s+)?partner|who\s+will\s+i\s+marry)\b",
        r"\b(?:7th\s+house\s+(?:partner|spouse)\s+(?:traits?|nature|personality))\b",
    ],
    "marriage_timing": [
        r"\b(?:when\s+(?:will|can)\s+i\s+get\s+married|when\s+will\s+marriage\s+happen)\b",
        r"\b(?:marriage\s+timing|timing\s+(?:of|for)\s+marriage|wedding\s+timing)\b",
        r"\b(?:meri\s+shadi\s+kab\s+hogi|shadi\s+kab\s+hogi|shaadi\s+kab\s+hogi|kab\s+hogi\s+meri\s+shadi)\b",
        r"\b(?:vivah\s+kab\s+hoga|kab\s+hoga\s+vivah|shadi\s+ki\s+umra|marriage\s+age)\b",
        r"\b(?:marriage\s+kab\s+hogi|when\s+will\s+i\s+find\s+my\s+(?:life\s+)?partner)\b",
        r"\b(?:timing\s+for\s+wedding|vivah\s+timing|shadi\s+ka\s+yog|marriage\s+yog)\b",
        r"\b(?:love\s+or\s+arranged\s+marriage|love\s+marriage\s+or\s+arranged)\b",
    ],
}


def classify_marriage_intent(query: str) -> Optional[str]:
    """
    Classifies a query into distinct marriage sub-intent categories:
      - 'marriage_remedies': Specific remedies for delays or obstacles.
      - 'partner_traits': Physical, emotional, and professional nature of the partner.
      - 'marriage_timing': When marriage/commitment will happen.
    Returns category name or None.
    """
    if not query or not str(query).strip():
        return None
    q = str(query).lower().strip()

    # 1. Remedies check (delays, obstacles, upay)
    for pat in MARRIAGE_SUB_INTENT_PATTERNS["marriage_remedies"]:
        if re.search(pat, q):
            return "marriage_remedies"

    # 2. Partner Traits / Nature check
    for pat in MARRIAGE_SUB_INTENT_PATTERNS["partner_traits"]:
        if re.search(pat, q):
            return "partner_traits"

    # 3. Marriage Timing check
    for pat in MARRIAGE_SUB_INTENT_PATTERNS["marriage_timing"]:
        if re.search(pat, q):
            return "marriage_timing"

    # General marriage query fallback to marriage_timing if it matches general marriage patterns
    if is_marriage_query(q):
        return "marriage_timing"

    return None


def classify_micro_intent(query: str) -> Optional[str]:
    """
    Unified micro-intent classifier breaking down broad domains into granular sub-intents:
      Career domain:
        - 'career_path': Best suited fields, domain expertise, 10th house authority.
        - 'job_timing': When to change jobs, transition windows, promotions.
        - 'business_vs_job': Entrepreneurship vs employment suitability (6th vs 7th/10th house).
        - 'gemstone_remedies': Career/fortune gemstones and ratna.
      Marriage domain:
        - 'partner_traits': Physical, emotional, and professional nature of the partner (7th house/Venus/Jupiter).
        - 'marriage_remedies': Specific remedies for delays or obstacles in marriage.
        - 'marriage_timing': When marriage/commitment will happen, 7th house transit activation.
    """
    if not query or not str(query).strip():
        return None

    # 1. Check marriage sub-intents (prioritize partner traits and remedies before broad timing)
    m_intent = classify_marriage_intent(query)
    if m_intent:
        return m_intent

    # 2. Check career sub-intents
    c_intent = classify_career_intent(query)
    if c_intent:
        return c_intent

    return None


CAREER_SUB_INTENT_PATTERNS: Dict[str, List[str]] = {
    "gemstone_remedies": [
        r"\b(?:which\s+gemstones?\s+suit\s+my\s+career|career\s+gemstones?|gemstone\s+for\s+career)\b",
        r"\b(?:gemstones?|ratna|lucky\s+stone|stones?|panna|emerald|ruby|sapphire|pukhraj|neelam|manik|heera|diamond|moti|pearl|rudraksha)\b",
        r"\b(?:remed(?:y|ies)|upay|upaye|totk[ae])\b",
    ],
    "business_vs_job": [
        r"\b(?:business\s+(?:or|vs|ya)\s+(?:a\s+)?job|job\s+(?:or|vs|ya)\s+(?:a\s+)?business)\b",
        r"\b(?:does\s+my\s+chart\s+favor\s+business\s+or\s+(?:a\s+)?job|favor\s+business\s+or\s+(?:a\s+)?job|favor\s+job\s+or\s+business)\b",
        r"\b(?:should\s+i\s+do\s+business\s+or\s+(?:a\s+)?job|start\s+a\s+business\s+or\s+(?:do\s+a\s+)?job)\b",
        r"\b(?:service\s+or\s+business|business\s+or\s+service)\b",
        r"\b(?:startup\s+vs\s+(?:corporate\s+)?job|job\s+vs\s+startup|freelance\s+vs\s+job)\b",
        r"\b(?:entrepreneurship\s+vs\s+employment|employment\s+vs\s+business)\b",
        r"\b(?:dhandha\s+ya\s+naukri|vyapar\s+ya\s+naukri|naukri\s+ya\s+(?:dhandha|vyapar))\b",
    ],
    "job_timing": [
        r"\b(?:when\s+(?:will|should|can)\s+i\s+(?:change|switch|get|leave)\s+(?:my\s+|a\s+)?(?:job|naukri))\b",
        r"\b(?:job\s+change\s+timing|timing\s+for\s+job\s+change|timing\s+of\s+job\s+change)\b",
        r"\b(?:when\s+is\s+the\s+right\s+time\s+for\s+job\s+change|right\s+time\s+to\s+change\s+job)\b",
        r"\b(?:when\s+will\s+i\s+get\s+(?:a\s+)?promotion|promotion\s+timing|timing\s+for\s+promotion|timing\s+of\s+promotion)\b",
        r"\b(?:job\s+switch\s+timing|timing\s+for\s+job\s+switch|when\s+to\s+switch\s+jobs?)\b",
        r"\b(?:when\s+will\s+i\s+get\s+(?:a\s+)?(?:new\s+)?job|job\s+kab\s+lagegi|naukri\s+kab\s+milegi)\b",
        r"\b(?:promotion\s+kab\s+hoga|appraisal\s+timing|salary\s+hike\s+timing)\b",
        r"\b(?:kab\s+(?:change|badlu)\s+(?:naukri|job))\b",
        # Hinglish natural-speech variants: 'mujhe acchi job kab milegi' etc.
        r"\b(?:mujhe\s+(?:acchi\s+|achi\s+|achhi\s+|nai\s+|new\s+)?(?:job|naukri)\s+kab\s+milegi)\b",
        r"\b(?:mujhe\s+(?:job|naukri)\s+kab\s+(?:milegi|lagegi|milega))\b",
        r"\b(?:naukri\s+(?:kab|kb)\s+(?:milegi|milega|lagegi|lagega|aayegi|ayegi))\b",
        r"\b(?:job\s+(?:kab|kb)\s+(?:milegi|milega|lagegi|lagega|aayegi|ayegi))\b",
        r"\b(?:acchi\s+(?:job|naukri)\s+kab|kab\s+milegi\s+(?:acchi\s+)?(?:job|naukri))\b",
        r"\b(?:interview\s+kab\s+hoga|offer\s+letter\s+kab\s+aayega|job\s+offer\s+kab)\b",
    ],
    "career_path": [
        r"\b(?:which\s+career\s+suits?\s+me(?:\s+best)?|what\s+career\s+suits?\s+me|best\s+career\s+for\s+me)\b",
        r"\b(?:career\s+direction|career\s+path|career\s+suitability|suitable\s+profession|best\s+profession)\b",
        r"\b(?:what\s+career\s+should\s+i\s+choose|which\s+profession\s+should\s+i\s+choose|which\s+field\s+suits?\s+me)\b",
        r"\b(?:ideal\s+career\s+path|career\s+choice|career\s+guidance)\b",
        r"\b(?:career\s+in\s+my\s+chart|10th\s+house\s+career|karma\s+bhava)\b",
        r"\b(?:kaunsa\s+career\s+sahi\s+rahega|career\s+kaisa\s+rahega)\b",
    ]
}


def classify_career_intent(query: str) -> Optional[str]:
    """
    Classifies a query into distinct career sub-intent categories:
      - 'gemstone_remedies'
      - 'business_vs_job'
      - 'job_timing'
      - 'career_path'
    Returns category name or None.
    """
    if not query or not str(query).strip():
        return None
    q = str(query).lower().strip()

    # 1. Gemstones & Remedies check
    for pat in CAREER_SUB_INTENT_PATTERNS["gemstone_remedies"]:
        if re.search(pat, q):
            return "gemstone_remedies"

    # 2. Business vs Job check
    for pat in CAREER_SUB_INTENT_PATTERNS["business_vs_job"]:
        if re.search(pat, q):
            return "business_vs_job"

    # 3. Job Change / Timing check
    for pat in CAREER_SUB_INTENT_PATTERNS["job_timing"]:
        if re.search(pat, q):
            return "job_timing"

    # 4. Career Path / Suitability check
    for pat in CAREER_SUB_INTENT_PATTERNS["career_path"]:
        if re.search(pat, q):
            return "career_path"

    # General career query fallback to career_path
    if is_career_query(q):
        return "career_path"

    return None


CAREER_PATTERNS = [
    r"\b(?:career|job|naukri|profession|business|work|employment)\b",
    r"\b(?:promotion|salary|appraisal|office|company|boss|switch)\b",
    r"\b(?:when\s+is\s+the\s+right\s+time\s+for\s+job\s+change|job\s+change|promotion\s+timing)\b",
    r"\b(?:which\s+gemstones?\s+suit\s+my\s+career|career\s+gemstones?|gemstone\s+for\s+career)\b",
    r"\b(?:does\s+my\s+chart\s+favor\s+business\s+or\s+(?:a\s+)?job|business\s+(?:or|vs|ya)\s+job|job\s+(?:or|vs|ya)\s+business)\b",
    r"\b(?:10th\s+house|tenth\s+house|karma\s+bhava?)\b",
]


def is_career_query(query: str) -> bool:
    """Checks if query belongs to career, job, promotion, or business domain."""
    if not query or not str(query).strip():
        return False
    clean_q = str(query).lower().strip()
    return any(re.search(pat, clean_q) for pat in CAREER_PATTERNS)


WEALTH_PATTERNS = [
    r"\b(?:wealth|money|finance|finances|financial|paisa|paise|dhan|savings|invest|investment|income)\b",
    r"\b(?:debt|loan|karz|karza|bachat|dhana\s+yoga)\b",
    r"\b(?:2nd\s+house|second\s+house|dhana\s+bhava?|11th\s+house|eleventh\s+house|labha\s+bhava?)\b",
]


def is_wealth_query(query: str) -> bool:
    """Checks if query belongs to wealth, money, or finance domain."""
    if not query or not str(query).strip():
        return False
    clean_q = str(query).lower().strip()
    return any(re.search(pat, clean_q) for pat in WEALTH_PATTERNS)


CAREER_LORD_SPECIALIZATIONS: Dict[str, Dict[str, str]] = {
    "Sun": {
        "fields_en": "government administration, civil services, executive leadership, institutional management, public policy, medicine, and corporate governance",
        "aptitudes_en": "sovereign executive authority, strategic decision-making, ethical command, and leading large organizations with decisive vision",
        "fields_hi": "prashasan (civil services), government sector, medical science, public policy, aur corporate leadership",
        "aptitudes_hi": "netritva kshamata, sthir vicharshilta, aur badi sansthaon ko nirdeshit karne ki swabhimani shakti",
    },
    "Moon": {
        "fields_en": "healthcare, public relations, hospitality, psychology, pharmaceuticals, human resources, creative writing, and mass media communication",
        "aptitudes_en": "deep emotional empathy, public resonance, intuitive crisis navigation, and high adaptability in human-centric domains",
        "fields_hi": "chikitsa (healthcare), hospitality, counseling, HR, public relations, creative media, aur samajik seva",
        "aptitudes_hi": "jan-sampark kshamata, bhavatmak samajh, aur logon ke sath gehra vishwas sthapit karne ka hunar",
    },
    "Mars": {
        "fields_en": "engineering, technology infrastructure, real estate development, defense operations, surgical medicine, enterprise management, and competitive athletics",
        "aptitudes_en": "tactical execution, high physical and intellectual stamina, fearless problem-solving, and decisive operational command under pressure",
        "fields_hi": "engineering, technical architecture, defense, real estate, surgery, operations management, aur sports",
        "aptitudes_hi": "drudh ichhashakti, kathan kaaryon ko chunouti ke roop mein poora karna, aur operational efficiency",
    },
    "Mercury": {
        "fields_en": "software development, data analytics, quantitative finance, fintech, journalism, digital commerce, accountancy, and strategic consulting",
        "aptitudes_en": "analytical clarity, rapid computational intelligence, agile negotiation, multi-domain synthesis, and crisp technical communication",
        "fields_hi": "software engineering, data analytics, banking & finance, digital marketing, CA/accounting, aur media consulting",
        "aptitudes_hi": "teekhshna buddhi, data ko samajhne ki gati, aur vyaparik batchit mein nipun samvaad",
    },
    "Jupiter": {
        "fields_en": "higher education, judicial law, wealth advisory, strategic consulting, philosophical research, publishing, and institutional banking",
        "aptitudes_en": "ethical mentorship, macro-economic foresight, profound conceptual depth, and guiding complex multi-stakeholder decisions with wisdom",
        "fields_hi": "uchh shiksha (higher education), kanoon (law), banking, wealth management, spiritual counseling, aur policy advisory",
        "aptitudes_hi": "gyan-vardhak drishtikon, dharmi netritva, aur door-darshi guidance pradan karne ki yoghyata",
    },
    "Venus": {
        "fields_en": "architectural design, creative media, luxury brand commerce, corporate law, finance & fintech, entertainment, hospitality, and public relations",
        "aptitudes_en": "refined aesthetic design sensibility, high emotional quotient, diplomatic negotiation, and creating captivating customer experiences",
        "fields_hi": "designing, media & entertainment, finance, corporate law, architecture, luxury branding, aur public relations",
        "aptitudes_hi": "kalaatmak drishti, rishton aur deal-making mein samanjasya, aur premium value nirman",
    },
    "Saturn": {
        "fields_en": "core engineering, heavy operations, supply chain logistics, legal compliance, civil infrastructure, mining, and institutional governance",
        "aptitudes_en": "rigorous structural discipline, enduring patience, fault-tolerant execution, and optimizing complex large-scale operational machinery",
        "fields_hi": "civil & mechanical engineering, operations management, supply chain, legal compliance, real estate, aur manufacturing",
        "aptitudes_hi": "parishram, sthir dharana, aur lambi aayu wale majboot systems ka nirman karne ki lagan",
    },
}


def format_career_rule_response(
    query: str,
    user_natal_context: Dict[str, Any],
    hinglish: Optional[bool] = None,
) -> str:
    """
    Intent-aware Career & 10th House / Dasha rule generator (0 LLM tokens, <10ms).
    Tailors response specifically to the classified sub-intent:
    'gemstone_remedies', 'business_vs_job', 'job_timing', or 'career_path'.
    Structures output into strictly 2 paragraphs (5-7 lines total) with source badge.
    """
    if hinglish is None:
        hinglish = is_hinglish_query(query)

    asc = user_natal_context.get("ascendant") or "Libra"
    dasha = user_natal_context.get("active_dasha") or "Rahu-Mars"
    rulerships = user_natal_context.get("house_rulerships") or calculate_house_rulerships(asc)
    h10_info = rulerships.get("10th_house", "Moon (Cancer)")
    intro = format_user_intro(user_natal_context, hinglish=hinglish)

    m = re.match(r"^(\w+)\s*\(([^)]+)\)$", str(h10_info).strip())
    h10_lord = m.group(1) if m else "Sun"
    h10_sign = m.group(2) if m else "Aries"

    sub_intent = classify_career_intent(query) or "career_path"

    # ---------------------------------------------------------------------------
    # Extract Antardasha (sub-period) from dasha string for hyper-specific timing
    # e.g. "Rahu-Mars" -> antardasha = "Mars"
    # ---------------------------------------------------------------------------
    dasha_parts = str(dasha).split("-")
    mahadasha = dasha_parts[0].strip() if dasha_parts else dasha
    antardasha = dasha_parts[1].strip() if len(dasha_parts) > 1 else ""

    # Lookup gemstone for Lagna lord and 10th lord (Vedic standard mapping)
    PLANET_GEMSTONE: Dict[str, Dict[str, str]] = {
        "Sun":     {"stone_en": "Ruby (Manik)",      "stone_hi": "Manik (Ruby)",     "metal_en": "gold",   "day_en": "Sunday",    "metal_hi": "sone",     "day_hi": "Ravivar"},
        "Moon":    {"stone_en": "Pearl (Moti)",      "stone_hi": "Moti (Pearl)",     "metal_en": "silver", "day_en": "Monday",    "metal_hi": "chaandi",  "day_hi": "Somvar"},
        "Mars":    {"stone_en": "Red Coral (Moonga)","stone_hi": "Moonga (Red Coral)","metal_en": "copper","day_en": "Tuesday",  "metal_hi": "taamba",   "day_hi": "Mangalvar"},
        "Mercury": {"stone_en": "Emerald (Panna)",   "stone_hi": "Panna (Emerald)",  "metal_en": "gold",   "day_en": "Wednesday", "metal_hi": "sone",     "day_hi": "Budhvar"},
        "Jupiter": {"stone_en": "Yellow Sapphire (Pukhraj)","stone_hi": "Pukhraj (Yellow Sapphire)","metal_en": "gold","day_en": "Thursday","metal_hi": "sone","day_hi": "Guruvar"},
        "Venus":   {"stone_en": "Diamond or White Sapphire (Heera/Safed Pukhraj)","stone_hi": "Heera ya Safed Pukhraj","metal_en": "platinum or silver","day_en": "Friday","metal_hi": "chaandi ya platinum","day_hi": "Shukravar"},
        "Saturn":  {"stone_en": "Blue Sapphire (Neelam)","stone_hi": "Neelam (Blue Sapphire)","metal_en": "iron or silver","day_en": "Saturday","metal_hi": "loha ya chaandi","day_hi": "Shanivar"},
        "Rahu":    {"stone_en": "Hessonite Garnet (Gomed)","stone_hi": "Gomed",       "metal_en": "silver", "day_en": "Saturday", "metal_hi": "chaandi",  "day_hi": "Shanivar"},
        "Ketu":    {"stone_en": "Cat's Eye (Lehsunia)","stone_hi": "Lehsunia",        "metal_en": "silver", "day_en": "Tuesday",  "metal_hi": "chaandi",  "day_hi": "Mangalvar"},
    }
    # Determine Lagna lord from ascendant sign
    lagna_lord = SIGN_LORDS.get(asc, "Jupiter")
    gem_info = PLANET_GEMSTONE.get(h10_lord, PLANET_GEMSTONE["Jupiter"])
    lagna_gem = PLANET_GEMSTONE.get(lagna_lord, PLANET_GEMSTONE["Jupiter"])

    if sub_intent == "gemstone_remedies":
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} aapke {asc} Lagna ke lord {lagna_lord} ke liye {lagna_gem['stone_hi']} sabse zyada upyukt ratna hai, "
                f"aur 10th house ke lord {h10_lord} ke liye {gem_info['stone_hi']} career vriddhi mein sahayak hoga. "
                f"Active {dasha} dasha mein yeh ratna aapki professional energy ko align karta hai."
            )
            p2 = (
                f"**Actionable Advice:** {lagna_gem['stone_hi']} ko {lagna_gem['metal_hi']} mein set karwake {lagna_gem['day_hi']} ko subah prana-pratishtha ke baad ring finger mein pehnein. "
                f"Keval certified, natural, unheated ratna hi dharan karein aur dharan karne se pehle expert jyotishi se {lagna_lord} ki bal-avastha zaroor verify karwayein."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} for your {asc} Lagna, {lagna_lord} is your Lagna lord—its strengthening stone is {lagna_gem['stone_en']}. "
                f"Your 10th house ruler {h10_lord} governs career elevation and responds to {gem_info['stone_en']}, particularly potent during your active {dasha} period."
            )
            p2 = (
                f"**Actionable Advice:** Wear {lagna_gem['stone_en']} set in {lagna_gem['metal_en']} on your ring finger on {lagna_gem['day_en']} after proper Vedic consecration. "
                f"Consult a certified jyotishi to confirm {lagna_lord}'s functional benefic status in your chart before purchasing—only natural, unheated stones are effective."
            )

    elif sub_intent == "business_vs_job":
        h6_info = rulerships.get("6th_house", "Jupiter (Pisces)")
        h7_info = rulerships.get("7th_house", "Mars (Aries)")
        m6 = re.match(r"^(\w+)\s*\(([^)]+)\)$", str(h6_info).strip())
        m7 = re.match(r"^(\w+)\s*\(([^)]+)\)$", str(h7_info).strip())
        h6_lord = m6.group(1) if m6 else "Jupiter"
        h7_lord = m7.group(1) if m7 else "Mars"
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} aapke {asc} Lagna chart mein salaried naukri (service) 6th house ke lord {h6_lord} se dekhi jaati hai, "
                f"jabki swatantra vyapar aur partnerships 7th house ({h7_lord}) aur 10th house ({h10_sign}, lord {h10_lord}) se chalti hain. "
                f"Active {dasha} dasha aapko risk aur stability ke beech sahi alignment dikhati hai."
            )
            p2 = (
                f"**Actionable Advice:** Pehle structured employment mein domain authority aur financial reserve banayein—phir low-capital pilot project se vyapar validate karein. "
                f"Jab tak 6 mahine ka emergency fund na ho, full-time business switch avoid karein."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} in your {asc} Lagna chart, salaried employment is governed by 6th house lord {h6_lord}, "
                f"while self-employment, trade, and partnerships fall under 7th house lord {h7_lord} and 10th house ({h10_sign}, ruled by {h10_lord}). "
                f"Your active {dasha} Dasha period is the key factor in timing any commercial pivot."
            )
            p2 = (
                f"**Actionable Advice:** Build domain expertise and a 6-month financial cushion in structured employment before executing any business switch. "
                f"Validate ideas with lean, low-capital pilots aligned with your {h10_lord}-governed vocational strengths first."
            )

    elif sub_intent == "job_timing":
        h11_info = rulerships.get("11th_house", "Sun (Leo)")
        m11 = re.match(r"^(\w+)\s*\(([^)]+)\)$", str(h11_info).strip())
        h11_lord = m11.group(1) if m11 else "Sun"
        h11_sign = m11.group(2) if m11 else "Leo"
        antardasha_str = f"{antardasha} Antardasha" if antardasha else f"{mahadasha} Dasha"
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} aapki job aur promotion ki sabse acchi timing tab aati hai jab Jupiter ya Saturn aapke 10th house ({h10_sign}) ya 11th house ({h11_sign}) par gochar karte hain, "
                f"aur {dasha} Dasha mein {antardasha_str} active hoti hai. "
                f"Agale 3-4 mahine aapke liye nayi opportunities aur interview calls ke liye favorable transition window hain."
            )
            p2 = (
                f"**Actionable Advice:** Abhi se apni resume update karein, quantified achievements highlight karein, aur targeted applications isi favorable window mein bhejein. "
                f"Impulsive resignation se bachein—pehle offer letter haath mein aaye, phir resign karein."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} your optimal job-change window opens when benefic planets transit your 10th house ({h10_sign}) and 11th house of gains ({h11_sign}, lord {h11_lord}), "
                f"coinciding with your {antardasha_str} within the {mahadasha} Mahadasha. "
                f"The next 3–4 months form a high-activation career transition window in your chart."
            )
            p2 = (
                f"**Actionable Advice:** Update your resume with quantified deliverables and submit targeted applications now during this transit window. "
                f"Secure your offer letter before resigning—never make an impulsive exit outside a supportive {h10_lord} transit period."
            )

    else:  # career_path
        spec = CAREER_LORD_SPECIALIZATIONS.get(h10_lord, CAREER_LORD_SPECIALIZATIONS["Sun"])
        # Sign-specific career trait addendum (makes each answer unique per 10th sign)
        SIGN_CAREER_TRAITS: Dict[str, str] = {
            "Aries": "pioneering leadership and speed-to-market execution",
            "Taurus": "financial stability, craftsmanship, and long-term value creation",
            "Gemini": "communication agility, multi-tasking, and information synthesis",
            "Cancer": "empathetic service, nurturing leadership, and public welfare",
            "Leo": "stage presence, creative authority, and brand-building charisma",
            "Virgo": "precision analysis, quality systems, and process optimization",
            "Libra": "diplomatic deal-making, aesthetic vision, and collaborative equity",
            "Scorpio": "deep research, transformational risk-taking, and investigative acuity",
            "Sagittarius": "philosophical breadth, cross-cultural reach, and high-vision strategy",
            "Capricorn": "executive discipline, institutional authority, and long-range planning",
            "Aquarius": "innovation, technology systems, and humanitarian reform",
            "Pisces": "intuitive creativity, spiritual service, and adaptive imagination",
        }
        SIGN_CAREER_TRAITS_HI: Dict[str, str] = {
            "Aries": "aagrah aur speed-driven execution",
            "Taurus": "sthir arthik nirman aur craftsmanship",
            "Gemini": "sanchar-kushalata aur information management",
            "Cancer": "bhavatmak netritva aur jan-seva",
            "Leo": "manch-pradarshan aur brand-building",
            "Virgo": "viश्लेषण (analysis) aur quality control",
            "Libra": "samajhasya aur aesthetic vision",
            "Scorpio": "shodh, risk-management aur investigative drishtikon",
            "Sagittarius": "darshanik vishaalata aur cross-cultural strategy",
            "Capricorn": "karyakaari anushaasan aur institutional leadership",
            "Aquarius": "innovation, technology aur samajik sudhar",
            "Pisces": "srishti-shakti, adhyatmik seva aur adaptive creativity",
        }
        sign_trait_en = SIGN_CAREER_TRAITS.get(h10_sign, "strategic excellence and domain mastery")
        sign_trait_hi = SIGN_CAREER_TRAITS_HI.get(h10_sign, "strategic excellence aur domain mastery")
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} aapka 10th house {h10_sign} rashi mein hai jiske lord {h10_lord} hain, "
                f"jo aapko {sign_trait_hi} mein natural strength deta hai. "
                f"Active {dasha} dasha ke saath {spec['fields_hi']} mein career sarvottam rahega."
            )
            p2 = (
                f"**Actionable Advice:** {h10_lord} ke domain ({spec['fields_hi']}) mein deep specialization build karein aur measurable results-driven roles mein lead karein. "
                f"Apne {asc} Lagna ki core strengths ko usi domain mein leverage karein jo long-term reputation banaye."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} your 10th house falls in {h10_sign}, ruled by {h10_lord}, channeling natural aptitude for {sign_trait_en}. "
                f"Under your active {dasha} Dasha, your highest-leverage career fields are {spec['fields_en']}."
            )
            p2 = (
                f"**Actionable Advice:** Pursue deep specialization in {h10_lord}-governed domains ({spec['fields_en'].split(',')[0].strip()}) and build a track record of measurable, milestone-driven results. "
                f"Leverage your {asc} Lagna constitution to anchor authority in your chosen field."
            )

    badge = format_source_badge("Local Database / FreeAstrologyAPI")
    return f"{p1}\n\n{p2}{badge}"


def format_wealth_rule_response(
    query: str,
    user_natal_context: Dict[str, Any],
    hinglish: Optional[bool] = None,
) -> str:
    """
    Deterministic Stage-8 Wealth & 2nd/11th House / Dasha rule generator (0 LLM tokens, <10ms).
    Structures output into strictly 2 paragraphs (5-7 lines total) with source badge.
    """
    if hinglish is None:
        hinglish = is_hinglish_query(query)

    asc = user_natal_context.get("ascendant") or "Libra"
    dasha = user_natal_context.get("active_dasha") or "Rahu-Mars"
    rulerships = user_natal_context.get("house_rulerships") or calculate_house_rulerships(asc)
    h2_info = rulerships.get("2nd_house", "Mars (Scorpio)")
    intro = format_user_intro(user_natal_context, hinglish=hinglish)

    m = re.match(r"^(\w+)\s*\(([^)]+)\)$", str(h2_info).strip())
    h2_lord = m.group(1) if m else "Mars"
    h2_sign = m.group(2) if m else "Scorpio"

    if hinglish:
        p1 = (
            f"**Astrological Insight:** {intro} dhan ka 2nd house {h2_sign} mein sthit hai jiske lord {h2_lord} hain, "
            f"jabki aapki active Dasha {dasha} chal rahi hai. Yeh planetary combination speculative risk ke bajay steady savings aur disciplined compounding ko favor karta hai."
        )
        p2 = (
            f"**Actionable Advice:** Apne dainik budget ko track karein, anavashyak kharche rokein, aur long-term savings buffer create karein. "
            f"Bade financial investments se pehle structured risk analysis zaroor karein."
        )
    else:
        p1 = (
            f"**Astrological Insight:** {intro} accumulated wealth is anchored in your 2nd house in {h2_sign} governed by {h2_lord}, "
            f"moving under your {dasha} Vimshottari period. Your planetary placements strongly favor methodical wealth accumulation, disciplined budgeting, and diversified compounding over speculative gambles."
        )
        p2 = (
            f"**Actionable Advice:** Track cash flow meticulously, maintain an emergency liquidity reserve, and avoid impulsive financial commitments. "
            f"Leverage this timing cycle to build steady long-term financial security through structured patience."
        )

    badge = format_source_badge("Local Database / FreeAstrologyAPI")
    return f"{p1}\n\n{p2}{badge}"


def resolve_followup_context(
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    session_data: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """
    CONVERSATIONAL FOLLOW-UP CONTEXT HANDLING:
    Maintains active user's chart context across follow-up turns in session state.
    For follow-up questions (e.g. 'Why is that?', 'When will this happen?', 'Tell me more about career'),
    resolves implicit pronouns and references using previous message context without re-querying raw birth parameters.
    Returns (resolved_query, domain).
    """
    if not query or not str(query).strip():
        return query, "general"

    raw_q = str(query).strip()
    clean_q = raw_q.lower()
    norm_q = normalize_hinglish(clean_q)
    words = set(re.sub(r"[^\w\s]", "", clean_q).split())

    # 1. Detect if query explicitly mentions a domain
    explicit_domain = None
    if any(k in words or k in clean_q for k in ("career", "job", "naukri", "profession", "business", "promotion", "work", "office", "salary", "interview")):
        explicit_domain = "career"
    elif any(k in words or k in clean_q for k in ("marriage", "shaadi", "shadi", "vivah", "spouse", "partner", "relationship", "wedding", "husband", "wife", "love")):
        explicit_domain = "marriage"
    elif any(k in words or k in clean_q for k in ("wealth", "money", "finance", "finances", "paisa", "paise", "dhan", "savings", "investment", "income", "debt")):
        explicit_domain = "wealth"
    elif any(k in words or k in clean_q for k in ("health", "disease", "illness", "sehat", "fitness", "wellness", "pain")):
        explicit_domain = "health"
    elif any(k in words or k in clean_q for k in ("travel", "abroad", "foreign", "visa", "videsh", "relocate")):
        explicit_domain = "travel"
    elif any(k in words or k in clean_q for k in ("education", "study", "exam", "college", "degree", "padhai")):
        explicit_domain = "education"

    # 2. Extract prior domain from session_profile topics_asked or recent chat history
    prior_domain = None
    if session_data:
        profile = session_data.get("profile") or {}
        topics = profile.get("topics_asked", [])
        for t in reversed(topics):
            if t in ("career", "marriage", "wealth", "health", "travel", "education"):
                prior_domain = t
                break
    if not prior_domain and chat_history:
        for msg in reversed(chat_history):
            content = (msg.get("content") or "").lower()
            if any(k in content for k in ("marriage", "shaadi", "shadi", "vivah", "spouse", "partner", "7th house")):
                prior_domain = "marriage"
                break
            elif any(k in content for k in ("career", "job", "naukri", "profession", "business", "promotion", "10th house")):
                prior_domain = "career"
                break
            elif any(k in content for k in ("wealth", "money", "finance", "paisa", "paise", "dhan", "2nd house")):
                prior_domain = "wealth"
                break
            elif any(k in content for k in ("health", "disease", "sehat", "6th house")):
                prior_domain = "health"
                break

    target_domain = explicit_domain or prior_domain or "general"

    # 3. Detect follow-up indicators
    is_why = any(p in clean_q for p in ("why is that", "why is it", "why so", "why", "kyun", "kyu", "aisa kyu", "aisa kyun", "karan kya"))
    is_when = any(p in clean_q for p in ("when will this happen", "when will it happen", "when will", "when", "kab hoga", "yeh kab hoga", "timing kya", "kab tak"))
    is_more = any(p in clean_q for p in ("tell me more about", "tell me more", "explain more", "aur batao", "aur detail", "thoda aur samjhao", "elaborate", "can you explain"))
    is_remedy = any(p in clean_q for p in ("remedy", "remedies", "gemstone", "stone", "upay", "upaye", "totka", "mantra", "ratna"))

    # If user explicitly asked "tell me more about career" or "tell me more about marriage"
    if is_more and explicit_domain:
        if explicit_domain == "career":
            return "Tell me more about career growth, profession, and 10th house strength", "career"
        elif explicit_domain == "marriage":
            return "Tell me more about marriage compatibility, relationship dynamics, and 7th house", "marriage"
        elif explicit_domain == "wealth":
            return "Tell me more about wealth accumulation, finances, and 2nd house support", "wealth"

    # If query has implicit pronouns or is elliptical (no explicit domain mentioned)
    if not explicit_domain and prior_domain:
        if is_why:
            if prior_domain == "marriage":
                return "Why is marriage timing influenced by 7th house and active dasha?", "marriage"
            elif prior_domain == "career":
                return "Why is career trajectory influenced by 10th house and active dasha?", "career"
            elif prior_domain == "wealth":
                return "Why is wealth accumulation governed by 2nd house and dasha?", "wealth"
            elif prior_domain == "health":
                return "Why is health vitality governed by 6th house and planetary dasha?", "health"
        elif is_when:
            if prior_domain == "marriage":
                return "When will marriage happen and favorable wedding timing in chart?", "marriage"
            elif prior_domain == "career":
                return "When is the right time for job change, promotion or career growth?", "career"
            elif prior_domain == "wealth":
                return "When will financial growth or wealth accumulation happen?", "wealth"
        elif is_more:
            if prior_domain == "marriage":
                return "Tell me more about marriage compatibility, relationship dynamics, and 7th house", "marriage"
            elif prior_domain == "career":
                return "Tell me more about career growth, profession, and 10th house strength", "career"
            elif prior_domain == "wealth":
                return "Tell me more about wealth, finances, and 2nd house planetary support", "wealth"
        elif is_remedy:
            if prior_domain == "marriage":
                return "What astrological remedies and gemstone strengthen marriage and 7th house?", "marriage"
            elif prior_domain == "career":
                return "Which gemstones and astrological remedies suit my career and 10th house?", "career"
            elif prior_domain == "wealth":
                return "What remedies and practices enhance wealth, savings and prosperity?", "wealth"

    return raw_q, target_domain


def format_source_badge(source: str) -> str:
    """
    Format standard source provenance badge appended at the bottom of every response:
      - Local DB / API hits: "\n\n[Local Database / FreeAstrologyAPI]"
      - LLM hits: "\n\n[LLM - <Clean Model Name>]" (e.g. "[LLM - Llama 3.3 70B]", "[LLM - Gemini 3.5 Flash]")
    """
    if not source:
        return "\n\n[Local Database / FreeAstrologyAPI]"
    s = str(source).strip()
    if any(k in s.lower() for k in ("local", "database", "freeastrology", "pyswisseph", "qa_database", "api")):
        return "\n\n[Local Database / FreeAstrologyAPI]"

    m = s
    if "LLM:" in m:
        m = m.split("LLM:", 1)[1].strip()
    m = re.sub(r"^[\(\[\s]+|[\)\]\s]+$", "", m).strip()

    m_lower = m.lower()
    if "llama-3.3-70b" in m_lower or "llama 3.3 70b" in m_lower:
        badge = "Llama 3.3 70B"
    elif "llama-3.1-8b" in m_lower or "llama 3.1 8b" in m_lower:
        badge = "Llama 3.1 8B"
    elif "gemini-3.5-flash" in m_lower:
        badge = "Gemini 3.5 Flash"
    elif "gemini-3.8-flash" in m_lower:
        badge = "Gemini 3.8 Flash"
    elif "gemini" in m_lower and "flash" in m_lower:
        badge = "Gemini 3.5 Flash"
    elif "gpt-oss-120b" in m_lower:
        badge = "GPT-OSS 120B"
    elif "gpt-oss-20b" in m_lower:
        badge = "GPT-OSS 20B"
    elif "qwen" in m_lower:
        badge = "Qwen 2.5 72B"
    elif "deepseek" in m_lower:
        badge = "DeepSeek R1"
    else:
        badge = m.split("/")[-1].replace("-", " ").replace("_", " ").title()

    return f"\n\n[LLM - {badge}]"


def format_direct_api_response(
    query: str,
    user_natal_context: Dict[str, Any],
    astro_features: Optional[List[int]] = None,
    hinglish: Optional[bool] = None,
) -> str:
    """
    Direct-API calculation response formatter (0 token cost, <10ms latency).
    Returns a strictly 5-7 line, 2-paragraph response with [Local Database / FreeAstrologyAPI] badge.
    Paragraph 1: "**Astrological Insight:**" (2-3 lines linking chart parameters/API facts to question)
    Paragraph 2: "**Actionable Advice:**" (2-3 lines providing practical grounded next steps)
    """
    clean_q = query.lower()
    if hinglish is None:
        hinglish = is_hinglish_query(query)

    asc = user_natal_context.get("ascendant") or "Virgo"
    moon = user_natal_context.get("moon_sign") or "Virgo"
    sun = user_natal_context.get("sun_sign") or "Taurus"
    dasha = user_natal_context.get("active_dasha") or "Jupiter-Saturn"
    intro = format_user_intro(user_natal_context, hinglish=hinglish)

    asks_moon = bool(re.search(r"\b(?:moon|chandra|rashi)\b", clean_q))
    asks_sun = bool(re.search(r"\b(?:sun|surya)\b", clean_q))
    asks_lagna = bool(re.search(r"\b(?:lagna|ascendant|rising)\b", clean_q))
    asks_dasha = bool(re.search(r"\b(?:dasha|mahadasha|antardasha|vimshottari)\b", clean_q))
    asks_planets = bool(re.search(r"\b(?:planetary\s*positions?|planet\s*positions?|planetary|graha)\b", clean_q))

    planets_str = ""
    if astro_features and len(astro_features) >= 10:
        names = ["Ascendant", "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
        pl_list = [f"{names[i]} in {ZODIAC_SIGNS[astro_features[i] % 12]}" for i in range(10)]
        planets_str = ", ".join(pl_list)

    # 1. Multi-parameter chart overview (e.g. moon sign + sun sign + dasha)
    if (asks_moon and asks_sun) or (asks_moon and asks_dasha) or (asks_lagna and asks_moon) or (asks_lagna and asks_dasha):
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} aapka Moon sign (Rashi) {moon} hai, "
                f"Sun sign {sun} hai, aur Lagna (Ascendant) {asc} hai. Abhi aap {dasha} Vimshottari Dasha dauran chal rahe hain, "
                f"jo aapke chart ke mukhya bhavon ko sakriya karta hai."
            )
            p2 = (
                f"**Actionable Advice:** Apne {asc} Lagna ke vicharshil gunon aur {sun} Sun ki sthirta ka upyog karke practical planning karein. "
                f"Is {dasha} timing mein disciplined self-development aur realistic goals par focus karna sarvottam rahega."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} your Moon sign (Rashi) is {moon}, "
                f"your Sun sign is {sun}, and your Ascendant (Lagna) is {asc}. You are currently running the active {dasha} "
                f"Vimshottari Dasha period, which activates key developmental themes across your personal and professional houses."
            )
            p2 = (
                f"**Actionable Advice:** Anchor your major initiatives in the practical strengths of your {asc} Lagna and {sun} Sun. "
                f"Use this {dasha} period to maintain steady discipline, prioritize structured execution, and avoid hasty compromises."
            )

    # 2. Planetary positions specifically
    elif asks_planets:
        pl_detail = planets_str if planets_str else f"Lagna in {asc}, Sun in {sun}, Moon in {moon}, active Dasha: {dasha}"
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} aapki kundali mein grahon ki sthiti is prakaar hai: {pl_detail}. "
                f"Yeh planetary placements aapke jeevan ke sabhi 12 bhavon mein karmic blueprint aur shaktiyon ko darshate hain."
            )
            p2 = (
                f"**Actionable Advice:** Apne Lagna lord aur supportive planets ko strengthen karne ke liye disciplined lifestyle banayein. "
                f"Apni natural planetary abilities ke anusaar continuous learning aur balanced habits par focus karein."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} your calculated sidereal planetary placements are: {pl_detail}. "
                f"These planetary coordinates define your fundamental strengths, behavioral patterns, and life potential across all 12 houses."
            )
            p2 = (
                f"**Actionable Advice:** Channel the supportive energies of your ruling planets through balanced daily habits and ethical conduct. "
                f"Focus on long-term personal mastery while working in harmony with your primary chart rulers."
            )

    # 3. Moon Sign / Rashi query
    elif asks_moon:
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} aapka Moon sign (Chandra Rashi) {moon} hai. "
                f"Vedic jyotish mein Moon aapke mann, bhavnaon, aur sochne ki prakriya ko nirdharit karta hai, jo aapko thoughtful aur intuitive banata hai."
            )
            p2 = (
                f"**Actionable Advice:** Apne mann ki shanti aur clarity ke liye regular routine aur mindful habits maintain karein. "
                f"Apni natural {moon} qualities ka upyog constructive planning aur meaningful relationships banane mein karein."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} your Moon sign (Chandra Rashi) is {moon}. "
                f"The Moon represents your emotional mind, cognitive instincts, and subconscious patterns, conferring a perceptive and thoughtful nature."
            )
            p2 = (
                f"**Actionable Advice:** Nurture your emotional equilibrium by maintaining balanced sleep, mindful routines, and healthy mental diet. "
                f"Channel your natural {moon} instincts into focused problem-solving and structured personal endeavors."
            )

    # 4. Sun Sign query
    elif asks_sun:
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} aapka calculated Vedic Sun sign (Surya Rashi) {sun} hai. "
                f"Vedic jyotish mein Sun aapki atma-shakti, willpower, vitality aur core personality ko represent karta hai, jo aapko sthirta deta hai."
            )
            p2 = (
                f"**Actionable Advice:** Apne {sun} Sun ke determination aur confidence ke saath apne career aur personal goals par aage badhein. "
                f"Consistency banaye rakhein aur unnecessary rigidity se bachte hue continuous progress par dhyan dein."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} your calculated Vedic sidereal Sun sign (Surya Rashi) is {sun}. "
                f"The Sun governs your core vitality, soul purpose, leadership instincts, and inner willpower, providing steady resilience."
            )
            p2 = (
                f"**Actionable Advice:** Ground your ambitions in consistent, value-driven execution that reflects the integrity of your {sun} placement. "
                f"Lead personal initiatives with steady patience while staying adaptable to constructive feedback."
            )

    # 5. Lagna / Ascendant query
    elif asks_lagna:
        lagna_lord = SIGN_LORDS.get(asc, "Mercury")
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} aapka calculated Ascendant (Lagna) {asc} hai, jiske ruling lord {lagna_lord} hain. "
                f"Lagna aapke physical constitution, personality, aur jeevan ke mukhya drishtikon ko darshata hai, jo aapko prabhavshali banata hai."
            )
            p2 = (
                f"**Actionable Advice:** Apne {asc} Lagna aur lord {lagna_lord} ke strengths ke anusaar proactive communication aur disciplined habits apnayein. "
                f"Apni physical vitality aur clarity of purpose ko hamesha priority par rakhein."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} your calculated Ascendant (Lagna) is {asc}, with {lagna_lord} functioning as your primary chart ruler. "
                f"Your Lagna establishes your outward persona, vitality, mindset, and the overarching framework of your life journey."
            )
            p2 = (
                f"**Actionable Advice:** Align your personal habits and daily routines with the natural strengths of your {asc} Ascendant. "
                f"Prioritize clear communication, continuous skill development, and purposeful execution in all major commitments."
            )

    # 6. Dasha / Mahadasha query
    else:
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} abhi aap {dasha} Vimshottari Dasha timing cycle ke dauran chal rahe hain. "
                f"Vedic timing mein yeh period aapke chart ke specific karmic lessons aur opportunities ko activate karta hai."
            )
            p2 = (
                f"**Actionable Advice:** Is timing period mein disciplined execution aur realistic goals par focus karein. "
                f"Kisi bhi hasty decision se bachte hue long-term stability aur structured effort par dhyan dein."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} you are currently navigating the {dasha} Vimshottari Dasha planetary cycle. "
                f"In Vedic timing science, this period emphasizes targeted learning, personal sovereignty, and adapting to shifting priorities across your active houses."
            )
            p2 = (
                f"**Actionable Advice:** Align major life initiatives with your active ruling planets rather than forcing misaligned external goals. "
                f"Maintain structured daily habits, measured patience, and focus on steady incremental milestones."
            )

    badge = format_source_badge("Local Database / FreeAstrologyAPI")
    return f"{p1}\n\n{p2}{badge}"


PARTNER_LORD_TRAITS: Dict[str, Dict[str, str]] = {
    "Mars": {
        "nature_en": "energetic, decisive, and ambitious with a courageous spirit and direct communication",
        "profession_en": "engineering, technology management, executive leadership, defense, entrepreneurship, or active sports",
        "nature_hi": "urjavan, nirdhayak aur ambitious swabhav ke honge, jinka direct communication aur protective nature hoga",
        "profession_hi": "engineering, management, technical fields, administration, ya business leadership",
    },
    "Venus": {
        "nature_en": "graceful, emotionally balanced, and refined with a strong aesthetic sense, natural charm, and harmony-seeking disposition",
        "profession_en": "creative arts, design, finance, hospitality, law, luxury commerce, or strategic consulting",
        "nature_hi": "akarshak, shant, aur kalaatmak vicharon wale honge jo rishton mein samanjasya aur prem ko prathmikta dete hain",
        "profession_hi": "creative arts, finance, media, hospitality, ya corporate consulting",
    },
    "Mercury": {
        "nature_en": "intellectually agile, articulate, witty, and adaptable with an analytical mindset and lively communicative warmth",
        "profession_en": "technology, communication, analytics, trade, media, accounting, data science, or education",
        "nature_hi": "buddhiman, baat karne mein nipun, aur fast learner honge jinme analytical aur practical soch hogi",
        "profession_hi": "IT, accounts, communication, media, ya marketing",
    },
    "Moon": {
        "nature_en": "deeply empathetic, nurturing, emotionally intuitive, and supportive with a strong connection to family values and emotional depth",
        "profession_en": "healthcare, psychology, public relations, hospitality, human resources, or creative management",
        "nature_hi": "bhavuk, caring, aur parivarik moolyon ko aadar dene wale honge jo aapke jazbaaton ko gehraai se samjhenge",
        "profession_hi": "counseling, medicine, HR, public relations, ya social services",
    },
    "Sun": {
        "nature_en": "charismatic, dignified, self-respecting, and principled with natural executive presence, integrity, and strong moral convictions",
        "profession_en": "governance, administrative services, executive management, law, or public authority",
        "nature_hi": "aatmavishwasi, swabhimani, aur netritva kshamata se yukta honge jinme uccha aadarsh aur satyavadi drishtikon hoga",
        "profession_hi": "government sector, administration, civil services, ya senior managerial roles",
    },
    "Jupiter": {
        "nature_en": "wise, philosophical, culturally grounded, generous, and optimistic with high moral integrity and a lifelong commitment to learning",
        "profession_en": "higher education, legal advisory, financial advisory, research, spiritual guidance, or academic consulting",
        "nature_hi": "gyani, dharmi, aur gyan-vardhak swabhav ke honge jo samajik aadarshon aur spiritual values ka palan karte hain",
        "profession_hi": "teaching, law, advisory, research, banking, ya consulting",
    },
    "Saturn": {
        "nature_en": "mature, deeply grounded, disciplined, and persevering with a practical outlook and steadfast loyalty through all life phases",
        "profession_en": "core engineering, operations, architecture, compliance, real estate, or structured corporate management",
        "nature_hi": "parishrami, gambhir, aur jimmedar swabhav ke honge jo shant rehkar jeevan ki har kathinai mein sath nibhate hain",
        "profession_hi": "industry, operations, civil engineering, law, ya organized management",
    },
}


def format_marriage_rule_response(
    query: str,
    user_natal_context: Dict[str, Any],
    hinglish: Optional[bool] = None,
) -> str:
    """
    Deterministic Stage-8 Marriage & 7th House / Dasha rule generator (0 LLM tokens, <10ms).
    Micro-Intent Aware:
      - 'partner_traits': Describes physical, emotional, and professional nature of spouse from 7th house sign, lord & Venus.
      - 'marriage_remedies': Diagnostic analysis and authentic Vedic remedies for marriage delays or obstacles.
      - 'marriage_timing': Planetary transits and Dasha activation windows for commitment.
    Structures output into strictly 2 paragraphs (5-7 lines total) with source badge.
    """
    if hinglish is None:
        hinglish = is_hinglish_query(query)

    asc = user_natal_context.get("ascendant") or "Libra"
    dasha = user_natal_context.get("active_dasha") or "Rahu-Mars"
    rulerships = user_natal_context.get("house_rulerships") or calculate_house_rulerships(asc)
    h7_info = rulerships.get("7th_house", "Mars (Aries)")
    intro = format_user_intro(user_natal_context, hinglish=hinglish)

    m = re.match(r"^(\w+)\s*\(([^)]+)\)$", str(h7_info).strip())
    if m:
        h7_lord, h7_sign = m.group(1), m.group(2)
    else:
        h7_lord, h7_sign = "Venus", "Taurus"

    sub_intent = classify_marriage_intent(query) or "marriage_timing"

    if sub_intent == "partner_traits":
        trait_info = PARTNER_LORD_TRAITS.get(h7_lord, PARTNER_LORD_TRAITS["Venus"])
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} aapka 7th house (vivah bhav) {h7_sign} rashi mein sthit hai jiske lord {h7_lord} hain, "
                f"aur Venus prem evan vivahik chemistry ke mukhya karaka hain. Aapke jeevansathi {trait_info['nature_hi']}. "
                f"Unka karyakshetra mukhya roop se {trait_info['profession_hi']} se juda hone ke majboot yog hain, "
                f"aur unka vyaktitva aapke {asc} Lagna ki urja ko poori tarah santulit karega."
            )
            p2 = (
                f"**Actionable Advice:** Rishte mein open communication aur mutual respect par dhyan dein. "
                f"Apne partner ki independent soch aur unke karyashaili ka samman karna aapke dampatya jeevan ko gehra aur madhur banayega."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} your 7th house of committed partnership is in {h7_sign} governed by {h7_lord}, "
                f"with Venus signifying relational grace and emotional harmony. Your future life partner will be {trait_info['nature_en']}, "
                f"naturally balancing your {asc} Lagna. Professionally, their planetary placements favor {trait_info['profession_en']}. "
                f"Under your active {dasha} Vimshottari cycle, you will attract someone who provides profound intellectual and emotional equilibrium."
            )
            p2 = (
                f"**Actionable Advice:** Prioritize genuine emotional authenticity and mutual values over superficial compatibility. "
                f"Foster clear communicative boundaries and mutual respect to anchor lasting relational harmony."
            )
    elif sub_intent == "marriage_remedies":
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} vivah mein vilamb ya sambandhon mein rukawat aksar 7th house ({h7_sign}), "
                f"uske swami {h7_lord}, ya Shukra/Guru par malefic drishti ke kaaran hoti hai. "
                f"Aapki vartaman {dasha} Vimshottari Dasha ke dauran functional malefic prabhavon ko shant karke aur 7th house lord ki urja badhakar vivah yog ko gati di ja sakti hai."
            )
            p2 = (
                f"**Actionable Advice:** Shukra aur vivah bhav ko balwan karne ke liye pratyek Shukrawar ko Goddess Lakshmi ki puja karein, "
                f"safed mithai ka bhog lagayein, aur 'Om Shum Shukraya Namaha' ka 108 baar jaap karein. Brihaspati ki kripa hetu Guruwar ko peeli vastuon ya chane ki daal ka daan karein."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} marriage delays or relationship turbulence frequently stem from planetary friction affecting your 7th house ({h7_sign}), "
                f"its lord {h7_lord}, or relational significators Venus and Jupiter. Under your active {dasha} Vimshottari Dasha, "
                f"harmonizing karmic debts and pacifying functional malefic transits removes energetic resistance to matrimony."
            )
            p2 = (
                f"**Actionable Advice:** Strengthen relational harmony by offering white sweets or flowers to Goddess Lakshmi on Fridays, "
                f"chanting the Shukra Beej Mantra ('Om Shum Shukraya Namaha') 108 times, and donating yellow items or lentils on Thursdays to propitiate Jupiter."
            )
    else:  # marriage_timing
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} 7th house (vivah bhav) {h7_sign} rashi mein sthit hai jiske lord {h7_lord} hain, "
                f"aur abhi aapki {dasha} Vimshottari Dasha chal rahi hai. Vedic timing ke anusaar jab 7th house aur Dasha cycle activate hote hain, "
                f"tab agle 4 se 8 mahino mein relationship commitment aur vivah ke yog majboot bante hain."
            )
            p2 = (
                f"**Actionable Advice:** Is samay bina jaldbazi ke partner ke saath transparent communication aur mutual understanding par focus karein. "
                f"Venus aur 7th house ki positive energy badhane ke liye Fridays ko Goddess Lakshmi ki aaradhna karein aur calm attitude rakhein."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} your 7th house of committed partnership is in {h7_sign} governed by {h7_lord}, "
                f"operating under your active {dasha} Vimshottari Dasha cycle. In Vedic timing, relationship commitments and marriage prospects "
                f"gain significant momentum as supportive planetary transits activate your 7th house rulers over the next 4 to 8 months."
            )
            p2 = (
                f"**Actionable Advice:** Prioritize clear mutual values, transparent communication, and emotional consistency rather than impulsive haste. "
                f"Strengthen your relational harmony by aligning major partnership decisions during auspicious planetary transit windows."
            )

    badge = format_source_badge("Local Database / FreeAstrologyAPI")
    return f"{p1}\n\n{p2}{badge}"


def generate_local_rule_fallback(
    query: str,
    user_natal_context: Optional[Dict[str, Any]] = None,
    hinglish: Optional[bool] = None,
) -> str:
    """
    Resilient local rule fallback engine invoked when ALL cascading LLMs are exhausted or rate limited (HTTP 429).
    Guarantees the user NEVER receives 'System Busy' or raw errors.
    Returns chart-grounded 2-paragraph response (5-7 lines) with source badge.
    """
    ctx = user_natal_context or DEFAULT_NATAL_CONTEXT
    if hinglish is None:
        hinglish = is_hinglish_query(query)

    clean_q = (query or "").lower().strip()

    # 1. Marriage / relationship check (with granular micro-intent handling)
    if is_marriage_query(clean_q) or classify_marriage_intent(clean_q):
        return format_marriage_rule_response(clean_q, ctx, hinglish=hinglish)

    # 2. Career check (with granular sub-intent handling: career_path, business_vs_job, job_timing, gemstone_remedies)
    if is_career_query(clean_q) or classify_career_intent(clean_q):
        return format_career_rule_response(clean_q, ctx, hinglish=hinglish)

    # 3. Wealth / finance check
    h2_info = rulerships.get("2nd_house", "Mars (Scorpio)")
    m2 = re.match(r"^(\w+)\s*\(([^)]+)\)$", str(h2_info).strip())
    h2_lord = m2.group(1) if m2 else "Mars"
    h2_sign = m2.group(2) if m2 else "Scorpio"

    is_wealth = any(k in clean_q for k in ("wealth", "money", "finance", "paisa", "financial", "dhan", "savings", "earning"))
    if is_wealth:
        if hinglish:
            p1 = (
                f"**Astrological Insight:** {intro} dhan ka 2nd house {h2_sign} mein hai jiske swami {h2_lord} hain, jabki aapki active Dasha {dasha} hai. "
                f"Yeh planetary yog speculative risk ke bajay steady savings aur disciplined financial compounding ko favor karta hai."
            )
            p2 = (
                f"**Actionable Advice:** Apne daily budget ko track karein, anavashyak kharche rokein, aur long-term savings buffer create karein. "
                f"Bade financial investments se pehle structured analysis zaroor karein."
            )
        else:
            p1 = (
                f"**Astrological Insight:** {intro} accumulated wealth is anchored in your 2nd house ({h2_sign}) governed by {h2_lord}, moving under the {dasha} period. "
                f"Your planetary placements strongly favor methodical accumulation, disciplined budgeting, and diversified compounding over speculative gambles."
            )
            p2 = (
                f"**Actionable Advice:** Track cash flow meticulously, maintain an emergency liquidity reserve, and avoid impulsive financial risks. "
                f"Leverage this timing cycle to build steady long-term financial security through structured patience."
            )
        badge = format_source_badge("Local Database / FreeAstrologyAPI")
        return f"{p1}\n\n{p2}{badge}"

    # 4. General / Life path fallback
    if hinglish:
        p1 = (
            f"**Astrological Insight:** {intro} {asc} Lagna aur {moon} Moon sign ke anusaar abhi aap {dasha} planetary timing cycle ke dauran chal rahe hain. "
            f"Yeh samay aapki inner strength aur patience ko test karte hue ek naye positive turning point ki taraf aage badha raha hai."
        )
        p2 = (
            f"**Actionable Advice:** Apne dainik routine mein mindful habits aur disciplined focus banaye rakhein. "
            f"Apne Lagna aur Moon sign ke natural strengths par bharosa karte hue consistent action lete rahein."
        )
    else:
        p1 = (
            f"**Astrological Insight:** {intro} your {asc} Ascendant and {moon} Moon coordinates indicate your life trajectory is actively guided by your {dasha} planetary cycle. "
            f"This astrological phase emphasizes inner resilience, clarity of purpose, and progressive alignment with your core developmental destiny."
        )
        p2 = (
            f"**Actionable Advice:** Maintain emotional equilibrium through structured routines, mindful habits, and steady goal pursuit. "
            f"Trust your natural astrological strengths and prioritize grounded decisions over short-term anxiety."
        )

    badge = format_source_badge("Local Database / FreeAstrologyAPI")
    return f"{p1}\n\n{p2}{badge}"


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

    user_name = user_context.get("name")
    birth_details = user_context.get("birth_details")
    birth_place = user_context.get("birth_place")

    # Substitution parameters for template markers
    subs: Dict[str, Any] = {
        "ascendant": ascendant,
        "moon_sign": moon_sign,
        "active_dasha": active_dasha,
        "name": user_name or "Seeker",
        "birth_details": birth_details or "",
        "birth_place": birth_place or "",
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

    # Core aliases for deterministic parameter injection
    subs["lagna"] = ascendant
    subs["dasha"] = active_dasha
    subs["current_dasha"] = active_dasha
    subs["moon"] = moon_sign
    subs["rashi"] = moon_sign
    subs["sun"] = user_context.get("sun_sign", "")
    subs["sun_sign"] = user_context.get("sun_sign", "")
    for i in range(1, 13):
        h_key = f"{i}th_house"
        if h_key in subs:
            subs[f"house_{i}"] = subs[h_key]
            subs[f"house_{i}_lord"] = subs.get(f"{h_key}_lord", "")
            subs[f"house_{i}_sign"] = subs.get(f"{h_key}_sign", "")

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

    is_hinglish = is_hinglish_query(core_analysis + " " + recommendation)
    intro = format_user_intro(user_context, hinglish=is_hinglish)

    sections = []
    insight_text = f"{intro} {core_analysis} {dasha_influence}".strip()
    if insight_text:
        sections.append(f"**Astrological Insight:** {insight_text}")
    if recommendation:
        sections.append(f"**Actionable Advice:** {recommendation.strip()}")
    badge = format_source_badge("Local Database / FreeAstrologyAPI")
    return "\n\n".join(sections) + badge


def get_local_qa_response(
    user_message: str,
    user_profile: Optional[Any] = None,
    user_natal_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Search local Q&A database for a granular sub-intent match (0 LLM tokens).
    Applies state tracking: if this specific template was recently served in the active conversation,
    forces fallback to the LLM pipeline (returns None) instead of repeating the template.
    """
    if not user_message or not user_message.strip():
        return None

    clean_msg = user_message.strip()

    # Compound queries or predictive questions must never be hijacked by local QA templates
    if should_bypass_local_qa(clean_msg, session_id=session_id):
        return None

    # 1. Granular sub-intent matching (keyword + semantic markers)
    match_result = match_sub_intent(clean_msg, qa_database_raw)
    template = None
    template_key = None

    if match_result:
        template_key, template = match_result
    else:
        # Fallback to normalized index lookup for exact definitions
        clean_lower = clean_msg.lower().strip("?.! ")
        template = qa_database_index.get(clean_lower, None)
        template_key = clean_lower
        if not template:
            for prefix in ("what is a ", "what is an ", "what is ", "define ", "meaning of ", "explain "):
                if clean_lower.startswith(prefix):
                    term = clean_lower[len(prefix):].strip("?.! ")
                    template = qa_database_index.get(term, None)
                    template_key = term
                    if template:
                        break

    if not template:
        return None

    # 2. State Tracking Check: If template was already served in this conversation, force fallback to LLM
    if is_template_recently_served(session_id, template_key, chat_history=chat_history):
        logger.info(
            "Local QA template '%s' was already served in session '%s'. Forcing fallback to LLM pipeline.",
            template_key, session_id or "anonymous"
        )
        return None

    # Resolve active natal context
    if not user_natal_context:
        user_natal_context = get_user_natal_context(user_profile=user_profile)

    # 3. Format and record template in session state
    if isinstance(template, dict):
        formatted_answer = format_local_qa_response(template, user_natal_context)
        record_served_template(session_id, template_key)
        return formatted_answer

    # Format plain text template if user_profile or user_natal_context is provided
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
            formatted_answer = template.format_map(SafeDict(**format_dict))
        except Exception:
            formatted_answer = template
        record_served_template(session_id, template_key)
        return formatted_answer

    record_served_template(session_id, template_key)
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


def store_last_token_stats(
    session_id: Optional[str],
    model: str,
    account_id: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    session_total_used: int,
    remaining_tokens: int,
    percentage_remaining: float,
    provider_remaining: Optional[int] = None,
    provider_limit: Optional[int] = None,
    provider_remaining_pct: Optional[float] = None,
) -> None:
    """
    Stores the most recent token usage stats for a session so that
    the streaming endpoint can append a [TOKEN_USAGE] metadata line.
    Thread-safe: uses _LAST_TOKEN_STATS_LOCK.
    """
    key = session_id or "__global__"
    stats: Dict[str, Any] = {
        "model": model,
        "account": account_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "session_total_used": session_total_used,
        "session_remaining_tokens": remaining_tokens,
        "session_quota": TOTAL_MODEL_QUOTA,
        "session_remaining_pct": round(percentage_remaining, 1),
    }
    if provider_remaining is not None:
        stats["provider_remaining_tokens"] = provider_remaining
    if provider_limit is not None:
        stats["provider_limit_tokens"] = provider_limit
    if provider_remaining_pct is not None:
        stats["provider_remaining_pct"] = round(provider_remaining_pct, 1)
    with _LAST_TOKEN_STATS_LOCK:
        _LAST_TOKEN_STATS[key] = stats


def get_last_token_stats(session_id: Optional[str]) -> Dict[str, Any]:
    """
    Returns the most recent token usage stats stored for session_id.
    Returns an empty dict if no stats have been recorded yet.
    """
    key = session_id or "__global__"
    with _LAST_TOKEN_STATS_LOCK:
        return dict(_LAST_TOKEN_STATS.get(key, {}))


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
    """Get or create cached AsyncGroq client for a specific account API key with connection pooling.
    Uses the official Groq SDK with GROQ_API_KEY.
    """
    if not api_key or not str(api_key).strip():
        return None
    cleaned_key = str(api_key).strip()
    if cleaned_key not in _GROQ_CLIENT_CACHE:
        try:
            _GROQ_CLIENT_CACHE[cleaned_key] = groq.AsyncGroq(
                api_key=cleaned_key,
                http_client=get_shared_httpx_client(),
            )
        except Exception:
            _GROQ_CLIENT_CACHE[cleaned_key] = groq.AsyncGroq(api_key=cleaned_key)
    return _GROQ_CLIENT_CACHE[cleaned_key]


def get_groq_openai_client_for_account(api_key: str) -> Optional[openai.AsyncOpenAI]:
    """
    Initializes an AsyncOpenAI client explicitly targeted at Groq's OpenAI-compatible endpoint.
    Strictly sets base_url='https://api.groq.com/openai/v1' to ensure requests
    intended for Groq NEVER hit default OpenAI endpoints (https://api.openai.com/v1).
    """
    if not api_key or not str(api_key).strip():
        return None
    cleaned_key = str(api_key).strip()
    return openai.AsyncOpenAI(
        api_key=cleaned_key,
        base_url=GROQ_BASE_URL,
        http_client=get_shared_httpx_client(),
    )


def get_groq_client() -> Optional[groq.AsyncGroq]:
    """Backward-compatible helper returning the primary Account 1 client."""
    accounts = get_groq_accounts()
    if accounts:
        return get_groq_client_for_account(accounts[0]["api_key"])
    return None


def get_openrouter_client() -> Optional[openai.AsyncOpenAI]:
    """
    Initialize AsyncOpenAI client for the OpenRouter fallback tier with connection pooling.
    - If OPENROUTER_API_KEY is configured, points to openrouter.ai (https://openrouter.ai/api/v1).
    - If OPENROUTER_API_KEY is not set, bridges using primary Groq account key
      against Groq's OpenAI-compatible base URL (https://api.groq.com/openai/v1).
    Guarantees requests intended for Groq or OpenRouter NEVER hit default OpenAI (api.openai.com).
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key and openrouter_api_key.strip():
        base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()
        return openai.AsyncOpenAI(
            api_key=openrouter_api_key.strip(),
            base_url=base_url,
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
            base_url=GROQ_BASE_URL,
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
    query: Optional[str] = None,
) -> str:
    """
    Step 2: Strict System Prompt Guardrails
    Binds the AI strictly to the provided natal context and enforces language matching:
    - If user asks in Hinglish -> respond ONLY in natural Hinglish (Roman script).
    - If user asks in English -> respond STRICTLY in clear English.
    """
    if not user_natal_context:
        user_natal_context = get_user_natal_context(user_profile=user_profile)

    # Determine language strictly from query if available, otherwise session profile
    if query is not None and str(query).strip():
        is_hinglish = is_hinglish_query(str(query))
    else:
        pref = (session_profile or {}).get("preferred_language", "english")
        is_hinglish = str(pref).casefold() == "hinglish"

    if is_hinglish:
        language_rule = (
            "5. LANGUAGE MANDATE (HINGLISH):\n"
            "   Write your ENTIRE response in natural, conversational Hinglish (Hindi written in Roman English alphabet). Headings '**Astrological Insight:**' and '**Actionable Advice:**' must remain as standard headers."
        )
    else:
        language_rule = (
            "5. LANGUAGE MANDATE (ENGLISH):\n"
            "   Write your ENTIRE response strictly in clear, professional English."
        )

    active_dasha_str = user_natal_context.get('active_dasha', 'current')
    seeker_name = user_natal_context.get('name') or 'Respected Seeker'
    birth_details_str = user_natal_context.get('birth_details') or 'Not specified'
    birth_place_str = user_natal_context.get('birth_place') or 'Not specified'

    # Pre-compute chart coordinates for injecting into micro-intent mandates
    _rulerships = user_natal_context.get("house_rulerships") or calculate_house_rulerships(
        user_natal_context.get("ascendant", "Libra")
    )
    _asc = user_natal_context.get("ascendant", "Libra")
    _lagna_lord = SIGN_LORDS.get(_asc, "Jupiter")
    _h7 = _rulerships.get("7th_house", "Venus (Taurus)")
    _h10 = _rulerships.get("10th_house", "Moon (Cancer)")
    _h11 = _rulerships.get("11th_house", "Sun (Leo)")
    _h6 = _rulerships.get("6th_house", "Jupiter (Pisces)")
    _m7 = re.match(r"^(\w+)\s*\(([^)]+)\)$", str(_h7).strip())
    _m10 = re.match(r"^(\w+)\s*\(([^)]+)\)$", str(_h10).strip())
    _m11 = re.match(r"^(\w+)\s*\(([^)]+)\)$", str(_h11).strip())
    _h7_lord = _m7.group(1) if _m7 else "Venus"
    _h7_sign = _m7.group(2) if _m7 else "Taurus"
    _h10_lord = _m10.group(1) if _m10 else "Moon"
    _h10_sign = _m10.group(2) if _m10 else "Cancer"
    _h11_lord = _m11.group(1) if _m11 else "Sun"
    _dasha_parts = str(active_dasha_str).split("-")
    _mahadasha = _dasha_parts[0].strip() if _dasha_parts else active_dasha_str
    _antardasha = _dasha_parts[1].strip() if len(_dasha_parts) > 1 else ""
    _antardasha_label = f"{_antardasha} Antardasha" if _antardasha else f"{_mahadasha} Dasha"

    micro_directive = ""
    if query:
        m_intent = classify_micro_intent(str(query))
        if m_intent == "partner_traits":
            micro_directive = (
                f"\n8. MICRO-INTENT MANDATE (PARTNER TRAITS):\n"
                f"   - The user is asking specifically about their life partner/spouse's physical, emotional, and professional nature.\n"
                f"   - Your {_asc} Lagna chart places the 7th house in {_h7_sign} ruled by {_h7_lord}—describe partner personality directly from this sign's nature and {_h7_lord}'s qualities.\n"
                f"   - Also reference Venus and Jupiter placements to shape emotional and physical partner traits.\n"
                f"   - DO NOT mention marriage timing, wedding dates, or career path in this response.\n"
            )
        elif m_intent == "marriage_timing":
            micro_directive = (
                f"\n8. MICRO-INTENT MANDATE (MARRIAGE TIMING):\n"
                f"   - The user is asking specifically WHEN marriage or commitment will happen.\n"
                f"   - Focus on 7th house ({_h7_sign}, lord {_h7_lord}) activation by Jupiter/Saturn transit and the active {active_dasha_str} Dasha sub-period windows.\n"
                f"   - Give a concrete timing window (e.g., specific months or Antardasha period) rather than vague statements.\n"
                f"   - DO NOT describe partner personality traits or generic marriage advice.\n"
            )
        elif m_intent == "marriage_remedies":
            micro_directive = (
                f"\n8. MICRO-INTENT MANDATE (MARRIAGE REMEDIES):\n"
                f"   - The user is asking for specific astrological remedies for delays or obstacles in marriage.\n"
                f"   - Tailor remedies to {_h7_lord} (7th house lord of {_h7_sign}): propitiation mantras, fasting day, donation, or gemstone strengthening.\n"
                f"   - Include at least one practical remedy (mantra/fasting/donation) and one lifestyle guideline.\n"
            )
        elif m_intent == "business_vs_job":
            micro_directive = (
                f"\n8. MICRO-INTENT MANDATE (BUSINESS VS JOB):\n"
                f"   - The user is asking whether their chart ({_asc} Lagna) favors independent business or structured employment.\n"
                f"   - Contrast 6th house ({_h6}) for service/employment versus 10th house ({_h10_sign}, lord {_h10_lord}) and 7th house ({_h7_sign}, lord {_h7_lord}) for commerce/partnership.\n"
                f"   - Give a clear verdict (business or job) backed by these house lords and the active {active_dasha_str} Dasha.\n"
            )
        elif m_intent == "job_timing":
            micro_directive = (
                f"\n8. MICRO-INTENT MANDATE (JOB CHANGE & PROMOTION TIMING):\n"
                f"   - The user is asking WHEN they will get a job, change jobs, or get a promotion.\n"
                f"   - Reference the 10th house ({_h10_sign}, lord {_h10_lord}) and 11th house of gains ({_h11_lord}) transit activation.\n"
                f"   - Cite the active {_antardasha_label} within {_mahadasha} Mahadasha as the timing anchor and give a concrete 3-4 month window.\n"
                f"   - Include specific interview preparation or application advice—NOT generic career fluff.\n"
            )
        elif m_intent == "career_path":
            micro_directive = (
                f"\n8. MICRO-INTENT MANDATE (CAREER PATH):\n"
                f"   - The user is asking which career or vocational field best suits their chart.\n"
                f"   - Map the 10th house sign ({_h10_sign}) and its lord ({_h10_lord}) directly to specific career disciplines and natural aptitudes.\n"
                f"   - Reference the {_asc} Lagna's constitutional strengths as the secondary vocational filter.\n"
                f"   - DO NOT give generic 'explore multiple options' advice—give a direct, chart-specific career verdict.\n"
            )
        elif m_intent == "gemstone_remedies":
            micro_directive = (
                f"\n8. MICRO-INTENT MANDATE (GEMSTONE REMEDIES):\n"
                f"   - The user is asking which gemstone suits them for career or fortune enhancement.\n"
                f"   - Name the EXACT gemstone for the {_asc} Lagna lord ({_lagna_lord}) and the 10th house lord ({_h10_lord}) explicitly.\n"
                f"   - Specify the metal, wearing day, and finger—do NOT give vague warnings without a concrete recommendation.\n"
            )

    system_prompt_str = f"""You are Astro AI, a precision Vedic Astrology assistant.

RULES FOR PERSONALIZED RESPONSES:
1. USER ISOLATION: The user's birth details provided below are unique to them. Tailor your response strictly to their specific Lagna, houses, and active Dasha. Never output generic or boilerplate text that could apply to anyone else.
2. INTENT ALIGNMENT: Directly answer the user's specific micro-intent (e.g., if they ask about partner traits, focus entirely on 7th house/Venus characteristics; do not drift into career or timing).
3. ZERO HALLUCINATION: Use only the exact planetary placements and timeline data supplied in the user profile below. Do not invent unverified predictions.
4. TOKEN BUDGET & LENGTH: Keep the response strictly between 5 to 7 lines total (50–65 words max). Never truncate mid-sentence.
5. STRUCTURE (mandatory for every response):
   **Astrological Insight:** [2–3 sentences connecting this specific user's unique chart facts to their question]
   **Actionable Advice:** [2 practical, grounded next steps tailored to their profile]

### USER NATAL DATA (bind all responses to these exact coordinates):
- Seeker Name: {seeker_name}
- Birth Details (Date & Time): {birth_details_str}
- Birth Place: {birth_place_str}
- Ascendant (Lagna): {user_natal_context.get('ascendant', 'Not specified')}
- Moon Sign: {user_natal_context.get('moon_sign', 'Not specified')}
- Sun Sign: {user_natal_context.get('sun_sign') or 'Not specified in profile'}
- Active Dasha: {active_dasha_str}
- Pre-Calculated House Lords: {user_natal_context.get('house_rulerships', 'Calculated based on Lagna')}

### ADDITIONAL FORMATTING CONSTRAINTS:
- TONE & VOCABULARY: Clear, beginner-friendly, and concise. Avoid dense jargon or rambling.
- SOURCE BADGE: The system wrapper appends the model/source tag automatically. Do NOT generate your own.
- {language_rule}
- COMPLETION & PUNCTUATION: Every paragraph must end with proper punctuation ('.'). Complete all sentences cleanly within the budget.
- PERSONALIZATION MANDATE: Reference {seeker_name}'s chart coordinates ({birth_details_str} | {birth_place_str}) in the **Astrological Insight** to establish this reading is customized specifically for their birth chart.{micro_directive}
"""
    return system_prompt_str


# ---------------------------------------------------------------------------
# [USER ISOLATION & DYNAMIC TWO-STEP PIPELINE]
# ---------------------------------------------------------------------------
def build_user_dynamic_payload(
    user_query: str,
    user_profile: Optional[Any] = None,
    user_natal_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Two-Step User-Isolated Pipeline:
    Step 1: Identify the exact micro-intent of the query ('career_path', 'job_timing', 'marriage_timing', 'partner_traits', etc.).
    Step 2: Extract active user's unique birth parameters from their session/profile state (Lagna, Moon Sign, 10th House, 7th House, active Dasha),
            and inject ONLY the user-specific chart factors relevant to that micro-intent.
    Anti-Hallucination Guardrail:
      - Restricts LLM to explicitly provided coordinates. Flags missing data to trigger calculation via FreeAstrologyAPI.
    Token Range Enforcement:
      - Enforces max_tokens=140 to fit inside total token budget while guaranteeing 5-7 lines, 50-65 words.
    """
    # 1. Session-Based User Isolation: resolve or extract natal context
    if not user_natal_context:
        user_natal_context = get_user_natal_context(user_profile=user_profile)

    asc = user_natal_context.get("ascendant") or "Libra"
    moon = user_natal_context.get("moon_sign") or "Gemini"
    sun = user_natal_context.get("sun_sign") or ""
    dasha = user_natal_context.get("active_dasha") or "Rahu-Mars"
    rulerships = user_natal_context.get("house_rulerships") or calculate_house_rulerships(asc)

    # Missing Data Detection for FreeAstrologyAPI Grounding Guardrail
    missing_data = []
    if not user_natal_context.get("ascendant") and not user_profile:
        missing_data.append("ascendant")
    if not user_natal_context.get("active_dasha") and not user_profile:
        missing_data.append("active_dasha")

    # Step 1: Precise Micro-Intent Classification
    micro_intent = classify_micro_intent(user_query) or "general"
    is_hinglish = is_hinglish_query(user_query)

    # Step 2: Inject ONLY relevant factors based on micro-intent
    relevant_factors: Dict[str, Any] = {
        "ascendant": asc,
        "active_dasha": dasha,
    }

    if micro_intent in ("career_path", "job_timing", "business_vs_job", "gemstone_remedies"):
        relevant_factors["10th_house"] = rulerships.get("10th_house", "Moon (Cancer)")
        if micro_intent == "business_vs_job":
            relevant_factors["6th_house"] = rulerships.get("6th_house", "Jupiter (Pisces)")
            relevant_factors["7th_house"] = rulerships.get("7th_house", "Mars (Aries)")
        elif micro_intent == "job_timing":
            relevant_factors["11th_house"] = rulerships.get("11th_house", "Sun (Leo)")
        elif micro_intent == "gemstone_remedies":
            relevant_factors["1st_house"] = rulerships.get("1st_house", f"Venus ({asc})")
            relevant_factors["5th_house"] = rulerships.get("5th_house", "Saturn (Aquarius)")
            relevant_factors["9th_house"] = rulerships.get("9th_house", "Mercury (Gemini)")
    elif micro_intent in ("partner_traits", "marriage_timing", "marriage_remedies"):
        relevant_factors["7th_house"] = rulerships.get("7th_house", "Mars (Aries)")
        relevant_factors["moon_sign"] = moon
        relevant_factors["venus_sign"] = user_natal_context.get("venus_sign", rulerships.get("7th_house", "Venus"))
        relevant_factors["jupiter_sign"] = user_natal_context.get("jupiter_sign", "Jupiter")
    else:
        relevant_factors["moon_sign"] = moon
        relevant_factors["sun_sign"] = sun

    # Build dynamically grounded system prompt
    system_prompt = build_system_prompt(
        user_natal_context=user_natal_context,
        query=user_query,
    )

    # Anti-Hallucination & Factual Grounding Mandate
    anti_hallucination_note = (
        "\n### ANTI-HALLUCINATION & FACTUAL GROUNDING MANDATE:\n"
        f"- Analyze strictly using the user's explicit factors: {json.dumps(relevant_factors)}.\n"
        "- Do NOT invent or fabricate planets, houses, or aspects not present in the user profile.\n"
        "- Ground your insights exclusively in sidereal Vedic principles matching their chart."
    )
    system_prompt += anti_hallucination_note

    # Sliding history limited to 2 turns
    filtered_history = filter_chat_history(chat_history or [], max_messages=2)

    return {
        "session_id": session_id or "default_session",
        "micro_intent": micro_intent,
        "is_hinglish": is_hinglish,
        "user_natal_context": user_natal_context,
        "relevant_factors": relevant_factors,
        "system_prompt": system_prompt,
        "chat_history": filtered_history,
        "max_tokens": MAX_TOKENS,  # 140 tokens
        "temperature": DEFAULT_TEMPERATURE,  # 0.4
        "missing_data": missing_data,
        "requires_api_calculation": len(missing_data) > 0,
    }


async def execute_dynamic_chat_pipeline(
    user_query: str,
    user_profile: Optional[Any] = None,
    user_natal_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[Tuple[str, str], None]:
    """
    Executes the user-isolated 2-step pipeline:
    1. Checks local rule/QA database rendered dynamically with the user's specific natal context.
    2. If bypassed or unserved, streams via cascading LLMs using the user-isolated payload (max_tokens=140).
    """
    payload = build_user_dynamic_payload(
        user_query=user_query,
        user_profile=user_profile,
        user_natal_context=user_natal_context,
        session_id=session_id,
        chat_history=chat_history,
    )

    # Check local QA/Stage-8 rule engine first
    from backend.main import check_qa_database
    local_resp = check_qa_database(
        user_query,
        profile=user_profile,
        user_natal_context=payload["user_natal_context"],
        session_id=session_id,
        chat_history=chat_history,
    )
    if local_resp:
        yield (local_resp, "Local Database")
        return

    # Stream dynamically via cascading LLM with user-isolated payload
    async for chunk, model_name in stream_cascading_router(
        prompt=user_query,
        system_prompt=payload["system_prompt"],
        history=payload["chat_history"],
        session_id=session_id,
        user_natal_context=payload["user_natal_context"],
    ):
        yield (chunk, model_name)


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

    # Step 2b: Strip any leaked AI badge, Astrological Guidance/Lagna headers, or Everyday Analogy sections
    cleaned = re.sub(r"^\s*(?:✨\s*)?(?:AI\s*[-:]\s*)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:^|\n)\s*\*\*[^*]*(?:Astrological Guidance|Lagna\s*\||\bMoon\b)[^*]*\*\*\s*(?:\n+|$)", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:^|\n)\s*(?:[-*•]\s*)?(?:\*\*|\b)?(?:Everyday\s+)?Analogy(?:\*\*|:)?\s*:[^\n]*(?:\n+|$)", "\n", cleaned, flags=re.IGNORECASE)

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


def ensure_two_paragraph_layout(text: str) -> str:
    """
    Guarantees the strict 2-paragraph layout:
    **Astrological Insight:** (2–3 sentences)
    **Actionable Advice:** (2 sentences)
    """
    if not text:
        return ""
    # If already contains the standard headers, format cleanly with double newline
    if "**Astrological Insight:**" in text and "**Actionable Advice:**" in text:
        parts = text.split("**Actionable Advice:**", 1)
        p1 = parts[0].strip()
        p2 = "**Actionable Advice:** " + parts[1].strip()
        return f"{p1}\n\n{p2}"

    # If headers are missing, split sentences and wrap into 2-paragraph layout
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if len(sentences) >= 4:
        insight = " ".join(sentences[:min(3, len(sentences)-2)])
        advice = " ".join(sentences[min(3, len(sentences)-2):])
        return f"**Astrological Insight:** {insight}\n\n**Actionable Advice:** {advice}"
    elif len(sentences) >= 2:
        insight = sentences[0]
        advice = " ".join(sentences[1:])
        return f"**Astrological Insight:** {insight}\n\n**Actionable Advice:** {advice}"
    else:
        return f"**Astrological Insight:** {text}\n\n**Actionable Advice:** Maintain steady focus and take grounded action aligned with your chart."


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
    3. Enforce the structured 2-paragraph layout (**Astrological Insight:** and **Actionable Advice:**).
    4. Strip server metadata unless include_suffix=True is explicitly requested.
    5. Append provenance source badge if requested.
    """
    if not text:
        return ""

    # 1. Run string cleaning / strip reasoning or meta-tags
    clean_text = clean_llm_output(text)

    # 2. Apply trim_to_last_sentence() so output always ends cleanly with '.', '!', or '?'
    clean_text = trim_to_last_sentence(clean_text)

    # 3. Enforce structured 2-paragraph layout
    clean_text = ensure_two_paragraph_layout(clean_text)

    # 4. Strip server metadata
    clean_text = strip_server_metadata(clean_text)

    # 5. Dynamic source provenance badge
    if include_suffix and model_id:
        return f"{clean_text}{format_source_badge(model_id)}"

    return clean_text


async def sanitize_stream(stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """
    Real-time streaming sanitizer:
    Buffers initial tokens to detect and strip <think> blocks and meta-preambles.
    Preserves bold headers (**Astrological Insight:** and **Actionable Advice:**)
    and paragraph breaks for the 2-paragraph layout.
    """
    buffer = ""
    in_think_block = False
    started_real_content = False

    async for chunk in stream:
        if started_real_content:
            if chunk:
                yield chunk
            continue

        buffer += chunk

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

        # Once buffer has non-empty text, flush buffer and start direct streaming
        if len(buffer) >= 15 or "\n" in buffer:
            started_real_content = True
            yield buffer
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
    during backend runtime ONLY when verified as decommissioned across all available accounts with valid keys.
    """
    # Guard: Verify that accounts with valid API keys actually exist before globally blacklisting
    accounts = get_groq_accounts()
    if not accounts:
        logger.warning(
            "[KEY AVAILABILITY] Skipping permanent blacklist for model %s: No active Groq accounts with valid API keys configured.",
            model_name
        )
        return

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
    """Marks a model as unavailable for a specific account without permanently blacklisting globally."""
    key = f"{account_id}::{model_name}"
    entry = ACCOUNT_MODEL_QUOTA_REGISTRY.setdefault(key, {})
    entry["is_exhausted"] = True
    entry["cooldown_until"] = time.time() + 3600.0  # 1-hour cooldown before re-evaluating
    entry["bypass_reason"] = f"Model unavailable on {account_id} ({reason})"
    logger.warning(
        "[MODEL STATUS] Model %s on %s marked unavailable: %s. Proactively bypassing for this account...",
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
            if account_id:
                mark_account_model_unavailable(account_id, model, reason=f"HTTP {sc}: {ase}")
        raise
    except Exception as exc:
        sc = getattr(exc, "status_code", None)
        exc_str = str(exc).lower()
        if sc == 429 or "429" in exc_str or "rate limit" in exc_str:
            if account_id:
                mark_account_rate_limited(account_id, 60.0)
                mark_account_model_rate_limited(account_id, model, 60.0)
        elif is_decommissioned_or_404_error(sc, exc):
            if account_id:
                mark_account_model_unavailable(account_id, model, reason=f"HTTP {sc or 'ERR'}: {exc}")
        # Re-raise so cascading router can try next model
        raise
    response_headers = raw_response.headers

    stream = await raw_response.parse()
    usage_logged = False
    usage_data = None
    tokens_streamed = 0

    # Stream chunks immediately with zero blocking overhead
    async for chunk in stream:
        try:
            choices = chunk.choices if chunk.choices else []
            delta = choices[0].delta if choices else None
        except (IndexError, AttributeError):
            delta = None
        if delta and delta.content:
            text = delta.content.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            if text:
                tokens_streamed += 1
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

        # Capture provider-level quota from Groq rate-limit headers
        prov_remaining = quota_entry.get("remaining_tokens")
        prov_limit = quota_entry.get("limit_tokens")
        prov_pct = quota_entry.get("remaining_pct")

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

        # Store stats so the streaming endpoint can emit a [TOKEN_USAGE] line
        store_last_token_stats(
            session_id, model, account_id,
            prompt_tokens, completion_tokens, total_tokens,
            session_total_used, remaining_tokens, percentage_remaining,
            provider_remaining=prov_remaining,
            provider_limit=prov_limit,
            provider_remaining_pct=prov_pct,
        )


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

        # Capture provider-level quota from OpenRouter rate-limit headers
        prov_remaining = quota_entry.get("remaining_tokens")
        prov_limit = quota_entry.get("limit_tokens")
        prov_pct = quota_entry.get("remaining_pct")

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

        # Store stats so the streaming endpoint can emit a [TOKEN_USAGE] line
        store_last_token_stats(
            session_id, model, None,
            prompt_tokens, completion_tokens, total_tokens,
            session_total_used, remaining_tokens, percentage_remaining,
            provider_remaining=prov_remaining,
            provider_limit=prov_limit,
            provider_remaining_pct=prov_pct,
        )


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
    try:
        async for chunk in response_stream:
            # Guard: skip chunks where the SDK would raise "model output must contain
            # either output text or tool calls" — happens on safety-blocked responses.
            try:
                chunk_text = chunk.text
            except Exception:
                chunk_text = None

            if chunk_text:
                yield chunk_text

            # Check finish_reason for safety blocks so we can surface a clear warning
            candidates = getattr(chunk, "candidates", None) or []
            for candidate in candidates:
                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason and str(finish_reason) in ("SAFETY", "RECITATION", "OTHER", "BLOCKLIST"):
                    logger.warning(
                        "[GEMINI BLOCKED] Model %s chunk finish_reason=%s. Yielding nothing for this chunk.",
                        model, finish_reason,
                    )

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

                # Store stats so the streaming endpoint can emit a [TOKEN_USAGE] line
                store_last_token_stats(
                    session_id, model, None,
                    prompt_tokens, completion_tokens, total_tokens,
                    session_total_used, remaining_tokens, percentage_remaining,
                )
    except Exception as gemini_stream_exc:
        exc_msg = str(gemini_stream_exc).lower()
        if "model output" in exc_msg and ("output text" in exc_msg or "tool calls" in exc_msg):
            # Re-raise with a clear tag so stream_cascading_router catches and cascades
            raise ValueError(
                f"model output must contain either output text or tool calls: {gemini_stream_exc}"
            ) from gemini_stream_exc
        raise


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
        system_prompt = build_system_prompt(user_natal_context=user_natal_context, query=prompt)
    elif system_prompt and "LANGUAGE REGISTER" not in system_prompt and "LANGUAGE MANDATE" not in system_prompt:
        is_hing = is_hinglish_query(prompt)
        if is_hing:
            system_prompt += (
                "\n\n### MANDATORY LANGUAGE MANDATE (HINGLISH):\n"
                "The user asked in HINGLISH. You MUST respond ENTIRELY in conversational HINGLISH (Hindi in Roman script). "
                "DO NOT respond in pure English."
            )
        else:
            system_prompt += (
                "\n\n### MANDATORY LANGUAGE MANDATE (ENGLISH):\n"
                "The user asked in ENGLISH. You MUST respond STRICTLY in clear, professional ENGLISH. "
                "DO NOT use Hindi or Hinglish words."
            )
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
                        mark_account_model_unavailable(acc_id, model_name, reason=f"HTTP {status_code}: {ase}")
                        if acc_idx + 1 < len(accounts):
                            next_acc = accounts[acc_idx + 1]["account_id"]
                            logger.warning(
                                "[MODEL ACCESS] Model %s not accessible on %s (HTTP %s in %.1fms). Failing over to %s...",
                                model_name, acc_id, status_code, elapsed_ms, next_acc
                            )
                            continue
                        else:
                            logger.warning(
                                "[MODEL UNAVAILABLE] Model %s unavailable across all %d configured Groq accounts (HTTP %s in %.1fms). Cascading to next model...",
                                model_name, len(accounts), status_code, elapsed_ms
                            )
                            if "decommission" in str(ase).lower():
                                mark_model_deprecated(model_name, reason=f"HTTP {status_code}: decommissioned")
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
                    exc_str = str(exc).lower()

                    # Handle "model output must contain either output text or tool calls" empty-response error
                    # Treat as a soft failure: cascade to next account/model instead of crashing
                    if "model output" in exc_str and ("output text" in exc_str or "tool calls" in exc_str):
                        logger.warning(
                            "[EMPTY OUTPUT] Groq model %s on %s returned empty response (%.1fms). Cascading to next account/model...",
                            model_name, acc_id, elapsed_ms
                        )
                        if tokens_yielded > 0:
                            return
                        continue

                    if is_decommissioned_or_404_error(sc, exc):
                        mark_account_model_unavailable(acc_id, model_name, reason=f"HTTP {sc or 'ERR'}: {exc}")
                        if acc_idx + 1 < len(accounts):
                            next_acc = accounts[acc_idx + 1]["account_id"]
                            logger.warning(
                                "[MODEL ACCESS] Model %s not accessible on %s (HTTP %s in %.1fms). Failing over to %s...",
                                model_name, acc_id, sc, elapsed_ms, next_acc
                            )
                            continue
                        else:
                            logger.warning(
                                "[MODEL UNAVAILABLE] Model %s unavailable across all %d configured Groq accounts (HTTP %s in %.1fms). Cascading to next model...",
                                model_name, len(accounts), sc, elapsed_ms
                            )
                            if "decommission" in str(exc).lower():
                                mark_model_deprecated(model_name, reason=f"HTTP {sc or 'ERR'}: decommissioned")
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
                exc_str = str(exc).lower()
                if "model output" in exc_str and ("output text" in exc_str or "tool calls" in exc_str):
                    logger.warning(
                        "[EMPTY OUTPUT] OpenRouter model %s returned empty response (%.1fms). Cascading...",
                        model_name, elapsed_ms
                    )
                else:
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
                sc = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                exc_str = str(exc).lower()
                is_429 = sc == 429 or "429" in exc_str or "rate limit" in exc_str or "resource_exhausted" in exc_str
                is_empty_output = "model output" in exc_str and ("output text" in exc_str or "tool calls" in exc_str)
                if is_429:
                    logger.warning(
                        "[GEMINI 429 RATE LIMIT] Gemini safety model %s hit 429 in %.1fms. Seamlessly rotating to next fallback...",
                        gemini_model, elapsed_ms
                    )
                elif is_empty_output:
                    logger.warning(
                        "[GEMINI EMPTY OUTPUT] Gemini model %s returned empty/blocked response in %.1fms. Cascading to next model...",
                        gemini_model, elapsed_ms
                    )
                else:
                    logger.warning(
                        "Gemini safety model %s failed in %.1fms: %s. Cascading...",
                        gemini_model, elapsed_ms, exc
                    )
                if tokens_yielded > 0:
                    return
                continue

    # Tier 4: Deterministic Local Calculation Fallback
    # If ALL LLM cascades fail or rate limit (HTTP 429), generate grounded Vedic advice instead of showing "System Busy"
    logger.warning("All LLM cascades exhausted / rate limited. Generating local calculation fallback response...")
    fallback_response = generate_local_rule_fallback(prompt, user_natal_context=user_natal_context)
    yield fallback_response, "Local Database"


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
            if account_id:
                mark_account_model_unavailable(account_id, model, reason=f"HTTP {sc}: {ase}")
        raise
    except Exception as exc:
        sc = getattr(exc, "status_code", None)
        if sc == 429 or "429" in str(exc) or "rate limit" in str(exc).lower():
            if account_id:
                mark_account_rate_limited(account_id, 60.0)
                mark_account_model_rate_limited(account_id, model, 60.0)
        elif is_decommissioned_or_404_error(sc, exc):
            if account_id:
                mark_account_model_unavailable(account_id, model, reason=f"HTTP {sc or 'ERR'}: {exc}")
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

    # Guard: accessing response.text raises ValueError when the response is blocked/empty.
    # Safely extract text and fall back to empty string so the caller can cascade.
    try:
        raw_text = response.text or ""
    except Exception as text_exc:
        exc_msg = str(text_exc).lower()
        if "model output" in exc_msg and ("output text" in exc_msg or "tool calls" in exc_msg):
            logger.warning(
                "[GEMINI EMPTY] Model %s returned empty/blocked non-streaming response: %s",
                model, text_exc,
            )
            raise ValueError(
                f"model output must contain either output text or tool calls: {text_exc}"
            ) from text_exc
        raise

    # Response Formatting Sequence: 1. clean_llm_output -> 2. trim_to_last_sentence -> 3. UI suffix
    cleaned_text = format_llm_response(raw_text, model_id=model, include_suffix=include_suffix)

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
                        mark_account_model_unavailable(acc_id, model_name, reason=f"HTTP {status_code}: {ase}")
                        if acc_idx + 1 < len(accounts):
                            next_acc = accounts[acc_idx + 1]["account_id"]
                            logger.warning(
                                "[MODEL ACCESS] Model %s not accessible on %s (HTTP %s). Failing over to %s...",
                                model_name, acc_id, status_code, next_acc
                            )
                            continue
                        else:
                            logger.warning(
                                "[MODEL UNAVAILABLE] Model %s unavailable across all %d configured Groq accounts (HTTP %s). Cascading to next model...",
                                model_name, len(accounts), status_code
                            )
                            if "decommission" in str(ase).lower():
                                mark_model_deprecated(model_name, reason=f"HTTP {status_code}: decommissioned")
                            break
                    continue
                except Exception as exc:
                    sc = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                    if is_decommissioned_or_404_error(sc, exc):
                        mark_account_model_unavailable(acc_id, model_name, reason=f"HTTP {sc or 'ERR'}: {exc}")
                        if acc_idx + 1 < len(accounts):
                            next_acc = accounts[acc_idx + 1]["account_id"]
                            logger.warning(
                                "[MODEL ACCESS] Model %s not accessible on %s (HTTP %s). Failing over to %s...",
                                model_name, acc_id, sc, next_acc
                            )
                            continue
                        else:
                            logger.warning(
                                "[MODEL UNAVAILABLE] Model %s unavailable across all %d configured Groq accounts (HTTP %s). Cascading to next model...",
                                model_name, len(accounts), sc
                            )
                            if "decommission" in str(exc).lower():
                                mark_model_deprecated(model_name, reason=f"HTTP {sc or 'ERR'}: decommissioned")
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

    # Tier 4: Deterministic Local Calculation Fallback
    logger.warning("All LLM cascades exhausted / rate limited (non-streaming). Generating local calculation fallback response...")
    fallback_response = generate_local_rule_fallback(prompt, user_natal_context=user_natal_context)
    return fallback_response, "Local Database", {
        "status": "local_fallback",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }
