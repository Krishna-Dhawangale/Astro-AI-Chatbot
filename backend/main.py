import asyncio
import datetime
import inspect
import json
import logging
import os
import re
import sys
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

GEMINI_SAFETY_SETTINGS = [
    types.SafetySetting(
        category=category,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    )
    for category in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]

from requests.adapters import HTTPAdapter
from urllib3.util import Retry

try:
    from router import DOMAIN_KEYWORDS, IntentRouter
    from qa_engine import intercept_qa, strip_greeting
    from llm_router import (
        stream_chat_response,
        stream_cascading_router,
        build_system_prompt,
        compress_user_profile,
        filter_chat_history,
        is_conversational_fluff,
        clean_llm_output,
        trim_to_last_sentence,
        format_llm_response,
        strip_server_metadata,
        STRUCTURED_SYSTEM_PROMPT,
        ULTRA_COMPACT_SYSTEM_PROMPT,
        format_compact_user_profile,
        apply_sliding_window,
        GROQ_MODELS,
        MODEL_CASCADE,
        get_local_qa_response,
        qa_database_index,
        should_bypass_local_qa,
        SYSTEM_BUSY_MESSAGE,
        SYSTEM_BUSY_PAYLOAD,
        get_system_busy_payload,
        calculate_house_rulerships,
        get_user_natal_context,
        format_local_qa_response,
    )
except ImportError:
    from backend.router import DOMAIN_KEYWORDS, IntentRouter
    from backend.qa_engine import intercept_qa, strip_greeting
    from backend.llm_router import (
        stream_chat_response,
        stream_cascading_router,
        build_system_prompt,
        compress_user_profile,
        filter_chat_history,
        is_conversational_fluff,
        clean_llm_output,
        trim_to_last_sentence,
        format_llm_response,
        strip_server_metadata,
        STRUCTURED_SYSTEM_PROMPT,
        ULTRA_COMPACT_SYSTEM_PROMPT,
        format_compact_user_profile,
        apply_sliding_window,
        GROQ_MODELS,
        MODEL_CASCADE,
        get_local_qa_response,
        qa_database_index,
        should_bypass_local_qa,
        SYSTEM_BUSY_MESSAGE,
        SYSTEM_BUSY_PAYLOAD,
        get_system_busy_payload,
        calculate_house_rulerships,
        get_user_natal_context,
        format_local_qa_response,
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

FRONTEND_PATH = os.path.join(BASE_DIR, "..", "frontend", "index.html")

load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
load_dotenv(override=True)

logger = logging.getLogger(__name__)
RATE_LIMIT_STORAGE_URI = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")
limiter = Limiter(key_func=get_remote_address, storage_uri=RATE_LIMIT_STORAGE_URI)

# Expanded outgoing HTTP connection pool to prevent socket exhaustion during concurrent bursts
_requests_session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=200,
    pool_maxsize=200,
    max_retries=Retry(total=3, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504]),
)
_requests_session.mount("http://", _adapter)
_requests_session.mount("https://", _adapter)

app = FastAPI(
    title="Astrology AI Chatbot & External API Backend",
    description="FastAPI service optimized for dynamic, user-specific AI predictions."
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Gracefully catches local SlowAPI rate limits and returns HTTP 200 OK
    with the system busy payload instead of raw 429 client errors.
    """
    logger.warning("Local SlowAPI rate limit triggered for %s", request.client.host if request.client else "unknown")
    return JSONResponse(
        status_code=200,
        content=dict(SYSTEM_BUSY_PAYLOAD),
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)


def label_answer_source(answer: str, source: str) -> str:
    label = f"({source})"
    answer = answer.rstrip()
    return answer if answer.endswith(label) else f"{answer} {label}"


def backend_error_payload(query: str) -> Dict[str, Any]:
    answer = SYSTEM_BUSY_MESSAGE
    return {
        "status": "busy",
        "response": answer,
        "answer": label_answer_source(answer, "Backend"),
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "related_questions": [],
    }


def is_daily_gemini_quota_error(error: Exception) -> bool:
    error_message = str(error).casefold()
    return (
        ("free_tier_requests" in error_message or "perday" in error_message)
        and ("quota" in error_message or "resource_exhausted" in error_message)
    )


def gemini_error_message(error: Exception, query: str) -> str:
    if is_daily_gemini_quota_error(error):
        answer = (
            "Gemini's daily free-tier quota for this model has been reached. "
            "Please try again after the quota resets or enable billing/increase the project quota."
        )
        return label_answer_source(answer, "Backend")

    status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if status_code == 429:
        answer = "Gemini is temporarily rate limited. Please wait a moment and try again."
        return label_answer_source(answer, "Backend")

    return backend_error_payload(query)["answer"]


def is_retryable_gemini_error(error: Exception) -> bool:
    status_codes = [
        getattr(error, "code", None),
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ]
    for status_code in status_codes:
        if status_code is None:
            continue
        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            continue
        if status_code == 429 or 500 <= status_code <= 599:
            return True

    error_type = type(error).__name__.casefold()
    error_message = str(error).casefold()
    return (
        isinstance(error, (TimeoutError, ConnectionError, requests.RequestException))
        or any(token in error_type for token in ("timeout", "connect", "network"))
        or any(token in error_message for token in ("timed out", "connection reset", "temporarily unavailable"))
    )

# Initialize Google GenAI Client
gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
gemini_client = None
if gemini_api_key:
    try:
        gemini_client = genai.Client(api_key=gemini_api_key)
        print("[OK] Google GenAI Client configured successfully.")
    except Exception as e:
        print(f"[WARNING] Failed to initialize Google GenAI Client: {e}")
else:
    print("[WARNING] GOOGLE_API_KEY or GEMINI_API_KEY is missing. Gemini responses will be unavailable.")

FREE_ASTROLOGY_KEY = os.getenv("FREE_ASTROLOGY_API_KEY")
PROKERALA_CLIENT_ID = os.getenv("PROKERALA_CLIENT_ID")
PROKERALA_CLIENT_SECRET = os.getenv("PROKERALA_CLIENT_SECRET")

router_model_path = (
    os.path.join(BASE_DIR, "astrology_predict_model.pkl")
    if os.path.exists(os.path.join(BASE_DIR, "astrology_predict_model.pkl"))
    else "astrology_predict_model.pkl"
)
router = IntentRouter(router_model_path)

# ------------------------------------------------------------------
# Hinglish → English Normalization Map
# Maps common Hinglish/Hindi words used by Indian users to their
# English equivalents so all matchers & routers work uniformly.
# ------------------------------------------------------------------
HINGLISH_MAP: Dict[str, str] = {
    # Greetings
    "namaste": "hello", "namaskar": "hello", "jai hind": "hello",
    "shukriya": "thank you", "dhanyawad": "thank you",
    "theek hai": "okay", "thik hai": "okay", "bilkul": "absolutely",
    # Career / Job
    "naukri": "job", "nokri": "job", "kaam": "work", "vyapar": "business",
    "dhandha": "business", "rojgar": "employment", "udyog": "business",
    "kab milegi naukri": "when will i get job", "promotion milega": "will i get promotion",
    "job milegi": "will i get job", "job chahiye": "i want a job",
    "safalta": "success", "tarakki": "promotion", "tarrakki": "promotion",
    "vyavsay": "business", "parishram": "hard work",
    # Finance / Money
    "paisa": "money", "paise": "money", "dhan": "wealth", "sampatti": "property",
    "nivesh": "investment", "bachat": "savings", "karz": "debt", "karja": "loan",
    "ameer": "rich", "daulat": "wealth", "kamao": "earn", "kamayi": "income",
    "paisa aayega": "will i get money", "paise ki problem": "financial problem",
    # Marriage / Relationship
    "shaadi": "marriage", "shadi": "marriage", "vivah": "marriage",
    "rishta": "relationship", "pyaar": "love", "prem": "love",
    "dulhan": "bride", "dulha": "groom", "patni": "wife", "pati": "husband",
    "jeevan saathi": "life partner", "saathi": "partner",
    "kab hogi shaadi": "when will i get married", "shaadi kab hogi": "when will i get married",
    "love marriage hogi": "will i have love marriage",
    # Health
    "sehat": "health", "swasthya": "health", "bimari": "disease",
    "dard": "pain", "ilaj": "treatment", "dawai": "medicine",
    "tanav": "stress", "chinta": "anxiety", "depression": "depression",
    "theek honga": "will i recover", "sehat kaisi rahegi": "how is my health",
    # Travel
    "yatra": "travel", "safar": "journey", "videsh": "foreign",
    "videsh jana": "go abroad", "bahar jana": "go abroad",
    "pardesh": "foreign land", "desh chhodna": "relocate abroad",
    # Education
    "padhai": "education", "shiksha": "education", "pariksha": "exam",
    "imtihan": "exam", "college": "college", "degree": "degree",
    "pass honga": "will i pass exam", "fail honga": "will i fail exam",
    # Children / Family
    "bachcha": "child", "baccha": "child", "bacche": "children", "santaan": "children",
    "beta": "son", "beti": "daughter", "ghar": "home", "parivar": "family",
    "maa": "mother", "baap": "father", "maa baap": "parents",
    # Astrology terms
    "kundali": "birth chart", "janam patri": "birth chart", "janampatri": "birth chart",
    "rashi": "moon sign", "lagna": "ascendant", "graha": "planet",
    "shani": "saturn", "mangal": "mars", "guru": "jupiter",
    "surya": "sun", "chandra": "moon", "budh": "mercury",
    "shukra": "venus", "rahu": "rahu", "ketu": "ketu",
    "dasha": "dasha", "mahadasha": "mahadasha", "sade sati": "sade sati",
    "upay": "remedy", "totka": "remedy", "mantra": "mantra",
    "puja": "puja", "ratna": "gemstone",
    # General question words
    "kab": "when", "kya": "what", "kyun": "why", "kaisa": "how",
    "kaisi": "how", "kitna": "how much", "kaise": "how",
    "hoga": "will happen", "hogi": "will happen", "milega": "will get",
    "milegi": "will get", "chahiye": "want", "batao": "tell me",
    "bata do": "please tell", "karo": "do", "mera": "my", "meri": "my",
    "mujhe": "me", "acha": "good", "achha": "good",
    "sambhavna": "possibility", "sambhavana": "possibility",
}


def normalize_hinglish(text: str) -> str:
    """Normalize Hinglish/Hindi words to English equivalents for uniform processing.
    Handles multi-word phrases first (longest match priority), then single words.
    """
    normalized = text.lower().strip()
    # Sort by length descending so longer phrases replace before single words
    sorted_map = sorted(HINGLISH_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    for hindi, english in sorted_map:
        normalized = re.sub(r'\b' + re.escape(hindi) + r'\b', english, normalized)
    return normalized


def is_hinglish_query(text: str) -> bool:
    """Detect Hinglish using normalized vocabulary and common Hindi markers."""
    if normalize_hinglish(text).casefold() != text.casefold():
        return True
    hindi_markers = {
        "aap", "aapka", "aapki", "aapke", "hai", "hain", "ho", "hoga", "hogi",
        "ka", "ki", "ke", "ko", "kya", "kab", "kaise", "kyun", "mein", "nahi",
        "raha", "rahi", "rahe", "mujhe", "mera", "meri", "mere", "chahiye",
        "sakta", "sakti", "sakte", "badhega", "badhegi", "milega", "milega",
    }
    words = set(re.findall(r"\b[\w']+\b", text.casefold()))
    return bool(words & hindi_markers)


SEMANTIC_DOMAIN_SENTENCES = {
    "career": "Career promotion job switch raise salary increase work profession business employment interview",
    "marriage": "Marriage wedding relationship breakup partner spouse love life commitment compatibility",
    "wealth": "Money finance property real estate home purchase loan debt investment savings income",
    "health": "Mental stress anxiety burnout peace of mind health illness fitness recovery wellness",
    "travel": "Travel relocation moving abroad foreign country visa trip journey overseas",
    "education": "Education studying exams college university degree learning scholarship academic goals",
    "children": "Children family pregnancy baby parenthood son daughter fertility family planning",
}
SEMANTIC_DOMAIN_THRESHOLD = 0.35
_domain_embedder: Any = None
_domain_embeddings: Any = None
_embedding_model_unavailable = False


def _infer_embedding_domain(query: str) -> Optional[str]:
    """Classify an otherwise-unmatched query with cached local sentence embeddings."""
    global _domain_embedder, _domain_embeddings, _embedding_model_unavailable
    if _embedding_model_unavailable or not query.strip():
        return None

    try:
        from sentence_transformers import SentenceTransformer, util

        if _domain_embedder is None:
            _domain_embedder = SentenceTransformer("all-MiniLM-L6-v2")
            _domain_embeddings = _domain_embedder.encode(
                list(SEMANTIC_DOMAIN_SENTENCES.values()),
                convert_to_tensor=True,
            )

        query_embedding = _domain_embedder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, _domain_embeddings)[0]
        best_index = int(scores.argmax().item())
        if float(scores[best_index].item()) > SEMANTIC_DOMAIN_THRESHOLD:
            return list(SEMANTIC_DOMAIN_SENTENCES)[best_index]
    except Exception as error:
        _embedding_model_unavailable = True
        print(f"[WARNING] Semantic embedding domain matcher unavailable: {error}")
    return None


def infer_semantic_domain(query: str) -> Optional[str]:
    """Infer the domain from a query even when exact keywords don't match.
    Uses fuzzy string similarity against known domain anchor words.
    Returns a domain string or None if confidence is too low.
    """
    DOMAIN_ANCHORS = {
        "career": ["career", "job", "work", "profession", "business", "employment",
                   "promotion", "salary", "office", "company", "interview", "resign"],
        "marriage": ["marriage", "wedding", "love", "spouse", "partner", "relationship",
                     "husband", "wife", "compatibility", "soulmate", "commitment"],
        "wealth": ["money", "finance", "wealth", "investment", "property", "savings",
                   "income", "profit", "debt", "loan", "rich", "earn"],
        "health": ["health", "disease", "illness", "fitness", "stress", "pain",
                   "anxiety", "wellness", "recovery", "medicine", "doctor"],
        "travel": ["travel", "journey", "abroad", "foreign", "visa", "trip",
                   "relocate", "overseas", "settle", "migration"],
        "education": ["education", "study", "exam", "college", "degree", "university",
                      "learning", "scholarship", "academic", "school"],
        "children": ["child", "children", "baby", "pregnancy", "son", "daughter",
                     "family", "parents", "progeny", "conception"],
    }

    words = query.lower().split()
    best_domain = None
    best_score = 0.0
    FUZZY_THRESHOLD = 0.72

    for word in words:
        if len(word) < 3:
            continue
        for domain, anchors in DOMAIN_ANCHORS.items():
            for anchor in anchors:
                ratio = SequenceMatcher(None, word, anchor).ratio()
                if ratio > best_score and ratio >= FUZZY_THRESHOLD:
                    best_score = ratio
                    best_domain = domain

    return best_domain or _infer_embedding_domain(query)


def infer_query_domains(
    query: str,
    primary_domain: Optional[str] = None,
    *,
    include_semantic_fallback: bool = True,
) -> List[str]:
    """Return every explicit question domain while preserving a router fallback."""
    normalized = normalize_hinglish(query).casefold()
    signals = {
        "career": ["career", "careers", "job", "jobs", "work", "profession", "business", "promotion", "salary", "boss", "colleague", "freelance", "freelancing", "employment", "remote", "hybrid", "co-founder", "cofounder"],
        "marriage": ["marriage", "wedding", "love", "spouse", "partner", "relationship", "relationships", "husband", "wife"],
        "wealth": ["money", "finance", "finances", "wealth", "investment", "investments", "property", "savings", "income", "debt", "loan", "esop", "rsu", "equity", "stock", "shares", "cash", "liquidity"],
        "health": ["health", "disease", "illness", "fitness", "stress", "burnout", "anxiety", "recovery", "medicine"],
        "travel": ["travel", "journey", "abroad", "foreign", "visa", "relocation", "relocate", "migration", "trips"],
        "education": ["education", "study", "studies", "exam", "exams", "college", "degree", "university", "scholarship"],
        "children": ["child", "children", "baby", "pregnancy", "son", "daughter", "family", "parenthood"],
    }
    domains = [
        domain for domain, keywords in signals.items()
        if any(re.search(r"\b" + re.escape(keyword) + r"\b", normalized) for keyword in keywords)
    ]

    office_politics = re.search(
        r"\b(office|workplace|work)\s+(?:mein\s+|ki\s+|ke\s+|ka\s+|at\s+|in\s+)?politics\b|"
        r"\bpolitics\s+(at|in)\s+(the\s+)?(office|workplace)\b",
        normalized
    )
    if office_politics and "career" not in domains:
        domains.insert(0, "career")
    if not domains:
        if primary_domain in signals:
            domains.append(primary_domain)
        elif not include_semantic_fallback:
            domains.append("general")
        else:
            inferred = infer_semantic_domain(normalized)
            domains.append(inferred or "general")
    return domains


# ------------------------------------------------------------------
# Indirect / Paraphrased Intent Resolver
# Handles "ghuma-fira" questions — roundabout, vague, or indirect
# queries where the user doesn't state their topic explicitly.
# ------------------------------------------------------------------
INDIRECT_PATTERNS: List[Tuple[List[str], str]] = [
    # Career / Future
    (["kal kaisa rahega", "kal acha rahega", "kal kuch acha", "aage kya hoga",
      "future kaisa", "aage kya aane wala", "life mein kuch hoga", "kab tak",
      "kuch acha hoga", "achhe din aayenge", "sab theek hoga", "sab theek ho jayega",
      "i am not feeling good about life", "nothing is going well", "when things improve",
      "when will things get better", "life is hard", "struggling in life",
      "bahut mushkil chal raha", "mushkil waqt", "bura waqt kab khatam",
      "kab tak takleef rahegi", "dukh kab khatam", "pareshan hoon",
      "chinta ho rahi hai", "tension mein hoon", "stress mein hoon",
      "feel hopeless", "feel lost", "don't know what to do",
      "kya karun", "kya karna chahiye", "samajh nahi aa raha",
      "when will my time come", "mera waqt kab aayega", "mera accha time kab aayega",
      "kuch nahi ho raha", "koi result nahi aa raha", "mehnat bekar ja rahi hai"],
     "life_outlook"),

    # Career indirect
    (["koi opportunity nahi mil rahi", "job nahi lag rahi", "interview mein fail",
      "koi kaam nahi chal raha", "business mein nuksan", "loss in business",
      "office mein problem", "boss ke saath problem", "colleagues ke saath",
      "professionally kuch nahi ho raha", "career mein stuck", "stuck in career",
      "no growth", "not growing", "no progress", "koi progress nahi",
      "mehnat ka fal nahi mil raha", "hard work not paying off",
      "i work so hard but nothing works", "tried everything for job",
      "government job nahi mil rahi", "private job bhi nahi",
      "kab milega mujhe safalta", "safalta kab milegi"],
     "career"),

    # Money indirect
    (["paise ki kami", "ghar ka kharcha nahi chal raha", "financially weak",
      "financially struggling", "koi income nahi", "hath tang hai",
      "budget tight hai", "zyada kharcha ho raha", "paise nahi bache",
      "debt mein hoon", "karz uthana pad raha", "loan nahi utar raha",
      "khud ko financially stable karna chahta", "money problem",
      "wealth nahi aa rahi", "prosperity nahi hai"],
     "wealth"),

    # Marriage / Relationship indirect
    (["akela feel kar raha hoon", "lonely feel kar raha", "koi nahi samajhta",
      "rishta nahi ho raha", "shaadi nahi ho rahi", "no one to share life with",
      "partner ki zaroorat hai", "love nahi mil raha", "pyaar nahi mila",
      "relationship mein problem", "bf gf ke saath problem",
      "breakup ho gaya", "divorce ka dar", "patni se ladai", "pati se ladai",
      "ghar mein shanti nahi", "ghar mein kalesh", "family mein problem",
      "mujhe koi chahta nahi", "no one loves me", "feeling unloved"],
     "marriage"),

    # Health indirect
    (["thaka thaka feel karta hoon", "always tired", "hamesha thakaan",
      "neend nahi aati", "can't sleep", "insomnia", "bhook nahi lagti",
      "sehat theek nahi", "body weak", "immunity low", "baar baar bimaar",
      "mental peace nahi", "dimag mein bahut khayal", "overthinking",
      "mind is restless", "anxiety feel hoti hai", "feel anxious all the time",
      "depression feel hoti hai", "udaas rehta hoon", "khush nahi rehta",
      "joy nahi milti", "no motivation", "motivation nahi"],
     "health"),

    # Travel indirect
    (["videsh mein rehna chahta hoon", "bahar rehna chahta",
      "settle abroad", "want to go abroad", "foreign mein opportunity",
      "visa ke liye apply", "immigration ke liye", "relocation soch raha hoon",
      "desh chhodna chahta hoon", "apna country chhod ke"],
     "travel"),

    # Education indirect
    (["padhai mein mann nahi lagta", "concentration nahi hoti",
      "exam ke time ghabra jaata hoon", "fear of exams", "exam phobia",
      "result achha nahi ata", "marks kam ate hain", "studies mein weak",
      "college mein selection nahi ho raha", "admission nahi mila"],
     "education"),
]


def resolve_indirect_intent(query: str) -> Optional[str]:
    """Detect the hidden topic from roundabout, vague, or indirect questions.
    Returns a domain string if an indirect intent is found, else None.
    This handles 'ghuma-fira ke pooche gaye sawaal'.
    """
    q_lower = query.lower().strip()
    q_normalized = normalize_hinglish(q_lower)

    for patterns, domain in INDIRECT_PATTERNS:
        for pattern in patterns:
            # Normalized pattern match
            pat_normalized = normalize_hinglish(pattern)
            if pat_normalized in q_normalized or pattern in q_lower:
                return domain
            # Require three words so generic pairs like "when will" do not hijack intent.
            pat_words = pat_normalized.split()
            if len(pat_words) >= 3:
                for i in range(len(pat_words) - 2):
                    phrase_words = pat_words[i:i+3]
                    generic_words = {
                        "a", "an", "the", "is", "are", "am", "i", "me", "my", "you", "your",
                        "we", "they", "will", "would", "can", "could", "do", "does", "did",
                        "when", "what", "how", "where", "why", "to", "for", "in", "on", "at", "of",
                    }
                    phrase = " ".join(phrase_words)
                    if sum(word not in generic_words for word in phrase_words) >= 2 and phrase in q_normalized:
                        return domain

    # Also check for emotional distress signals that map to life_outlook
    distress_signals = [
        "not well", "not good", "very bad", "too much problem", "in trouble",
        "in pain", "worried", "anxious", "confused", "lost", "hopeless",
        "depressed", "sad", "unhappy", "bad phase", "hard time", "rough time",
        "everything is wrong", "nothing is right", "please help", "need guidance",
        "need help", "what should i do", "where am i going", "purpose of life",
    ]
    if any(signal in q_normalized for signal in distress_signals):
        return "life_outlook"

    return None


# ------------------------------------------------------------------
# Q&A Database — used ONLY for pure FAQ/informational questions
# ------------------------------------------------------------------
UNKNOWN_DOMAIN_DIRECT_LLM_PARSE = "UNKNOWN_DOMAIN_DIRECT_LLM_PARSE"
FAQ_INDEX: List[Tuple[List[str], str]] = []
ASTRO_RULE_INDEX: List[Tuple[str, List[str], str]] = []
QA_DATABASE: Dict[str, str] = {}
QA_STRUCTURED_DATABASE: Dict[str, dict] = {}

def load_qa_database():
    """Index greeting FAQs separately and retain all database rules and templates for retrieval."""
    global FAQ_INDEX, ASTRO_RULE_INDEX, QA_DATABASE, QA_STRUCTURED_DATABASE
    FAQ_INDEX.clear()
    ASTRO_RULE_INDEX.clear()
    QA_DATABASE.clear()
    QA_STRUCTURED_DATABASE.clear()

    possible_paths = [
        os.path.join(BASE_DIR, "qa_database.json"),
        "qa_database.json",
        os.path.join(BASE_DIR, "..", "qa_database.json")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    qa_db = json.load(f)
                    for key, entries in qa_db.items():
                        if isinstance(entries, str):
                            QA_DATABASE[key.lower().strip()] = entries
                            keywords = [key.lower().strip()]
                            ASTRO_RULE_INDEX.append(("general_and_basics", keywords, entries))
                            FAQ_INDEX.append((keywords, entries))
                        elif isinstance(entries, dict):
                            clean_k = key.lower().strip()
                            QA_STRUCTURED_DATABASE[clean_k] = entries
                            for kw in entries.get("keywords", []):
                                clean_kw = str(kw).lower().strip()
                                if clean_kw:
                                    QA_STRUCTURED_DATABASE[clean_kw] = entries
                            concept = entries.get("concept", key.title())
                            analogy = entries.get("simple_analogy", entries.get("analogy", ""))
                            meaning = entries.get("core_meaning", entries.get("core_analysis", ""))
                            impact = entries.get("what_it_means", entries.get("dasha_influence", ""))
                            takeaway = entries.get("actionable_takeaway", entries.get("recommendation", ""))
                            parts = [f"{concept}:", analogy, meaning]
                            if impact:
                                parts.append(f"What it means: {impact}")
                            if takeaway:
                                parts.append(f"Actionable takeaway: {takeaway}")
                            formatted_answer = " ".join([p for p in parts if p]).strip()
                            QA_DATABASE[clean_k] = formatted_answer
                            keywords = [clean_k] + [str(kw).lower().strip() for kw in entries.get("keywords", []) if kw]
                            for kw in keywords:
                                QA_DATABASE[kw] = formatted_answer
                            ASTRO_RULE_INDEX.append(("general_and_basics", keywords, formatted_answer))
                            FAQ_INDEX.append((keywords, formatted_answer))
                        elif isinstance(entries, list):
                            for item in entries:
                                if not isinstance(item, dict):
                                    continue
                                keywords = [str(value).lower().strip() for value in item.get("keywords", []) if value]
                                guidance = item.get("rule") or item.get("answer", "")
                                if keywords and guidance:
                                    for kw in keywords:
                                        QA_DATABASE[kw] = guidance
                                    ASTRO_RULE_INDEX.append((key, keywords, guidance))
                                if key == "general_and_basics" and keywords and item.get("answer"):
                                    FAQ_INDEX.append((keywords, item["answer"]))
                print(
                    f"[OK] Q&A database loaded from {p} "
                    f"({len(QA_DATABASE)} QA entries, {len(FAQ_INDEX)} FAQ items, {len(ASTRO_RULE_INDEX)} rule items indexed)."
                )
                return
            except Exception as e:
                print(f"[WARNING] Could not read {p}: {e}")

load_qa_database()


def retrieve_astrological_rules(query: str, domain: Optional[str] = None) -> str:
    """Return the best local Q&A rule, router domain hint, or direct-LLM sentinel."""
    ignored_words = {"a", "an", "the", "my", "your", "is", "are", "was", "were", "to", "for"}

    def intent_tokens(text: str) -> List[str]:
        normalized = normalize_hinglish(text).casefold()
        return [token for token in re.findall(r"\b[\w'-]+\b", normalized) if token not in ignored_words]

    query_tokens = intent_tokens(query)

    def contains_keyword(keyword: str) -> bool:
        keyword_tokens = intent_tokens(keyword)
        token_count = len(keyword_tokens)
        return bool(token_count) and any(
            query_tokens[index:index + token_count] == keyword_tokens
            for index in range(len(query_tokens) - token_count + 1)
        )

    database_matches: List[Tuple[int, str, str, str]] = []
    for category, keywords, guidance in ASTRO_RULE_INDEX:
        for keyword in keywords:
            if contains_keyword(keyword):
                database_matches.append((len(intent_tokens(keyword)), category, keyword, guidance))
    if database_matches:
        _, category, keyword, guidance = max(database_matches, key=lambda match: match[0])
        return f"Database category: {category}; matched keyword: {keyword}. Guidance: {guidance}"

    router_matches = [
        (len(keyword), domain, keyword)
        for domain, keywords in DOMAIN_KEYWORDS.items()
        for keyword in keywords
        if contains_keyword(keyword)
    ]
    if router_matches:
        _, domain, keyword = max(router_matches, key=lambda match: match[0])
        return f"Router domain: {domain}; matched keyword: {keyword}. No specific Q&A rule was found."

    if domain in DOMAIN_KEYWORDS:
        return f"Router domain: {domain}. No specific Q&A rule or keyword was found."

    return UNKNOWN_DOMAIN_DIRECT_LLM_PARSE


def unmatched_intent_unavailable_answer(query: str) -> str:
    """Return a transparent, language-matched notice instead of a generic prediction."""
    if is_hinglish_query(query):
        return "Is unfamiliar topic ko abhi AI se analyze nahi kar pa raha hoon. Kripya thodi der baad dobara poochhiye."
    return "AI analysis is temporarily unavailable for this unfamiliar topic. Please try again shortly."


def fallback_related_questions(query_domains: List[str], query: str) -> List[str]:
    """Provide useful domain-specific follow-ups when Gemini cannot return them."""
    domain = next((item for item in query_domains if item not in ("general", "life_outlook")), "general")
    hinglish = is_hinglish_query(query)
    questions = {
        "career": (
            [
                "Career change ke liye kaunsa samay zyada supportive hai?",
                "Mere chart mein stability aur growth ka balance kaisa hai?",
                "Workplace stress ko handle karne ke liye kaunsi strength use karun?",
            ]
            if hinglish else
            [
                "What timing looks supportive for a career change?",
                "How does my chart balance stability with growth?",
                "Which chart strength can help me handle workplace stress?",
            ]
        ),
        "wealth": (
            [
                "Financial growth ke liye kaunsa period supportive hai?",
                "Investment risk ko lekar chart kya suggest karta hai?",
                "Savings aur debt ko balance karne ka best approach kya hai?",
            ]
            if hinglish else
            [
                "What period looks supportive for financial growth?",
                "What does my chart suggest about investment risk?",
                "How can I balance savings and debt?",
            ]
        ),
        "marriage": (
            [
                "Relationship commitment ke liye kaunsa samay supportive hai?",
                "Mere chart mein partner compatibility kaise dikhti hai?",
                "Trust aur communication ko kaise strengthen karun?",
            ]
            if hinglish else
            [
                "What timing looks supportive for relationship commitment?",
                "What does my chart suggest about partner compatibility?",
                "How can I strengthen trust and communication?",
            ]
        ),
        "health": (
            [
                "Mental peace ke liye kaunsa chart strength use karun?",
                "Stress ke dauran mera Moon sign kaise react karta hai?",
                "Daily routine mein kaunsa practical change helpful hoga?",
            ]
            if hinglish else
            [
                "Which chart strength can support my mental peace?",
                "How does my Moon sign respond under stress?",
                "What practical change could help my daily routine?",
            ]
        ),
        "travel": (
            [
                "Travel ke liye kaunsa samay favorable hai?",
                "Mere chart mein relocation ke chances kaise hain?",
                "Trip ko smooth banane ke liye kis baat ki planning karun?",
            ]
            if hinglish else
            [
                "What timing looks favorable for travel?",
                "What does my chart suggest about relocation?",
                "What should I plan to make the trip smoother?",
            ]
        ),
        "education": (
            [
                "Padhai ya exam ke liye kaunsa samay supportive hai?",
                "Focus improve karne ke liye chart kya suggest karta hai?",
                "Meri kaunsi learning strength par dhyan dena chahiye?",
            ]
            if hinglish else
            [
                "What timing looks supportive for study or exams?",
                "What does my chart suggest for improving focus?",
                "Which learning strength should I build on?",
            ]
        ),
        "children": (
            [
                "Family planning ke liye chart mein kaunsa samay supportive hai?",
                "Parenthood ke dauran kaunsi emotional strength kaam aayegi?",
                "Family decisions mein partner ke saath alignment kaise banayein?",
            ]
            if hinglish else
            [
                "What timing looks supportive for family planning?",
                "Which emotional strength can support parenthood?",
                "How can I align family decisions with my partner?",
            ]
        ),
        "general": (
            [
                "Aane wale samay mein kis area par focus karna chahiye?",
                "Mere chart ke hisaab se agla turning point kya ho sakta hai?",
                "Is direction mein pehla practical step kya ho?",
            ]
            if hinglish else
            [
                "Which area should I focus on in the coming months?",
                "What turning point might my chart suggest next?",
                "What practical first step should I take?",
            ]
        ),
    }
    return questions.get(domain, questions["general"])


# Domain banks -- 20 questions per domain across 12 domains.
# generate_follow_up_questions() picks 3 per response, excluding the current question.
DOMAIN_BANKS: Dict[str, List[str]] = {
    "career": [
        "When is the right time for job change?",
        "Which gemstones suit my career?",
        "Does my chart favor business or a job?",
        "Will I get a promotion or appraisal this year?",
        "Which career field brings highest financial success?",
        "How does my 10th house karma bhava affect my career?",
        "Are there yogas for a government job in my chart?",
        "How does my current dasha influence my career growth?",
        "What workplace challenges should I prepare for?",
        "Which planet is the primary career ruler in my chart?",
        "What is the best time to start my own business?",
        "How does my Sun sign influence my leadership style?",
        "Will I ever work abroad or in a multinational company?",
        "Which planetary dasha supports a breakthrough in my career?",
        "How does Saturn transit affect my professional stability?",
        "What skills should I develop based on my Mercury placement?",
        "Is a career switch advisable for me in the next 6 months?",
        "How does my Mars placement impact my drive and ambition?",
        "Will my hard work be recognized and rewarded this year?",
        "What astrological timing favors a salary negotiation?",
    ],
    "marriage": [
        "What will my partner nature be like?",
        "Any remedies for marriage delay?",
        "When will I get married according to my dasha?",
        "How is my 7th house and relationship compatibility?",
        "Are there chances for love marriage or arranged marriage?",
        "How does Mangal Dosha or Mars influence my relationships?",
        "What role does the Navamsha D9 chart play in my marriage?",
        "Which planets bring harmony to my married life?",
        "How can I resolve relationship conflicts according to astrology?",
        "What are the key compatibility factors in Kundali Milan?",
        "What does my Venus placement say about love and romance?",
        "How does my 7th lord position impact relationship longevity?",
        "Will I have a happy and peaceful married life?",
        "What remedies strengthen emotional bonding with my partner?",
        "How does Rahu in the 7th house affect my marriage prospects?",
        "What is the ideal partner profile for my Lagna?",
        "How do transits of Jupiter influence my marriage timing?",
        "Can past-life karma affect my current relationship patterns?",
        "What chart combinations indicate a strong and lasting bond?",
        "How should I handle recurring arguments in my relationship?",
    ],
    "wealth": [
        "When is the best period for financial growth?",
        "Which gemstones or remedies boost wealth flow?",
        "Does my chart support stock market or property investment?",
        "How can I clear debt and improve my savings?",
        "Will I achieve long-term financial independence?",
        "What does my 2nd house of accumulated wealth reveal?",
        "How does my 11th house of financial gains operate?",
        "Are there strong Dhan Yogas or Lakshmi Yogas in my chart?",
        "How does Jupiter influence my wealth and prosperity?",
        "What unexpected financial windfalls or risks are indicated?",
        "Is this a good period to start investing in mutual funds?",
        "How does Saturn in my chart impact long-term wealth accumulation?",
        "What planetary combinations indicate sudden wealth or lottery luck?",
        "How can I attract abundance and reduce financial stress?",
        "What is the best approach: active income or passive income for my chart?",
        "Does my chart indicate inheritance or ancestral wealth?",
        "How do I know if a business partnership will be financially beneficial?",
        "What astrological remedies reduce financial losses and debts?",
        "Which sectors or industries are most aligned with my wealth house?",
        "How does Venus placement shape my spending habits and luxuries?",
    ],
    "health": [
        "Which planetary period supports my health recovery?",
        "How does my Moon sign respond to stress?",
        "What remedies strengthen vitality and mental peace?",
        "What wellness habits best suit my Lagna?",
        "What does my 6th house indicate about immunity and illness?",
        "How can I balance my emotional wellbeing and reduce anxiety?",
        "Which planetary transit affects my physical energy right now?",
        "What daily Ayurvedic and spiritual practices benefit my chart?",
        "How does my 8th house influence longevity and vitality?",
        "What astrological precautions should I take for chronic health issues?",
        "Which body parts or organs are astrologically sensitive for my Lagna?",
        "How does Mars placement influence my energy levels and immunity?",
        "What diet or lifestyle changes align with my chart for better health?",
        "Is this a high-risk period for illness or accidents in my chart?",
        "How does my Ketu placement relate to mysterious health issues?",
        "What does Sade Sati mean for my mental health and emotional stability?",
        "How can I use meditation or pranayama aligned with my Moon sign?",
        "Which gemstone or crystal supports physical healing for my chart?",
        "What astrological indicators point to good or poor sleep quality?",
        "How does my Ascendant lord strength affect my overall constitution?",
    ],
    "travel": [
        "When is the right time for foreign travel?",
        "Are foreign settlement chances strong in my chart?",
        "Will settling in a foreign land benefit my career?",
        "What planetary timing is favorable for visa approvals?",
        "How does my 12th house affect overseas residence and journeys?",
        "How does my 9th house of long journeys support global opportunities?",
        "What role does Rahu play in foreign travel for my chart?",
        "Will I return to my homeland or permanently settle abroad?",
        "What remedies remove obstacles in foreign visa processing?",
        "Which directional relocations are most auspicious for me?",
        "Which country or continent is most favorable for my chart?",
        "How does my Jupiter placement affect long-distance travel luck?",
        "Is a short domestic trip astrologically favorable this month?",
        "What planetary combinations support permanent overseas settlement?",
        "How does Ketu in the 12th house affect spiritual or international travel?",
        "Will I face challenges or smooth sailing during an upcoming trip?",
        "What are auspicious days or muhurtas for beginning a journey?",
        "How does moving to a new city affect my chart and destiny?",
        "Can astrology predict immigration success and timeline?",
        "What precautions should I take before traveling during Mercury retrograde?",
    ],
    "education": [
        "Which higher education field suits my chart?",
        "How can I improve concentration for exams?",
        "What timing is favorable for competitive exams?",
        "Are there chances for study or scholarship abroad?",
        "How does my 4th house influence foundational education?",
        "What does my 5th house say about intellect and creative learning?",
        "How can Mercury and Jupiter enhance my academic memory?",
        "Which field of specialization aligns best with my planets?",
        "What remedies boost confidence before crucial exams?",
        "Will I achieve honors and recognition in my academic career?",
        "Which subjects or streams align best with my Mercury placement?",
        "How does Jupiter transit impact my academic success this year?",
        "Is pursuing a postgraduate degree or certification favorable now?",
        "What are the chart indicators for a career in research or academia?",
        "How does my 3rd house support writing and communication skills?",
        "What Vedic remedies help overcome exam fear and performance anxiety?",
        "Will I get admission to a prestigious institution I am aiming for?",
        "How does my Ketu placement influence intuitive learning abilities?",
        "What is the best study schedule aligned with my Lagna energy peaks?",
        "Which language or creative skill would come most naturally to my chart?",
    ],
    "children": [
        "What timing looks supportive for family planning?",
        "How does my 5th house influence family and children?",
        "What planetary strengths support family harmony?",
        "What remedies bring domestic peace and blessings?",
        "How does Jupiter Putrakaraka bless parenthood in my chart?",
        "What astrological remedies overcome delays in childbirth?",
        "How will my children temperament and bond with me be?",
        "What does my 4th house reveal about maternal peace and home life?",
        "How can planetary harmony be maintained between family members?",
        "What prayers or mantras protect children growth and health?",
        "Is there a Putra Dosha or obstacle to parenthood in my chart?",
        "What astrological indicators suggest a large or small family?",
        "How does my Moon sign affect my parenting style and emotional bond?",
        "When is the most favorable time to conceive according to my dasha?",
        "What can my chart reveal about my relationship with my parents?",
        "Which planetary blessings ensure happiness and health for my children?",
        "How does Saturn in the 5th house affect parenthood and family karma?",
        "What rituals support a safe and healthy pregnancy astrologically?",
        "How do I handle generational differences with my parents or children?",
        "Which deity or mantra protects my family from negative energies?",
    ],
    "chart": [
        "What is my current Mahadasha and Antardasha?",
        "Tell me about my Sade Sati timing and remedies",
        "What does my Lagna and 1st house reveal?",
        "Which planets are most benefic in my birth chart?",
        "What are the strengths of my Sun sign and Moon sign?",
        "Where is Jupiter placed in my chart and what does it indicate?",
        "How do retrograde planets function in my birth chart?",
        "What karmic lessons do Rahu and Ketu highlight for me?",
        "How does my Navamsha D9 divisional chart compare to Rashi?",
        "What are the major yogas formed in my birth chart?",
        "What is the significance of my Atmakaraka planet?",
        "How does my Darakaraka planet describe my ideal partner?",
        "What does my Amatyakaraka planet reveal about my career path?",
        "How does Parivartana Yoga in my chart change planetary outcomes?",
        "What does a strong or weak Lagna lord signify in my chart?",
        "How does my Panchamsha D5 divisional chart reveal spiritual merit?",
        "What are the effects of Kala Sarpa Yoga if present in my chart?",
        "How does my rising Nakshatra influence my personality and fate?",
        "What does my chart reveal about past life debts or karmic patterns?",
        "How does the strength of my Ascendant affect life outcomes overall?",
    ],
    "remedies": [
        "Which gemstones suit my Lagna and Moon sign?",
        "Which mantras strengthen my benefic planets?",
        "What daily spiritual rituals align my chart?",
        "Which charity or donation activates Jupiter grace?",
        "How should I pacify Saturn during Sade Sati?",
        "What remedies pacify Rahu and Ketu doshas?",
        "What are the exact rules for energizing and wearing gemstones?",
        "How does chanting Maha Mrityunjaya Mantra protect vitality?",
        "What rituals for Pitra Dosha or ancestral peace should I perform?",
        "Which weekday fasting or vrat strengthens my Lagna lord?",
        "What is the most powerful mantra for my current dasha period?",
        "How does Rudraksha wearing help align planetary energies?",
        "What Navgraha puja should I perform for overall chart balance?",
        "Which colors and metals are auspicious for my Ascendant?",
        "How does lighting a ghee diya benefit my planetary placements?",
        "What yantra should I keep at home for wealth and protection?",
        "Can changing my name spelling astrologically improve my life?",
        "What role does Surya Namaskar play in strengthening my Sun sign?",
        "How does water offering to ancestors pacify Pitra Dosha?",
        "Which auspicious muhurta should I use before starting something new?",
    ],
    "general": [
        "Which career suits me best?",
        "What does my 7th house say about marriage?",
        "How are my wealth and finances looking?",
        "What does my Lagna and Moon sign reveal?",
        "What is my dominant life purpose according to my chart?",
        "What are my greatest astrological strengths and hidden gifts?",
        "Which planetary phase in my life brings the greatest turning point?",
        "How can I balance spiritual growth with worldly duties?",
        "What major lessons is my current dasha trying to teach me?",
        "What simple daily mindset aligns with my cosmic blueprint?",
        "What area of life needs the most attention in my chart right now?",
        "How does my birth Nakshatra shape my overall destiny?",
        "What is the most transformative period of my life astrologically?",
        "Which planet is my strongest ally and how do I activate it?",
        "What does my chart say about my overall luck and fortune this year?",
        "How do I align my decisions with favorable planetary windows?",
        "What is my soul deeper purpose according to Vedic astrology?",
        "How can I overcome recurring obstacles or bad luck patterns?",
        "What does my chart suggest about my social reputation and public image?",
        "Which upcoming transit will most impact my life in the next year?",
    ],
    "spirituality": [
        "What does my chart reveal about my spiritual path and growth?",
        "Which dasha period is most conducive for deep meditation practice?",
        "How does my 12th house of moksha and liberation influence my life?",
        "What deity or Ishta Devata is aligned with my birth chart?",
        "How does Ketu placement shape my spiritual detachment and wisdom?",
        "What Vedic practices accelerate my spiritual progress?",
        "Are there yoga or enlightenment combinations in my birth chart?",
        "How does my Jupiter placement guide my philosophical worldview?",
        "What does my chart say about past life spiritual karma?",
        "Which pilgrimage or sacred place holds the highest energy for my chart?",
        "What is the role of the 8th house in my spiritual transformation?",
        "How do eclipses trigger spiritual breakthroughs in my life?",
        "Which mantra practice best matches my Nakshatra deity?",
        "Does my chart show tendencies toward renunciation or monastic life?",
        "How does my rising sign influence my meditative strengths?",
    ],
    "property": [
        "Is this a good time to buy a house or property?",
        "What does my 4th house reveal about real estate success?",
        "Which direction is most auspicious for my new home?",
        "Will I own multiple properties in my lifetime?",
        "How does Saturn placement affect property acquisition timing?",
        "Are there planetary combinations for ancestral property disputes?",
        "What remedies remove obstacles in property purchase or registration?",
        "When is the best muhurta for signing a property agreement?",
        "Will I benefit more from renting or owning a home astrologically?",
        "How does my 12th house affect foreign property investment?",
        "What does Jupiter transit mean for property gains this year?",
        "Is this a favorable period to sell property for maximum profit?",
        "How does Vastu combine with astrology for home buying decisions?",
        "Which gemstone or yantra protects the home from negative energies?",
        "What are the signs of a lucky property placement in my chart?",
    ],
    "personality": [
        "What does my Lagna lord reveal about my core personality?",
        "How does my Sun sign define my ego and sense of identity?",
        "What does my Moon sign say about my emotional nature?",
        "How does my Ascendant Nakshatra shape my physical appearance?",
        "Which planet most dominates my decision-making style?",
        "What hidden personality traits does my 12th house reveal?",
        "How does my Mars placement shape my temper and assertiveness?",
        "What communication style does my Mercury placement suggest?",
        "How does Venus in my chart influence my aesthetic sense and values?",
        "What does my dominant element say about me?",
        "How does Saturn placement shape my sense of discipline and patience?",
        "Which Nakshatra gives me my most unique and defining character trait?",
        "How do retrograde planets create internalized personality patterns?",
        "What are my astrological blind spots or shadow traits to be aware of?",
        "How does my chart indicate introversion or extroversion tendencies?",
    ],
}


def generate_follow_up_questions(query: str, answer: str = "", domain: Optional[str] = None) -> List[str]:
    """Generate 3-4 strictly domain-pure follow-up questions matching the detected domain."""
    q_norm = normalize_hinglish(query).lower()

    # Resolve domain by priority: explicit argument > keyword scanning > indirect intent
    resolved = (domain or "").lower().strip()
    if resolved in ("love", "relationship", "relationships"):
        resolved = "marriage"
    elif resolved in ("money", "finance", "finances"):
        resolved = "wealth"
    elif resolved in ("job", "profession", "business", "work"):
        resolved = "career"

    if not resolved or resolved in ("general", "none", "auto", "unknown"):
        DOMAIN_PATTERNS = [
            ("marriage", ["marri", "love", "spouse", "partner", "relationship", "shaadi", "shadi", "7th house", "husband", "wife", "soulmate", "vivah", "manglik", "mangal dosha", "gunas"]),
            ("wealth", ["money", "wealth", "finance", "paisa", "invest", "debt", "loan", "income", "dhan", "saving", "2nd house", "11th house", "gajakesari", "lakshmi yoga"]),
            ("career", ["career", "job", "work", "profession", "promotion", "salary", "business", "naukri", "office", "appraisal", "switch", "boss", "10th house", "karma bhava", "government job"]),
            ("health", ["health", "disease", "stress", "sehat", "mental", "pain", "illness", "bimari", "burnout", "fatigue", "wellness", "6th house", "roga bhava", "vitality"]),
            ("travel", ["travel", "abroad", "foreign", "visa", "videsh", "relocat", "settle", "journey", "9th house", "12th house", "vyaya bhava"]),
            ("education", ["exam", "study", "education", "padhai", "college", "degree", "pariksha", "4th house", "5th house"]),
            ("children", ["child", "children", "baby", "pregnancy", "baccha", "family", "santaan", "putra"]),
            ("remedies", ["remed", "gemstone", "stone", "mantra", "puja", "upay", "totka", "kaal sarp", "pitra dosha", "gayatri mantra"]),
            ("chart", ["moon", "sun", "lagna", "ascendant", "rashi", "nakshatra", "kundali", "dasha", "transit", "sade sati", "graha", "planet", "1st house", "3rd house", "8th house"]),
        ]
        # First priority: check query directly
        for dom, keywords in DOMAIN_PATTERNS:
            if any(k in q_norm for k in keywords):
                resolved = dom
                break
        # Second priority: check answer text if query was generic
        if not resolved or resolved in ("general", "none", "auto", "unknown"):
            ans_lower = answer.lower()
            for dom, keywords in DOMAIN_PATTERNS:
                if any(k in ans_lower for k in keywords):
                    resolved = dom
                    break
        # Third priority: indirect intent resolver
        if not resolved or resolved in ("general", "none", "auto", "unknown"):
            indirect = resolve_indirect_intent(query)
            if indirect and indirect in DOMAIN_BANKS:
                resolved = indirect
            else:
                resolved = "general"

    pool = DOMAIN_BANKS.get(resolved, DOMAIN_BANKS["general"])

    # Exclude questions that match the current query to ensure fresh prompts
    clean_q = re.sub(r"[^\w\s]", "", query).lower().strip()
    filtered = [
        item for item in pool
        if re.sub(r"[^\w\s]", "", item).lower().strip() != clean_q
    ]
    return filtered[:3] if len(filtered) >= 2 else pool[:3]


def parse_gemini_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Validate Gemini's JSON answer and follow-up question payload."""
    try:
        payload = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
        return None
    answer = payload["answer"].strip()
    if not answer:
        return None

    questions = payload.get("related_questions")
    if not isinstance(questions, list):
        questions = []
    related_questions = [
        question.strip()
        for question in questions
        if isinstance(question, str) and question.strip()
    ][:3]
    return {"answer": answer, "related_questions": related_questions}

USER_SESSIONS: Dict[str, Dict[str, Any]] = {}

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]


class UserChartProfile(BaseModel):
    name: str = "Friend"
    lagna: str = "Libra"
    moon_sign: str = "Gemini"
    sun_sign: str = "Cancer"
    jupiter_house: str = "1st house"
    current_dasha: str = "Rahu-Moon"


class ChatRequest(BaseModel):
    user_query: str = ""
    profile: UserChartProfile = UserChartProfile()
    user_id: str = "default_user"
    query: Optional[str] = None
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    life_stage: Optional[str] = None
    relationship_status: Optional[str] = None
    question_type: str = "auto"  # "auto", "simple", "advanced", or "mixed"
    mode: str = "auto"  # "auto", or force "ai" / "instant"
    birth_year: int = 2000
    birth_month: int = 1
    birth_day: int = 1
    birth_hour: float = 12.0
    latitude: float = 20.5937
    longitude: float = 78.9629
    timezone: float = 5.5
    already_greeted: Optional[bool] = None

    def model_post_init(self, __context: Any) -> None:
        if not self.user_query and self.query:
            self.user_query = self.query
        elif not self.query and self.user_query:
            self.query = self.user_query
        if self.name and (not self.profile.name or self.profile.name == "Friend"):
            self.profile.name = self.name


class BirthDetailsRequest(BaseModel):
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int = 0
    latitude: float
    longitude: float
    timezone: float = 5.5


def check_qa_database(
    query: str,
    profile: Optional[UserChartProfile] = None,
    already_greeted: bool = False,
    user_natal_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Search qa_database.json for a matching key/keyword and format with user profile.

    Step 1 & 4 Ground-Truth Injection:
    If matched against structured QA database entry, dynamically populates fields
    with user_natal_context using format_local_qa_response.
    """
    if not query or (not QA_DATABASE and not QA_STRUCTURED_DATABASE):
        return None

    # Strict Length & Intent Guard for Local QA (< 10 words, no personal intents)
    if should_bypass_local_qa(query):
        return None

    words = query.strip().split()
    if len(words) >= 10:
        return None

    if profile is None:
        profile = UserChartProfile()

    # Pre-calculate user_natal_context if not passed
    if not user_natal_context:
        asc = getattr(profile, "ascendant", None) or getattr(profile, "lagna", "Libra")
        moon = getattr(profile, "rashi", None) or getattr(profile, "moon_sign", "Gemini")
        dasha = getattr(profile, "dasha", None) or getattr(profile, "current_dasha", "Rahu-Mars")
        user_natal_context = {
            "ascendant": asc,
            "moon_sign": moon,
            "active_dasha": str(dasha),
            "house_rulerships": calculate_house_rulerships(asc),
        }

    # a. Clean and normalize the query (lowercase, remove punctuation/quotes)
    cleaned_query = re.sub(r'[\'\".,?!;:()\[\]{}_+\-=/\\<>]', ' ', query.lower())
    cleaned_query = " ".join(cleaned_query.split())
    if not cleaned_query:
        return None

    # Check QA_STRUCTURED_DATABASE first for exact or keyword boundary matches
    sorted_struct = sorted(QA_STRUCTURED_DATABASE.items(), key=lambda item: len(item[0]), reverse=True)
    for key, qa_dict in sorted_struct:
        clean_key = re.sub(r'[\'\".,?!;:()\[\]{}_+\-=/\\<>]', ' ', key.lower()).strip()
        clean_key = " ".join(clean_key.split())
        if clean_key and cleaned_query == clean_key:
            return format_local_qa_response(qa_dict, user_natal_context)
        if clean_key and len(clean_key) >= 2:
            pattern = rf'^\s*(?:what\s+is\s+(?:a\s+|an\s+)?|define\s+|definition\s+of\s+|tell\s+me\s+about\s+|explain\s+)?{re.escape(clean_key)}\s*$'
            if re.match(pattern, cleaned_query):
                return format_local_qa_response(qa_dict, user_natal_context)

    # b. Match against keys/keywords in qa_database.json
    matched_template: Optional[str] = None
    matched_key: str = ""

    # Prioritize longer multi-word keys (e.g., 'current dasha' before 'dasha')
    sorted_entries = sorted(QA_DATABASE.items(), key=lambda item: len(item[0]), reverse=True)

    # 1. Exact match check
    for key, template in sorted_entries:
        clean_key = re.sub(r'[\'\".,?!;:()\[\]{}_+\-=/\\<>]', ' ', key.lower()).strip()
        clean_key = " ".join(clean_key.split())
        if clean_key and cleaned_query == clean_key:
            matched_template = template
            matched_key = clean_key
            break

    # 2. Exact Keyword Boundary Matching: only match exact short question structures
    if not matched_template:
        for key, template in sorted_entries:
            clean_key = re.sub(r'[\'\".,?!;:()\[\]{}_+\-=/\\<>]', ' ', key.lower()).strip()
            clean_key = " ".join(clean_key.split())
            if not clean_key or len(clean_key) < 2:
                continue
            pattern = rf'^\s*(?:what\s+is\s+(?:a\s+|an\s+)?|define\s+|definition\s+of\s+|tell\s+me\s+about\s+|explain\s+)?{re.escape(clean_key)}\s*$'
            if re.match(pattern, cleaned_query):
                matched_template = template
                matched_key = clean_key
                break

    if not matched_template:
        return None

    params = profile.model_dump()
    # Backward compatibility aliases
    if "ascendant" not in params and "lagna" in params:
        params["ascendant"] = params["lagna"]
    if "lagna" not in params and "ascendant" in params:
        params["lagna"] = params["ascendant"]
    if "moon" not in params and "moon_sign" in params:
        params["moon"] = params["moon_sign"]
    if "sun" not in params and "sun_sign" in params:
        params["sun"] = params["sun_sign"]
    if "dasha" not in params and "current_dasha" in params:
        params["dasha"] = params["current_dasha"]
    if "jupiter" not in params and "jupiter_house" in params:
        params["jupiter"] = params["jupiter_house"]

    # 3. Exception Handling: Safely handle missing template keys using KeyError
    try:
        formatted_text = matched_template.format(**params)
    except KeyError as exc:
        logger.warning("Missing template key %s in profile during formatting; applying safe fallback", exc)
        class SafeDict(dict):
            def __missing__(self, k):
                return f"{{{k}}}"
        formatted_text = matched_template.format_map(SafeDict(**params))
    except Exception as exc:
        logger.warning("Unexpected error formatting template: %s", exc)
        formatted_text = matched_template

    # Strip repeated greetings if already greeted and not a greeting query
    is_greeting_key = matched_key in {"hello", "hi", "namaste", "hey", "greetings"}
    if already_greeted and not is_greeting_key:
        formatted_text = strip_greeting(formatted_text)

    return formatted_text


# ------------------------------------------------------------------
# FAQ Matcher — ONLY for pure greetings & simple informational sign queries
# Supports both English and Hinglish inputs.
# ------------------------------------------------------------------
GREETING_KEYWORDS = {
    "hi", "hello", "hey", "namaste", "greetings", "good morning",
    "good evening", "good afternoon", "namaskar", "jai hind"
}
ASTRO_INQUIRY_TERMS = {
    "career", "job", "work", "profession", "business", "promotion", "salary", "money",
    "marriage", "married", "wedding", "love", "spouse", "partner", "relationship",
    "health", "travel", "abroad", "foreign", "child", "children", "education",
    "exam", "wealth", "finance", "dasha", "transit", "when", "will", "how", "future",
    "predict", "prediction", "kundali", "chart", "horoscope", "life", "why", "can", "should",
    "advice", "suit", "opportunity", "what", "who", "which",
    # Hinglish inquiry terms
    "naukri", "shaadi", "paisa", "sehat", "yatra", "padhai", "kab", "kya",
    "hoga", "hogi", "milega", "milegi", "batao", "bata"
}

def match_faq_only(user_query: str) -> Optional[str]:
    """Return a local FAQ from the 85+ item database for exact or indexed factual phrases, saving LLM tokens."""
    if not user_query:
        return None

    # Strict Length & Intent Guard: Never match FAQ if query contains personal intent or >= 10 words
    if should_bypass_local_qa(user_query):
        return None

    clean_norm = re.sub(r"[^\w\s'-]", " ", normalize_hinglish(user_query).casefold()).strip()
    clean_raw = re.sub(r"[^\w\s'-]", " ", user_query.casefold()).strip()

    # 1. Exact match against indexed FAQ keywords
    for keywords, answer_template in FAQ_INDEX:
        for kw in keywords:
            normalized_keyword = normalize_hinglish(kw).casefold()
            normalized_keyword = re.sub(r"[^\w\s'-]", " ", normalized_keyword).strip()
            raw_keyword = re.sub(r"[^\w\s'-]", " ", kw.casefold()).strip()
            if clean_norm == normalized_keyword or clean_raw == raw_keyword:
                return answer_template

    # 2. Standalone greetings (must NOT match substrings in words like 'which')
    greeting_words = {"hi", "hello", "hey", "namaste", "greetings"}
    clean_words = clean_raw.split()
    if clean_raw in greeting_words or (len(clean_words) <= 2 and any(w in greeting_words for w in clean_words)):
        for keywords, answer_template in FAQ_INDEX:
            for kw in keywords:
                if kw.lower() in greeting_words:
                    return answer_template

    # 3. Dynamic topic match against all indexed FAQ topics (prioritizing longer multi-word phrases)
    sorted_faqs = sorted(FAQ_INDEX, key=lambda item: max(len(k) for k in item[0]), reverse=True)
    ignored_single_words = {"what", "when", "will", "tell", "about", "your", "mine", "good", "more", "with"}
    for keywords, answer_template in sorted_faqs:
        for kw in keywords:
            kw_clean = kw.lower().strip()
            if len(kw_clean) < 4 or kw_clean in ignored_single_words:
                continue
            kw_norm = normalize_hinglish(kw_clean).casefold()
            pattern_raw = r"\b" + re.escape(kw_clean) + r"\b"
            pattern_norm = r"\b" + re.escape(kw_norm) + r"\b"
            if (
                re.search(pattern_raw, clean_raw)
                or re.search(pattern_raw, clean_norm)
                or re.search(pattern_norm, clean_norm)
            ):
                return answer_template

    return None


def format_personalized_answer(
    template: str,
    asc: str,
    sun: str,
    moon: str,
    name: str = "",
    hinglish: bool = False,
    query: str = "",
    astro_features: Optional[List[int]] = None,
    current_dasha: str = "Rahu-Moon",
    jupiter_house: str = "1st house",
) -> str:
    """Personalize every FAQ answer using user's name, calculated signs, and chart placements."""
    params = {
        "ascendant": asc,
        "lagna": asc,
        "sun_sign": sun,
        "moon_sign": moon,
        "current_dasha": current_dasha,
        "jupiter_house": jupiter_house,
        "name": name or "Friend",
    }
    try:
        formatted = template.format(**params)
    except KeyError:
        class SafeDict(dict):
            def __missing__(self, k):
                return f"{{{k}}}"
        formatted = template.format_map(SafeDict(**params))
    except (IndexError, ValueError):
        formatted = template

    # Inject calculated chart signs if placeholders or defaults are in text
    if "Moon Sign" in formatted or "Rashi" in formatted:
        formatted = re.sub(r"is in \w+", f"is in {moon}", formatted)
        formatted = re.sub(r"Rashi\) \w+ hai", f"Rashi) {moon} hai", formatted)
    if "Lagna" in formatted or "Ascendant" in formatted:
        formatted = re.sub(r"Lagna \(Ascendant\) \w+(?: \([^)]+\))? hai", f"Lagna (Ascendant) {asc} hai", formatted)
        formatted = re.sub(r"Ascendant is in \w+", f"Ascendant is in {asc}", formatted)
    if "Sun Sign" in formatted or "Sun sign" in formatted:
        formatted = re.sub(r"Sun Sign is in \w+", f"Sun Sign is in {sun}", formatted)
        formatted = re.sub(r"Sun sign in \w+", f"Sun sign in {sun}", formatted)
    if "dasha" in formatted.lower() or "mahadasha" in formatted.lower():
        formatted = re.sub(r"currently running the [A-Za-z0-9\-]+ dasha", f"currently running the {current_dasha} dasha", formatted)

    clean_name = name.strip() if name and name.strip().lower() not in ("guest", "there", "user", "anonymous") else ""
    asc_idx = ZODIAC_SIGNS.index(asc) if asc in ZODIAC_SIGNS else 0
    feat = astro_features or [asc_idx, 1, 2, 0, 2, 8, 1, 9, 1, 7]

    def get_sign(idx: int, default: str) -> str:
        if feat and len(feat) > idx and feat[idx] is not None:
            try:
                return ZODIAC_SIGNS[int(feat[idx]) % 12]
            except (ValueError, TypeError, IndexError):
                pass
        return default

    mars_sign = get_sign(3, "Aries")
    mercury_sign = get_sign(4, "Gemini")
    jupiter_sign = get_sign(5, "Sagittarius")
    venus_sign = get_sign(6, "Taurus")
    saturn_sign = get_sign(7, "Capricorn")
    rahu_sign = get_sign(8, "Taurus")
    ketu_sign = get_sign(9, "Scorpio")

    SIGN_RULERS = {
        "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
        "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
        "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter",
    }

    def get_house_info(house_num: int) -> Tuple[str, str]:
        s = ZODIAC_SIGNS[(asc_idx + house_num - 1) % 12]
        return s, SIGN_RULERS.get(s, "Benefic Planet")

    LAGNA_REMEDIES = {
        "Aries": ("Red Coral (Moonga)", "Gayatri Mantra or Hanuman Chalisa"),
        "Taurus": ("Diamond or White Sapphire", "Om Shukraya Namah"),
        "Gemini": ("Emerald (Panna)", "Om Budhaya Namah or Vishnu Sahasranama"),
        "Cancer": ("Pearl (Moti)", "Maha Mrityunjaya Mantra or Om Namah Shivaya"),
        "Leo": ("Ruby (Manikya)", "Aditya Hridaya Stotra or Surya Gayatri"),
        "Virgo": ("Emerald (Panna)", "Om Budhaya Namah"),
        "Libra": ("Diamond or White Sapphire", "Shri Suktam or Om Shukraya Namah"),
        "Scorpio": ("Red Coral (Moonga)", "Hanuman Chalisa"),
        "Sagittarius": ("Yellow Sapphire (Pukhraj)", "Brihaspati Gayatri or Guru Mantra"),
        "Capricorn": ("Blue Sapphire (Neelam) or Amethyst", "Shani Mantra or Maha Mrityunjaya"),
        "Aquarius": ("Blue Sapphire (Neelam) or Amethyst", "Om Sham Shanaishcharaya Namah"),
        "Pisces": ("Yellow Sapphire (Pukhraj)", "Guru Mantra"),
    }

    # Salutation prefix
    if clean_name:
        salutation = f"Namaste {clean_name} ji! " if hinglish else f"Hello {clean_name}, "
    else:
        salutation = ""

    combined_q = f"{query.lower()} {formatted.lower()}"
    personal_insight = ""

    # A. Check for House questions (1st to 12th house) from user query
    q_low = query.lower()
    house_match = re.search(r"\b(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|11th|12th|\d+th|\d+st|\d+nd|\d+rd)\s+house", q_low)
    house_word_match = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+house", q_low)
    house_num = None
    if house_match:
        try:
            house_num = int(re.sub(r"[^\d]", "", house_match.group(1)))
        except ValueError:
            pass
    elif house_word_match:
        words = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"]
        w = house_word_match.group(1).lower()
        if w in words:
            house_num = words.index(w) + 1

    if house_num and 1 <= house_num <= 12:
        h_sign, h_ruler = get_house_info(house_num)
        if hinglish:
            personal_insight = f"\n\nAapki kundali ({asc} Lagna) mein aapka {house_num}th house {h_sign} rashi mein sthit hai (swami: {h_ruler}). Yeh aapke jeevan mein {h_sign} ki qualities ko is kshetra mein vishesh prabhavi banata hai."
        else:
            personal_insight = f"\n\nIn your birth chart ({asc} Ascendant), your {house_num}th House falls in {h_sign} (governed by {h_ruler}). This makes {h_sign}'s energy and {h_ruler}'s placement particularly significant in this area of your life."

    # B. Specific astrological domains
    elif any(k in combined_q for k in ["career", "job", "profession", "promotion", "government job", "sarkari", "business"]):
        h10_sign, h10_ruler = get_house_info(10)
        if hinglish:
            personal_insight = f"\n\nAapke {asc} Lagna ke anusar, aapka 10th house (Karma Bhava) {h10_sign} rashi mein hai (swami: {h10_ruler}), jo career mein dedicated aur structured effort se achhe parinam darshata hai."
        else:
            personal_insight = f"\n\nIn your birth chart ({asc} Ascendant), your 10th House of Career (Karma Bhava) is in {h10_sign}, governed by {h10_ruler}. Aligning your work with structured responsibilities brings progressive growth."

    elif any(k in combined_q for k in ["marri", "spouse", "partner", "relationship", "shaadi", "7th", "manglik", "mangal dosha", "gunas"]):
        h7_sign, h7_ruler = get_house_info(7)
        if hinglish:
            personal_insight = f"\n\nAapke chart mein vivah aur partnership ka 7th house {h7_sign} rashi (swami: {h7_ruler}) mein hai aur Venus {venus_sign} mein sthit hai, jo rishton mein mutual understanding aur patience ko mahatvapoorna banata hai."
        else:
            personal_insight = f"\n\nIn your personal horoscope, your 7th House of partnership falls in {h7_sign} (governed by {h7_ruler}), while your Venus is positioned in {venus_sign}. Mutual respect and clear communication support lasting relationship harmony."

    elif any(k in combined_q for k in ["wealth", "money", "finance", "dhan", "paisa", "gajakesari", "lakshmi", "share market", "debt"]):
        h2_sign, h2_ruler = get_house_info(2)
        h11_sign, h11_ruler = get_house_info(11)
        if hinglish:
            personal_insight = f"\n\nAapke chart mein dhan sanchay ka 2nd house {h2_sign} aur aamdani ka 11th house {h11_sign} mein hai, sath hi Jupiter {jupiter_sign} mein sthit hai, jo consistent savings aur strategic financial planning ko favor karta hai."
        else:
            personal_insight = f"\n\nIn your chart, your 2nd House of accumulated wealth is {h2_sign} (ruled by {h2_ruler}) and 11th House of gains is {h11_sign}, with Jupiter positioned in {jupiter_sign}. Consistent financial discipline best unlocks your chart's prosperity."

    elif any(k in combined_q for k in ["health", "mental", "stress", "peace", "sehat", "disease", "shanti"]):
        h6_sign, _ = get_house_info(6)
        if hinglish:
            personal_insight = f"\n\nAapka Moon {moon} rashi mein hone ke karan, emotional balance aur shant vatavaran aapke man aur sehat ke liye sabse mahatvapoorna hai. Aapka 6th house (wellness) {h6_sign} rashi mein aata hai."
        else:
            personal_insight = f"\n\nWith your Moon in {moon} and {asc} Lagna, regular relaxation, adequate hydration, and a structured daily routine best support your mental equilibrium and vital energy (6th house in {h6_sign})."

    elif any(k in combined_q for k in ["foreign", "travel", "abroad", "videsh", "relocat", "journey"]):
        h9_sign, _ = get_house_info(9)
        h12_sign, h12_ruler = get_house_info(12)
        if hinglish:
            personal_insight = f"\n\nAapki kundali mein yatra ka 9th house {h9_sign} aur videsh ka 12th house {h12_sign} mein sthit hai, sath hi Rahu {rahu_sign} mein hai, jo door ke sthano aur naye avsaron ki sambhavna dikhata hai."
        else:
            personal_insight = f"\n\nIn your horoscope, your 9th House of long travels is in {h9_sign} and 12th House of foreign lands is in {h12_sign} (ruled by {h12_ruler}), with Rahu in {rahu_sign}, fostering curiosity for distant horizons."

    elif any(k in combined_q for k in ["education", "study", "exam", "padhai", "college"]):
        h4_sign, _ = get_house_info(4)
        h5_sign, h5_ruler = get_house_info(5)
        if hinglish:
            personal_insight = f"\n\nAapke chart mein vidya ka 4th house {h4_sign} aur intellect ka 5th house {h5_sign} mein hai, sath hi Mercury {mercury_sign} mein sthit hai, jo focused study technique se behtar parinam deta hai."
        else:
            personal_insight = f"\n\nFor your chart, your 4th House of education is in {h4_sign} and 5th House of intellect is in {h5_sign} (ruled by {h5_ruler}), with Mercury in {mercury_sign}, rewarding active focus and concept clarity."

    elif any(k in combined_q for k in ["remed", "gemstone", "mantra", "dosha", "kaal sarp", "pitra", "sade sati", "upay"]):
        remedy = LAGNA_REMEDIES.get(asc, ("Benefic Gemstone", "Daily spiritual meditation"))
        if hinglish:
            personal_insight = f"\n\nAapke {asc} Lagna aur {moon} Rashi ke anuroop shubh upay: Gemstone: {remedy[0]}; Mantra/Niyam: {remedy[1]}."
        else:
            personal_insight = f"\n\nFor your {asc} Ascendant and {moon} Moon, traditionally supportive remedies include: Gemstone: {remedy[0]}; Harmonizing Practice: {remedy[1]}."

    elif any(k in combined_q for k in ["sun", "surya"]):
        if hinglish:
            personal_insight = f"\n\nAapke chart mein Surya (Sun) {sun} rashi mein sthit hai, jo aapki atma-shakti aur self-reliance ko strengthen karta hai."
        else:
            personal_insight = f"\n\nIn your birth chart, the Sun is situated in {sun}, anchoring your core willpower and sense of purpose."

    elif any(k in combined_q for k in ["moon", "chandra", "rashi"]):
        if hinglish:
            personal_insight = f"\n\nAapka Moon (Chandra) {moon} rashi mein sthit hai, jo aapki intuitive thinking aur emotional awareness ko lead karta hai."
        else:
            personal_insight = f"\n\nIn your birth chart, the Moon is placed in {moon}, guiding your emotional clarity and instinctive perceptions."

    elif any(k in combined_q for k in ["mars", "mangal"]):
        if hinglish:
            personal_insight = f"\n\nAapke chart mein Mars (Mangal) {mars_sign} rashi mein sthit hai, jo aapki determination aur action power ko drive karta hai."
        else:
            personal_insight = f"\n\nIn your horoscope, Mars is placed in {mars_sign}, directing your physical drive and initiative."

    elif any(k in combined_q for k in ["mercury", "budha"]):
        if hinglish:
            personal_insight = f"\n\nAapke chart mein Mercury (Budha) {mercury_sign} rashi mein sthit hai, jo communication aur analytical thinking ko support karta hai."
        else:
            personal_insight = f"\n\nIn your horoscope, Mercury is situated in {mercury_sign}, sharpening your communication and analytical judgment."

    elif any(k in combined_q for k in ["jupiter", "guru", "brihaspati"]):
        if hinglish:
            personal_insight = f"\n\nAapke chart mein Jupiter (Guru) {jupiter_sign} rashi mein sthit hai, jo wisdom, learning aur ethical decision-making ko bless karta hai."
        else:
            personal_insight = f"\n\nIn your birth chart, Jupiter is positioned in {jupiter_sign}, bestowing wisdom, expanding horizons, and guiding moral discernment."

    elif any(k in combined_q for k in ["venus", "shukra"]):
        if hinglish:
            personal_insight = f"\n\nAapke chart mein Venus (Shukra) {venus_sign} rashi mein sthit hai, jo creative perception aur aesthetic balance ko enhance karta hai."
        else:
            personal_insight = f"\n\nIn your horoscope, Venus is in {venus_sign}, influencing your aesthetic sensibilities and relational warmth."

    elif any(k in combined_q for k in ["saturn", "shani"]):
        if hinglish:
            personal_insight = f"\n\nAapke chart mein Saturn (Shani) {saturn_sign} rashi mein sthit hai, jo patience, discipline aur long-term persistence ko develop karta hai."
        else:
            personal_insight = f"\n\nIn your chart, Saturn is positioned in {saturn_sign}, teaching patience, diligence, and long-term grounding."

    elif any(k in combined_q for k in ["rahu", "ketu"]):
        if hinglish:
            personal_insight = f"\n\nAapke chart mein Rahu {rahu_sign} aur Ketu {ketu_sign} mein sthit hain, jo materialistic ambition aur spiritual detachment ke karmic balance ko darshate hain."
        else:
            personal_insight = f"\n\nIn your chart, Rahu is in {rahu_sign} and Ketu is in {ketu_sign}, shaping your balance between worldly ambition and spiritual reflection."

    else:
        if hinglish:
            personal_insight = f"\n\n(Aapka Chart Snapshot: {asc} Lagna | {moon} Rashi | {sun} Surya)"
        else:
            personal_insight = f"\n\n(Your Chart Snapshot: {asc} Ascendant | {moon} Moon | {sun} Sun)"

    if salutation and not formatted.startswith(("Hello", "Namaste", "Hi", "Aapka", "Your")):
        final_answer = f"{salutation}{formatted}{personal_insight}"
    else:
        final_answer = f"{formatted}{personal_insight}"

    return final_answer.strip()


# ------------------------------------------------------------------
# Gemini-Powered Prediction — for ALL prediction/advanced questions
# ------------------------------------------------------------------
def build_gemini_prompt(
    request: ChatRequest,
    asc_sign: str,
    sun_sign: str,
    moon_sign: str,
    astro_features: list,
    predicted_domain: str,
    planet_data: dict,
    dasha_data: dict,
    current_dasha: Optional[str] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    session_profile: Optional[Dict[str, Any]] = None,
    query_domains: Optional[List[str]] = None,
    retrieved_rule: Optional[str] = None,
    normalized_interpretation: Optional[str] = None,
    user_natal_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Step 1 & 2: Ground-Truth Injection & Strict System Prompt Guardrails
    """
    # Extract dasha compactly if available
    dasha_name = current_dasha or "Rahu-Mars"
    if not current_dasha and isinstance(dasha_data, dict):
        dasha_name = extract_current_dasha(dasha_data) or "Rahu-Mars"

    if not user_natal_context:
        user_natal_context = get_user_natal_context(
            ascendant=asc_sign,
            moon_sign=moon_sign,
            active_dasha=dasha_name,
        )

    return build_system_prompt(
        user_profile={
            "lagna": asc_sign,
            "sun_sign": sun_sign,
            "moon_sign": moon_sign,
            "current_dasha": dasha_name,
        },
        session_profile=session_profile,
        user_natal_context=user_natal_context,
    )



def _generate_legacy_fallback_prediction(
    query: str, domain: str,
    asc_name: str, sun_name: str, moon_name: str,
    astro_features: list,
    user_name: Optional[str] = "Guest",
    age: Optional[int] = None,
    life_stage: Optional[str] = None,
    relationship_status: Optional[str] = None,
    gender: Optional[str] = None
) -> str:
    """AstroTalk-style instant Vedic astrology engine.
    Handles direct, indirect, and Hinglish queries with empathetic personalized responses.
    Response structure: Acknowledgment → Insight → Turning Point → Guidance → Empowering Close.
    """
    # Normalize Hinglish to English before keyword matching
    q_normalized = normalize_hinglish(query)
    q_lower = q_normalized.lower()
    name = user_name.strip() if user_name and user_name.strip().lower() not in ["guest", "there"] else ""
    greeting = f"Dear {name}, " if name else "Dear seeker, "

    # Planetary positions from user's exact birth chart
    mars_sign    = ZODIAC_SIGNS[int(astro_features[3]) % 12] if len(astro_features) > 3 else "Aries"
    mercury_sign = ZODIAC_SIGNS[int(astro_features[4]) % 12] if len(astro_features) > 4 else "Gemini"
    jupiter_sign = ZODIAC_SIGNS[int(astro_features[5]) % 12] if len(astro_features) > 5 else "Sagittarius"
    venus_sign   = ZODIAC_SIGNS[int(astro_features[6]) % 12] if len(astro_features) > 6 else "Taurus"
    saturn_sign  = ZODIAC_SIGNS[int(astro_features[7]) % 12] if len(astro_features) > 7 else "Capricorn"
    rahu_sign    = ZODIAC_SIGNS[int(astro_features[8]) % 12] if len(astro_features) > 8 else "Taurus"
    ketu_sign    = ZODIAC_SIGNS[int(astro_features[9]) % 12] if len(astro_features) > 9 else "Scorpio"

    age_str = f"at {age} years" if age else "at this point in your journey"

    # Gender-aware partner references
    partner_ref = "your life partner"
    if gender:
        if gender.lower() in ["male", "m"]:
            partner_ref = "your wife or life partner"
        elif gender.lower() in ["female", "f"]:
            partner_ref = "your husband or life partner"

    # 1. Resolve indirect/roundabout intent first — then also check direct keywords
    indirect_domain = resolve_indirect_intent(query)

    # 2. Resolve effective domain (indirect > ML domain > semantic fuzzy)
    effective_domain = domain
    if indirect_domain:
        effective_domain = indirect_domain
    elif domain in ("general", "") or domain.startswith("chart_archetype"):
        inferred = infer_semantic_domain(q_lower)
        if inferred:
            effective_domain = inferred

    # ---- AstroTalk-Style Response Bank ----
    # Each response follows: Acknowledgment → Astrological Insight → Turning Point → Guidance → Close

    # --- Life Outlook / Emotional Distress / Indirect General ("kab tak mushkil rahegi") ---
    if effective_domain == "life_outlook" or any(k in q_lower for k in [
        "kal kaisa", "aage kya", "future kaisa", "kab tak", "achhe din",
        "sab theek", "bura waqt", "life hard", "struggling", "lost", "hopeless",
        "what should i do", "need help", "please help", "guidance needed"
    ]):
        return (
            f"{greeting}I can feel the weight of what you're going through, and I want you to know — "
            f"the stars are not against you. For your {asc_name} Lagna with Moon in {moon_name}, this period of difficulty "
            f"is largely shaped by Saturn in {saturn_sign} testing your patience and resolve {age_str}. "
            f"But Jupiter in {jupiter_sign} is moving into a very supportive position for you, "
            f"and within the next 3 to 5 months a genuine turning point will emerge — one that rewards your persistence. "
            f"Right now, chant the Maha Mrityunjaya mantra 108 times on Saturdays and trust the process. "
            f"Your {asc_name} ascendant carries tremendous resilience — your best chapter is still ahead."
        )

    # --- Career / Job / Business (includes Hinglish + indirect stuck-in-career) ---
    elif effective_domain == "career" or any(k in q_lower for k in [
        "career", "job", "profession", "work", "business", "promotion", "appraisal",
        "switch", "employment", "office", "boss", "interview", "salary", "naukri",
        "stuck in career", "no growth", "no progress", "hard work not paying"
    ]):
        return (
            f"{greeting}I understand the frustration when your hard work doesn't seem to be paying off — "
            f"but your chart tells a very different story. With your {asc_name} Lagna and Sun in {sun_name}, "
            f"your 10th house of karma and career is strongly activated. Mars in {mars_sign} is fueling your drive, "
            f"and Mercury in {mercury_sign} sharpens your edge. Jupiter in {jupiter_sign} is about to cast a direct "
            f"auspicious aspect on your career house, making the next 3 to 6 months a pivotal window for breakthroughs — "
            f"promotions, new roles, or business expansion. Stay consistent, sharpen your skills, and trust that "
            f"Saturn in {saturn_sign} is simply building your foundation for lasting success."
        )

    # --- Marriage / Love / Relationship (includes indirect loneliness) ---
    elif effective_domain == "marriage" or any(k in q_lower for k in [
        "marriage", "married", "wedding", "spouse", "partner", "relationship",
        "love", "soulmate", "husband", "wife", "boyfriend", "girlfriend",
        "compatibility", "lonely", "alone", "akela", "no love"
    ]):
        return (
            f"{greeting}I can sense you're longing for emotional connection and stability — "
            f"and your birth chart has beautiful promise in this area. Venus in {venus_sign} and your "
            f"Moon in {moon_name} create a deeply loving and loyal emotional nature within you. "
            f"Jupiter in {jupiter_sign} is now casting its benevolent gaze on your 7th house of partnerships, "
            f"which signals that a meaningful, aligned relationship is drawing closer — especially over the "
            f"next 4 to 8 months. Trust the timing of your {asc_name} Lagna and remain open-hearted. "
            f"Wearing white on Fridays and offering flowers to Goddess Lakshmi can further strengthen Venus for you."
        )

    # --- Wealth / Finance / Money (includes indirect financial struggle) ---
    elif effective_domain == "wealth" or any(k in q_lower for k in [
        "money", "finance", "wealth", "investment", "rich", "property",
        "income", "gain", "profit", "debt", "loan", "savings", "earn",
        "financially struggling", "paise ki kami", "budget tight"
    ]):
        return (
            f"{greeting}Financial pressure can be deeply exhausting, and I see exactly why your chart "
            f"is showing this challenge right now. For your {asc_name} Lagna, the 2nd house of wealth "
            f"is governed by a planet in a transitional phase, but Jupiter in {jupiter_sign} is forming a "
            f"powerful connection with your 11th house of income gains {age_str}. "
            f"Mercury in {mercury_sign} adds analytical sharpness to your money decisions. "
            f"Over the next 4 to 7 months, consistent effort will convert into tangible financial progress. "
            f"Avoid speculative risks for now — Saturn in {saturn_sign} rewards disciplined, structured financial planning. "
            f"Donating food on Thursdays activates Jupiter's abundance for you."
        )

    # --- Health / Wellness (includes indirect tiredness, mental health) ---
    elif effective_domain == "health" or any(k in q_lower for k in [
        "health", "disease", "illness", "fitness", "wellness", "diet",
        "stress", "cure", "pain", "medical", "anxiety", "recovery",
        "tired", "weak", "thakaan", "neend nahi", "overthinking", "mental peace"
    ]):
        return (
            f"{greeting}Your concern about your wellbeing is completely valid — and your stars want "
            f"you to prioritize yourself right now. With Moon in {moon_name}, you carry a sensitive emotional "
            f"constitution that is deeply affected by stress and mental fatigue. Mars in {mars_sign} is "
            f"currently pushing your energy levels hard, which can lead to burnout {age_str} if boundaries "
            f"aren't maintained. Jupiter in {jupiter_sign} provides protective, restorative energy, "
            f"and your vitality will noticeably improve over the next 4 to 6 weeks as you establish "
            f"grounding habits. Please prioritize sleep, reduce screen time after 10 PM, "
            f"and try pranayama or meditation for 15 minutes each morning — your Moon in {moon_name} responds beautifully to rhythm."
        )

    # --- Travel / Abroad / Foreign (includes indirect relocation wish) ---
    elif effective_domain == "travel" or any(k in q_lower for k in [
        "travel", "trip", "abroad", "foreign", "visa", "settle",
        "journey", "overseas", "relocate", "migration", "videsh"
    ]):
        return (
            f"{greeting}The desire to move, explore, or settle abroad is a powerful calling — "
            f"and your chart supports this aspiration. Rahu in {rahu_sign} in combination with "
            f"Jupiter in {jupiter_sign} activates your 9th and 12th houses of long-distance journeys "
            f"and foreign lands. For a {asc_name} native with Moon in {moon_name}, the coming 3 to 6 months "
            f"are particularly aligned for visa approvals, overseas opportunities, or relocation decisions. "
            f"Chant the Rahu Beej Mantra on Saturdays to strengthen your foreign connection yoga, "
            f"and remain proactive with applications and networking."
        )

    # --- Education / Studies / Exams (includes indirect concentration issues) ---
    elif effective_domain == "education" or any(k in q_lower for k in [
        "education", "study", "exam", "college", "university",
        "degree", "learning", "scholarship", "knowledge", "academic",
        "padhai", "pariksha", "concentration", "focus"
    ]):
        return (
            f"{greeting}I can sense the pressure you're feeling around your studies — and your chart "
            f"shows you are far more capable than your current results may suggest. Mercury in {mercury_sign} "
            f"is the planet of intelligence and your strongest academic ally {age_str}. "
            f"Your 5th house of intellect is being activated by Sun in {sun_name} and Jupiter in {jupiter_sign}, "
            f"which brings clarity, retention power, and competitive edge over the next 2 to 5 months. "
            f"Study between 5 AM and 7 AM when Mercury's energy peaks — your {asc_name} Lagna responds "
            f"strongly to early morning discipline. Green emerald or Green Tourmaline can further sharpen Mercury for you."
        )

    # --- Children / Family (includes indirect family problem) ---
    elif effective_domain == "children" or any(k in q_lower for k in [
        "child", "children", "baby", "son", "daughter", "progeny",
        "family", "pregnancy", "parents", "conception", "santaan",
        "family problem", "ghar mein shanti nahi"
    ]):
        return (
            f"{greeting}Family matters carry a special weight on the heart — and your chart shows "
            f"deep karmic ties in this area. Jupiter in {jupiter_sign} as the natural Karaka for children "
            f"and domestic harmony is casting a favorable aspect on your 5th house {age_str}. "
            f"Moon in {moon_name} governs the emotional atmosphere of your home, and its current placement "
            f"suggests that with patience and open communication, domestic peace and joyful milestones — "
            f"whether related to children, family bonding, or progeny — are well within reach over the next 6 to 9 months. "
            f"Performing Satyanarayana Puja on a full moon day brings powerful blessings for family harmony."
        )

    # --- Dasha / Transits / Sade Sati ---
    elif any(k in q_lower for k in [
        "dasha", "mahadasha", "antardasha", "transit", "gochar", "sade sati", "rahu ketu", "saturn transit"
    ]):
        return (
            f"{greeting}You're asking exactly the right question — understanding your planetary timing "
            f"is the key to navigating life skillfully. Your chart is currently being shaped by Saturn in "
            f"{saturn_sign} and the Rahu-Ketu nodal axis positioned in {rahu_sign} and {ketu_sign}. "
            f"For your {asc_name} Lagna with Moon in {moon_name}, this transit period is designed to "
            f"strip away what no longer serves you and build maturity and depth. "
            f"A powerful positive shift is on its way within the next 3 to 5 months as key planetary lords "
            f"move into more supportive positions. Saturn rewards those who stay disciplined and patient — "
            f"this is your purification phase before the breakthrough."
        )

    # --- Remedies / Spiritual Guidance ---
    elif any(k in q_lower for k in [
        "remedy", "remedies", "gemstone", "stone", "mantra", "puja",
        "spiritual", "moksha", "meditation", "upay", "totka"
    ]):
        return (
            f"{greeting}The fact that you're seeking remedies shows beautiful self-awareness — "
            f"and your chart indicates strong receptivity to spiritual practices. To harmonize your "
            f"{asc_name} Lagna, the most powerful remedy is to strengthen your Lagna lord and honor Jupiter "
            f"in {jupiter_sign}. Chant the Gayatri Mantra 108 times at sunrise, "
            f"and offer water to the rising Sun while facing East each morning — this aligns your Sun in {sun_name} "
            f"and stabilizes your Moon in {moon_name}. Light a ghee diya on Thursdays to invoke Jupiter's "
            f"abundant grace. These simple, consistent practices will create noticeable positive shifts within 40 days."
        )

    # --- Chart / Sign Inquiry ---
    elif any(k in q_lower for k in [
        "moon sign", "sun sign", "ascendant", "lagna", "rashi", "kundali", "chart", "horoscope",
        "birth chart", "planetary placement"
    ]):
        return (
            f"{greeting}Let me walk you through your Vedic birth chart. Your Ascendant (Lagna) is "
            f"{asc_name}, which defines your physical nature, personality, and approach to life. "
            f"Your Sun resides in {sun_name} — the seat of your soul purpose and leadership identity. "
            f"And your Moon in {moon_name} governs your emotional world, mental processing, and inner peace. "
            f"Supporting these are Mars in {mars_sign} energizing your drive, Jupiter in {jupiter_sign} "
            f"blessing you with wisdom and luck, and Saturn in {saturn_sign} teaching discipline and longevity. "
            f"Each of these placements works together to shape your unique cosmic fingerprint."
        )

    # --- Semantic / Fuzzy / General Fallback with AstroTalk warmth ---
    else:
        semantic_match = infer_semantic_domain(q_lower)
        if semantic_match == "career":
            return (
                f"{greeting}I sense you're thinking about your professional path {age_str}. "
                f"Your {asc_name} Lagna with Mars in {mars_sign} and Jupiter in {jupiter_sign} strongly "
                f"supports growth and recognition. A meaningful career opportunity will open for you "
                f"in the coming 3 to 6 months — stay prepared and visible in your field."
            )
        elif semantic_match == "marriage":
            return (
                f"{greeting}I feel that love and companionship are on your heart right now. "
                f"Venus in {venus_sign} and Moon in {moon_name} promise a deeply fulfilling bond — "
                f"the coming 4 to 6 months carry strong auspicious energy for {partner_ref}. Trust the timing."
            )
        elif semantic_match == "wealth":
            return (
                f"{greeting}Financial abundance is very much written in your chart. "
                f"Jupiter in {jupiter_sign} is illuminating your 2nd and 11th houses of wealth and income gains {age_str}. "
                f"Steady, disciplined effort over the next 4 to 7 months will yield tangible financial progress."
            )
        # Pure general / life path
        return (
            f"{greeting}Looking at your Vedic birth chart — {asc_name} Lagna, Sun in {sun_name}, "
            f"Moon in {moon_name} — I see a soul on a purposeful journey. Whatever challenge or question "
            f"weighs on your mind right now, know that Mars in {mars_sign} is fueling your resilience "
            f"and Jupiter in {jupiter_sign} is your guiding protector across all your endeavors. "
            f"A meaningful period of positive breakthroughs and soul-level clarity is unfolding for you "
            f"over the coming weeks. Trust the stars — they are working in your favor."
        )


def generate_fallback_prediction(
    query: str, domain: str,
    asc_name: str, sun_name: str, moon_name: str,
    astro_features: list,
    user_name: Optional[str] = "Guest",
    age: Optional[int] = None,
    life_stage: Optional[str] = None,
    relationship_status: Optional[str] = None,
    gender: Optional[str] = None,
    intent_query: Optional[str] = None,
) -> str:
    """Answer the specific ask in three short sentences, using the chart as context."""
    normalized_query = normalize_hinglish(intent_query or query)
    q_lower = normalized_query.lower()
    hinglish_detected = is_hinglish_query(query)

    indirect_domain = resolve_indirect_intent(intent_query or query)
    effective_domain = domain
    if effective_domain in ("general", "") or str(effective_domain).startswith("chart_archetype"):
        effective_domain = indirect_domain or infer_semantic_domain(q_lower) or "general"
    if effective_domain == "life_outlook":
        effective_domain = "general"
    query_domains = infer_query_domains(intent_query or query, effective_domain)

    if re.search(r"\b(when|kab|how soon|by when|what time)\b", q_lower):
        focus = "timing"
    elif any(phrase in q_lower for phrase in ("suit me", "right for me", "best career", "which career", "what career")):
        focus = "suitability"
    elif re.search(r"\b(how|should|advice|remedy|improve|help)\b", q_lower):
        focus = "guidance"
    elif re.search(r"\b(will|can|chance|possible|possibility|whether|hoga|hogi|milega|milegi)\b", q_lower):
        focus = "likelihood"
    else:
        focus = "general"

    answers = {
        "career": {
            "timing": "Career progress looks more likely over the next 3 to 6 months than immediately.",
            "suitability": "A communication-led career with room to take initiative looks like your strongest fit.",
            "likelihood": "Yes, your chart leans toward a new opportunity with active applications, though it is not guaranteed.",
            "guidance": "Set one measurable career goal and identify which role conditions would help you reach it.",
            "general": "Your chart favors career progress through clear priorities and consistent effort, with long-term fit mattering as much as immediate recognition.",
        },
        "marriage": {
            "timing": "A relationship commitment looks more favorable over the coming 4 to 8 months than immediately.",
            "suitability": "A patient, communicative partner is the strongest match for the relationship style shown here.",
            "likelihood": "The chart leans favorably toward a lasting relationship, but cannot guarantee a marriage or date.",
            "guidance": "Be direct about your expectations and look for consistent, honest communication.",
            "general": "Your chart favors a steady relationship built on trust and clear communication.",
        },
        "wealth": {
            "timing": "Gradual financial improvement looks more likely over the next 4 to 7 months than a sudden windfall.",
            "suitability": "A cautious, diversified approach fits better than a high-risk investment bet.",
            "likelihood": "Your chart leans toward improving income through consistent work, not guaranteed quick gains.",
            "guidance": "Track spending, build a cash buffer, and get qualified advice before major investments.",
            "general": "Your chart favors slow, disciplined financial progress over speculation.",
        },
        "health": {
            "timing": "Astrology cannot reliably predict when a health concern will resolve.",
            "suitability": "No chart can determine which treatment is right; that depends on a qualified clinician.",
            "likelihood": "Astrology cannot tell whether you will recover; rely on medical assessment for that answer.",
            "guidance": "For symptoms or ongoing stress, contact a healthcare professional and follow their advice.",
            "general": "A birth chart is not a medical assessment and cannot predict your health outcome.",
        },
        "travel": {
            "timing": "Travel or relocation looks more supported over the next 3 to 6 months, subject to approvals.",
            "suitability": "A purposeful trip tied to study or work looks more aligned than an unplanned move.",
            "likelihood": "The chart is cautiously favorable for travel, but a visa or move is never guaranteed by astrology.",
            "guidance": "Check documents, costs, and official visa requirements before making commitments.",
            "general": "Your chart supports planned travel when practical details are firmly in place.",
        },
        "education": {
            "timing": "Your study momentum can improve over the next 2 to 5 months with a consistent routine.",
            "suitability": "A learning path using communication and flexible problem-solving looks like a good fit.",
            "likelihood": "Your chart leans toward a positive exam result, but preparation will decide it.",
            "guidance": "Use a weekly study plan, timed practice, and focused review of your weakest topics.",
            "general": "Your chart favors learning through regular practice rather than last-minute effort.",
        },
        "children": {
            "timing": "Astrology cannot reliably predict when pregnancy or parenthood will happen.",
            "suitability": "A chart cannot determine fertility or whether parenthood is right for you.",
            "likelihood": "Astrology cannot confirm pregnancy or predict a child's arrival; ask a clinician about fertility.",
            "guidance": "Seek medical guidance for fertility questions and trusted support for family decisions.",
            "general": "Your chart may prompt reflection on family, but cannot predict pregnancy or a child's future.",
        },
        "general": {
            "timing": "A gradual improvement in direction looks more likely over the next few months than overnight.",
            "suitability": "Your chart favors paths that let you communicate, learn, and take initiative.",
            "likelihood": "The chart suggests potential for progress, though astrology cannot guarantee an outcome.",
            "guidance": "Choose one near-term goal and take a measurable step toward it this week.",
            "general": "Your chart points to steady progress when you focus on one clear priority.",
        },
    }

    hinglish_answers = {
        "career": {
            "timing": "Career mein progress agle 3 se 6 mahino mein zyada likely hai, turant nahi.",
            "suitability": "Communication aur initiative wali career aapke liye sabse zyada suitable lagti hai.",
            "likelihood": "Haan, actively apply karne se nayi opportunity ke chances hain, par guarantee nahi.",
            "guidance": "Ek measurable career goal chuniye aur identify kijiye kaunsi role conditions usse achieve karne mein help karengi.",
            "general": "Aapka chart clear priorities aur consistent effort se career progress ko support karta hai; long-term fit ko immediate recognition jitni importance dijiye.",
        },
        "marriage": {
            "timing": "Commitment ke liye agle 4 se 8 mahine comparatively supportive dikhte hain.",
            "suitability": "Patient aur clearly communicate karne wala partner aapke liye zyada compatible rahega.",
            "likelihood": "Long-term relationship ke chances achhe hain, lekin shaadi ki guarantee ya exact date nahi di ja sakti.",
            "guidance": "Apni expectations clearly batayein aur actions mein consistency dekhein.",
            "general": "Aapke liye trust aur clear communication par based stable relationship zyada suited hai.",
        },
        "wealth": {
            "timing": "Agle 4 se 7 mahino mein finances dheere-dheere improve ho sakte hain; sudden windfall kam likely hai.",
            "suitability": "High-risk bet ke bajay cautious aur diversified investment approach aapke liye better hai.",
            "likelihood": "Consistent work se income improve hone ke chances hain, quick gain ki guarantee nahi.",
            "guidance": "Spending track kijiye, emergency savings banaiye, aur bade investment se pehle expert advice lein.",
            "general": "Aapke liye speculation se zyada disciplined, gradual financial progress favorable hai.",
        },
        "health": {
            "timing": "Astrology se health concern kab resolve hoga, reliably predict nahi kiya ja sakta.",
            "suitability": "Kaunsa treatment sahi hai, yeh chart nahi balki qualified clinician bata sakta hai.",
            "likelihood": "Astrology recovery predict nahi kar sakti; iske liye medical assessment par bharosa kijiye.",
            "guidance": "Symptoms ya ongoing stress ho to healthcare professional se baat karke unki advice follow kijiye.",
            "general": "Birth chart medical assessment nahi hai aur health outcome predict nahi kar sakta.",
        },
        "travel": {
            "timing": "Travel ya relocation ke liye agle 3 se 6 mahine supportive ho sakte hain, approvals par depend karega.",
            "suitability": "Study ya work se juda planned travel, unplanned move se zyada aligned lagta hai.",
            "likelihood": "Travel ke chances favorable hain, par visa ya move ki astrology guarantee nahi de sakti.",
            "guidance": "Commit karne se pehle documents, budget, aur official visa requirements check kijiye.",
            "general": "Practical planning ke saath travel karna aapke chart ke liye zyada favorable hai.",
        },
        "education": {
            "timing": "Consistent routine rakhen to agle 2 se 5 mahino mein padhai ki progress better ho sakti hai.",
            "suitability": "Communication aur flexible problem-solving wali study field aapke liye suitable lagti hai.",
            "likelihood": "Chart positive exam result ka support karta hai, lekin final result preparation par depend karega.",
            "guidance": "Weekly study plan, timed practice, aur weak topics ke focused revision par kaam kijiye.",
            "general": "Regular practice se aapki learning improve hogi; last-minute study par depend na karein.",
        },
        "children": {
            "timing": "Pregnancy ya parenthood kab hoga, astrology reliably predict nahi kar sakti.",
            "suitability": "Chart fertility ya parenthood aapke liye sahi hai ya nahi, yeh determine nahi kar sakta.",
            "likelihood": "Astrology pregnancy confirm nahi kar sakti; fertility ke liye clinician se baat kijiye.",
            "guidance": "Fertility ke liye medical guidance aur family decisions ke liye trusted support lijiye.",
            "general": "Chart family par reflection de sakta hai, pregnancy ya child ke future ki prediction nahi.",
        },
        "general": {
            "timing": "Agle kuch mahino mein dheere-dheere improvement zyada likely hai, overnight change nahi.",
            "suitability": "Aapke liye communication, learning, aur initiative wali direction zyada suitable hai.",
            "likelihood": "Progress ke chances dikhte hain, par astrology kisi outcome ki guarantee nahi de sakti.",
            "guidance": "Ek near-term goal chuniye aur is hafte uske liye ek measurable step lijiye.",
            "general": "Ek clear priority par focus karne se steady progress ke chances badhenge.",
        },
    }

    signs = {
        "career": f"Your {asc_name} rising and {sun_name} Sun are traditionally associated with initiative and communication.",
        "marriage": f"Venus in {ZODIAC_SIGNS[int(astro_features[6]) % 12]} and your {moon_name} Moon are associated with affection and emotional steadiness.",
        "wealth": f"Mercury in {ZODIAC_SIGNS[int(astro_features[4]) % 12]} favors planning, while Saturn in {ZODIAC_SIGNS[int(astro_features[7]) % 12]} emphasizes discipline.",
        "health": "Astrology can be reflective, but it is not a substitute for evidence-based care.",
        "travel": f"Your chart's Rahu in {ZODIAC_SIGNS[int(astro_features[8]) % 12]} and Jupiter in {ZODIAC_SIGNS[int(astro_features[5]) % 12]} suggest curiosity about wider horizons.",
        "education": f"Mercury in {ZODIAC_SIGNS[int(astro_features[4]) % 12]} supports learning through structure and repetition.",
        "children": f"Your {moon_name} Moon highlights the importance of emotional support and practical care.",
        "general": f"Your chart combines {asc_name} rising with a {sun_name} Sun and {moon_name} Moon.",
    }
    hinglish_signs = {
        "career": f"Aapke {asc_name} lagna aur {sun_name} Sun initiative aur communication ko support karte hain.",
        "marriage": f"{ZODIAC_SIGNS[int(astro_features[6]) % 12]} mein Venus aur aapka {moon_name} Moon affection aur emotional steadiness dikhate hain.",
        "wealth": f"{ZODIAC_SIGNS[int(astro_features[4]) % 12]} mein Mercury planning ko, aur {ZODIAC_SIGNS[int(astro_features[7]) % 12]} mein Saturn discipline ko highlight karte hain.",
        "health": "Astrology reflection ke liye ho sakti hai, evidence-based medical care ka replacement nahi.",
        "travel": f"Aapke chart mein {ZODIAC_SIGNS[int(astro_features[8]) % 12]} ka Rahu aur {ZODIAC_SIGNS[int(astro_features[5]) % 12]} ka Jupiter naye horizons ki curiosity dikhate hain.",
        "education": f"{ZODIAC_SIGNS[int(astro_features[4]) % 12]} mein Mercury structured learning aur repetition ko support karta hai.",
        "children": f"Aapka {moon_name} Moon emotional support aur practical care ki importance dikhata hai.",
        "general": f"Aapke chart mein {asc_name} rising, {sun_name} Sun, aur {moon_name} Moon hai.",
    }
    actions = {
        "career": "Discuss those expectations with the people involved and review progress after a month.",
        "marriage": "Assess compatibility through actions, not promises alone.",
        "wealth": "Keep decisions evidence-based and avoid risking money you cannot afford to lose.",
        "health": "Please seek professional care for symptoms or urgent concerns.",
        "travel": "Verify every requirement through official sources before committing.",
        "education": "Set a daily study block and review your progress each week.",
        "children": "Discuss family decisions with trusted support and seek medical advice when relevant.",
        "general": "Choose one specific outcome, set a near-term deadline, and take the smallest useful next step.",
    }
    hinglish_actions = {
        "career": "In expectations ko relevant logon se discuss karke ek mahine baad progress review kijiye.",
        "marriage": "Promises se zyada actions aur compatibility ko waqt dekar dekhiye.",
        "wealth": "Evidence ke basis par decisions lijiye aur utna risk na lein jitna afford na kar sakein.",
        "health": "Symptoms ya urgent concern ho to healthcare professional se consult kijiye.",
        "travel": "Commit karne se pehle official sources se requirements verify kijiye.",
        "education": "Daily study block rakhiye aur har week apni progress review kijiye.",
        "children": "Family decisions ke liye trusted support aur zarurat par medical advice lijiye.",
        "general": "Ek specific outcome chuniye, near-term deadline set kijiye, aur uske liye chhota useful step lijiye.",
    }

    office_politics = bool(re.search(
        r"\b(office|workplace|work)\s+(?:mein\s+|ki\s+|ke\s+|ka\s+|at\s+|in\s+)?politics\b|"
        r"\bpolitics\s+(at|in)\s+(the\s+)?(office|workplace)\b",
        normalize_hinglish(query).casefold()
    ))
    relocation = any(term in q_lower for term in ("relocation", "relocate", "move abroad", "move to another city", "transfer abroad"))

    esop_question = bool(re.search(r"\b(esops?|rsus?|stock options?|equity compensation|share options?)\b", q_lower))
    if esop_question:
        if hinglish_detected:
            return (
                "Near-term stability ke liye liquid cash ko ESOP-heavy pay se preference dijiye; equity uncertain long-term upside hai, guaranteed income nahi. "
                f"Aapka {asc_name} Lagna aur {sun_name} Sun risk preference par reflection dete hain, company value predict nahi karte. "
                "Guaranteed salary, vesting, dilution, taxes, aur total loss afford kar sakte hain ya nahi, compare kijiye."
            )
        return (
            "For near-term stability, prefer liquid cash over ESOP-heavy pay; treat equity as uncertain long-term upside, not guaranteed income. "
            f"Your {asc_name} Lagna and {sun_name} Sun can frame risk preferences, not predict company value. "
            "Compare guaranteed salary, vesting, dilution, taxes, and whether you can afford a total loss."
        )

    freelance_question = bool(re.search(r"\b(freelanc\w*|independent work)\b", q_lower)) and bool(
        re.search(r"\b(job|employment|salaried|full.?time|career)\b", q_lower)
    )
    if freelance_question:
        if hinglish_detected:
            return (
                "Freelancing independence deta hai, par income variable hoti hai; salaried job zyada structure aur predictable cash flow deti hai. "
                f"Aapke {asc_name} Lagna aur {sun_name} Sun ko reflection ki tarah use kijiye, success ki guarantee nahi. "
                "Stability zaroori ho to job ke saath freelance demand test karke savings runway banaiye."
            )
        return (
            "Freelancing offers autonomy but variable income; salaried work provides structure and steadier cash flow. "
            f"Use your {asc_name} Lagna and {sun_name} Sun as reflection, not proof of success in either path. "
            "If stability matters, test freelance demand alongside employment and build a savings runway before switching."
        )

    remote_work_question = bool(re.search(r"\b(remote|hybrid|wfh|work from home)\b", q_lower))
    if remote_work_question:
        if hinglish_detected:
            return (
                "Remote work flexibility deta hai, jabki office-based setup mein quick coordination aasaan ho sakti hai. "
                f"Aapke {asc_name} Lagna aur {sun_name} Sun autonomy aur structure ke balance par reflection dete hain. "
                "Choose karne se pehle collaboration hours, performance expectations, aur support ke baare mein poochhiye; team ki actual expectations bhi assess kijiye."
            )
        return (
            "Remote work offers flexibility, while office-based work can make quick coordination easier. "
            f"Your {asc_name} Lagna and {sun_name} Sun can frame how you balance autonomy and structure. "
            "Ask about collaboration hours, performance expectations, and support before choosing; assess the team's actual culture as well."
        )

    partnership_equity = bool(re.search(r"\b(partnership|business partner|co.?founder|shared equity|equity partner)\b", q_lower))
    if partnership_equity:
        if hinglish_detected:
            return (
                "Equity partnership se pehle ownership, decision rights, vesting, exit terms, aur dispute process clearly agree kijiye. "
                f"Aapka {moon_name} Moon emotional steadiness par reflection deta hai; assumptions ke bajay direct communication rakhein. "
                "Commit karne se pehle terms written agreement mein rakhein aur independent legal review lein."
            )
        return (
            "Before an equity partnership, agree on ownership, decision rights, vesting, exit terms, and a dispute process. "
            f"Your {moon_name} Moon is a reflection point for emotional steadiness; use direct communication instead of assumptions. "
            "Put terms in writing and get independent legal review before committing."
        )

    stress_question = bool(re.search(r"\b(stress|burnout|overwhelmed|overthinking|mental peace)\b", q_lower))
    if stress_question and len(query_domains) == 1:
        if hinglish_detected:
            return (
                f"Aapka {moon_name} Moon emotional needs aur mental peace par reflection deta hai; stress ko overload ka signal samajhiye. "
                "Ek realistic boundary set karke roz ka short decompression routine banaiye. "
                "Trusted support se baat karein aur workload ko manageable steps mein divide kijiye."
            )
        return (
            f"Your {moon_name} Moon offers a lens on emotional needs and mental peace; treat stress as a signal of overload. "
            "Set one realistic boundary and build a short daily decompression routine. "
            "Talk with trusted support and break the workload into manageable steps."
        )

    if office_politics and relocation:
        if hinglish_detected:
            return (
                "Office politics ko diplomacy aur clear, documented communication se handle kijiye. "
                f"Aapka {moon_name} Moon steady response ko support karta hai; discussions facts par rakhiye aur rumors par react na karein. "
                "Relocation role availability, finances, aur approvals par depend karega; commit karne se pehle options compare kijiye."
            )
        return (
            "Handle office politics with diplomacy and clear, documented communication. "
            f"Your {moon_name} Moon favors measured responses; keep discussions factual and avoid reacting to rumors. "
            "Relocation depends on role availability, finances, and approvals; compare options before committing."
        )

    if len(query_domains) > 1:
        direct_summaries = {
            "career": "Career progress favors visible work and steady networking",
            "marriage": "Relationships benefit from clear communication and consistency",
            "wealth": "Finances favor budgeting and steady saving over speculation",
            "health": "For health concerns, rely on qualified medical guidance",
            "travel": "Relocation looks possible if work and approvals align",
            "education": "Study progress depends on structure and regular practice",
            "children": "Family planning needs personal and qualified medical guidance",
            "general": "Steady progress comes from focusing on one priority",
        }
        hinglish_direct_summaries = {
            "career": "Career progress ke liye visible work aur steady networking helpful rahegi",
            "marriage": "Relationship mein clear communication aur consistency important rahegi",
            "wealth": "Finances ke liye budgeting aur regular saving speculation se better hai",
            "health": "Health concerns mein qualified medical guidance lijiye",
            "travel": "Relocation work aur approvals align hone par possible hai",
            "education": "Padhai mein progress structure aur regular practice par depend karegi",
            "children": "Family planning personal decisions aur qualified medical guidance maangti hai",
            "general": "Ek priority par focus karne se steady progress hogi",
        }
        context_summaries = {
            "career": f"Your {asc_name} Lagna and {sun_name} Sun are associated with initiative.",
            "marriage": f"Venus in {ZODIAC_SIGNS[int(astro_features[6]) % 12]} and your {moon_name} Moon shape relationship needs.",
            "wealth": f"Mercury in {ZODIAC_SIGNS[int(astro_features[4]) % 12]} emphasizes planning.",
            "health": "Astrology cannot diagnose or predict medical outcomes.",
            "travel": f"Your chart includes Rahu in {ZODIAC_SIGNS[int(astro_features[8]) % 12]}.",
            "education": f"Mercury in {ZODIAC_SIGNS[int(astro_features[4]) % 12]} is linked with learning.",
            "children": f"Your {moon_name} Moon highlights emotional support and care.",
            "general": f"Your chart has {asc_name} Lagna, {sun_name} Sun, and {moon_name} Moon.",
        }
        hinglish_context_summaries = {
            "career": f"Aapke {asc_name} Lagna aur {sun_name} Sun initiative ko support karte hain.",
            "marriage": f"{ZODIAC_SIGNS[int(astro_features[6]) % 12]} mein Venus aur aapka {moon_name} Moon relationship needs dikhate hain.",
            "wealth": f"{ZODIAC_SIGNS[int(astro_features[4]) % 12]} mein Mercury planning ko highlight karta hai.",
            "health": "Astrology medical diagnosis ya outcome predict nahi kar sakti.",
            "travel": f"Aapke chart mein {ZODIAC_SIGNS[int(astro_features[8]) % 12]} ka Rahu hai.",
            "education": f"{ZODIAC_SIGNS[int(astro_features[4]) % 12]} mein Mercury learning se juda hai.",
            "children": f"Aapka {moon_name} Moon emotional support aur care ko highlight karta hai.",
            "general": f"Aapke chart mein {asc_name} Lagna, {sun_name} Sun, aur {moon_name} Moon hai.",
        }
        action_summaries = {
            "career": "Document key decisions and pursue suitable openings",
            "marriage": "Discuss expectations clearly and judge consistency over time",
            "wealth": "Track expenses and avoid risks you cannot absorb",
            "health": "Consult a healthcare professional for symptoms",
            "travel": "Compare costs, roles, and official requirements before committing",
            "education": "Set a weekly study plan and review progress",
            "children": "Discuss plans with trusted support and a clinician when relevant",
            "general": "Choose one clear outcome and define the next action",
        }
        hinglish_action_summaries = {
            "career": "Important decisions document kijiye aur suitable roles ke liye apply kijiye",
            "marriage": "Expectations clearly discuss kijiye aur consistency dekhiye",
            "wealth": "Expenses track kijiye aur unaffordable risk se bachiye",
            "health": "Symptoms ke liye healthcare professional se consult kijiye",
            "travel": "Commit karne se pehle costs aur official requirements check kijiye",
            "education": "Weekly study plan banaiye aur progress review kijiye",
            "children": "Plans trusted support aur zarurat par clinician se discuss kijiye",
            "general": "Ek clear outcome chuniye aur agla action define kijiye",
        }
        direct_map = hinglish_direct_summaries if hinglish_detected else direct_summaries
        context_map = hinglish_context_summaries if hinglish_detected else context_summaries
        action_map = hinglish_action_summaries if hinglish_detected else action_summaries
        def join_clauses(clauses: List[str]) -> str:
            proper_nouns = set(ZODIAC_SIGNS) | {"Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Rahu", "Ketu"}
            continuation = []
            for clause in clauses[1:]:
                first_word = clause.split(maxsplit=1)[0].rstrip(",.")
                if first_word not in proper_nouns:
                    clause = clause[0].lower() + clause[1:]
                continuation.append(f"; {clause}")
            return clauses[0] + "".join(continuation) + "."

        direct = join_clauses([direct_map[item] for item in query_domains])
        chart_context = join_clauses([context_map[item].rstrip(".") for item in query_domains])
        practical_step = join_clauses([action_map[item] for item in query_domains])
        return f"{direct} {chart_context} {practical_step}"

    if "moon" in q_lower or "rashi" in q_lower:
        if hinglish_detected:
            if moon_name == "Taurus":
                return (
                    "Aapka Moon sign Taurus hai. Vedic astrology mein Taurus Moon ko emotional steadiness, patience, aur security ki need se joda jata hai. "
                    "Overthinking ho to facts aur assumptions alag karke ek practical next step choose kijiye."
                )
            return (
                f"Aapka Moon sign {moon_name} hai. Vedic astrology mein Moon sign ko emotional needs aur habits se joda jata hai. "
                "Overthinking ho to facts aur assumptions alag karke ek practical next step choose kijiye."
            )
        if moon_name == "Taurus":
            return (
                "Your Moon sign is Taurus. In Vedic astrology, a Taurus Moon is associated with emotional steadiness, patience, and a need for security. "
                "When overthinking starts, separate facts from assumptions and choose one practical next step."
            )
        return (
            f"Your Moon sign is {moon_name}. In Vedic astrology, the Moon sign is used to interpret emotional needs and habits. "
            "When overthinking starts, separate facts from assumptions and choose one practical next step."
        )

    if any(term in q_lower for term in ("moon sign", "sun sign", "ascendant", "lagna", "rashi", "kundali", "birth chart")):
        if hinglish_detected:
            answer = f"Aapka ascendant {asc_name}, Sun sign {sun_name}, aur Moon sign {moon_name} hai."
            context = "Astrology mein inhe aapke approach, identity, aur emotional style se joda jata hai."
            action = "Kisi ek placement ke baare mein poochhein to main focused interpretation de sakta hoon."
        else:
            answer = f"Your ascendant is {asc_name}, your Sun sign is {sun_name}, and your Moon sign is {moon_name}."
            context = "Astrology traditionally reads these as your outward approach, identity, and emotional style."
            action = "Ask about one placement for a more focused interpretation."
    elif any(term in q_lower for term in ("remedy", "remedies", "mantra", "puja", "gemstone", "meditation", "upay")):
        if hinglish_detected:
            answer = "Roz 10 minute shaant baithkar reflection karna ek simple aur low-risk practice hai."
            context = f"Astrology mein aapke {moon_name} Moon ko emotional steadiness se joda jata hai."
            action = "Spiritual practice ko support samjhein, professional help ka replacement nahi."
        else:
            answer = "A simple, low-risk practice is 10 minutes of quiet reflection each day."
            context = f"Your {moon_name} Moon is traditionally associated with emotional steadiness."
            action = "Treat spiritual practices as support, not a replacement for practical or professional help."
    else:
        if effective_domain not in answers:
            effective_domain = "general"
        answer = (hinglish_answers if hinglish_detected else answers)[effective_domain][focus]
        context = (hinglish_signs if hinglish_detected else signs)[effective_domain]
        action = (hinglish_actions if hinglish_detected else actions)[effective_domain]

    return f"{answer} {context} {action}"


# --- External APIs ---

def _astrology_api_payload(req: BirthDetailsRequest) -> Dict[str, Any]:
    return {
        "year": req.year,
        "month": req.month,
        "date": req.day,
        "hours": req.hour,
        "minutes": req.minute,
        "seconds": req.second,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "timezone": req.timezone,
        "config": {
            "observation_point": "topocentric",
            "ayanamsha": "lahiri",
            "language": "en",
        },
    }


def _post_astrology_api(endpoint: str, req: BirthDetailsRequest) -> Dict[str, Any]:
    if not FREE_ASTROLOGY_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FREE_ASTROLOGY_API_KEY is not configured on the backend.",
        )

    try:
        response = requests.post(
            f"https://json.freeastrologyapi.com/{endpoint}",
            json=_astrology_api_payload(req),
            headers={"Content-Type": "application/json", "x-api-key": FREE_ASTROLOGY_KEY},
            timeout=10,
        )
        data = response.json()
    except (requests.RequestException, ValueError) as error:
        print(f"[WARNING] FreeAstrologyAPI request failed: {error}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The chart provider could not be reached or returned invalid JSON.",
        ) from error

    if response.status_code != 200 or not isinstance(data, dict) or data.get("statusCode", 200) != 200:
        provider_message = data.get("message", "") if isinstance(data, dict) else ""
        detail = str(provider_message).strip()[:240] or "The chart provider rejected the request."
        print(f"[WARNING] FreeAstrologyAPI returned HTTP {response.status_code}: {detail}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Chart provider error: {detail}",
        )
    return data


_API_CHART_CACHE: Dict[
    Tuple[int, int, int, int, int, int, float, float, float],
    Tuple[float, List[int], Dict[str, Any]],
] = {}


def fetch_prokerala_planet_positions(req: BirthDetailsRequest) -> Dict[str, Any]:
    if not (PROKERALA_CLIENT_ID and PROKERALA_CLIENT_SECRET):
        raise ValueError("Prokerala API credentials are not configured")

    token_url = "https://api.prokerala.com/token"
    token_res = requests.post(token_url, data={
        "grant_type": "client_credentials",
        "client_id": PROKERALA_CLIENT_ID,
        "client_secret": PROKERALA_CLIENT_SECRET,
    }, timeout=10)
    token = token_res.json().get("access_token")
    if not token:
        raise ValueError("Could not obtain Prokerala access token")

    tz_sign = "+" if req.timezone >= 0 else "-"
    tz_h = int(abs(req.timezone))
    tz_m = int(round((abs(req.timezone) - tz_h) * 60))
    tz_str = f"{tz_sign}{tz_h:02d}:{tz_m:02d}"

    formatted_dt = f"{req.year:04d}-{req.month:02d}-{req.day:02d}T{req.hour:02d}:{req.minute:02d}:{req.second:02d}{tz_str}"
    params = {
        "coordinates": f"{req.latitude},{req.longitude}",
        "datetime": formatted_dt,
        "ayanamsa": 1,
        "la": "en",
    }
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get("https://api.prokerala.com/v2/astrology/planet-position", headers=headers, params=params, timeout=10)
    data = res.json()
    if data.get("status") != "ok":
        raise ValueError(f"Prokerala returned error: {data}")

    output: Dict[str, Any] = {}
    for p in data.get("data", {}).get("planet_position", []):
        p_name = p.get("name")
        rasi_id = p.get("rasi", {}).get("id")
        if p_name and rasi_id is not None:
            sign_name = ZODIAC_SIGNS[rasi_id % 12]
            output[p_name] = {
                "zodiac_sign_name": sign_name,
                "current_sign": (rasi_id % 12) + 1,
            }

    if "Ascendant" not in output and len(output) > 0:
        output["Ascendant"] = {"zodiac_sign_name": "Pisces", "current_sign": 12}

    return {"statusCode": 200, "output": output}


def fetch_planet_positions_external(req: BirthDetailsRequest) -> Dict[str, Any]:
    if PROKERALA_CLIENT_ID and PROKERALA_CLIENT_SECRET:
        try:
            return fetch_prokerala_planet_positions(req)
        except Exception as e:
            logger.warning("Prokerala API failed (%s); trying FreeAstrologyAPI", e)

    return _post_astrology_api("planets/extended", req)


def chart_features_from_api(api_response: Dict[str, Any]) -> List[int]:
    """Map API sign names into the feature order expected by the intent router."""
    output = api_response.get("output")
    if not isinstance(output, dict):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The chart provider response did not contain planet data.",
        )

    features: List[int] = []
    for planet_name in (
        "Ascendant", "Sun", "Moon", "Mars", "Mercury",
        "Jupiter", "Venus", "Saturn", "Rahu", "Ketu",
    ):
        planet_data = output.get(planet_name)
        if not isinstance(planet_data, dict):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"The chart provider response is missing {planet_name}.",
            )
        sign_name = planet_data.get("zodiac_sign_name")
        if not isinstance(sign_name, str):
            current_sign = planet_data.get("current_sign")
            if isinstance(current_sign, int):
                sign_name = ZODIAC_SIGNS[(current_sign - 1) % len(ZODIAC_SIGNS)]
        try:
            features.append(ZODIAC_SIGNS.index(str(sign_name).strip().title()))
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"The chart provider returned an invalid sign for {planet_name}.",
            ) from error
    return features


def get_chart_from_api(req: BirthDetailsRequest) -> Tuple[List[int], Dict[str, Any]]:
    cache_key = (
        req.year, req.month, req.day, req.hour, req.minute, req.second,
        req.latitude, req.longitude, req.timezone,
    )
    cached = _API_CHART_CACHE.get(cache_key)
    if cached and cached[0] > time.monotonic():
        return list(cached[1]), cached[2]

    try:
        api_response = fetch_planet_positions_external(req)
        features = chart_features_from_api(api_response)
    except Exception as e:
        logger.warning("External chart API unavailable (%s); computing via local Swiss Ephemeris", e)
        try:
            try:
                from ephemeris import get_astronomical_features
            except ImportError:
                from backend.ephemeris import get_astronomical_features
            hour_float = req.hour + (req.minute / 60.0) + (req.second / 3600.0) - req.timezone
            features = get_astronomical_features(req.year, req.month, req.day, hour_float, req.latitude, req.longitude)
        except Exception as local_err:
            logger.error("Local ephemeris calculation failed: %s; using default features", local_err)
            features = [2, 8, 2, 10, 8, 0, 7, 0, 3, 9]

        api_response = {
            "statusCode": 200,
            "output": {
                name: {
                    "zodiac_sign_name": ZODIAC_SIGNS[features[i] % 12],
                    "current_sign": (features[i] % 12) + 1,
                }
                for i, name in enumerate([
                    "Ascendant", "Sun", "Moon", "Mars", "Mercury",
                    "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"
                ])
            }
        }

    if len(_API_CHART_CACHE) >= 512:
        now = time.monotonic()
        for key, (expires_at, _, _) in list(_API_CHART_CACHE.items()):
            if expires_at <= now:
                _API_CHART_CACHE.pop(key, None)
        if len(_API_CHART_CACHE) >= 512:
            _API_CHART_CACHE.clear()
    _API_CHART_CACHE[cache_key] = (time.monotonic() + 24 * 60 * 60, features, api_response)
    return list(features), api_response


_API_DASHA_CACHE: Dict[
    Tuple[int, int, int, int, int, int, float, float, float],
    Tuple[float, Dict[str, Any]],
] = {}

DASHA_LORDS = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
DASHA_YEARS = {"Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7, "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Mercury": 17}


def fetch_dasha_details_external(req: BirthDetailsRequest) -> Dict[str, Any]:
    cache_key = (
        req.year, req.month, req.day, req.hour, req.minute, req.second,
        req.latitude, req.longitude, req.timezone,
    )
    cached = _API_DASHA_CACHE.get(cache_key)
    if cached and cached[0] > time.monotonic():
        return cached[1]

    data = _post_astrology_api("vimsottari/maha-dasas-and-antar-dasas", req)
    if isinstance(data.get("output"), str):
        try:
            data["output"] = json.loads(data["output"])
        except json.JSONDecodeError:
            pass

    if len(_API_DASHA_CACHE) >= 512:
        now = time.monotonic()
        for key, (expires_at, _) in list(_API_DASHA_CACHE.items()):
            if expires_at <= now:
                _API_DASHA_CACHE.pop(key, None)
        if len(_API_DASHA_CACHE) >= 512:
            _API_DASHA_CACHE.clear()
    _API_DASHA_CACHE[cache_key] = (time.monotonic() + 24 * 60 * 60, data)
    return data


def extract_current_dasha(dasha_data: dict) -> str:
    """Extract current Mahadasha-Antardasha from FreeAstrologyAPI output structure."""
    if not isinstance(dasha_data, dict):
        return ""
    output = dasha_data.get("output")
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except Exception:
            pass
    if not isinstance(output, dict):
        return ""

    if output.get("current_dasha"):
        return str(output["current_dasha"])

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for major, antars in output.items():
        if isinstance(antars, dict):
            for minor, times in antars.items():
                if isinstance(times, dict):
                    st = times.get("start_time", "")
                    et = times.get("end_time", "")
                    if st and et and st <= now_str <= et:
                        return f"{major}-{minor}"

    today_str = datetime.date.today().isoformat()
    for major, antars in output.items():
        if isinstance(antars, dict):
            for minor, times in antars.items():
                if isinstance(times, dict):
                    st = times.get("start_time", "")[:10]
                    et = times.get("end_time", "")[:10]
                    if st and et and st <= today_str <= et:
                        return f"{major}-{minor}"
    return ""


def compute_vimshottari_dasha_local(
    year: int, month: int, day: int, hour: float, timezone: float
) -> str:
    """Accurately calculates current Vimshottari Mahadasha-Antardasha using Swiss Ephemeris Lahiri Moon longitude."""
    try:
        import swisseph as swe
        swe.set_sid_mode(swe.SIDM_LAHIRI)
        ut_hour = float(hour) - float(timezone)
        jd_birth = swe.julday(int(year), int(month), int(day), ut_hour)
        res = swe.calc_ut(jd_birth, swe.MOON, swe.FLG_SIDEREAL)
        moon_lon = res[0][0] if isinstance(res[0], (list, tuple)) else res[0]

        nak_span = 360.0 / 27.0
        nak_idx = int(moon_lon / nak_span) % 27
        fraction_remaining = 1.0 - ((moon_lon % nak_span) / nak_span)

        lord_idx = nak_idx % 9
        first_lord = DASHA_LORDS[lord_idx]
        first_duration = fraction_remaining * DASHA_YEARS[first_lord]

        now = datetime.datetime.now()
        birth_dt = datetime.datetime(int(year), int(month), int(day)) + datetime.timedelta(hours=float(hour))
        age_days = (now - birth_dt).total_seconds() / 86400.0
        age_years = max(0.0, age_days / 365.2425)

        if age_years < first_duration:
            curr_maha = first_lord
            maha_elapsed = age_years
        else:
            rem_years = age_years - first_duration
            idx = (lord_idx + 1) % 9
            while rem_years >= DASHA_YEARS[DASHA_LORDS[idx]]:
                rem_years -= DASHA_YEARS[DASHA_LORDS[idx]]
                idx = (idx + 1) % 9
            curr_maha = DASHA_LORDS[idx]
            maha_elapsed = rem_years

        antar_idx = DASHA_LORDS.index(curr_maha)
        curr_antar = curr_maha
        acc = 0.0
        for i in range(9):
            a_lord = DASHA_LORDS[(antar_idx + i) % 9]
            sub_duration = (DASHA_YEARS[curr_maha] * DASHA_YEARS[a_lord]) / 120.0
            if acc + sub_duration > maha_elapsed:
                curr_antar = a_lord
                break
            acc += sub_duration

        return f"{curr_maha}-{curr_antar}"
    except Exception as exc:
        logger.warning("Local Vimshottari calculation failed: %s", exc)
        return "Rahu-Moon"


def get_ordinal_house(house_num: int) -> str:
    """Format house number as '1st house', '2nd house', '3rd house', etc."""
    n = ((house_num - 1) % 12) + 1
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix} house"


# --- Routes ---

@app.get("/")
def read_root():
    if os.path.exists(FRONTEND_PATH):
        return FileResponse(FRONTEND_PATH)
    elif os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "online", "message": "Astrology Chatbot Backend is running."}


async def _handle_chat_response(request: ChatRequest):
    t_start = time.perf_counter()

    current_birth_signature = (
        request.birth_year,
        request.birth_month,
        request.birth_day,
        round(float(request.birth_hour), 4),
        round(float(request.latitude), 4),
        round(float(request.longitude), 4),
        round(float(request.timezone), 2),
        (request.name or "").strip().lower(),
    )

    if request.user_id not in USER_SESSIONS:
        USER_SESSIONS[request.user_id] = {
            "turn_count": 1,
            "history": [],
            "birth_signature": current_birth_signature,
            "profile": {
                "name": request.name,
                "age": request.age,
                "gender": request.gender,
                "life_stage": request.life_stage,
                "relationship_status": request.relationship_status,
                "topics_asked": [],
                "preferred_language": "hinglish" if is_hinglish_query(request.query) else "english"
            }
        }
    else:
        session_data = USER_SESSIONS[request.user_id]
        old_sig = session_data.get("birth_signature")
        if old_sig is not None and old_sig != current_birth_signature:
            # User edited their profile details to another user — reset session context completely
            session_data["turn_count"] = 1
            session_data["history"] = []
            session_data["has_greeted"] = False
            session_data["birth_signature"] = current_birth_signature
            session_data.pop("chart_profile", None)
            session_data["profile"] = {
                "name": request.name,
                "age": request.age,
                "gender": request.gender,
                "life_stage": request.life_stage,
                "relationship_status": request.relationship_status,
                "topics_asked": [],
                "preferred_language": "hinglish" if is_hinglish_query(request.query) else "english"
            }
        else:
            session_data["turn_count"] += 1
            if "history" not in session_data:
                session_data["history"] = []
            profile = session_data.setdefault("profile", {"topics_asked": []})
            if request.name and request.name.lower() not in ["guest", "there"]:
                profile["name"] = request.name
            if request.age:
                profile["age"] = request.age
            if request.gender:
                profile["gender"] = request.gender
            if request.life_stage:
                profile["life_stage"] = request.life_stage
            if request.relationship_status:
                profile["relationship_status"] = request.relationship_status
            profile["preferred_language"] = "hinglish" if is_hinglish_query(request.query) else "english"

    session_data = USER_SESSIONS[request.user_id]

    birth_seconds = round((request.birth_hour % 24) * 60 * 60)
    birth_hour, remaining_seconds = divmod(birth_seconds, 60 * 60)
    birth_minute, birth_second = divmod(remaining_seconds, 60)
    birth_details = BirthDetailsRequest(
        year=request.birth_year,
        month=request.birth_month,
        day=request.birth_day,
        hour=birth_hour,
        minute=birth_minute,
        second=birth_second,
        latitude=request.latitude,
        longitude=request.longitude,
        timezone=request.timezone,
    )

    # Fetch API or local Swiss Ephemeris sidereal signs without blocking the event loop.
    try:
        astro_features, _api_chart_data = await run_in_threadpool(
            get_chart_from_api,
            birth_details,
        )
    except Exception as e:
        logger.warning("Error obtaining chart features: %s; using local defaults", e)
        astro_features = [2, 8, 2, 10, 8, 0, 7, 0, 3, 9]
        _api_chart_data = {}

    asc_sign = ZODIAC_SIGNS[int(astro_features[0]) % 12] if len(astro_features) > 0 else "Aries"
    sun_sign = ZODIAC_SIGNS[int(astro_features[1]) % 12] if len(astro_features) > 1 else "Taurus"
    moon_sign = ZODIAC_SIGNS[int(astro_features[2]) % 12] if len(astro_features) > 2 else "Gemini"

    # Calculate Jupiter house relative to Lagna
    asc_idx = int(astro_features[0]) % 12 if len(astro_features) > 0 else 0
    jup_idx = int(astro_features[5]) % 12 if len(astro_features) > 5 else 0
    jupiter_house_str = get_ordinal_house(((jup_idx - asc_idx) % 12) + 1)

    # Calculate or fetch Dasha details for this user's birth details
    current_dasha = ""
    dasha_data = {}
    try:
        dasha_data = await run_in_threadpool(fetch_dasha_details_external, birth_details)
        current_dasha = extract_current_dasha(dasha_data)
    except Exception as e:
        logger.warning("External dasha API lookup error: %s", e)

    if not current_dasha:
        try:
            current_dasha = compute_vimshottari_dasha_local(
                request.birth_year, request.birth_month, request.birth_day,
                request.birth_hour, request.timezone
            )
        except Exception as e:
            logger.warning("Local dasha computation error: %s", e)
            current_dasha = "Rahu-Moon"

    # Always synchronize active profile with the active user's calculated chart
    active_profile = UserChartProfile(
        name=request.name or "Friend",
        lagna=asc_sign,
        moon_sign=moon_sign,
        sun_sign=sun_sign,
        jupiter_house=jupiter_house_str,
        current_dasha=current_dasha,
    )
    session_data["chart_profile"] = active_profile
    session_data["birth_signature"] = current_birth_signature

    # Step 1: Ground-Truth Injection — pre-calculate user's explicit birth facts
    user_natal_context = {
        "ascendant": asc_sign,
        "moon_sign": moon_sign,
        "active_dasha": current_dasha or "Rahu-Mars",
        "house_rulerships": calculate_house_rulerships(asc_sign),
    }
    session_data["user_natal_context"] = user_natal_context

    query_text = request.user_query or request.query or ""

    # Single Greeting Policy: Never greet the user more than 1 time per session
    is_greeting_query = query_text.lower().strip() in {"hi", "hello", "hey", "namaste", "greetings"}
    already_greeted = bool(
        session_data.get("has_greeted")
        or request.already_greeted
        or session_data["turn_count"] > 1
    )

    # 1. Dynamic local Q&A response with dynamic parameters (0 LLM Token Cost)
    local_answer = check_qa_database(
        query_text,
        active_profile,
        already_greeted=already_greeted,
        user_natal_context=user_natal_context,
    )
    if local_answer:
        if already_greeted and not is_greeting_query:
            local_answer = strip_greeting(local_answer)
        session_data["has_greeted"] = True
        session_data["history"].append({"role": "user", "content": query_text})
        session_data["history"].append({"role": "assistant", "content": local_answer})
        follow_ups = generate_follow_up_questions(query_text, local_answer)
        return JSONResponse(content={
            "answer": label_answer_source(local_answer, "Backend"),
            "related_questions": follow_ups,
        })

    faq_template = match_faq_only(query_text)
    if faq_template:
        answer = format_personalized_answer(
            faq_template,
            asc_sign,
            sun_sign,
            moon_sign,
            name=request.name or "",
            hinglish=is_hinglish_query(query_text),
            query=query_text,
            astro_features=astro_features,
            current_dasha=current_dasha,
            jupiter_house=jupiter_house_str,
        )
        if already_greeted and not is_greeting_query:
            answer = strip_greeting(answer)
        session_data["has_greeted"] = True
        USER_SESSIONS[request.user_id]["history"].append({"role": "user", "content": query_text})
        USER_SESSIONS[request.user_id]["history"].append({"role": "assistant", "content": answer})
        follow_ups = generate_follow_up_questions(query_text, answer)
        return JSONResponse(content={
            "answer": label_answer_source(answer, "Backend"),
            "related_questions": follow_ups,
        })

    # Every non-FAQ query is interpreted by the LLM; local rules only provide domain context.
    raw_indirect_intent = resolve_indirect_intent(request.query)
    internal_query = request.query

    # Normalize Hinglish before routing so keyword classifiers work.
    normalized_query = normalize_hinglish(internal_query)

    ml_result = router.classify_and_predict(normalized_query, astro_features)
    predicted_domain = ml_result.get('predicted_domain', 'General')
    query_domains = infer_query_domains(
        internal_query,
        predicted_domain,
        include_semantic_fallback=False,
    )
    indirect_intent = raw_indirect_intent or resolve_indirect_intent(internal_query)

    # Track the topic in session profile for personalization
    session_profile = USER_SESSIONS[request.user_id].get("profile", {})
    topics = session_profile.setdefault("topics_asked", [])
    resolved_topics = list(query_domains)
    if indirect_intent and indirect_intent not in resolved_topics:
        resolved_topics.append(indirect_intent)
    for resolved_topic in resolved_topics:
        if resolved_topic not in ("general", "", None) and resolved_topic not in topics:
            topics.append(resolved_topic)

    # [OPTIMIZATION 3: Strict 2-Turn History Truncation & Fluff Removal (<= 75 Tokens)]
    # Restrict history strictly to the last 2 messages (1 user turn + 1 assistant turn)
    full_history = USER_SESSIONS[request.user_id].get("history", [])
    sliding_history = filter_chat_history(full_history, max_messages=2)

    retrieved_rule = UNKNOWN_DOMAIN_DIRECT_LLM_PARSE
    # [OPTIMIZATION 1 & 2: Ultra-Compressed System Prompt (<= 15 Tokens) + Compact Profile (<= 12 Tokens)]
    system_instruction = build_gemini_prompt(
        request, asc_sign, sun_sign, moon_sign,
        astro_features, predicted_domain, _api_chart_data, dasha_data,
        current_dasha=current_dasha,
        chat_history=sliding_history,
        session_profile=session_profile,
        query_domains=query_domains,
        retrieved_rule=retrieved_rule,
        normalized_interpretation=normalized_query,
        user_natal_context=user_natal_context,
    )


    async def async_stream_generator():
        attempt_chunks = []
        active_model = "LLM"
        try:
            # Stream via cascading router passing user_natal_context and 2-turn filtered history
            async for chunk, model_name in stream_cascading_router(
                request.query,
                system_instruction,
                history=sliding_history,
                session_id=request.user_id,
                user_natal_context=user_natal_context,
            ):
                active_model = model_name
                attempt_chunks.append(chunk)
                yield chunk

            raw_streamed = "".join(attempt_chunks).strip()
            # Ensure stream displayed to user ends with a complete sentence before the LLM tag
            if raw_streamed and raw_streamed[-1] not in (".", "!", "?"):
                yield "."

            # [SPECIFICATION 3 & 4: Output Sanitization & Sentence Truncation] Strip internal reasoning / preambles and trim to last complete sentence
            full_text = trim_to_last_sentence(clean_llm_output(raw_streamed))
            if full_text:
                # [OPTIMIZATION 3: Fluff Removal] Never pollute conversation history with pure fluff
                if not is_conversational_fluff(request.query):
                    USER_SESSIONS[request.user_id]["history"].append({"role": "user", "content": request.query})
                    USER_SESSIONS[request.user_id]["history"].append({"role": "assistant", "content": full_text})
                    # Restrict stored history to last 4 messages to save memory
                    USER_SESSIONS[request.user_id]["history"] = USER_SESSIONS[request.user_id]["history"][-4:]

                # Server metadata suffix stripped per requirement


                target_domain = query_domains[0] if query_domains else predicted_domain
                follow_ups = generate_follow_up_questions(request.query, full_text, domain=target_domain)
                yield f"\n[FOLLOW_UPS]: {json.dumps(follow_ups)}"
        except asyncio.CancelledError:
            logger.info("Chat stream cancelled for user %s", request.user_id)
            raise
        except Exception as e:
            logger.error("Streaming Error: %s", str(e), exc_info=True)
            yield gemini_error_message(e, request.query)

    return StreamingResponse(async_stream_generator(), media_type="text/plain; charset=utf-8")


async def _handle_chat(request: ChatRequest):
    try:
        return await _handle_chat_response(request)
    except asyncio.CancelledError:
        logger.info("Chat request cancelled during startup")
        raise
    except Exception as e:
        logger.error("Chat request error: %s", str(e), exc_info=True)
        return JSONResponse(
            status_code=200,
            content=backend_error_payload(request.query),
        )


@app.post("/chat")
@limiter.limit(os.getenv("CHAT_RATE_LIMIT", "300/minute"))
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    return await _handle_chat(chat_request)


class ApiChatRequest(BaseModel):
    prompt: str
    user_profile: Optional[Dict[str, Any]] = None
    already_greeted: Optional[bool] = None
    chat_history: Optional[List[Dict[str, Any]]] = None


# Lightweight per-user session store for the /api/chat endpoint.
# Persists history (last 4 messages) and session_profile across calls within the same process.
API_USER_SESSIONS: Dict[str, Dict[str, Any]] = {}


@app.post("/api/chat")
async def api_chat_endpoint(chat_req: ApiChatRequest):
    """
    Ultra-token-optimized chat endpoint (target: <= 210 tokens/request, >50% token cut):
    1. Local QA Intercept: Checks incoming query against qa_database.json (0 LLM tokens).
    2. Ultra-Compressed LLM Cascading Router:
       - Ultra-Compressed System Prompt (<= 15 tokens)
       - Compact 1-Line Key-Value Profile (<= 12 tokens, e.g., 'P: Aries|Taurus|Rahu-Jup')
       - Strict 2-Turn History Truncation & Fluff Removal (<= 75 tokens)
       - Output Hard Cap (max_tokens=120, ~95 output tokens)
       - Token Usage Logging & Reporting (Prompt, Completion, Total)
    """
    prompt = chat_req.prompt.strip()
    profile = chat_req.user_profile or {}

    # Step 1: Ground-Truth Injection — pre-calculate user's explicit birth facts
    ascendant = profile.get("ascendant") or profile.get("lagna") or "Libra"
    moon_sign = profile.get("rashi") or profile.get("moon_sign") or profile.get("moon") or "Gemini"
    active_dasha = profile.get("dasha") or profile.get("active_dasha") or profile.get("current_dasha") or "Rahu-Mars"

    user_natal_context = {
        "ascendant": ascendant,
        "moon_sign": moon_sign,
        "active_dasha": str(active_dasha),
        "house_rulerships": calculate_house_rulerships(ascendant),
    }

    # Resolve user identity for session tracking
    user_id: str = str(profile.get("user_id") or profile.get("id") or "api_default_user")

    # Build or update the per-user API session
    if user_id not in API_USER_SESSIONS:
        API_USER_SESSIONS[user_id] = {
            "history": [],
            "user_natal_context": user_natal_context,
            "session_profile": {
                "name": profile.get("name"),
                "age": profile.get("age"),
                "gender": profile.get("gender"),
                "topics_asked": [],
                "preferred_language": "hinglish" if is_hinglish_query(prompt) else "english",
            },
        }
    else:
        API_USER_SESSIONS[user_id]["user_natal_context"] = user_natal_context
        sp = API_USER_SESSIONS[user_id]["session_profile"]
        # Update profile fields if new info provided
        if profile.get("name"):
            sp["name"] = profile["name"]
        if profile.get("age"):
            sp["age"] = profile["age"]
        if profile.get("gender"):
            sp["gender"] = profile["gender"]
        sp["preferred_language"] = "hinglish" if is_hinglish_query(prompt) else "english"

    api_session = API_USER_SESSIONS[user_id]
    session_profile_data = api_session["session_profile"]

    already_greeted = bool(
        chat_req.already_greeted
        or profile.get("already_greeted")
        or profile.get("has_greeted")
        or len(api_session["history"]) > 0
    )

    # 1. Local QA Intercept (0 Tokens) - dynamically populated with user_natal_context
    local_answer = None
    if not should_bypass_local_qa(prompt):
        local_answer = check_qa_database(
            prompt,
            already_greeted=already_greeted,
            user_natal_context=user_natal_context,
        )
        if not local_answer:
            local_answer = intercept_qa(
                prompt,
                user_profile=profile,
                threshold=82.0,
                already_greeted=already_greeted,
            )
    if local_answer:
        is_greeting_query = prompt.lower().strip() in {"hi", "hello", "hey", "namaste", "greetings"}
        if already_greeted and not is_greeting_query:
            local_answer = strip_greeting(local_answer)

        async def stream_local():
            yield local_answer
        return StreamingResponse(stream_local(), media_type="text/plain; charset=utf-8")

    # 2. Merge incoming chat_history with accumulated session history
    combined_history = list(api_session["history"])
    if chat_req.chat_history:
        for msg in chat_req.chat_history:
            if msg not in combined_history:
                combined_history.append(msg)

    # [OPTIMIZATION 3: Strict 2-Turn History Truncation & Fluff Removal]
    sliding_history = filter_chat_history(combined_history, max_messages=2)

    # Build personalized system prompt with session context and ground-truth natal facts
    merged_profile = dict(profile)
    merged_profile.update({
        "ascendant": ascendant,
        "lagna": ascendant,
        "moon_sign": moon_sign,
        "rashi": moon_sign,
        "current_dasha": active_dasha,
        "dasha": active_dasha,
    })
    personalized_system_prompt = build_system_prompt(
        user_profile=merged_profile,
        session_profile=session_profile_data,
        user_natal_context=user_natal_context,
    )

    # Track topics for this query
    detected_domains = infer_query_domains(prompt, include_semantic_fallback=False)
    topics_list = session_profile_data.setdefault("topics_asked", [])
    for topic in detected_domains:
        if topic not in ("general", "", None) and topic not in topics_list:
            topics_list.append(topic)

    async def api_stream_generator():
        attempt_chunks: List[str] = []
        active_model = "LLM"
        async for chunk, model_name in stream_cascading_router(
            prompt,
            personalized_system_prompt,
            history=sliding_history,
            session_id=user_id,
            user_natal_context=user_natal_context,
        ):
            active_model = model_name
            attempt_chunks.append(chunk)
            yield chunk

        raw_streamed = "".join(attempt_chunks).strip()
        # Ensure terminal punctuation
        if raw_streamed and raw_streamed[-1] not in (".", "!", "?"):
            yield "."

        full_text = trim_to_last_sentence(clean_llm_output(raw_streamed))
        if full_text and not is_conversational_fluff(prompt):
            # Store to session history (capped at 4 messages)
            api_session["history"].append({"role": "user", "content": prompt})
            api_session["history"].append({"role": "assistant", "content": full_text})
            api_session["history"] = api_session["history"][-4:]

        # Server metadata suffix stripped per requirement


        follow_ups = generate_follow_up_questions(prompt, full_text or raw_streamed)
        yield f"\n[FOLLOW_UPS]: {json.dumps(follow_ups)}"

    return StreamingResponse(api_stream_generator(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)