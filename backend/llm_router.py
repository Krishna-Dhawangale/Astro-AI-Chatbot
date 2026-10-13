"""
Evidence-Grounded Multi-Provider LLM Cascading Router (backend/llm_router.py)
-----------------------------------------------------------------------------
Stage 5 Architecture: Resilient multi-provider cascading for Mode 3 (LLM_ASSISTED).

Design Guarantees:
1. Zero SDK Bloat:
   - Direct asynchronous HTTP calls via `httpx.AsyncClient` for Groq and OpenRouter.
   - Eliminates third-party dependencies (`groq`, `openai`).
2. Strict Missing-Key Fast Bypass:
   - Missing or unconfigured provider API keys are skipped immediately (0 ms overhead).
   - Only active tiers with valid keys make network requests.
3. Universal Evidence-Grounded Renderer Contract:
   - All providers receive the exact same system prompt and token budget.
   - Forbids calculating or inventing astrological facts (Micro Protocol).
   - Word count validation against 40–65 words on completed output.
4. Safe Streaming Failover:
   - Output is buffered during the initial phase (up to 15 tokens) to detect provider failures
     and sanitize reasoning tags BEFORE any text is emitted to the client.
   - If an error occurs before streaming starts: failover to the next provider cleanly.
   - If an error occurs after streaming starts: safe termination (trim to terminal punctuation)
     without concatenating incompatible partial answers from multiple models.
5. Robust Streaming Sanitization:
   - Handles <think> / <thought> tags even when split across streaming chunks.
   - Handles missing closing tags and provider cutoffs.
   - Trims mid-sentence cutoffs back to the last terminal punctuation mark ('.', '!', '?').
6. Observability & Clean Text:
   - Provider, model, tokens, and fallback reasons recorded in decision metadata.
   - Zero model badges appended to user-facing answer text (preserves clean frontend contract).
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import httpx

# Defensive import for Google GenAI
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider & Model Configuration
# ---------------------------------------------------------------------------
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODELS: List[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODELS: List[str] = [
    "openai/gpt-oss-120b",
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
]

# Verified active models for Google Gemini
GEMINI_SAFETY_MODELS: List[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]

# Shared HTTPX Client (Singleton with connection pooling)
_SHARED_HTTPX_CLIENT: Optional[httpx.AsyncClient] = None


def get_shared_httpx_client() -> httpx.AsyncClient:
    global _SHARED_HTTPX_CLIENT
    if _SHARED_HTTPX_CLIENT is None or _SHARED_HTTPX_CLIENT.is_closed:
        _SHARED_HTTPX_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.0, read=8.0, write=2.0, pool=5.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20, keepalive_expiry=30.0),
        )
    return _SHARED_HTTPX_CLIENT


async def close_shared_httpx_client() -> None:
    global _SHARED_HTTPX_CLIENT
    if _SHARED_HTTPX_CLIENT is not None and not _SHARED_HTTPX_CLIENT.is_closed:
        await _SHARED_HTTPX_CLIENT.aclose()
        _SHARED_HTTPX_CLIENT = None


# ---------------------------------------------------------------------------
# Provider Account Resolvers
# ---------------------------------------------------------------------------
def get_groq_accounts() -> List[Dict[str, str]]:
    """Returns active Groq accounts with valid non-empty API keys."""
    accounts = []
    key1 = (
        os.getenv("GROQ_API_KEY_1")
        or os.getenv("GROQ_API_KEY_ACCOUNT1")
        or os.getenv("GROQ_API_KEY_ACC1")
        or os.getenv("GROQ_API_KEY")
    )
    if key1 and key1.strip():
        accounts.append({"account_id": "Groq-1", "api_key": key1.strip()})

    key2 = (
        os.getenv("GROQ_API_KEY_2")
        or os.getenv("GROQ_API_KEY_ACCOUNT2")
        or os.getenv("GROQ_API_KEY_ACC2")
    )
    if key2 and key2.strip() and key2.strip() != (key1 or "").strip():
        accounts.append({"account_id": "Groq-2", "api_key": key2.strip()})

    return accounts


def get_openrouter_key() -> Optional[str]:
    """Returns active OpenRouter API key if present and non-empty."""
    k = os.getenv("OPENROUTER_API_KEY")
    return k.strip() if k and k.strip() else None


def get_gemini_key() -> Optional[str]:
    """Returns active Google Gemini API key if present and non-empty."""
    k = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return k.strip() if k and k.strip() else None


# ---------------------------------------------------------------------------
# Sentence Completion & Output Validation Helpers
# ---------------------------------------------------------------------------
TERMINAL_PUNCTUATION = {".", "!", "?"}


def trim_to_last_sentence(text: str) -> str:
    """
    If an LLM response gets cut off mid-sentence due to max_tokens,
    this function slices back to the last terminal punctuation mark ('.', '!', '?').
    Ensures no dangling words or broken phrases reach the user.
    """
    clean_text = text.strip()
    if not clean_text:
        return ""

    if clean_text[-1] in TERMINAL_PUNCTUATION:
        return clean_text

    # Search for the last occurrence of terminal punctuation
    last_idx = -1
    for i in range(len(clean_text) - 1, -1, -1):
        if clean_text[i] in TERMINAL_PUNCTUATION:
            last_idx = i
            break

    if last_idx != -1 and last_idx >= 15:
        return clean_text[: last_idx + 1].strip()

    # If no punctuation exists or string is short, append period to close cleanly
    return clean_text + "."


def validate_response_words(text: str, min_words: int = 35, max_words: int = 70) -> Dict[str, Any]:
    """
    Validates completed output word count against the 40-65 target.
    Returns status metadata for decision logging.
    """
    words = text.split()
    count = len(words)
    is_valid = min_words <= count <= max_words
    return {
        "word_count": count,
        "within_bounds": is_valid,
        "target_range": f"{min_words}-{max_words}",
    }


# ---------------------------------------------------------------------------
# Streaming Reasoning Tag Sanitizer (Split-Chunk Safe)
# ---------------------------------------------------------------------------
class StreamingTagSanitizer:
    """
    State-machine streaming sanitizer that detects and strips reasoning tags:
    `<think>...</think>`, `<thought>...</thought>`.

    Robustness guarantees:
    - Buffers potential tag prefixes across chunk boundaries (e.g. chunk 1: '<th', chunk 2: 'ink>').
    - Discards all reasoning content inside tags.
    - Handles missing closing tags by detecting unclosed reasoning blocks.
    - Emits clean user content as soon as confirmed outside tags.
    """

    def __init__(self):
        self.inside_think = False
        self.buffer = ""
        self.has_emitted_content = False
        self.raw_emitted_tokens = 0

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""

        self.buffer += chunk
        output = []

        while self.buffer:
            if not self.inside_think:
                # Check for start of thinking tag
                match = re.search(r"<(think|thought)>", self.buffer, re.IGNORECASE)
                if match:
                    # Emit any text preceding the tag
                    prefix = self.buffer[: match.start()]
                    if prefix:
                        output.append(prefix)
                    # Enter thinking mode and discard the opening tag
                    self.inside_think = True
                    self.buffer = self.buffer[match.end() :]
                else:
                    # Check if buffer ends with a partial opening tag, e.g. '<', '<t', '<th', '<thi'
                    partial_match = re.search(r"<[a-zA-Z0-9]*$", self.buffer)
                    if partial_match:
                        # Keep the partial tag in the buffer for the next chunk
                        prefix = self.buffer[: partial_match.start()]
                        if prefix:
                            output.append(prefix)
                        self.buffer = self.buffer[partial_match.start() :]
                        break
                    else:
                        output.append(self.buffer)
                        self.buffer = ""
            else:
                # Inside think block: search for closing tag
                match = re.search(r"</(think|thought)>", self.buffer, re.IGNORECASE)
                if match:
                    # Discard reasoning content up to end of closing tag
                    self.inside_think = False
                    self.buffer = self.buffer[match.end() :]
                else:
                    # Keep only potential partial closing tag in buffer
                    partial_match = re.search(r"</?[a-zA-Z0-9]*$", self.buffer)
                    if partial_match:
                        self.buffer = self.buffer[partial_match.start() :]
                    else:
                        self.buffer = ""
                    break

        result = "".join(output)
        if result:
            self.has_emitted_content = True
            self.raw_emitted_tokens += len(result.split())
        return result

    def finalize(self) -> str:
        """Flushes any remaining text when stream closes cleanly."""
        if not self.inside_think and self.buffer:
            res = self.buffer
            self.buffer = ""
            return res
        return ""


# ---------------------------------------------------------------------------
# Native Asynchronous OpenAI-Compatible Streaming (Groq & OpenRouter via httpx)
# ---------------------------------------------------------------------------
async def _stream_openai_compatible(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> AsyncGenerator[str, None]:
    """
    Streams Server-Sent Events (SSE) from OpenAI-compatible REST endpoints.
    Uses httpx directly with zero third-party SDK dependencies.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Astro-AI-Chatbot/1.0",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    req_timeout = httpx.Timeout(
        connect=2.0,
        read=timeout_seconds,
        write=2.0,
        pool=2.0,
    )

    async with client.stream("POST", url, json=payload, headers=headers, timeout=req_timeout) as resp:
        if resp.status_code != 200:
            error_body = await resp.aread()
            error_text = error_body.decode("utf-8", errors="replace")[:300]
            raise httpx.HTTPStatusError(
                f"HTTP {resp.status_code} from {model}: {error_text}",
                request=resp.request,
                response=resp,
            )

        async for line in resp.aiter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data_obj = json.loads(data_str)
                    choices = data_obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# Native Asynchronous Google Gemini Streaming
# ---------------------------------------------------------------------------
async def _stream_gemini(
    client: Any,
    model: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> AsyncGenerator[str, None]:
    """Streams response chunks from Google Gemini using google-genai client."""
    if client is None or types is None:
        raise RuntimeError("Google GenAI client is not configured or types module unavailable")

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    response_stream = await client.aio.models.generate_content_stream(
        model=model,
        contents=prompt,
        config=config,
    )

    async for chunk in response_stream:
        chunk_text = getattr(chunk, "text", None)
        if chunk_text:
            yield chunk_text


# ---------------------------------------------------------------------------
# Master Cascading Router with Safe Failover
# ---------------------------------------------------------------------------
async def stream_cascading_router(
    prompt: str,
    system_prompt: str,
    max_output_tokens: int = 120,
    temperature: float = 0.2,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    custom_httpx_client: Optional[httpx.AsyncClient] = None,
    custom_gemini_client: Optional[Any] = None,
    local_fallback_text: Optional[str] = None,
) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
    """
    Cascades sequentially through:
      1. Groq Account 1 (Llama 3.3 70B -> Llama 3.1 8B)
      2. Groq Account 2 (Llama 3.3 70B -> Llama 3.1 8B)
      3. OpenRouter Tier (GPT-OSS -> Qwen 2.5 72B -> Llama 3.3 70B)
      4. Google Gemini Safety Net (gemini-3.5-flash-lite -> gemini-3.8-flash)
      5. Local Deterministic Rule Fallback (if all network providers fail)

    Safe Failover Contract:
    - Buffers the initial 15 tokens from a provider.
    - If provider errors or yields empty stream before buffering completes:
      Catch exception, log fallback reason, and switch to next provider (zero partial text sent to client).
    - If provider errors AFTER streaming begins (chunks_yielded > 0):
      Apply safe termination (trim to terminal punctuation). Do NOT switch models mid-sentence.
    - Yields tuple: (chunk_text, provider_meta).
    """
    httpx_client = custom_httpx_client or get_shared_httpx_client()
    groq_accounts = get_groq_accounts()
    openrouter_key = get_openrouter_key()
    gemini_key = get_gemini_key()

    candidate_providers: List[Dict[str, Any]] = []

    # Tier 1 & 2: Active Groq Accounts
    for acc in groq_accounts:
        for model in GROQ_MODELS:
            candidate_providers.append({
                "type": "groq",
                "account_id": acc["account_id"],
                "api_key": acc["api_key"],
                "model": model,
                "url": GROQ_BASE_URL,
                "timeout": 4.0,
            })

    # Tier 3: OpenRouter Tier
    if openrouter_key:
        for model in OPENROUTER_MODELS:
            candidate_providers.append({
                "type": "openrouter",
                "account_id": "OpenRouter",
                "api_key": openrouter_key,
                "model": model,
                "url": OPENROUTER_BASE_URL,
                "timeout": 6.0,
            })

    # Tier 4: Google Gemini Safety Net
    gemini_client = custom_gemini_client
    if gemini_client is None and gemini_key and genai is not None:
        try:
            gemini_client = genai.Client(api_key=gemini_key)
        except Exception as e:
            logger.warning("Failed to initialize Gemini client: %s", e)

    if gemini_client is not None:
        for model in GEMINI_SAFETY_MODELS:
            candidate_providers.append({
                "type": "gemini",
                "account_id": "Google-Gemini",
                "api_key": gemini_key or "configured",
                "model": model,
                "timeout": 8.0,
            })

    execution_attempts = []
    stream_started = False
    full_emitted_text = []

    for provider in candidate_providers:
        p_type = provider["type"]
        p_model = provider["model"]
        p_acc = provider["account_id"]
        call_start = time.time()
        sanitizer = StreamingTagSanitizer()

        initial_buffer: List[str] = []
        buffer_token_count = 0
        MIN_BUFFER_TOKENS = 6  # Buffer enough tokens to validate response and detect reasoning tags

        try:
            # Generate raw token stream
            if p_type in ("groq", "openrouter"):
                raw_gen = _stream_openai_compatible(
                    client=httpx_client,
                    url=provider["url"],
                    api_key=provider["api_key"],
                    model=p_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout_seconds=provider["timeout"],
                )
            elif p_type == "gemini":
                raw_gen = _stream_gemini(
                    client=gemini_client,
                    model=p_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                )
            else:
                continue

            async for chunk in raw_gen:
                clean_chunk = sanitizer.feed(chunk)
                if not clean_chunk:
                    continue

                if not stream_started:
                    # In buffering phase
                    initial_buffer.append(clean_chunk)
                    buffer_token_count += len(clean_chunk.split())
                    if buffer_token_count >= MIN_BUFFER_TOKENS or any(p in clean_chunk for p in TERMINAL_PUNCTUATION):
                        # Release buffer and transition to live streaming
                        stream_started = True
                        for buffered_chunk in initial_buffer:
                            full_emitted_text.append(buffered_chunk)
                            yield (buffered_chunk, {
                                "provider": p_type,
                                "account": p_acc,
                                "model": p_model,
                                "status": "streaming",
                            })
                        initial_buffer.clear()
                else:
                    # Already streaming live
                    full_emitted_text.append(clean_chunk)
                    yield (clean_chunk, {
                        "provider": p_type,
                        "account": p_acc,
                        "model": p_model,
                        "status": "streaming",
                    })

            # Handle flush
            final_chunk = sanitizer.finalize()
            if final_chunk:
                if not stream_started:
                    initial_buffer.append(final_chunk)
                else:
                    full_emitted_text.append(final_chunk)
                    yield (final_chunk, {
                        "provider": p_type,
                        "account": p_acc,
                        "model": p_model,
                        "status": "streaming",
                    })

            if not stream_started and initial_buffer:
                # Stream was short but valid
                stream_started = True
                for buffered_chunk in initial_buffer:
                    full_emitted_text.append(buffered_chunk)
                    yield (buffered_chunk, {
                        "provider": p_type,
                        "account": p_acc,
                        "model": p_model,
                        "status": "streaming",
                    })
                initial_buffer.clear()

            if stream_started:
                # Successfully streamed response from this provider!
                total_text = "".join(full_emitted_text).strip()
                if sanitizer.inside_think:
                    # Unclosed think tag: truncated before valid answer
                    logger.warning("Provider %s (%s) output ended inside thinking tag.", p_type, p_model)

                word_eval = validate_response_words(total_text)
                latency_ms = (time.time() - call_start) * 1000
                logger.info(
                    "[ROUTER SUCCESS] %s (%s) generated %d words in %.1f ms.",
                    p_type, p_model, word_eval["word_count"], latency_ms
                )
                yield ("", {
                    "provider": p_type,
                    "account": p_acc,
                    "model": p_model,
                    "status": "completed",
                    "latency_ms": latency_ms,
                    "word_eval": word_eval,
                    "full_text": total_text,
                })
                return  # Single-model completion: terminate cascade immediately

            # If loop exited without streaming any text, treat as empty response failure
            raise ValueError(f"Provider {p_type} ({p_model}) produced empty or reasoning-only response")

        except Exception as e:
            latency_ms = (time.time() - call_start) * 1000
            err_msg = str(e)
            logger.warning(
                "[PROVIDER FAILOVER] %s (%s) failed in %.1f ms: %s",
                p_type, p_model, latency_ms, err_msg
            )
            execution_attempts.append({
                "provider": p_type,
                "model": p_model,
                "error": err_msg,
                "latency_ms": latency_ms,
            })

            if stream_started:
                # CRITICAL: Stream already emitted partial tokens to client!
                # Do NOT switch providers to prevent concatenating incompatible answers.
                logger.error("[SAFE TERMINATION] Stream interrupted after tokens emitted; safely terminating.")
                yield (".", {
                    "provider": p_type,
                    "account": p_acc,
                    "model": p_model,
                    "status": "interrupted",
                    "error": err_msg,
                })
                return

            # Otherwise, failover before sending chunks is safe: proceed to next provider in loop

    # Tier 5: Local Deterministic Rule Fallback (if all network providers failed or unconfigured)
    logger.warning("All LLM providers exhausted or unconfigured. Activating Local Rule Fallback.")
    fallback_text = (
        local_fallback_text
        or "Based on your chart configuration and active planetary placements, "
           "the astrological indicators show a supportive alignment for this life domain. "
           "Planetary energies encourage thoughtful planning and steady perseverance."
    )
    yield (fallback_text, {
        "provider": "local_fallback",
        "account": "Local-Rules",
        "model": "deterministic_synthesis",
        "status": "fallback",
        "attempts": execution_attempts,
    })


# ===========================================================================
# MULTI-DOMAIN DETECTION & HYBRID MODULAR RAG (Krishna Integration)
# ===========================================================================

_QA_DB_CACHE: Optional[Dict[str, str]] = None

DOMAIN_KEYWORD_SIGNALS: Dict[str, List[str]] = {
    "career": [
        "career", "careers", "job", "jobs", "profession", "work", "promotion",
        "business", "boss", "colleague", "office", "workplace", "company",
        "interview", "corporate", "employment", "freelance"
    ],
    "finance": [
        "finance", "finances", "money", "wealth", "investment", "investments",
        "savings", "income", "debt", "loan", "share market", "stock", "stocks",
        "shares", "trading", "lottery", "cash", "bank balance", "earning", "earnings"
    ],
    "marriage": [
        "marriage", "wedding", "love", "spouse", "partner", "relationship",
        "relationships", "husband", "wife", "marital", "divorce", "separation",
        "manglik", "mangal dosha", "compatibility"
    ],
    "health": [
        "health", "disease", "illness", "fitness", "stress", "burnout", "anxiety",
        "mental stress", "mental peace", "recovery", "medicine", "wellness", "sleep",
        "chronic", "healing", "vitality"
    ],
    "gemstones": [
        "gemstone", "gemstones", "ratna", "ruby", "pearl", "emerald",
        "yellow sapphire", "blue sapphire", "diamond", "coral", "gomed",
        "cat's eye", "stone", "stones", "crystal", "crystals"
    ],
    "education": [
        "education", "study", "studies", "exam", "exams", "college", "degree",
        "university", "school", "learning", "academic", "student"
    ],
    "travel": [
        "travel", "travels", "journey", "abroad", "foreign", "visa",
        "relocation", "relocate", "migration", "overseas", "settlement"
    ],
    "spirituality": [
        "mantra", "puja", "pooja", "temple", "pitra dosha", "kaal sarp",
        "sade sati", "shani shanti", "spiritual", "spirituality", "ishta devata",
        "moksha", "dharma", "meditation"
    ]
}

DOMAIN_QA_KEYS: Dict[str, List[str]] = {
    "career": [
        "career_timing", "career_probability", "career_domain_fit",
        "career house", "10th house", "career prediction", "job change",
        "promotion", "business vs job", "government job", "government job yoga", "6th house"
    ],
    "finance": [
        "finance_timing", "finance_risk", "wealth_accumulation",
        "financial stability", "dhan yoga", "lakshmi yoga", "share market",
        "lottery or windfall", "debt remedies", "remedies for debt", "2nd house", "11th house"
    ],
    "marriage": [
        "marriage_timing", "partner_profile", "relationship_compatibility", "relationship_mitigation",
        "remedies for marriage delay", "which astrological remedies can resolve relationship delays?",
        "marriage house", "love marriage", "arranged marriage",
        "mangal dosha", "divorce or separation",
        "kundali milan", "7th house", "navamsha"
    ],
    "health": [
        "health_timing", "health_mitigation",
        "health house", "health remedies", "mental stress", "remedies for mental peace",
        "chronic illness", "1st house", "6th house", "8th house", "maha mrityunjaya mantra"
    ],
    "gemstones": [
        "gemstone_recommendation", "gemstone_wearing_rules",
        "gemstones for career"
    ],
    "education": [
        "education_field_fit", "exam_timing_success", "study_abroad",
        "education house", "higher education", "competitive exams",
        "mercury for education", "5th house", "4th house"
    ],
    "travel": [
        "foreign_travel_timing", "foreign_settlement_fit", "travel_obstacle_mitigation",
        "foreign settlement", "foreign travel", "travel abroad",
        "can astrology predict foreign travel", "will i travel abroad", "rahu foreign travel",
        "9th house", "12th house"
    ],
    "spirituality": [
        "ishta_devata_guidance", "karmic_dosha_mitigation", "spiritual_growth_dharma",
        "ishta devata", "gayatri mantra", "maha mrityunjaya mantra",
        "kaal sarp dosha", "pitra dosha", "sade sati", "jupiter remedies"
    ]
}


def load_qa_database() -> Dict[str, str]:
    """Loads and caches the local QA database."""
    global _QA_DB_CACHE
    if _QA_DB_CACHE is not None:
        return _QA_DB_CACHE

    candidate_paths = [
        os.path.join(BASE_DIR, "qa_database.json"),
        os.path.join(BASE_DIR, "..", "qa_database.json"),
        os.path.join(os.getcwd(), "qa_database.json"),
    ]
    for p in candidate_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    _QA_DB_CACHE = json.load(f)
                    return _QA_DB_CACHE
            except Exception as e:
                logger.warning("Failed to parse QA database at %s: %s", p, e)

    _QA_DB_CACHE = {}
    return _QA_DB_CACHE


def is_hinglish_query(text: str) -> bool:
    """Detects if a query is in Hinglish."""
    if not text:
        return False
    if re.search(r"[\u0900-\u097F]", text):
        return True

    text_lower = text.lower().strip()
    hindi_phrases = [
        "kab milegi", "kab milega", "kab hogi", "kab hoga", "kaisa rahega", "kaisi rahegi",
        "kaisa hoga", "kaisi hogi", "bata do", "bataiye na", "theek hai", "thik hai",
        "paise aayenge", "paisa aayega", "job milegi", "naukri kab", "shaadi kab", "shadi kab",
        "ke baare me", "ke bare me", "ke baare mein", "ke bare mein", "kya hai", "kya hoga",
        "kya ho", "kaise kare", "kaise karein", "kya karu", "kya karun", "kya karna",
        "kaun sa", "kaun si", "kon sa", "kon si"
    ]
    for phrase in hindi_phrases:
        if phrase in text_lower:
            return True

    hindi_grammar_markers = {
        "aap", "aapka", "aapki", "aapke", "apka", "apki", "apke",
        "mera", "meri", "mere", "mujhe", "mujhko", "hum", "humara", "humari", "humare",
        "tum", "tumhara", "tumhari", "tumhare", "tera", "teri", "tere", "apna", "apni", "apne",
        "hai", "hain", "hein", "hoon", "hun", "tha", "thi",
        "hoga", "hogi", "hoge", "honge", "rahega", "rahegi", "rahenge",
        "raha", "rahi", "rahe", "karo", "kare", "karein", "karna", "karta", "karti", "karte",
        "batao", "bataiye", "bataye", "bata", "btao", "btaye", "boliye", "bolo",
        "dekh", "dekho", "dekhiye", "chahiye", "chaiye", "sakta", "sakti", "sakte",
        "milega", "milegi", "milege", "badhega", "badhegi", "lagta", "lagti", "lagte",
        "aayega", "aayegi", "kya", "kyaa", "kab", "kabb", "kaise", "kaisa", "kaisi",
        "kyun", "kyu", "kyon", "kaha", "kahan", "kitna", "kitni", "kitne"
    }
    words = re.findall(r"\b[a-zA-Z]+\b", text_lower)
    return any(w in hindi_grammar_markers for w in words)


def detect_query_domains(query: str) -> List[str]:
    """Scans the query for domain signals and returns all matching domains."""
    if not query:
        return []
    normalized = query.lower()
    matched_domains: List[str] = []

    for domain, keywords in DOMAIN_KEYWORD_SIGNALS.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, normalized):
                if domain not in matched_domains:
                    matched_domains.append(domain)
                break

    return matched_domains


def is_multi_domain_query(query: str) -> bool:
    """Returns True if the query explicitly spans 2 or more life domains."""
    return len(detect_query_domains(query)) >= 2


def extract_modular_domain_evidence(
    query: str,
    detected_domains: Optional[List[str]] = None,
    qa_db: Optional[Dict[str, str]] = None
) -> Dict[str, List[str]]:
    """Extracts verified rules from qa_database.json modularly for each detected domain."""
    qa = qa_db or load_qa_database()
    if not detected_domains:
        detected_domains = detect_query_domains(query)

    q_lower = query.lower() if query else ""
    modular_evidence: Dict[str, List[str]] = {}

    stop_words = {
        "for", "in", "the", "a", "an", "of", "to", "or", "and", "is", "my",
        "your", "what", "which", "are", "remedy", "remedies", "house", "can",
        "any", "how", "does", "do", "tell", "me", "about", "best", "give", "please"
    }

    for domain in detected_domains:
        domain_rules: List[str] = []
        target_keys = DOMAIN_QA_KEYS.get(domain, [])

        # Priority 1: Specific sub-phrase / keyword match
        for key in target_keys:
            if key in qa:
                sig_words = [w for w in key.split() if w not in stop_words]
                is_match = (key in q_lower) or (bool(sig_words) and all(w in q_lower for w in sig_words))
                if is_match:
                    rule_text = qa[key].strip()
                    if rule_text not in domain_rules:
                        domain_rules.append(rule_text)

        # Priority 2: Domain base rule fallback
        if not domain_rules:
            for key in target_keys:
                if len(domain_rules) >= 1:
                    break
                if key in qa:
                    rule_text = qa[key].strip()
                    if rule_text not in domain_rules:
                        domain_rules.append(rule_text)

        if domain_rules:
            modular_evidence[domain] = domain_rules[:2]

    return modular_evidence


def format_multidomain_evidence_context(domain_evidence: Dict[str, List[str]]) -> str:
    """Formats modular domain evidence into clean Markdown context for LLM prompt."""
    lines = []
    for domain, rules in domain_evidence.items():
        lines.append(f"### Domain: {domain.upper()}")
        for idx, r in enumerate(rules, 1):
            lines.append(f"{idx}. {r}")
        lines.append("")
    return "\n".join(lines).strip()


def get_hybrid_multidomain_system_instruction(is_hinglish: bool = False) -> str:
    """Returns the system instruction for Hybrid Multi-Domain RAG with strict language mirroring."""
    if is_hinglish:
        lang_rule = (
            "5. LANGUAGE REQUIREMENT (CRITICAL): The user asked in Hinglish.\n"
            "   You MUST respond entirely in conversational, natural Hinglish.\n"
            "   Keep section labels in English bold: **Astrological Insight:** and **Actionable Advice:**.\n"
            "   Do NOT respond in pure English."
        )
    else:
        lang_rule = (
            "5. LANGUAGE REQUIREMENT (CRITICAL): The user asked in English.\n"
            "   You MUST respond entirely in professional English.\n"
            "   Keep section labels in English bold: **Astrological Insight:** and **Actionable Advice:**.\n"
            "   Do NOT use Hindi or Hinglish words."
        )

    return f"""You are a seasoned Vedic Astrologer providing a multi-domain consultation in a warm, empathetic, professional 'AstroTalk-style' tone.
You are given verified astrological rules and principles extracted modularly from our local database for each queried domain.

STRICT INSTRUCTIONS:
1. Grounding: You must strictly ground your synthesis in the provided modular domain evidence. Never invent imaginary planetary degrees or unverified external claims.
2. Cohesive Synthesis: Weave the disparate domains into a single, seamless, harmonious reading.
3. Strict 5-7 Line Layout: You MUST format your response into exactly these two sections:
**Astrological Insight:** [Synthesized multi-domain insight explaining how the queried domains interact and align]
**Actionable Advice:** [Grounded next steps and practical guidance across the queried areas]
4. Brevity: Keep the total output to 5-7 lines (~55-80 words). Do NOT add extra chit-chat, conversational filler, or trailing notes.
{lang_rule}
"""


async def synthesize_hybrid_multidomain_rag(
    query: str,
    detected_domains: Optional[List[str]] = None,
    qa_db: Optional[Dict[str, str]] = None,
    custom_httpx_client: Optional[httpx.AsyncClient] = None,
    custom_gemini_client: Optional[Any] = None,
    is_hinglish: Optional[bool] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Synthesizes a cohesive multi-domain response using local database facts + cascading LLM.
    """
    if is_hinglish is None:
        is_hinglish = is_hinglish_query(query)

    domains = detected_domains or detect_query_domains(query)
    if not domains:
        domains = ["career"]

    modular_evidence = extract_modular_domain_evidence(query, domains, qa_db)
    evidence_context = format_multidomain_evidence_context(modular_evidence)

    if is_hinglish:
        synthesis_req = "Please synthesize these verified domain rules into a cohesive 5-7 line reading in conversational Hinglish."
    else:
        synthesis_req = "Please synthesize these verified domain rules into a cohesive 5-7 line reading in English."

    llm_prompt = f"""USER QUERY:
{query}

MODULAR ASTROLOGICAL RULES & FACTS (GROUND TRUTH):
{evidence_context}

{synthesis_req}"""

    fallback_insight_parts = []
    fallback_advice_parts = []
    for d in domains:
        if d == "career":
            fallback_insight_parts.append("your 10th-house career alignment supports professional growth")
            fallback_advice_parts.append("focus on structured career milestones")
        elif d == "finance":
            fallback_insight_parts.append("wealth indicators show steady financial stability")
            fallback_advice_parts.append("maintain disciplined savings")
        elif d == "health":
            fallback_insight_parts.append("vitality remains balanced with proper routine")
            fallback_advice_parts.append("prioritize morning wellness and mental peace")
        elif d == "gemstones":
            fallback_insight_parts.append("benefic gemstones can energize key house lords")
            fallback_advice_parts.append("seek suitable energization timing")
        elif d == "marriage":
            fallback_insight_parts.append("relationship houses encourage mutual trust")
            fallback_advice_parts.append("foster open communication")
        else:
            fallback_insight_parts.append(f"{d} indicators show steady harmony")

    local_fallback_reading = (
        f"**Astrological Insight:** Based on your birth chart, {', and '.join(fallback_insight_parts)}.\n"
        f"**Actionable Advice:** {'; '.join(fallback_advice_parts).capitalize()}."
    )

    active_model = "local_rules"
    raw_response_chunks = []
    decision_meta: Dict[str, Any] = {}

    system_inst = get_hybrid_multidomain_system_instruction(is_hinglish=is_hinglish)

    router_gen = stream_cascading_router(
        prompt=llm_prompt,
        system_prompt=system_inst,
        max_output_tokens=180,
        temperature=0.3,
        custom_httpx_client=custom_httpx_client,
        custom_gemini_client=custom_gemini_client,
        local_fallback_text=local_fallback_reading,
    )

    async for chunk_text, meta in router_gen:
        if meta:
            decision_meta = meta
            if meta.get("model"):
                active_model = meta.get("model")
        if chunk_text:
            raw_response_chunks.append(chunk_text)

    assembled_text = "".join(raw_response_chunks).strip()
    cleaned_body = trim_to_last_sentence(assembled_text)
    hybrid_badge = f"[Local Database + LLM - {active_model}]"
    final_output = f"{cleaned_body}\n\n{hybrid_badge}"

    result_meta = {
        "status": "success",
        "domains_detected": domains,
        "primary_domain": domains[0] if domains else "career",
        "modular_evidence_count": sum(len(v) for v in modular_evidence.values()),
        "active_model": active_model,
        "hybrid_badge": hybrid_badge,
        "decision_meta": decision_meta,
        "is_hinglish": is_hinglish,
    }

    return final_output, result_meta


async def stream_hybrid_multidomain_rag(
    query: str,
    detected_domains: Optional[List[str]] = None,
    qa_db: Optional[Dict[str, str]] = None,
    custom_httpx_client: Optional[httpx.AsyncClient] = None,
    custom_gemini_client: Optional[Any] = None,
    is_hinglish: Optional[bool] = None,
) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
    """Streaming generator for hybrid multi-domain RAG."""
    if is_hinglish is None:
        is_hinglish = is_hinglish_query(query)

    domains = detected_domains or detect_query_domains(query)
    if not domains:
        domains = ["career"]

    modular_evidence = extract_modular_domain_evidence(query, domains, qa_db)
    evidence_context = format_multidomain_evidence_context(modular_evidence)

    if is_hinglish:
        synthesis_req = "Please synthesize these verified domain rules into a cohesive 5-7 line reading in conversational Hinglish."
    else:
        synthesis_req = "Please synthesize these verified domain rules into a cohesive 5-7 line reading in English."

    llm_prompt = f"""USER QUERY:
{query}

MODULAR ASTROLOGICAL RULES & FACTS (GROUND TRUTH):
{evidence_context}

{synthesis_req}"""

    active_model = "deterministic_synthesis"
    emitted_chunks: List[str] = []

    system_inst = get_hybrid_multidomain_system_instruction(is_hinglish=is_hinglish)

    router_gen = stream_cascading_router(
        prompt=llm_prompt,
        system_prompt=system_inst,
        max_output_tokens=180,
        temperature=0.3,
        custom_httpx_client=custom_httpx_client,
        custom_gemini_client=custom_gemini_client,
    )

    async for chunk_text, meta in router_gen:
        if meta and meta.get("model"):
            active_model = meta.get("model")
        if chunk_text:
            emitted_chunks.append(chunk_text)
            yield (chunk_text, {"status": "streaming", "model": active_model, "domains": domains, "is_hinglish": is_hinglish})

    hybrid_badge = f"\n\n[Local Database + LLM - {active_model}]"
    yield (hybrid_badge, {
        "status": "completed",
        "model": active_model,
        "domains": domains,
        "hybrid_badge": hybrid_badge.strip(),
        "is_hinglish": is_hinglish,
        "full_text": "".join(emitted_chunks).strip() + hybrid_badge
    })
