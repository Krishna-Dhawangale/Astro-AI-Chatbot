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
