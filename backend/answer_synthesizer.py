"""
backend/answer_synthesizer.py
Multi-Domain LLM Answer Synthesizer & Prose Refiner
===================================================
1. Mandatory LLM Synthesis for multi-domain inquiries.
2. Formulates context_payload containing verified raw chart facts & database rules.
3. Ultra-lean Prose Refiner system prompt (~50 tokens) with strict 350-400 total token budget.
4. Word-to-token ratio (1.3x) token estimator and conditional token logging.
5. Real-time Model Token Tracker & Remaining Token Observability.
6. Generates dynamic calculation footer:
   *[Calculation: SwissEphemeris_VimshottariEngine | Multi-Domain Hybrid Synthesis: Career & Finance]*
"""

import os
import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("AstroChatbot.AnswerSynthesizer")

ULTRA_LEAN_SYSTEM_PROMPT = (
    "You are a Vedic astrology synthesizer. Use ONLY the provided facts. "
    "Synthesize all requested domains into one cohesive reading. Never invent astrology. "
    "Be concise: 65–85 words. Return only the final consultation prose."
)

# ---------------------------------------------------------------------------
# Per-Model Token Quota & Remaining Token Tracking
# ---------------------------------------------------------------------------
DEFAULT_MODEL_BUDGETS: Dict[str, int] = {
    "gemini-3.5-flash-lite": int(os.getenv("GEMINI_FLASH_LITE_TOKEN_BUDGET", "250000")),
    "gemini-3.8-flash": int(os.getenv("GEMINI_FLASH_TOKEN_BUDGET", "250000")),
    "gemini-2.5-flash": int(os.getenv("GEMINI_25_FLASH_TOKEN_BUDGET", "250000")),
    "qwen/qwen3.8-27b": int(os.getenv("GROQ_QWEN_TOKEN_BUDGET", "100000")),
    "llama-3.3-70b-versatile": int(os.getenv("GROQ_LLAMA70B_TOKEN_BUDGET", "100000")),
    "llama-3.1-8b-instant": int(os.getenv("GROQ_LLAMA8B_TOKEN_BUDGET", "100000")),
    "openai/gpt-oss-120b": int(os.getenv("GROQ_GPTOSS_TOKEN_BUDGET", "100000")),
}
DEFAULT_FALLBACK_BUDGET = 100000

# In-memory tracking state: model_name -> {"used_tokens": int, "call_count": int}
_MODEL_TOKEN_USAGE: Dict[str, Dict[str, int]] = {}


def record_model_tokens(model_name: str, tokens: int) -> Dict[str, Any]:
    """
    Records consumed tokens for a given model and calculates remaining token budget.
    """
    clean_model = (model_name or "unknown").strip().lower()
    if clean_model not in _MODEL_TOKEN_USAGE:
        _MODEL_TOKEN_USAGE[clean_model] = {"used_tokens": 0, "call_count": 0}
    
    _MODEL_TOKEN_USAGE[clean_model]["used_tokens"] += tokens
    _MODEL_TOKEN_USAGE[clean_model]["call_count"] += 1
    
    budget = DEFAULT_MODEL_BUDGETS.get(clean_model, DEFAULT_FALLBACK_BUDGET)
    used = _MODEL_TOKEN_USAGE[clean_model]["used_tokens"]
    remaining = max(0, budget - used)
    
    return {
        "model": clean_model,
        "used_tokens": used,
        "budget_tokens": budget,
        "remaining_tokens": remaining,
        "call_count": _MODEL_TOKEN_USAGE[clean_model]["call_count"],
    }


def get_model_remaining_tokens(model_name: str) -> int:
    """Returns the remaining tokens available for a given model."""
    clean_model = (model_name or "unknown").strip().lower()
    budget = DEFAULT_MODEL_BUDGETS.get(clean_model, DEFAULT_FALLBACK_BUDGET)
    used = _MODEL_TOKEN_USAGE.get(clean_model, {}).get("used_tokens", 0)
    return max(0, budget - used)


def get_all_models_token_status() -> Dict[str, Dict[str, Any]]:
    """Returns a full snapshot of token usage and remaining tokens for all registered models."""
    status = {}
    all_models = set(DEFAULT_MODEL_BUDGETS.keys()).union(_MODEL_TOKEN_USAGE.keys())
    for model in sorted(all_models):
        budget = DEFAULT_MODEL_BUDGETS.get(model, DEFAULT_FALLBACK_BUDGET)
        used = _MODEL_TOKEN_USAGE.get(model, {}).get("used_tokens", 0)
        calls = _MODEL_TOKEN_USAGE.get(model, {}).get("call_count", 0)
        status[model] = {
            "budget_tokens": budget,
            "used_tokens": used,
            "remaining_tokens": max(0, budget - used),
            "calls": calls,
        }
    return status


def estimate_tokens(text: str) -> int:
    """
    Estimates token count using a word-to-token ratio (~1.3 multiplier).
    """
    if not text or not text.strip():
        return 0
    words = len(text.strip().split())
    return max(1, int(round(words * 1.3)))


def format_domains_title(domains: List[str]) -> str:
    """
    Formats a list of domain names into clean, capitalized title strings.
    E.g. ['career', 'finance'] -> 'Career & Finance'
         ['career', 'finance', 'dasha'] -> 'Career & Finance'
    """
    raw_lower = [d.lower().strip() for d in domains if d]
    has_career = "career" in raw_lower
    has_finance = "finance" in raw_lower
    other_substantive = [d for d in raw_lower if d not in ["career", "finance", "dasha", "timing", "general"]]

    if has_career and has_finance and not other_substantive:
        return "Career & Finance"

    clean_domains = []
    domain_order = ["career", "finance", "marriage", "health", "education", "remedies", "travel", "dasha"]
    
    sorted_domains = sorted(
        set(raw_lower),
        key=lambda x: domain_order.index(x) if x in domain_order else 99
    )
    
    for d in sorted_domains:
        d_cap = d.capitalize()
        if d == "dasha":
            d_cap = "Dasha"
        elif d == "remedies":
            d_cap = "Remedies"
        clean_domains.append(d_cap)

    if not clean_domains:
        return "Multi-Domain"
    elif len(clean_domains) == 1:
        return clean_domains[0]
    elif len(clean_domains) == 2:
        return f"{clean_domains[0]} & {clean_domains[1]}"
    else:
        return ", ".join(clean_domains[:-1]) + f" & {clean_domains[-1]}"


def print_terminal_token_report(
    mode: str,
    model_name: str,
    inp_tokens: int,
    out_tokens: int,
    total_tokens: int,
    query_remaining: int,
    model_budget: int,
    model_remaining: int,
) -> None:
    """Prints clean observability metrics to the terminal only."""
    report_lines = [
        "==================================================",
        "        TERMINAL LLM TOKEN & REMAINING TRACKER    ",
        "==================================================",
        f"Execution Mode     : {mode}",
        f"Model Used         : {model_name}",
        f"Input Tokens       : {inp_tokens}",
        f"Output Tokens      : {out_tokens}",
        f"Total Tokens       : {total_tokens} (Budget: 350-400)",
        f"Query Remainder    : {query_remaining} tokens",
        f"Model Total Budget : {model_budget:,} tokens",
        f"Model Tokens Left  : {model_remaining:,} tokens",
        "=================================================="
    ]
    print("\n" + "\n".join(report_lines) + "\n", flush=True)


def format_multidomain_footer(
    domains: List[str],
    engine: str = "FreeAstrologyAPI (Dasha) + SwissEphemeris (Gochar Timing)"
) -> str:
    """
    Constructs the standardized multi-domain calculation metadata footer.
    Example: *[Calculation: FreeAstrologyAPI (Dasha) + SwissEphemeris (Gochar Timing) | Multi-Domain Hybrid Synthesis: Career & Finance]*
    """
    domain_title = format_domains_title(domains)
    return f"*[Calculation: {engine} | Multi-Domain Hybrid Synthesis: {domain_title}]*"


def append_multidomain_footer(answer: str, domains: List[str]) -> str:
    """
    Appends the standardized calculation metadata footer with exactly two newlines,
    preventing duplicate footer accumulation.
    """
    if not answer:
        return ""
    body = answer.strip()
    body = re.sub(r"\n*\s*\*\[Calculation:[^\]]+\]\*", "", body).strip()
    footer = format_multidomain_footer(domains)
    return f"{body}\n\n{footer}"


def build_multidomain_system_prompt(
    domains: Optional[List[str]] = None,
    is_hinglish: bool = False
) -> str:
    """
    Generates ultra-compact prose refiner system prompt (~50 tokens).
    """
    if is_hinglish:
        return (
            "You are a Vedic astrology synthesizer. Use ONLY the provided facts. "
            "Synthesize all requested domains into one cohesive reading in conversational Hinglish. "
            "Never invent astrology. Be concise: 65–85 words. Return only the final consultation prose."
        )
    return ULTRA_LEAN_SYSTEM_PROMPT


def build_context_payload(
    raw_context: str,
    detected_intents: List[str],
    user_profile: Dict[str, Any]
) -> str:
    """
    Returns clean, non-redundant context_payload without duplicating birth chart fields.
    """
    return raw_context.strip()


async def synthesizer_multidomain_response(
    query: str,
    context_payload: str,
    detected_intents: List[str],
    domains: List[str],
    user_profile: Dict[str, Any],
    session_id: str = "default_user",
    custom_gemini_client: Any = None,
    is_hinglish: bool = False
) -> Dict[str, Any]:
    """
    Executes LLM synthesis over multi-domain context_payload with strict token optimization,
    hard output token cap (120), and conditional remaining-token tracking per model.
    """
    system_prompt = build_multidomain_system_prompt(domains=domains, is_hinglish=is_hinglish)
    user_prompt = (
        f"Q: {query}\n\n"
        f"Evidence:\n"
        f"{context_payload}"
    )

    refined_prose = ""
    active_model = ""
    meta_info = {}
    t_start = time.time()

    try:
        from backend.llm_router import stream_cascading_router
        chunks = []
        async for chunk, meta in stream_cascading_router(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_output_tokens=120,
            temperature=0.2,
            session_id=session_id,
            custom_gemini_client=custom_gemini_client
        ):
            if meta:
                if meta.get("model"):
                    active_model = meta.get("model")
                if meta.get("status") == "completed":
                    meta_info = meta
            if chunk:
                chunks.append(chunk)
        raw_prose = "".join(chunks).strip()
        from backend.llm_router import trim_to_last_sentence
        refined_prose = trim_to_last_sentence(raw_prose)
    except Exception as e:
        logger.warning("LLM cascade refinement exception in AnswerSynthesizer: %s", e)

    latency_ms = (time.time() - t_start) * 1000

    # Provenance source handling
    try:
        from backend.response_formatter import prepend_source_prefix
    except ImportError:
        from response_formatter import prepend_source_prefix

    if refined_prose:
        used_model = active_model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        source_badge_type = "HYBRID"
        llm_calls = 1
        inp_tokens = meta_info.get("input_tokens") or estimate_tokens(f"{system_prompt}\n{user_prompt}")
        out_tokens = meta_info.get("output_tokens") or estimate_tokens(refined_prose)
        total_tokens = inp_tokens + out_tokens

        # Record tokens against model budget & calculate remaining
        model_stats = record_model_tokens(used_model, total_tokens)
        model_tokens_remaining = model_stats["remaining_tokens"]
        query_remaining_tokens = max(0, 400 - total_tokens)

        # Conditional token tracking and threshold alerts for LLM / Hybrid mode
        if inp_tokens > 260:
            logger.warning(
                "[TOKEN ALERT] Input tokens (%d) exceed safe threshold of 260! Total Tokens: %d | Prompt snippet: %s...",
                inp_tokens, total_tokens, user_prompt[:90].replace("\n", " ")
            )
        print_terminal_token_report(
            mode="LOCAL_DATABASE_PLUS_LLM_MULTI_DOMAIN",
            model_name=used_model,
            inp_tokens=inp_tokens,
            out_tokens=out_tokens,
            total_tokens=total_tokens,
            query_remaining=query_remaining_tokens,
            model_budget=model_stats["budget_tokens"],
            model_remaining=model_tokens_remaining,
        )
    else:
        # Graceful deterministic fallback synthesizing all domains if LLM offline
        used_model = None
        source_badge_type = "LOCAL"
        llm_calls = 0
        inp_tokens = 0
        out_tokens = 0
        total_tokens = 0
        query_remaining_tokens = 400
        model_tokens_remaining = 0
        asc = user_profile.get("ascendant_sign", "Virgo")
        moon = user_profile.get("moon_sign", "Taurus")
        maha = user_profile.get("mahadasha", "Rahu")
        antar = user_profile.get("antardasha", "Saturn")

        is_health_or_sleep = any(d in domains for d in ["health", "sleep", "rest"]) or any(w in query.lower() for w in ["sleep", "rest", "neend", "insomnia", "health", "fatigue", "tired"])
        is_property_or_vehicle = any(d in domains for d in ["property", "vehicle", "asset"]) or any(w in query.lower() for w in ["car", "vehicle", "gadi", "property", "makan", "ghar", "land", "plot", "flat", "vahana"])
        is_travel = any(d in domains for d in ["travel", "foreign"]) or any(w in query.lower() for w in ["travel", "foreign", "abroad", "videsh", "visa", "pr", "relocation", "overseas"])

        if is_property_or_vehicle:
            if is_hinglish:
                refined_prose = (
                    f"Aapki kundali ({asc} Lagna, {moon} Rashi) ke anusaar gadi aur property acquisition 4th house "
                    f"(Sukha aur Vahana Bhava) se govern hota hai, jisme Venus vehicles aur Mars real estate ka karaka hai. "
                    f"Active **{maha}** Mahadasha aur **{antar}** Antardasha ke dauran 4th house par supportive transits "
                    f"vehicle purchase aur property security ke liye anukool alignments create karte hain."
                )
            else:
                refined_prose = (
                    f"According to your birth chart ({asc} Ascendant, {moon} Moon sign), property and vehicle acquisition "
                    f"is governed by your 4th house (Sukha and Vahana Bhava), with Venus signifying conveyances and Mars signifying "
                    f"real estate. Under your active **{maha}** Mahadasha and **{antar}** Antardasha, transits supporting the 4th house "
                    f"create favorable conditions for securing vehicle comfort and fixed property investments."
                )
        elif is_travel:
            if is_hinglish:
                refined_prose = (
                    f"Aapki kundali ({asc} Lagna, {moon} Rashi) ke anusaar videsh yatra aur overseas settlement 9th house "
                    f"(long journeys) aur 12th house (foreign lands) se govern hoti hai, jisme Rahu aur Moon primary karakas hain. "
                    f"Active **{maha}** Mahadasha aur **{antar}** Antardasha ke dauran yeh transits foreign travel, visa clearances, "
                    f"aur international relocations ke liye favorable opportunities open karte hain."
                )
            else:
                refined_prose = (
                    f"According to your birth chart ({asc} Ascendant, {moon} Moon sign), international travel and overseas settlement "
                    f"are ruled by the 9th house (long-distance journeys) and 12th house (foreign lands), energized by Rahu and the Moon. "
                    f"Navigating your active **{maha}** Mahadasha and **{antar}** Antardasha activates cross-border prospects, documentation "
                    f"clearances, and auspicious foreign travel opportunities."
                )
        elif is_health_or_sleep:
            if is_hinglish:
                refined_prose = (
                    f"Aapki kundali ({asc} Lagna, {moon} Rashi) ke anusaar active **{maha}** Mahadasha aur **{antar}** Antardasha "
                    f"ke dauran mental peace aur physical rest ke liye 12th house (shayan sukha) aur 6th house (vitality) ko balance karna "
                    f"zaroori hai. Late-night overthinking se bachein aur Moon ko strengthen karne ke liye evening relaxation aur "
                    f"consistent sleep routine maintain karein. Yeh planetary cycle conscious rest ke sath deep rejuvenation support karega."
                )
            else:
                refined_prose = (
                    f"Based on your birth chart ({asc} Ascendant, {moon} Moon sign), the active planetary influences of your "
                    f"**{maha}** Mahadasha and **{antar}** Antardasha highlight the 12th house (rest, sleep quality, and subconscious peace) "
                    f"and 6th house (immunity and recovery). To optimize restorative rest, maintain a grounded evening routine, "
                    f"reduce overstimulation before bed, and align with calming lunar practices to nurture vitality throughout this cycle."
                )
        elif is_hinglish:
            refined_prose = (
                f"Aapki kundali ({asc} Lagna, {moon} Rashi) ke anusaar **{maha}** Mahadasha aur **{antar}** Antardasha ka prabhav "
                f"career growth mein dynamic opportunities ke sath structural challenges lata hai. 10th house karma bhava indicate karta hai "
                f"ki career prospects ko enhance karne ke liye patient dedication zaroori hai. Financial stability ke context mein, 2nd aur 11th "
                f"house factors conservative planning aur impulsive expenditure se bachne ki salah dete hain. Yeh period steady growth provide karega."
            )
        else:
            refined_prose = (
                f"According to your birth chart ({asc} Ascendant, {moon} Moon sign), navigating the active **{maha}** Mahadasha "
                f"and **{antar}** Antardasha creates ambitious career aspirations paired with structural discipline. For career growth prospects, "
                f"your 10th house signifies steady elevation achieved through methodical perseverance. Regarding financial stability, the 2nd and "
                f"11th houses emphasize prudent savings and avoiding speculative risks during this planetary cycle to ensure lasting prosperity."
            )

    # 1. Append standardized calculation metadata footer with domain specifics (clean, no tokens)
    final_answer = append_multidomain_footer(refined_prose, domains=domains)

    # 2. Prepend source badge [Local Database + LLM: {model_name}]
    final_answer = prepend_source_prefix(final_answer, source_type=source_badge_type, model_name=used_model)

    footer_str = format_multidomain_footer(domains=domains)

    return {
        "answer": final_answer,
        "answer_source": "HYBRID_SYNTHESIS",
        "answer_mode": "MULTI_DOMAIN",
        "detected_intents": detected_intents,
        "domains": domains,
        "model_name": used_model,
        "context_payload": context_payload,
        "attribution_badge": footer_str,
        "llm_calls": llm_calls,
        "input_tokens": inp_tokens,
        "output_tokens": out_tokens,
        "total_tokens": total_tokens,
        "remaining_query_tokens": query_remaining_tokens,
        "remaining_model_tokens": model_tokens_remaining,
        "all_models_token_status": get_all_models_token_status(),
        "latency_ms": latency_ms,
    }


# Backwards-compatibility alias
synthesize_multidomain_answer = synthesizer_multidomain_response
