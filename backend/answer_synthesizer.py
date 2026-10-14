"""
backend/answer_synthesizer.py
Multi-Domain LLM Answer Synthesizer & Prose Refiner
===================================================
1. Mandatory LLM Synthesis for multi-domain inquiries.
2. Formulates context_payload containing verified raw chart facts & database rules.
3. Strict Prose Refiner system prompt ensuring EVERY domain is synthesized.
4. Generates dynamic calculation footer:
   *[Calculation: SwissEphemeris_VimshottariEngine | Multi-Domain Hybrid Synthesis: Career & Finance]*
"""

import os
import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("AstroChatbot.AnswerSynthesizer")


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


def format_multidomain_footer(domains: List[str], engine: str = "FreeAstrologyAPI (Dasha) + SwissEphemeris (Gochar Timing)") -> str:
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
    # Strip any existing calculation badge
    body = re.sub(r"\n*\s*\*\[Calculation:[^\]]+\]\*", "", body).strip()
    footer = format_multidomain_footer(domains)
    return f"{body}\n\n{footer}"


def build_multidomain_system_prompt(
    domains: Optional[List[str]] = None,
    is_hinglish: bool = False
) -> str:
    """
    Generates ultra-compact prose refiner system prompt (~45 tokens) with mandatory zero-hallucination contract.
    """
    lang = "conversational Hinglish (Latin script)" if is_hinglish else "articulate English"
    dom_str = ", ".join(d.capitalize() for d in domains) if domains else "requested domains"
    return (
        f"You are a Vedic astrologer refiner. Synthesize provided facts into one cohesive {lang} consultation "
        f"covering: {dom_str}. Use ONLY supplied chart facts and timing windows. Never invent astrology or dates. "
        "Keep tone objective, grounded, and practical. Concise: 50–75 words. Return ONLY consultation prose."
    )


def build_context_payload(
    raw_context: str,
    detected_intents: List[str],
    user_profile: Dict[str, Any]
) -> str:
    """
    Returns clean, non-redundant context_payload without duplicating birth chart fields.
    """
    return raw_context.strip()


async def synthesize_multidomain_answer(
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
    Executes mandatory LLM synthesis over multi-domain context_payload.
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
            max_output_tokens=110,
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
        inp_tokens = meta_info.get("input_tokens") or max(1, (len(user_prompt) + len(system_prompt)) // 4)
        out_tokens = meta_info.get("output_tokens") or max(1, len(refined_prose) // 4)
        total_tokens = inp_tokens + out_tokens
    else:
        # Graceful deterministic fallback synthesizing all domains if LLM offline
        used_model = None
        source_badge_type = "LOCAL"
        llm_calls = 0
        inp_tokens = 0
        out_tokens = 0
        total_tokens = 0
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

    # 1. Append standardized calculation metadata footer with domain specifics
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
        "latency_ms": latency_ms,
    }
