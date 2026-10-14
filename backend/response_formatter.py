"""
backend/response_formatter.py
Standardized Technical Calculation Metadata Footer Pipeline
============================================================
Appends professional calculation metadata / provenance badges to astrological
responses with clean Markdown formatting, proper line-break isolation,
and dynamic domain/engine awareness.
"""

import re
from typing import Dict, Any, Optional


def build_calculation_metadata_footer(
    domain: str = "general",
    intent: str = "general",
    engine: str = "SwissEphemeris_VimshottariEngine",
    details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Constructs an italicized technical calculation metadata string
    customized by domain, active planetary timing, and transit activations.
    """
    details = details or {}
    domain_clean = (domain or "general").lower().strip()
    intent_clean = (intent or "general").lower().strip()

    # 1. Career Timing & Job Change
    if domain_clean == "career" and ("timing" in intent_clean or "change" in intent_clean or "switch" in intent_clean):
        window_note = details.get("window", "10th House Jupiter-Saturn Gochar")
        timing_engine = engine if engine != "SwissEphemeris_VimshottariEngine" else "SwissEphemeris_GocharTransitEngine"
        return f"*[Calculation: {timing_engine} | Transit Window: {window_note}]*"

    # 2. Marriage & Relationship Timing
    elif domain_clean == "marriage" and ("timing" in intent_clean or "when" in intent_clean):
        window_note = details.get("window", "7th House Jupiter-Venus Gochar")
        timing_engine = engine if engine != "SwissEphemeris_VimshottariEngine" else "SwissEphemeris_GocharTransitEngine"
        return f"*[Calculation: {timing_engine} | Transit Window: {window_note}]*"

    # 3. Direct Dasha / Mahadasha Lookups (from FreeAstrologyAPI key)
    elif "dasha" in intent_clean or "period" in intent_clean:
        dasha_engine = engine if engine != "SwissEphemeris_VimshottariEngine" else "FreeAstrologyAPI_VimshottariEngine"
        dasha_label = details.get("dasha_label")
        if dasha_label:
            return f"*[Calculation: {dasha_engine} | Active Period: {dasha_label}]*"
        return f"*[Calculation: {dasha_engine} | Vimshottari Dasha Hierarchy]*"

    # 4. Wealth & Financial Growth
    elif domain_clean in ["finance", "wealth"]:
        timing_engine = engine if engine != "SwissEphemeris_VimshottariEngine" else "SwissEphemeris_GocharTransitEngine"
        return f"*[Calculation: {timing_engine} | Transit Window: 2nd & 11th House Dhana Gochar]*"

    # 5. Health & Vitality
    elif domain_clean == "health":
        timing_engine = engine if engine != "SwissEphemeris_VimshottariEngine" else "SwissEphemeris_GocharTransitEngine"
        return f"*[Calculation: {timing_engine} | Natal Lagna & 6th House Gochar]*"

    # 6. General Ephemeris Natal Chart Placement
    else:
        return f"*[Calculation: SwissEphemeris_LahiriAyanamsha | Natal Planetary Positions]*"


def append_calculation_footer(
    answer: str,
    footer: Optional[str] = None,
    domain: str = "general",
    intent: str = "general",
    engine: str = "SwissEphemeris_VimshottariEngine",
    details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Appends a standardized italicized metadata footer to an astrological answer.
    Guarantees:
    - Exactly two newlines (\\n\\n) separating the body and footer.
    - Idempotency: Strips previous duplicate calculation tags to prevent footer stacking.
    - Clean rendering across frontend Markdown and stream consumers.
    """
    if not answer:
        return ""

    body = answer.strip()

    # Strip existing footer if already present to prevent duplicate accumulation
    body = re.sub(r"\n*\s*\*\[Calculation:[^\]]+\]\*", "", body).strip()

    # Generate metadata footer if not explicitly provided
    if not footer:
        footer = build_calculation_metadata_footer(
            domain=domain,
            intent=intent,
            engine=engine,
            details=details
        )

    footer = footer.strip()
    return f"{body}\n\n{footer}"


import os


def build_source_prefix(source_type: str, model_name: Optional[str] = None) -> str:
    """
    Constructs provenance badge prefix:
    - Local database + LLM: [Local Database + LLM: {model_name}]
    - Only Local database: [Local Database]
    - Only LLM: [LLM: {model_name}]
    """
    st = (source_type or "").upper().strip()
    clean_model = (model_name or "").strip()
    default_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    if st in ["HYBRID", "LOCAL_PLUS_LLM", "HYBRID_SYNTHESIS"]:
        target_model = clean_model or default_model
        return f"[Local Database + LLM: {target_model}]"
    elif st in ["LOCAL", "DATABASE", "LOCAL_DATABASE", "DIRECT", "FAQ", "RULE_BASED", "DETERMINISTIC_API"]:
        return "[Local Database]"
    elif st in ["LLM", "LLM_ONLY", "EVIDENCE_GROUNDED_LLM"]:
        target_model = clean_model or default_model
        return f"[LLM: {target_model}]"
    else:
        return f"[{source_type}]"


def prepend_source_prefix(
    answer: str,
    source_type: str,
    model_name: Optional[str] = None
) -> str:
    """
    Prepends the standardized provenance prefix to an answer.
    Guarantees exactly two newlines (\\n\\n) separation and prevents duplicate stacking.
    Preserves any existing model name if model_name is not explicitly passed.
    """
    if not answer:
        return ""

    body = answer.strip()
    # Check if answer already has a model name in its prefix
    if not model_name:
        existing_match = re.match(r"^\[(?:Local Database\s*\+\s*LLM|LLM):\s*([^\]]+)\]", body)
        if existing_match:
            model_name = existing_match.group(1).strip()

    # Strip existing prefix if already present to prevent duplicate accumulation
    body = re.sub(r"^\[(Local Database(?:\s*\+\s*LLM(?::\s*[^\]]+)?)?|LLM(?::\s*[^\]]+)?|[^\]]+)\]\s*\n*", "", body).strip()

    prefix = build_source_prefix(source_type, model_name)
    return f"{prefix}\n\n{body}"


