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


def clean_astrology_response(text: str) -> str:
    """
    Cleans astrological text:
    - Removes internal provenance tags: [Local Database + LLM: ...], [Local Database], [LLM: ...]
    - Removes calculation footers: *[Calculation: ...]*
    - Removes markdown asterisks (* and **) from headings/text
    """
    if not text:
        return ""
    # 1. Strip provenance prefixes ([Local Database...], [LLM...], etc.)
    res = re.sub(
        r"^\s*\[(?:Local Database(?:\s*\+\s*LLM(?::\s*[^\]]+)?)?|LLM(?::\s*[^\]]+)?|Backend|DIRECT|RULE_BASED|Deterministic[^\]]*|Hybrid[^\]]*|Timing[^\]]*|FAQ[^\]]*)\]\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE
    ).strip()
    # 2. Strip calculation footers
    res = re.sub(r"\n*\s*\*?\[Calculation:[^\]]+\]\*?", "", res, flags=re.IGNORECASE).strip()
    # 3. Strip all asterisks (* and **)
    res = res.replace("*", "").strip()
    return res


def build_calculation_metadata_footer(
    domain: str = "general",
    intent: str = "general",
    engine: str = "SwissEphemeris_VimshottariEngine",
    details: Optional[Dict[str, Any]] = None
) -> str:
    """Returns empty string so calculation footers are omitted from customer-facing answers."""
    return ""


def append_calculation_footer(
    answer: str,
    footer: Optional[str] = None,
    domain: str = "general",
    intent: str = "general",
    engine: str = "SwissEphemeris_VimshottariEngine",
    details: Optional[Dict[str, Any]] = None
) -> str:
    """Strips any existing calculation footers and asterisks, returning clean prose."""
    return clean_astrology_response(answer)


import os


def build_source_prefix(source_type: str, model_name: Optional[str] = None) -> str:
    """Returns empty string so provenance badges are omitted from customer-facing answers."""
    return ""


def prepend_source_prefix(
    answer: str,
    source_type: str,
    model_name: Optional[str] = None
) -> str:
    """Returns clean answer without provenance badges, calculation footers, or asterisks."""
    return clean_astrology_response(answer)



