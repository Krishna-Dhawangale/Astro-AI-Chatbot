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


# Patterns that deflect the user to an external astrologer (disclaimers to strip)
ASTROLOGER_DISCLAIMER_PATTERNS = [
    # English disclaimers
    r"(?i)(?:please\s+)?(?:consult|seek|reach\s+out\s+to|speak\s+with)\s+(?:a|an)?\s*(?:qualified|professional|experienced|certified|practicing)?\s*(?:vedic\s+)?astrologer(?:\s+for\s+[^.!?\n]+)?[.!?]?",
    r"(?i)(?:for\s+(?:a\s+)?(?:detailed|comprehensive|in-depth|personalized)\s+(?:analysis|reading|interpretation)[^.!?\n]*,?\s*)?(?:please\s+)?(?:consult|speak\s+with|reach\s+out\s+to)\s+(?:a|an)?\s*(?:professional|qualified|experienced|vedic)?\s*astrologer[.!?]?",
    r"(?i)(?:it\s+is\s+(?:recommended|advised|best)\s+to\s+)?consult\s+(?:with\s+)?(?:a|an)?\s*(?:qualified|professional|experienced|vedic)?\s*astrologer(?:\s+for\s+[^.!?\n]+)?[.!?]?",
    r"(?i)consult\s+a\s+professional\s+astrologer\s+for\s+a\s+detailed,?\s*personalized\s+analysis\s+of\s+your\s+chart[.!?]?",
    r"(?i)(?:please\s+)?consult\s+(?:an|a)\s+astrologer\s+for\s+(?:a\s+)?detailed,?\s*personalized\s+analysis[.!?]?",
    r"(?i)for\s+a\s+detailed,?\s*personalized\s+analysis\s+of\s+your\s+chart,?\s*(?:please\s+)?consult\s+a\s+professional\s+astrologer[.!?]?",

    # Hindi / Hinglish disclaimers
    r"(?i)(?:kripya\s+)?(?:kisi\s+)?(?:qualified|professional|experienced)?\s*(?:vedic\s+)?astrologer\s+se\s+(?:sampark|paramarsh|consult|baat)\s*(?:karein|kare|kijiye|karna\s+chahiye)[.!?]?",
    r"(?i)(?:detailed|personalized)\s*(?:chart\s*)?(?:analysis|reading)?\s*ke\s+liye\s*(?:kisi\s+)?(?:qualified\s+)?(?:vedic\s+)?astrologer\s+se\s+(?:sampark|paramarsh|consult)\s*(?:karein|kare|kijiye)[.!?]?",
    r"(?i)(?:kripya\s+)?(?:kisi\s+)?(?:jyotishi|pandit\s*ji|vidwan)\s+se\s+(?:sampark|paramarsh)\s*(?:karein|kare|kijiye)[.!?]?",
    r"(?i)kripya\s+kisi\s+qualified\s+vedic\s+astrologer\s+se\s+sampark\s+karein[.!?]?",
    r"(?i)kisi\s+qualified\s+astrologer\s+se\s+sampark\s+karein[.!?]?",
]


def strip_astrologer_disclaimers(text: str) -> str:
    """Removes all phrases advising the user to consult an external astrologer."""
    if not text:
        return ""
    res = text
    for pat in ASTROLOGER_DISCLAIMER_PATTERNS:
        res = re.sub(pat, "", res)
    return re.sub(r"\s+", " ", res).strip()


def clean_astrology_response(text: str, is_hinglish: bool = False) -> str:
    """
    Cleans astrological text:
    - Removes internal provenance tags: [Local Database + LLM: ...], [Local Database], [LLM: ...]
    - Removes calculation footers: *[Calculation: ...]*
    - Removes markdown asterisks (* and **) from headings/text
    - Strips external astrologer referral disclaimers
    - Guarantees authentic astrological fallback if disclaimer removal leaves the answer empty
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
    # 4. Strip external astrologer referral disclaimers
    res = strip_astrologer_disclaimers(res)

    # 5. Fallback if response became empty or trivial after removing disclaimers
    if len(res) < 15:
        if is_hinglish:
            return (
                "Aapki kundali ke anusaar, aapke planetary indicators positive transformation aur gradual stability ko support kar rahe hain. "
                "Current planetary cycle aapke favor mein align ho rahi hai—apne karma aur lakshya par focused rahiye."
            )
        return (
            "Based on your birth chart placements, your planetary indicators highlight constructive momentum, gradual stabilization, and favorable opportunities. "
            "Maintaining disciplined focus aligns harmoniously with your current planetary cycle."
        )

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



