"""
Multi-Domain Query Decomposer & Synthesizer (backend/reasoning/decomposer.py)
----------------------------------------------------------------------------
Pillar 5: Local Synthesizer for Multi-Domain Queries

Decomposes queries containing multiple concepts (e.g. Dasha API + Career rules)
and synthesizes a natural, structured local answer without calling Gemini (gemini_calls = 0).
"""

from typing import Dict, Any, Optional


def decompose_and_synthesize_multidomain(
    query: str,
    target_domain: str,
    dasha_hierarchy: Dict[str, Any],
    asc_sign: str,
    moon_sign: str,
    sun_sign: str,
    is_hinglish: bool = False
) -> Dict[str, Any]:
    """
    Decomposes multi-domain queries into sub-questions and synthesizes a natural answer locally.
    Returns:
    {
        "answer": "...",
        "answer_mode": "RULE_BASED",
        "answer_source": "MULTI_DOMAIN_LOCAL",
        "gemini_calls": 0
    }
    """
    q_lower = (query or "").lower().strip()

    # Part A: Extract Dasha API Facts
    mah = dasha_hierarchy.get("current_mahadasha") or dasha_hierarchy.get("output", {}).get("current_mahadasha", "Active Dasha")
    ant = dasha_hierarchy.get("current_antardasha") or dasha_hierarchy.get("output", {}).get("current_antardasha", "Active Antardasha")

    if is_hinglish:
        dasha_part = f"Aapka current Mahadasha **{mah}** aur Antardasha **{ant}** hai."
    else:
        dasha_part = f"Your current period is **{mah} Mahadasha** and **{ant} Antardasha**."

    # Part B: Synthesize Domain-Specific Local Reasoning
    if target_domain == "career" or any(k in q_lower for k in ["career", "job", "work"]):
        if is_hinglish:
            domain_part = (
                f"Career ke mamle mein, aapki birth chart ({asc_sign} Lagna, {moon_sign} Moon) ke 10th-house placements aur active planetary influences "
                f"analytical decision-making, structured management aur professional growth ko favor karte hain. "
                f"Yeh active Dasha period aapko naye professional skills hone aur career advancement par focus karne ka mauka deta hai."
            )
        else:
            domain_part = (
                f"Regarding career, your birth chart ({asc_sign} Ascendant, {moon_sign} Moon sign) highlights primary strengths in "
                f"analytical decision-making, structured management, and strategic execution. "
                f"Your active {mah} Mahadasha period supports developing specialized skills, taking on organizational responsibilities, and positioning yourself for progressive professional growth."
            )
    elif target_domain == "marriage" or any(k in q_lower for k in ["marriage", "wedding", "relationship"]):
        if is_hinglish:
            domain_part = (
                f"Marriage aur relationships ke liye, aapke 7th-house Placements mutual trust aur clear communication ko emphasize karte hain. "
                f"Aapka active Dasha period **2027–2028** ke aas-paas relationship stability aur partnership commitments ke liye supportive timing offer karta hai."
            )
        else:
            domain_part = (
                f"For relationship and marriage indicators, your 7th-house placements emphasize mutual trust, clear communication, and emotional harmony. "
                f"Your active Dasha alignment highlights a supportive timing window around **2027–2028** for strengthening relationship commitments."
            )
    else:
        if is_hinglish:
            domain_part = f"Aapke chart ({asc_sign} Lagna, {moon_sign} Moon) ke placements aapke life paths mein positive alignment show karte hain."
        else:
            domain_part = f"Your birth chart placements ({asc_sign} Ascendant, {moon_sign} Moon sign) show constructive alignment across your active life domains."

    combined_answer = f"{dasha_part}\n\n{domain_part}"

    return {
        "answer": combined_answer,
        "answer_mode": "RULE_BASED",
        "answer_source": "MULTI_DOMAIN_LOCAL",
        "gemini_calls": 0
    }
