"""
LLM Renderer (backend/reasoning/llm_renderer.py)
-------------------------------------------------
MODE 3 — LLM_ASSISTED (Minimal LLM Tokens, Strictly Evidence-Bound Renderer)

Constructs a compact evidence JSON package and builds a strict renderer prompt
that forbids the LLM from performing astrology calculations or inferring missing facts.
"""

import json
from typing import Dict, Any, List


STRICT_RENDERER_SYSTEM_INSTRUCTION = (
    "You are a response renderer for a Vedic Astrology system.\n"
    "The backend has already performed all astrological calculations and reasoning.\n\n"
    "Use ONLY the supplied evidence JSON.\n\n"
    "Rules:\n"
    "1. Do not calculate astrology.\n"
    "2. Do not infer missing chart facts.\n"
    "3. Do not introduce planets, houses, Dashas, signs, aspects, or placements absent from the evidence.\n"
    "4. Do not contradict the evidence.\n"
    "5. Do not provide medical diagnosis or factual medical causation.\n"
    "6. Do not add generic astrological claims unless explicitly present in the evidence.\n"
    "7. Maximum 75 words.\n"
    "8. If the evidence is insufficient, state that the available chart evidence is insufficient."
)


def build_compact_evidence_package(
    domain: str,
    intent: str,
    chart_data: Dict[str, Any],
    matched_rules: List[Dict[str, Any]],
    dasha_hierarchy: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Builds a minimal evidence JSON containing ONLY backend-calculated facts and matched rules.
    """
    planets = chart_data.get("planets", {})
    ascendant = chart_data.get("ascendant", {})

    evidence_items = []
    for r in matched_rules:
        if r.get("matched", True):
            r_id = r.get("rule_id", "")
            ev = r.get("evidence", {})
            interp = r.get("interpretation_key", "")
            evidence_items.append(f"{r_id}: {interp} ({ev})")

    dasha_str = "N/A"
    if dasha_hierarchy:
        m = dasha_hierarchy.get("mahadasha", {}).get("planet", "")
        a = dasha_hierarchy.get("antardasha", {}).get("planet", "")
        dasha_str = f"{m} Mahadasha / {a} Antardasha" if m and a else m or "N/A"

    return {
        "domain": domain,
        "intent": intent,
        "chart_summary": {
            "ascendant": ascendant.get("rashi"),
            "current_dasha": dasha_str,
        },
        "matched_evidence": evidence_items[:5]
    }


def format_llm_assisted_prompt(question: str, evidence_package: Dict[str, Any]) -> str:
    """
    Formats the final prompt sent to Gemini in MODE 3.
    """
    evidence_json = json.dumps(evidence_package, indent=2)
    return (
        f"USER QUESTION: {question}\n\n"
        f"SUPPLIED EVIDENCE JSON:\n{evidence_json}\n\n"
        f"Generate a concise, user-friendly response (max 75 words) using ONLY the supplied evidence."
    )
