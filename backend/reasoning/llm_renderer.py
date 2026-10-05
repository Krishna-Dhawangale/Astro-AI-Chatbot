"""
LLM Renderer (backend/reasoning/llm_renderer.py)
-------------------------------------------------
MODE 3 — LLM_ASSISTED (Minimal LLM Tokens, Strictly Evidence-Bound Renderer)

Constructs a compact evidence JSON package and builds a strict renderer prompt
that forbids the LLM from performing astrology calculations or inferring missing facts.
"""

import json
from typing import Dict, Any, List, Tuple


STRICT_RENDERER_SYSTEM_INSTRUCTION = (
    "You are the final-response renderer.\n"
    "The backend has already performed:\n"
    "- domain classification\n"
    "- intent classification\n"
    "- chart calculation\n"
    "- Dasha calculation\n"
    "- transit calculation\n"
    "- deterministic reasoning\n\n"
    "Do NOT recalculate astrology.\n"
    "Use only the supplied evidence.\n"
    "Answer only the unresolved portion of the user's question.\n"
    "Do not invent missing chart information.\n"
    "Keep the answer concise.\n"
    "If evidence is insufficient, explicitly state the limitation."
)


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of tokens (1 token ~ 4 characters).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def build_compact_evidence_package(
    domain: str,
    intent: str,
    chart_data: Dict[str, Any],
    matched_rules: List[Dict[str, Any]],
    dasha_hierarchy: Dict[str, Any] = None,
    unresolved_subquestion: str = None
) -> Dict[str, Any]:
    """
    Builds a minimal evidence JSON package containing ONLY backend-calculated facts and matched rules.
    If unresolved_subquestion is provided, restricts evidence focus to the unresolved portion.
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

    pkg = {
        "domain": domain,
        "intent": intent,
        "chart_summary": {
            "ascendant": ascendant.get("rashi"),
            "current_dasha": dasha_str,
        },
        "matched_evidence": evidence_items[:5]
    }
    if unresolved_subquestion:
        pkg["unresolved_subquestion"] = unresolved_subquestion

    return pkg


def format_llm_assisted_prompt(question: str, evidence_package: Dict[str, Any]) -> Tuple[str, int]:
    """
    Formats the final prompt sent to Gemini in MODE 3 / Fallback.
    Returns (prompt_text, estimated_input_tokens).
    """
    evidence_json = json.dumps(evidence_package, indent=2)
    unresolved = evidence_package.get("unresolved_subquestion", question)
    
    prompt = (
        f"UNRESOLVED QUESTION PORTION: {unresolved}\n\n"
        f"SUPPLIED EVIDENCE JSON:\n{evidence_json}\n\n"
        f"Generate a concise, user-friendly response (max 75 words) using ONLY the supplied evidence."
    )
    
    full_text = f"{STRICT_RENDERER_SYSTEM_INSTRUCTION}\n\n{prompt}"
    input_tokens = estimate_tokens(full_text)
    return prompt, input_tokens
