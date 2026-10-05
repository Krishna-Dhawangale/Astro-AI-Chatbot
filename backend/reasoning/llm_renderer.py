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
    Builds an ultra-compact evidence package containing ONLY essential established backend facts.
    """
    planets = chart_data.get("planets", {})
    ascendant = chart_data.get("ascendant", {})

    facts = []
    if ascendant and ascendant.get("rashi"):
        facts.append(f"Ascendant: {ascendant.get('rashi')}")
    
    if dasha_hierarchy:
        m = dasha_hierarchy.get("mahadasha", {}).get("planet", "")
        a = dasha_hierarchy.get("antardasha", {}).get("planet", "")
        if m:
            facts.append(f"Current Dasha: {m}{'/' + a if a else ''}")

    for r in matched_rules[:3]:
        if r.get("matched", True):
            r_id = r.get("rule_id", "")
            interp = r.get("interpretation_key", "")
            facts.append(f"{r_id}: {interp}")

    return {
        "domain": domain,
        "intent": intent,
        "established_facts": facts,
        "unresolved": unresolved_subquestion or ""
    }


def format_llm_assisted_prompt(question: str, evidence_package: Dict[str, Any]) -> Tuple[str, int]:
    """
    Formats an ultra-compact prompt sent to Gemini in Phase 21.
    Returns (prompt_text, estimated_input_tokens).
    """
    domain = evidence_package.get("domain", "astrology")
    intent = evidence_package.get("intent", "general")
    facts = evidence_package.get("established_facts", [])
    unresolved = evidence_package.get("unresolved") or question

    facts_str = "; ".join(facts) if facts else "Chart evidence established"

    prompt = (
        f"You are a response renderer for domain '{domain}' ({intent}).\n"
        f"Established Backend Facts: {facts_str}\n"
        f"Task: Render a concise response (max 50 words) for: \"{unresolved}\"\n"
        f"Rules: Use ONLY established facts. Do NOT recalculate astrology. Do NOT introduce new claims."
    )

    input_tokens = estimate_tokens(prompt)
    return prompt, input_tokens
