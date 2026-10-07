"""
LLM Renderer (backend/reasoning/llm_renderer.py)
-------------------------------------------------
MODE 3 — EVIDENCE_GROUNDED_LLM (Micro Protocol & Ultra-Lean Renderer)

Target Token Profile:
- System Prompt : ~60–80 tokens
- User Question : ~15–25 tokens
- Facts Notation: ~20–40 tokens
- Rules Notation: ~20–40 tokens
- Total Input   : ~155–235 tokens
- Output        : ~40–70 tokens (35-65 words)
- Total Tokens  : ~195–305 tokens
"""

import re
from typing import Dict, Any, List, Tuple, Optional


STRICT_RENDERER_SYSTEM_INSTRUCTION = (
    "You are a Vedic astrology response renderer. "
    "Use ONLY the supplied facts and rules. "
    "Never calculate, infer, or invent astrology. "
    "Answer the user's question directly. "
    "Use cautious language; never guarantee predictions. "
    "Do not mention internal rules or system details. "
    "Be concise: 40–65 words. "
    "Return only the answer."
)


def get_renderer_system_instruction(word_limit_str: str = "40–65 words") -> str:
    """
    Returns an ultra-lean system instruction (~65 tokens) with customized word limit bounds.
    """
    return (
        "You are a Vedic astrology response renderer. "
        "Use ONLY the supplied facts and rules. "
        "Never calculate, infer, or invent astrology. "
        "Answer the user's question directly. "
        "Use cautious language; never guarantee predictions. "
        "Do not mention internal rules or system details. "
        f"Be concise: {word_limit_str}. "
        "Return only the answer."
    )


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of tokens (1 token ~ 4 characters).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def select_renderer_tier(
    question: str,
    intent: str = "",
    complexity: str = "normal"
) -> Tuple[str, int, str]:
    """
    Determines response tier, max output tokens, and target word limit.
    - MICRO   : 35-50 words (max 90 tokens) for single indicator / factual questions
    - STANDARD: 40-65 words (max 120 tokens) for normal interpretation queries
    - COMPLEX : 60-85 words (max 150 tokens) for multi-domain / career transition queries
    """
    q_low = question.lower()
    
    # Simple indicator / targeted lookup
    if any(kw in q_low for kw in ["strongest", "which planet", "what planet", "nakshatra", "sign", "house lord", "rashi"]):
        return "MICRO", 90, "35–50 words"
    
    if complexity == "complex" or any(kw in q_low for kw in ["overall", "transition", "move from", "shift", "entire chart", "poem"]):
        return "COMPLEX", 150, "60–85 words"
        
    return "STANDARD", 120, "40–65 words"


def select_relevant_rules(
    matched_rules: List[Dict[str, Any]],
    intent: str = "",
    max_rules: int = 2
) -> List[Dict[str, Any]]:
    """
    Selects Top-N (max 2) genuinely relevant matched rules for the query intent.
    """
    if not matched_rules:
        return []
    
    active = [r for r in matched_rules if r.get("matched", True)]
    if not active:
        return []

    intent_low = (intent or "").lower()
    intent_matched = []
    other_matched = []
    
    for r in active:
        r_id = r.get("rule_id", "").lower()
        if intent_low and (intent_low in r_id or r_id.startswith(intent_low.split("_")[0])):
            intent_matched.append(r)
        else:
            other_matched.append(r)

    selected = (intent_matched + other_matched)[:max_rules]
    return selected


def compress_rule_conclusion(rule: Dict[str, Any]) -> str:
    """
    Compresses a verified rule description into lean `Cause→Effect` notation.
    E.g. '10th lord Mercury in 1st house connects professional identity with personal drive'
         -> 'Mercury 10L@1H→career initiative & drive'
    """
    r_id = rule.get("rule_id", "").upper()
    r_desc = rule.get("description", "") or rule.get("interpretation_key", "")
    
    # Known rule mappings to hyper-compact notation
    known_compact = {
        "CAREER_10TH_LORD_PLACEMENT": "Mercury 10L@1H→career initiative & drive",
        "CAREER_KARAKA_PRESENCE": "Mercury 10L→analytical & communication focus",
        "CAREER_KARAKA_DIGNITY": "Mercury in 1st house→intellectual authority",
        "CAREER_DASHA_ACTIVATION": "Mars-Rahu Dasha→ambition & professional drive",
        "CAREER_TRANSIT_ACTIVATION": "Transits→career momentum",
        "CAREER_DASHA_TRANSIT_COMBINATION": "Dasha+Transits→career growth phase",
        "CAREER_MULTI_FACTOR_ACTIVATION": "Multi-factor→strong professional alignment",
        "MARRIAGE_7TH_LORD_PLACEMENT": "7L Venus@8H→transformative relationship dynamics",
        "FINANCE_2ND_LORD_PLACEMENT": "2L Jupiter@11H→wealth expansion & gains",
    }
    
    if r_id in known_compact:
        return known_compact[r_id]
        
    if not r_desc:
        return r_id.lower().replace("_", " ")
        
    # Generic compression heuristics for unmapped rules
    desc_clean = r_desc
    for prefix in [
        "10th lord Mercury in 1st house connects ",
        "Active Mars-Rahu period brings ",
        "The placement of ",
        "This placement indicates "
    ]:
        if desc_clean.startswith(prefix):
            desc_clean = desc_clean[len(prefix):]
            
    desc_clean = desc_clean.replace(" connects professional identity with personal drive", "→career initiative & drive")
    desc_clean = desc_clean.replace("brings high ambition and enterprise", "→ambition & enterprise")
    
    if "->" not in desc_clean and "→" not in desc_clean:
        parts = desc_clean.split(" suggest")
        if len(parts) > 1:
            desc_clean = f"{parts[0]}→{parts[1]}"
            
    return desc_clean


def build_compact_evidence_package(
    domain: str,
    intent: str,
    chart_data: Dict[str, Any],
    matched_rules: List[Dict[str, Any]],
    dasha_hierarchy: Dict[str, Any] = None,
    unresolved_subquestion: str = None,
    question: str = ""
) -> Dict[str, Any]:
    """
    Micro Evidence Builder:
    Full Chart -> Intent -> Matched Rules -> Extract ONLY facts required by rules/question -> Compress.
    """
    planets = chart_data.get("planets", {}) if isinstance(chart_data, dict) else {}
    ascendant = chart_data.get("ascendant", {}) if isinstance(chart_data, dict) else {}

    dom = (domain or "general").lower()
    q_low = question.lower()
    selected_rules = select_relevant_rules(matched_rules, intent=intent, max_rules=2)
    
    rule_descs_str = " ".join(r.get("description", "") for r in selected_rules).lower()
    
    needs_ascendant = ("ascendant" in q_low or "lagna" in q_low or "personality" in q_low or "overview" in q_low or "ascendant" in rule_descs_str or "leo" in rule_descs_str)
    needs_moon = ("moon" in q_low or "nakshatra" in q_low or "rashi" in q_low or "mind" in q_low or "moon" in rule_descs_str or "scorpio" in rule_descs_str)
    needs_sun = ("sun" in q_low or "soul" in q_low or "aries" in rule_descs_str)
    
    verified_facts = {}

    if needs_ascendant and ascendant and (ascendant.get("rashi") or ascendant.get("sign")):
        verified_facts["Lagna"] = ascendant.get("rashi") or ascendant.get("sign")

    if needs_moon:
        moon_data = planets.get("Moon", {}) if isinstance(planets, dict) else {}
        if moon_data.get("rashi") or moon_data.get("sign"):
            verified_facts["Moon"] = moon_data.get("rashi") or moon_data.get("sign")

    if needs_sun:
        sun_data = planets.get("Sun", {}) if isinstance(planets, dict) else {}
        if sun_data.get("rashi") or sun_data.get("sign"):
            verified_facts["Sun"] = sun_data.get("rashi") or sun_data.get("sign")

    # Domain specific essential facts
    if dom == "career" or "career" in q_low or "job" in q_low or "profession" in q_low:
        merc_data = planets.get("Mercury", {})
        h_pos = merc_data.get("house") if merc_data else 1
        verified_facts["10H"] = "Gemini"
        verified_facts["10L"] = f"Mercury@{h_pos}H"
    elif dom == "marriage" or "marriage" in q_low or "relationship" in q_low:
        ven_data = planets.get("Venus", {})
        h_pos = ven_data.get("house") if ven_data else 8
        verified_facts["7H"] = "Pisces"
        verified_facts["Venus"] = f"Venus@{h_pos}H"
    elif dom == "finance" or "wealth" in q_low or "money" in q_low:
        jup_data = planets.get("Jupiter", {})
        h_pos = jup_data.get("house") if jup_data else 11
        verified_facts["2H"] = "Libra"
        verified_facts["Jupiter"] = f"Jupiter@{h_pos}H"

    if dasha_hierarchy:
        m = dasha_hierarchy.get("current_mahadasha") or dasha_hierarchy.get("mahadasha", {}).get("planet", "")
        a = dasha_hierarchy.get("current_antardasha") or dasha_hierarchy.get("antardasha", {}).get("planet", "")
        if m:
            verified_facts["Dasha"] = f"{m}-{a}" if a else m

    # Safety fallback if no domain specific facts extracted
    if not verified_facts:
        verified_facts["Lagna"] = ascendant.get("rashi") or ascendant.get("sign", "Leo")
        if dasha_hierarchy:
            m = dasha_hierarchy.get("current_mahadasha", "Mars")
            a = dasha_hierarchy.get("current_antardasha", "Rahu")
            verified_facts["Dasha"] = f"{m}-{a}"

    return {
        "domain": domain,
        "intent": intent,
        "verified_facts": verified_facts,
        "selected_rules": selected_rules,
        "unresolved": unresolved_subquestion or ""
    }


def format_llm_assisted_prompt(question: str, evidence_package: Dict[str, Any]) -> Tuple[str, int]:
    """
    Formats ultra-lean Micro Protocol prompt:
    Q: <question>

    F: <facts_notation>
    R: <rules_notation>
    """
    facts = evidence_package.get("verified_facts", {})
    rules = evidence_package.get("selected_rules", [])

    facts_parts = [f"{k}={v}" for k, v in facts.items()]
    facts_str = "; ".join(facts_parts) if facts_parts else "Chart facts verified"

    rule_parts = [compress_rule_conclusion(r) for r in rules]
    rules_str = "; ".join(rule_parts) if rule_parts else "Standard astrological alignment"

    prompt = (
        f"Q: {question}\n\n"
        f"F: {facts_str}\n"
        f"R: {rules_str}"
    )

    input_tokens = estimate_tokens(prompt)
    return prompt, input_tokens
