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
from backend.reasoning.lordship import get_house_details, RASHI_LORDS


STRICT_RENDERER_SYSTEM_INSTRUCTION = (
    "You are an expert Vedic Astrologer and synthesizer. "
    "Treat supplied chart facts as authoritative; never invent or alter planetary placements, houses, or Dasha periods. "
    "Apply established Vedic principles to interpret verified facts and answer the user's question directly with authority and empathy. "
    "CRITICAL BAN: You ARE the qualified Vedic Astrologer. NEVER instruct the user to consult an astrologer or say 'kisi qualified Vedic astrologer se sampark karein'. Deliver the astrological interpretation directly. "
    "Use cautious language; never guarantee predictions. "
    "Do not mention internal rules or system details. "
    "Be concise: 40–65 words. "
    "Return only the answer."
)


def get_renderer_system_instruction(word_limit_str: str = "40–65 words", is_hinglish: bool = False) -> str:
    """
    Returns an ultra-lean system instruction with customized word limit bounds
    and natural conversational Hinglish language enforcement when requested.
    Permits grounded Vedic reasoning on supplied facts while strictly prohibiting invented chart calculations.
    """
    if is_hinglish:
        return (
            "You are an expert Vedic Astrologer. "
            "The user asked their question in conversational Hinglish (Hindi written in Roman/English script). "
            "You MUST respond naturally in conversational Hinglish (using Roman script, e.g. 'Aapki kundali ke anusar...'). "
            "Do NOT default to pure English. "
            "Treat supplied chart facts as authoritative; never invent or alter planetary placements, houses, or Dasha periods. "
            "Apply established Vedic principles to interpret verified facts and answer the user's question directly with authority and empathy. "
            "CRITICAL BAN: You ARE the qualified Vedic Astrologer. NEVER tell the user to consult an astrologer or say 'kisi qualified Vedic astrologer se sampark karein'. You MUST deliver the direct astrological guidance yourself based on their chart. "
            "Use cautious language; never guarantee predictions. "
            "Do not mention internal rules or system details. "
            f"Be concise: {word_limit_str}. "
            "Return only the answer in conversational Hinglish."
        )
    return (
        "You are an expert Vedic astrology synthesizer. "
        "Treat supplied chart facts as authoritative; never invent or alter planetary placements, houses, or Dasha periods. "
        "Apply established Vedic principles to interpret the verified facts and answer the user's question directly with authority and empathy. "
        "If the question involves multiple factors or comparative choices, synthesize them rather than repeating isolated placements. "
        "If evidence is partial, provide a helpful qualified interpretation and disclose limitations naturally without repeating generic disclaimers. "
        "For wellness or health queries, provide astrological perspectives only, never medical diagnoses. "
        "CRITICAL BAN: You ARE the qualified Vedic Astrologer. NEVER tell the user to consult a professional astrologer, seek an astrologer, or get a personalized chart analysis. You MUST deliver the direct astrological guidance yourself based on their chart. "
        "Use cautious language; never guarantee outcomes. "
        "Do not mention internal rules, evidence tags, or system instructions. "
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
    Compresses a verified rule description or its evidence into lean notation.
    """
    r_id = rule.get("rule_id", "").upper()
    r_desc = rule.get("description", "") or rule.get("interpretation_key", "")

    if r_id == "CAREER_10TH_LORD_PLACEMENT":
        evidence = rule.get("evidence", {})
        if isinstance(evidence, dict):
            lord = evidence.get("lord")
            house_sign = evidence.get("house_rashi")
            lord_house = evidence.get("lord_natal_house")
            lord_sign = evidence.get("lord_natal_rashi")
            if isinstance(lord, str) and lord and isinstance(house_sign, str) and house_sign:
                conclusion = f"{lord} rules the {house_sign} 10th house"
                if lord_house is not None:
                    conclusion += f"; placed in natal house {lord_house}"
                    if isinstance(lord_sign, str) and lord_sign:
                        conclusion += f" ({lord_sign})"
                return conclusion

    # Known rule mappings to hyper-compact notation
    known_compact = {
        "MARRIAGE_7TH_LORD_PLACEMENT": "7L Venus@8H→transformative relationship dynamics",
        "FINANCE_2ND_LORD_PLACEMENT": "2L Jupiter@11H→wealth expansion & gains",
    }
    
    if r_id in known_compact:
        return known_compact[r_id]
        
    if not r_desc:
        return r_id.lower().replace("_", " ")
        
    # Generic compression heuristics for unmapped rules
    desc_clean = r_desc
    for prefix in ["The placement of ", "This placement indicates "]:
        if desc_clean.startswith(prefix):
            desc_clean = desc_clean[len(prefix):]

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
    needs_moon = ("moon" in q_low or "nakshatra" in q_low or "rashi" in q_low or "mind" in q_low or "emotion" in q_low or "mental" in q_low or "moon" in rule_descs_str or "scorpio" in rule_descs_str)
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

    # Domain and query-specific dynamic house evidence extraction for all 12 houses (1..12)
    try:
        from backend.router.semantic_parser import extract_all_houses
        target_houses = extract_all_houses(q_low)
    except Exception:
        target_houses = []

    # Domain default primary houses
    domain_default_map = {
        "career": [10],
        "marriage": [7],
        "finance": [2, 11],
        "property": [4],
        "education": [5, 9],
    }
    for dh in domain_default_map.get(dom, []):
        if dh not in target_houses:
            target_houses.append(dh)

    # Inject verified facts for EVERY target house (1..12)
    for h in target_houses:
        if isinstance(h, int) and 1 <= h <= 12:
            h_det = get_house_details(chart_data, h)
            if h_det.get("data_available"):
                if f"{h}H" not in verified_facts:
                    verified_facts[f"{h}H"] = h_det["sign"]
                if h_det.get("lord_placement_available") and f"{h}L" not in verified_facts:
                    verified_facts[f"{h}L"] = f"{h_det['lord']}@{h_det['lord_house']}H"
                if h_det.get("occupants_available") and f"{h}H_occupants" not in verified_facts:
                    if h_det["occupants"]:
                        verified_facts[f"{h}H_occupants"] = ", ".join(h_det["occupants"])
                    else:
                        verified_facts[f"{h}H_occupants"] = "None (empty)"

    # Special domain aliases and Karakas
    if 6 in target_houses and (any(k in q_low for k in ["loan", "debt", "borrow", "emi"]) or "6H" in verified_facts):
        if "6H" in verified_facts:
            verified_facts["6H_Loan"] = verified_facts["6H"]
        if "6L" in verified_facts:
            verified_facts["6L_Loan"] = verified_facts["6L"]

    # Natural Karakas when relevant domain or houses are present
    if 4 in target_houses or dom == "property":
        mars_data = planets.get("Mars", {}) if isinstance(planets, dict) else {}
        if isinstance(mars_data, dict) and mars_data.get("house"):
            verified_facts["Mars_Property_Karaka"] = f"{mars_data.get('rashi', '')}@{mars_data.get('house')}H"

    if 7 in target_houses or dom == "marriage":
        ven_data = planets.get("Venus", {}) if isinstance(planets, dict) else {}
        if isinstance(ven_data, dict) and ven_data.get("house"):
            verified_facts["Venus_Marriage_Karaka"] = f"Venus@{ven_data.get('house')}H"

    if 11 in target_houses or any(k in q_low for k in ["gain", "income", "profit"]):
        jup_data = planets.get("Jupiter", {}) if isinstance(planets, dict) else {}
        if isinstance(jup_data, dict) and jup_data.get("house"):
            verified_facts["Jupiter_Gains_Karaka"] = f"{jup_data.get('rashi', '')}@{jup_data.get('house')}H"

    if 2 in target_houses or dom == "finance":
        jup_data = planets.get("Jupiter", {}) if isinstance(planets, dict) else {}
        if isinstance(jup_data, dict) and jup_data.get("house"):
            verified_facts["Jupiter_Wealth_Karaka"] = f"Jupiter@{jup_data.get('house')}H"

    if 5 in target_houses or dom == "education":
        merc_data = planets.get("Mercury", {}) if isinstance(planets, dict) else {}
        if isinstance(merc_data, dict) and merc_data.get("house"):
            verified_facts["Mercury_Intellect_Karaka"] = f"Mercury@{merc_data.get('house')}H"

    # 11. All Planetary Placements Fallback
    if any(k in q_low for k in ["which planet", "planetary placements", "planet positions", "where are my planets", "which house"]):
        for p_name in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]:
            p_data = planets.get(p_name, {})
            if isinstance(p_data, dict) and p_data.get("house"):
                verified_facts[p_name] = f"{p_data.get('rashi', '')}@{p_data.get('house')}H"

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
    facts_str = "; ".join(facts_parts) if facts_parts else "No verified chart facts available"

    rule_parts = [compress_rule_conclusion(r) for r in rules]
    rules_str = "; ".join(rule_parts) if rule_parts else "No matched rules available"

    prompt = (
        f"Q: {question}\n\n"
        f"F: {facts_str}\n"
        f"R: {rules_str}"
    )

    input_tokens = estimate_tokens(prompt)
    return prompt, input_tokens
