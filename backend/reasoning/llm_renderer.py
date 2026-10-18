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
    "You are an expert Vedic astrology synthesizer. "
    "Treat supplied chart facts as authoritative; never invent or alter planetary placements, houses, or Dasha periods. "
    "Apply established Vedic principles to interpret verified facts and answer the user's question directly. "
    "Use cautious language; never guarantee predictions. "
    "Do not mention internal rules or system details. "
    "Be concise: 40–65 words. "
    "Return only the answer."
)


def get_renderer_system_instruction(word_limit_str: str = "40–65 words") -> str:
    """
    Returns an ultra-lean system instruction with customized word limit bounds.
    Permits grounded Vedic reasoning on supplied facts while strictly prohibiting invented chart calculations.
    """
    return (
        "You are an expert Vedic astrology synthesizer. "
        "Treat supplied chart facts as authoritative; never invent or alter planetary placements, houses, or Dasha periods. "
        "Apply established Vedic principles to interpret the verified facts and answer the user's question directly. "
        "If the question involves multiple factors or comparative choices, synthesize them rather than repeating isolated placements. "
        "If evidence is partial, provide a helpful qualified interpretation and disclose limitations naturally without repeating generic disclaimers. "
        "For wellness or health queries, provide astrological perspectives only, never medical diagnoses. "
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

    # Domain and query-specific dynamic house evidence extraction
    # 1. Career (10H)
    if dom == "career" or any(k in q_low for k in ["career", "job", "profession", "work", "working", "10th house", "10th"]):
        tenth_lord_rule = next(
            (
                rule for rule in matched_rules
                if rule.get("rule_id") == "CAREER_10TH_LORD_PLACEMENT"
                and rule.get("matched", True)
            ),
            None,
        )
        rule_evidence = tenth_lord_rule.get("evidence", {}) if tenth_lord_rule else {}
        if isinstance(rule_evidence, dict):
            house_sign = rule_evidence.get("house_rashi")
            lord = rule_evidence.get("lord")
            if isinstance(house_sign, str) and house_sign and isinstance(lord, str) and lord:
                verified_facts["10H"] = house_sign
                verified_facts["10L"] = lord
                lord_house = rule_evidence.get("lord_natal_house")
                if lord_house is not None:
                    verified_facts["10L_placement"] = f"{lord_house}H"
                    lord_sign = rule_evidence.get("lord_natal_rashi")
                    if isinstance(lord_sign, str) and lord_sign:
                        verified_facts["10L_placement"] += f" ({lord_sign})"
        if "10H" not in verified_facts:
            h10 = get_house_details(chart_data, 10)
            if h10["data_available"]:
                verified_facts["10H"] = h10["sign"]
                if h10["lord_placement_available"]:
                    verified_facts["10L"] = f"{h10['lord']}@{h10['lord_house']}H"
                if h10["occupants"]:
                    verified_facts["10H_occupants"] = ", ".join(h10["occupants"])

    # 2. Property & Conveyance (4H)
    is_prop_query = dom == "property" or any(k in q_low for k in ["property", "real estate", "house purchase", "buying a house", "purchasing a house", "buy a house", "purchase a house", "land", "flat", "plot", "conveyance", "vehicle", "4th house", "4h"])
    if is_prop_query:
        h4 = get_house_details(chart_data, 4)
        if h4["data_available"]:
            verified_facts["4H"] = h4["sign"]
            if h4["lord_placement_available"]:
                verified_facts["4L"] = f"{h4['lord']}@{h4['lord_house']}H"
            if h4["occupants"]:
                verified_facts["4H_occupants"] = ", ".join(h4["occupants"])
        mars_data = planets.get("Mars", {}) if isinstance(planets, dict) else {}
        if isinstance(mars_data, dict) and mars_data.get("house"):
            verified_facts["Mars_Property_Karaka"] = f"{mars_data.get('rashi', '')}@{mars_data.get('house')}H"


    # 3. Debt, Loans & Financial Obstacles (6H)
    is_loan_query = any(k in q_low for k in ["loan", "debt", "emi", "mortgage", "borrow", "karz", "6th house", "6h"])
    if is_loan_query:
        h6 = get_house_details(chart_data, 6)
        if h6["data_available"]:
            verified_facts["6H_Loan"] = h6["sign"]
            if h6["lord_placement_available"]:
                verified_facts["6L_Loan"] = f"{h6['lord']}@{h6['lord_house']}H"
            if h6["occupants"]:
                verified_facts["6H_occupants"] = ", ".join(h6["occupants"])

    # 4. Higher Education, Study Abroad, Fortune & Long Travel (9H)
    is_nine_query = any(k in q_low for k in ["higher education", "study abroad", "studying abroad", "travel", "father", "fortune", "dharma", "9th house", "9h"])
    if is_nine_query:
        h9 = get_house_details(chart_data, 9)
        if h9["data_available"]:
            verified_facts["9H"] = h9["sign"]
            if h9["lord_placement_available"]:
                verified_facts["9L"] = f"{h9['lord']}@{h9['lord_house']}H"
            if h9["occupants"]:
                verified_facts["9H_occupants"] = ", ".join(h9["occupants"])

    # 5. Foreign Settlement, Relocation & Sleep (12H)
    is_twelve_query = any(k in q_low for k in ["foreign", "abroad", "relocat", "overseas", "country", "another country", "sleep", "neend", "insomnia", "restlessness", "expense", "12th house", "12h"]) or bool(re.search(r"\b12\s*th\s*house\b", q_low))
    if is_twelve_query:
        h12 = get_house_details(chart_data, 12)
        if h12["data_available"]:
            verified_facts["12H"] = h12["sign"]
            if h12["lord_placement_available"]:
                verified_facts["12L"] = f"{h12['lord']}@{h12['lord_house']}H"
            if h12["occupants"]:
                verified_facts["12H_occupants"] = ", ".join(h12["occupants"])

    # 6. Financial Gains & Income (11H)
    is_eleven_query = any(k in q_low for k in ["gain", "income", "profit", "wealth expansion", "11th house", "11h"])
    if is_eleven_query:
        h11 = get_house_details(chart_data, 11)
        if h11["data_available"]:
            verified_facts["11H"] = h11["sign"]
            if h11["lord_placement_available"]:
                verified_facts["11L"] = f"{h11['lord']}@{h11['lord_house']}H"
            if h11["occupants"]:
                verified_facts["11H_occupants"] = ", ".join(h11["occupants"])
        jup_data = planets.get("Jupiter", {}) if isinstance(planets, dict) else {}
        if isinstance(jup_data, dict) and jup_data.get("house"):
            verified_facts["Jupiter_Gains_Karaka"] = f"{jup_data.get('rashi', '')}@{jup_data.get('house')}H"

    # 7. Marriage & Relationships (7H)
    is_marriage_query = dom == "marriage" or any(k in q_low for k in ["marriage", "relationship", "spouse", "partner", "love life", "7th house", "7h"])
    if is_marriage_query:
        h7 = get_house_details(chart_data, 7)
        if h7["data_available"]:
            verified_facts["7H"] = h7["sign"]
            if h7["lord_placement_available"]:
                verified_facts["7L"] = f"{h7['lord']}@{h7['lord_house']}H"
            if h7["occupants"]:
                verified_facts["7H_occupants"] = ", ".join(h7["occupants"])
        ven_data = planets.get("Venus", {}) if isinstance(planets, dict) else {}
        if isinstance(ven_data, dict) and ven_data.get("house"):
            verified_facts["Venus_Marriage_Karaka"] = f"Venus@{ven_data.get('house')}H"

    # 8. Wealth & Savings (2H)
    is_finance_query = dom == "finance" or any(k in q_low for k in ["wealth", "saving", "finance", "money", "2nd house", "2h"])
    if is_finance_query:
        h2 = get_house_details(chart_data, 2)
        if h2["data_available"]:
            verified_facts["2H"] = h2["sign"]
            if h2["lord_placement_available"]:
                verified_facts["2L"] = f"{h2['lord']}@{h2['lord_house']}H"
            if h2["occupants"]:
                verified_facts["2H_occupants"] = ", ".join(h2["occupants"])
        jup_data = planets.get("Jupiter", {}) if isinstance(planets, dict) else {}
        if isinstance(jup_data, dict) and jup_data.get("house"):
            verified_facts["Jupiter_Wealth_Karaka"] = f"Jupiter@{jup_data.get('house')}H"

    # 9. Intellect & Education (5H)
    is_five_query = dom == "education" or any(k in q_low for k in ["intellect", "buddhi", "memory", "discernment", "speculation", "children", "5th house", "5h", "5th-house"]) or bool(re.search(r"\b5\s*th\s*house\b", q_low))
    if is_five_query:
        h5 = get_house_details(chart_data, 5)
        if h5["data_available"]:
            verified_facts["5H"] = h5["sign"]
            if h5["lord_placement_available"]:
                verified_facts["5L"] = f"{h5['lord']}@{h5['lord_house']}H"
            if h5["occupants"]:
                verified_facts["5H_occupants"] = ", ".join(h5["occupants"])

    # 10. Direct Numbered House Fallback (for any other house 1..12 e.g. 1H, 3H, 8H)
    m_house = re.search(r"\b(?:what does my|what is my|in my|tell me about my)?\s*(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d{1,2}\s*(?:st|nd|rd|th)?)\s*house\b", q_low)
    if m_house:
        from backend.reasoning.direct_fact_engine import WORD_TO_HOUSE
        h_num = WORD_TO_HOUSE.get(re.sub(r"\s+", "", m_house.group(1).lower()))
        if h_num and f"{h_num}H" not in verified_facts:
            hn = get_house_details(chart_data, h_num)
            if hn["data_available"]:
                verified_facts[f"{h_num}H"] = hn["sign"]
                if hn["lord_placement_available"]:
                    verified_facts[f"{h_num}L"] = f"{hn['lord']}@{hn['lord_house']}H"
                if hn["occupants"]:
                    verified_facts[f"{h_num}H_occupants"] = ", ".join(hn["occupants"])

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
