"""
Intent to Required Evidence Matrix Module
=========================================
Module: backend/reasoning/intent_evidence_matrix.py

Defines the authoritative INTENT_REQUIRED_EVIDENCE_MAP for all 23 intents,
distinguishing MANDATORY, OPTIONAL SUPPORTING, and TIMING evidence categories.

Provides the evidence evaluator: evaluate_intent_evidence_sufficiency()
supporting the 3 evidence states: SUFFICIENT, PARTIAL, UNRESOLVED, as well as
EVIDENCE_CONFLICT detection for router safety.
"""

from typing import Dict, Any, List, Optional, Tuple

INTENT_REQUIRED_EVIDENCE_MAP: Dict[str, Dict[str, Any]] = {
    "career_general": {
        "domain": "career",
        "mandatory": ["10th_house", "10th_lord"],
        "optional_supporting": ["career_karakas", "1st_house", "5th_house", "9th_house", "planetary_dignity"],
        "timing": [],
        "api_requirement": "planetary_positions",
        "reasoning_module": "career_rules",
        "locally_answerable_without_optional": True,
    },
    "career_business": {
        "domain": "career",
        "mandatory": ["10th_house", "10th_lord", "7th_house", "7th_lord"],
        "optional_supporting": ["mercury_karaka", "3rd_house", "11th_house", "planetary_dignity"],
        "timing": [],
        "api_requirement": "planetary_positions",
        "reasoning_module": "career_rules",
        "locally_answerable_without_optional": True,
    },
    "career_change": {
        "domain": "career",
        "mandatory": ["10th_house", "10th_lord", "5th_house", "9th_house"],
        "optional_supporting": ["8th_house", "career_karakas", "planetary_dignity"],
        "timing": ["mahadasha", "antardasha"],
        "api_requirement": "planetary_positions + dasha",
        "reasoning_module": "career_rules + dasha_timing",
        "locally_answerable_without_optional": True,
    },
    "career_interview": {
        "domain": "career",
        "mandatory": ["10th_house", "10th_lord", "6th_house", "6th_lord"],
        "optional_supporting": ["mercury_karaka", "sun_karaka", "3rd_house"],
        "timing": ["mahadasha", "antardasha"],
        "api_requirement": "planetary_positions + dasha",
        "reasoning_module": "career_rules + dasha_timing",
        "locally_answerable_without_optional": True,
    },
    "career_job": {
        "domain": "career",
        "mandatory": ["10th_house", "10th_lord", "6th_house", "6th_lord"],
        "optional_supporting": ["saturn_karaka", "sun_karaka", "planetary_dignity"],
        "timing": ["mahadasha", "antardasha"],
        "api_requirement": "planetary_positions + dasha",
        "reasoning_module": "career_rules + dasha_timing",
        "locally_answerable_without_optional": True,
    },
    "career_promotion": {
        "domain": "career",
        "mandatory": ["10th_house", "10th_lord", "11th_house", "11th_lord"],
        "optional_supporting": ["sun_karaka", "saturn_karaka", "planetary_dignity"],
        "timing": ["mahadasha", "antardasha", "jupiter_transit", "saturn_transit"],
        "api_requirement": "planetary_positions + dasha + transits",
        "reasoning_module": "career_rules + dasha_transit",
        "locally_answerable_without_optional": True,
    },
    "compatibility": {
        "domain": "marriage",
        "mandatory": ["7th_house", "7th_lord", "moon_sign", "venus_karaka"],
        "optional_supporting": ["1st_house", "5th_house", "jupiter_karaka", "ashtakoota_points"],
        "timing": [],
        "api_requirement": "planetary_positions + moon_nakshatra",
        "reasoning_module": "marriage_rules + relationships",
        "locally_answerable_without_optional": True,
    },
    "dasha": {
        "domain": "other",
        "mandatory": ["mahadasha", "antardasha"],
        "optional_supporting": ["paryantardasha", "dasha_lord_placement", "dasha_lord_dignity"],
        "timing": ["dasha_start_date", "dasha_end_date"],
        "api_requirement": "dasha_response",
        "reasoning_module": "dasha_timing",
        "locally_answerable_without_optional": True,
    },
    "definition": {
        "domain": "other",
        "mandatory": ["astrological_concept_terms"],
        "optional_supporting": [],
        "timing": [],
        "api_requirement": "faq_dictionary",
        "reasoning_module": "direct_fact_engine",
        "locally_answerable_without_optional": True,
    },
    "finance_general": {
        "domain": "finance",
        "mandatory": ["2nd_house", "2nd_lord", "11th_house", "11th_lord"],
        "optional_supporting": ["jupiter_karaka", "dhana_yogas", "9th_house", "planetary_dignity"],
        "timing": [],
        "api_requirement": "planetary_positions",
        "reasoning_module": "finance_rules",
        "locally_answerable_without_optional": True,
    },
    "financial_stability": {
        "domain": "finance",
        "mandatory": ["2nd_house", "2nd_lord", "11th_house", "11th_lord"],
        "optional_supporting": ["8th_house", "12th_house", "jupiter_karaka", "planetary_dignity"],
        "timing": ["mahadasha", "antardasha"],
        "api_requirement": "planetary_positions + dasha",
        "reasoning_module": "finance_rules + dasha_timing",
        "locally_answerable_without_optional": True,
    },
    "health_general": {
        "domain": "health",
        "mandatory": ["1st_house", "1st_lord", "6th_house", "6th_lord"],
        "optional_supporting": ["sun_karaka", "moon_karaka", "8th_house", "planetary_dignity"],
        "timing": [],
        "api_requirement": "planetary_positions",
        "reasoning_module": "health_rules",
        "locally_answerable_without_optional": True,
    },
    "health_period": {
        "domain": "health",
        "mandatory": ["1st_house", "1st_lord", "6th_house", "6th_lord"],
        "optional_supporting": ["8th_house", "12th_house", "saturn_karaka", "planetary_dignity"],
        "timing": ["mahadasha", "antardasha"],
        "api_requirement": "planetary_positions + dasha",
        "reasoning_module": "health_rules + dasha_timing",
        "locally_answerable_without_optional": True,
    },
    "income": {
        "domain": "finance",
        "mandatory": ["11th_house", "11th_lord", "2nd_house", "2nd_lord"],
        "optional_supporting": ["mercury_karaka", "jupiter_karaka", "planetary_dignity"],
        "timing": ["mahadasha", "antardasha"],
        "api_requirement": "planetary_positions + dasha",
        "reasoning_module": "finance_rules + dasha_timing",
        "locally_answerable_without_optional": True,
    },
    "life_overview": {
        "domain": "other",
        "mandatory": ["ascendant", "1st_lord", "moon_sign", "sun_sign"],
        "optional_supporting": ["10th_house", "7th_house", "2nd_house", "mahadasha"],
        "timing": [],
        "api_requirement": "planetary_positions",
        "reasoning_module": "generic_rules",
        "locally_answerable_without_optional": True,
    },
    "love_marriage": {
        "domain": "marriage",
        "mandatory": ["5th_house", "5th_lord", "7th_house", "7th_lord"],
        "optional_supporting": ["venus_karaka", "mars_karaka", "planetary_dignity"],
        "timing": [],
        "api_requirement": "planetary_positions",
        "reasoning_module": "marriage_rules",
        "locally_answerable_without_optional": True,
    },
    "marriage_timing": {
        "domain": "marriage",
        "mandatory": ["7th_house", "7th_lord", "venus_karaka", "jupiter_karaka"],
        "optional_supporting": ["2nd_house", "1st_house", "planetary_dignity"],
        "timing": ["mahadasha", "antardasha", "jupiter_transit", "saturn_transit"],
        "api_requirement": "planetary_positions + dasha + transits",
        "reasoning_module": "marriage_rules + dasha_transit",
        "locally_answerable_without_optional": True,
    },
    "married_life": {
        "domain": "marriage",
        "mandatory": ["7th_house", "7th_lord", "2nd_house", "venus_karaka"],
        "optional_supporting": ["8th_house", "jupiter_karaka", "planetary_dignity"],
        "timing": [],
        "api_requirement": "planetary_positions",
        "reasoning_module": "marriage_rules",
        "locally_answerable_without_optional": True,
    },
    "multi_domain": {
        "domain": "multi_domain",
        "mandatory": ["sub_intent_1_mandatory", "sub_intent_2_mandatory"],
        "optional_supporting": ["sub_intent_1_supporting", "sub_intent_2_supporting"],
        "timing": ["mahadasha", "antardasha"],
        "api_requirement": "planetary_positions + dasha",
        "reasoning_module": "multi_domain",
        "locally_answerable_without_optional": True,
    },
    "personality": {
        "domain": "other",
        "mandatory": ["ascendant", "1st_lord", "moon_sign"],
        "optional_supporting": ["sun_sign", "mercury_placement", "planetary_aspects_on_1st"],
        "timing": [],
        "api_requirement": "planetary_positions",
        "reasoning_module": "generic_rules",
        "locally_answerable_without_optional": True,
    },
    "relationship": {
        "domain": "marriage",
        "mandatory": ["5th_house", "5th_lord", "venus_karaka"],
        "optional_supporting": ["7th_house", "mars_karaka", "moon_karaka"],
        "timing": [],
        "api_requirement": "planetary_positions",
        "reasoning_module": "relationships",
        "locally_answerable_without_optional": True,
    },
    "wealth": {
        "domain": "finance",
        "mandatory": ["2nd_house", "2nd_lord", "11th_house", "11th_lord", "9th_house", "9th_lord"],
        "optional_supporting": ["jupiter_karaka", "dhana_yogas", "laxmi_yogas"],
        "timing": [],
        "api_requirement": "planetary_positions",
        "reasoning_module": "finance_rules",
        "locally_answerable_without_optional": True,
    },
    "wellness": {
        "domain": "health",
        "mandatory": ["1st_house", "1st_lord", "moon_sign", "moon_karaka"],
        "optional_supporting": ["6th_house", "jupiter_karaka", "sun_karaka"],
        "timing": [],
        "api_requirement": "planetary_positions",
        "reasoning_module": "health_rules",
        "locally_answerable_without_optional": True,
    },
}


def evaluate_intent_evidence_sufficiency(
    intent: str,
    domain: str,
    normalized_chart: Dict[str, Any],
    matched_rules: Optional[List[Dict[str, Any]]] = None,
    question: str = ""
) -> Dict[str, Any]:
    """
    Evaluates evidence completeness for a given intent against normalized chart fields and Stage 8 matched rules.
    Returns structured audit payload including evidence state (SUFFICIENT, PARTIAL, UNRESOLVED)
    and conflict detection (EVIDENCE_CONFLICT).
    """
    q_lower = (question or "").lower().strip()
    unsupported_kws = ["past life", "previous life", "lottery", "gambling", "sports prediction", "sports match", "stock picking", "crypto prediction"]
    if any(kw in q_lower for kw in unsupported_kws):
        return {
            "intent": intent,
            "domain": domain,
            "expected_domain": "unsupported",
            "evidence_state": "UNRESOLVED",
            "is_conflict": True,
            "conflict_reason": "Query requests insights outside Vedic astrology chart boundaries (Unsupported / OOD).",
            "mandatory_complete": False,
            "mandatory_required": [],
            "available_evidence": [],
            "missing_evidence": ["unsupported_query_boundary"],
            "timing_required": [],
            "api_requirement": "none",
            "reasoning_module": "none"
        }

    spec = INTENT_REQUIRED_EVIDENCE_MAP.get(intent)
    if not spec:
        return {
            "intent": intent,
            "domain": domain,
            "evidence_state": "UNRESOLVED",
            "is_conflict": True,
            "conflict_reason": f"Intent '{intent}' is not registered in INTENT_REQUIRED_EVIDENCE_MAP.",
            "mandatory_complete": False,
            "available_evidence": [],
            "missing_evidence": ["unregistered_intent"],
            "reason": f"Intent '{intent}' is not registered in INTENT_REQUIRED_EVIDENCE_MAP."
        }

    expected_domain = spec["domain"]
    
    # Step 7B: Router Safety & Domain Mismatch / Conflict Detection
    is_conflict = False
    conflict_reason = None
    if domain != "multi_domain" and expected_domain != "other" and domain != expected_domain:
        # Check if chart evidence strongly supports expected_domain rather than predicted domain
        is_conflict = True
        conflict_reason = f"Router predicted domain '{domain}' but intent '{intent}' requires domain '{expected_domain}'."

    planets = normalized_chart.get("planets", {})
    ascendant = normalized_chart.get("ascendant", {})
    dasha_info = normalized_chart.get("dasha_hierarchy", {})

    available_evidence = []
    missing_evidence = []

    # 1. Check Mandatory Fields
    for req in spec["mandatory"]:
        has_field = False
        if req in ["10th_house", "10th_lord", "1st_house", "1st_lord", "7th_house", "7th_lord", "2nd_house", "2nd_lord", "11th_house", "11th_lord", "6th_house", "6th_lord", "5th_house", "5th_lord", "9th_house", "9th_lord"]:
            # Chart planets and ascendant present
            has_field = bool(planets and ascendant and ascendant.get("rashi"))
        elif req in ["moon_sign", "moon_karaka"]:
            has_field = "Moon" in planets
        elif req in ["venus_karaka"]:
            has_field = "Venus" in planets
        elif req in ["ascendant", "sun_sign"]:
            has_field = bool(ascendant and ascendant.get("rashi") and "Sun" in planets)
        elif req in ["mahadasha", "antardasha"]:
            has_field = bool(dasha_info or normalized_chart.get("current_dasha"))
        elif req in ["astrological_concept_terms", "sub_intent_1_mandatory"]:
            has_field = True

        if has_field:
            available_evidence.append(req)
        else:
            missing_evidence.append(req)

    # 2. Check Optional Supporting Fields
    for opt in spec["optional_supporting"]:
        has_opt = False
        if opt in ["career_karakas", "saturn_karaka"]:
            has_opt = "Saturn" in planets
        elif opt in ["jupiter_karaka", "sun_karaka", "mercury_karaka", "mars_karaka"]:
            has_opt = any(p in planets for p in ["Jupiter", "Sun", "Mercury", "Mars"])
        elif opt == "planetary_dignity":
            has_opt = bool(planets)
        elif opt == "ashtakoota_points":
            has_opt = False  # Optional supporting field requiring dual-chart compatibility payload

        if has_opt:
            available_evidence.append(opt)

    mandatory_complete = len(missing_evidence) == 0

    # Step 7 State Classification
    if mandatory_complete:
        if spec.get("locally_answerable_without_optional", True):
            evidence_state = "SUFFICIENT"
        else:
            evidence_state = "PARTIAL"
    elif available_evidence:
        evidence_state = "PARTIAL"
    else:
        evidence_state = "UNRESOLVED"

    return {
        "intent": intent,
        "domain": domain,
        "expected_domain": expected_domain,
        "evidence_state": evidence_state,
        "is_conflict": is_conflict,
        "conflict_reason": conflict_reason,
        "mandatory_complete": mandatory_complete,
        "mandatory_required": spec["mandatory"],
        "available_evidence": available_evidence,
        "missing_evidence": missing_evidence,
        "timing_required": spec["timing"],
        "api_requirement": spec["api_requirement"],
        "reasoning_module": spec["reasoning_module"],
    }
