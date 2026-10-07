"""
Question Registry & Strategy Contract (backend/reasoning/question_registry.py)
--------------------------------------------------------------------------------
Pillar 3: Question Registry = Evidence / Strategy Contract

Defines required evidence, Stage 8 rules, and default answer strategies for
known astrology question intents. Enables dynamic answer synthesis without hardcoded answers.
"""

from typing import Dict, Any, List, Optional


QUESTION_STRATEGIES: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Direct Astrology Facts (Pillar 1 Authority: FreeAstrologyAPI)
    # ------------------------------------------------------------------
    "nakshatra_lookup": {
        "required_evidence": ["moon_nakshatra"],
        "rules": [],
        "answer_strategy": "DIRECT_API",
        "domain": "astrology_fact"
    },
    "moon_rashi_lookup": {
        "required_evidence": ["moon_rashi"],
        "rules": [],
        "answer_strategy": "DIRECT_API",
        "domain": "astrology_fact"
    },
    "sun_sign_lookup": {
        "required_evidence": ["sun_rashi"],
        "rules": [],
        "answer_strategy": "DIRECT_API",
        "domain": "astrology_fact"
    },
    "lagna_lookup": {
        "required_evidence": ["ascendant_rashi"],
        "rules": [],
        "answer_strategy": "DIRECT_API",
        "domain": "astrology_fact"
    },
    "zodiac_lookup": {
        "required_evidence": ["moon_rashi", "sun_rashi"],
        "rules": [],
        "answer_strategy": "DIRECT_API",
        "domain": "astrology_fact"
    },
    "current_dasha_lookup": {
        "required_evidence": ["mahadasha", "antardasha"],
        "rules": [],
        "answer_strategy": "DIRECT_API",
        "domain": "astrology_fact"
    },
    "moon_longitude_lookup": {
        "required_evidence": ["moon_longitude"],
        "rules": [],
        "answer_strategy": "DIRECT_API",
        "domain": "astrology_fact"
    },

    # ------------------------------------------------------------------
    # Single Domain Rules (Pillar 2: Stage 8 Deterministic Reasoning)
    # ------------------------------------------------------------------
    "career_general": {
        "required_evidence": ["10th_house", "10th_lord", "career_karakas"],
        "rules": ["CAREER_10TH_LORD_PLACEMENT", "CAREER_KARAKA_DIGNITY", "CAREER_KARAKA_PRESENCE"],
        "answer_strategy": "LOCAL_RULE_SYNTHESIS",
        "domain": "career"
    },
    "career_timing": {
        "required_evidence": ["10th_house", "mahadasha", "antardasha"],
        "rules": ["CAREER_DASHA_ACTIVATION", "CAREER_TRANSIT_ACTIVATION"],
        "answer_strategy": "LOCAL_TIMING_SYNTHESIS",
        "domain": "career"
    },
    "marriage_general": {
        "required_evidence": ["7th_house", "7th_lord", "venus"],
        "rules": ["MARRIAGE_7TH_LORD_PLACEMENT", "MARRIAGE_KARAKA_DIGNITY", "MARRIAGE_KARAKA_PRESENCE"],
        "answer_strategy": "LOCAL_RULE_SYNTHESIS",
        "domain": "marriage"
    },
    "marriage_timing": {
        "required_evidence": ["7th_house", "mahadasha", "antardasha"],
        "rules": ["MARRIAGE_DASHA_ACTIVATION", "MARRIAGE_TRANSIT_ACTIVATION"],
        "answer_strategy": "LOCAL_TIMING_SYNTHESIS",
        "domain": "marriage"
    },
    "finance_general": {
        "required_evidence": ["2nd_house", "11th_house"],
        "rules": ["FINANCE_2ND_LORD_PLACEMENT", "FINANCE_11TH_LORD_PLACEMENT"],
        "answer_strategy": "LOCAL_RULE_SYNTHESIS",
        "domain": "finance"
    },
    "health_general": {
        "required_evidence": ["ascendant_rashi", "1st_house"],
        "rules": [],
        "answer_strategy": "LOCAL_RULE_SYNTHESIS",
        "domain": "health"
    },
    "education_general": {
        "required_evidence": ["4th_house", "5th_house"],
        "rules": ["EDUCATION_4TH_LORD_PLACEMENT", "EDUCATION_5TH_LORD_PLACEMENT"],
        "answer_strategy": "LOCAL_RULE_SYNTHESIS",
        "domain": "education"
    },

    # ------------------------------------------------------------------
    # Multi-Domain Combinations
    # ------------------------------------------------------------------
    "dasha_career_interaction": {
        "required_evidence": ["mahadasha", "antardasha", "10th_house"],
        "rules": ["CAREER_DASHA_ACTIVATION"],
        "answer_strategy": "MULTI_DOMAIN_LOCAL_SYNTHESIS",
        "domain": "career"
    },
    "multi_domain": {
        "required_evidence": ["mahadasha", "antardasha", "10th_house"],
        "rules": ["MULTI_DOMAIN_RULE_MERGER"],
        "answer_strategy": "MULTI_DOMAIN_LOCAL_SYNTHESIS",
        "domain": "career"
    }
}


def get_question_strategy(intent: str, question: str = "") -> Dict[str, Any]:
    """
    Returns the strategy contract dict for a given intent or question.
    """
    q_lower = (question or "").lower().strip()

    # Pattern overrides based on question phrasing
    if any(k in q_lower for k in ["business or job", "job or business", "business vs job", "job vs business"]):
        return {
            "required_evidence": ["10th_house", "career_karakas"],
            "rules": ["CAREER_10TH_LORD_PLACEMENT"],
            "answer_strategy": "LOCAL_RULE_SYNTHESIS",
            "domain": "career"
        }

    if intent in QUESTION_STRATEGIES:
        return QUESTION_STRATEGIES[intent]

    # Default fallback strategy definition
    return {
        "required_evidence": ["ascendant_rashi", "moon_rashi"],
        "rules": [],
        "answer_strategy": "LOCAL_RULE_SYNTHESIS",
        "domain": "general"
    }
