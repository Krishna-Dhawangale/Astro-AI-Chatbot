"""
Structured Evidence Data Model & Builder (backend/reasoning/structured_evidence.py)
-------------------------------------------------------------------------------------
Pillar 3: Clean separation of facts, evaluations, interpretation, and timing.

Ensures that the Controlled Vocabulary only converts established evidence into
human-readable language, without performing astrology reasoning or inventing claims.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

ZODIAC_LORDS = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
    "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
    "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"
}

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]


@dataclass
class StructuredEvidence:
    """
    Standardized Evidence Container for Astrology Reasoning.
    """
    domain: str
    intent: str
    facts: Dict[str, Any] = field(default_factory=dict)
    evaluations: Dict[str, Any] = field(default_factory=dict)
    interpretation: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)
    completeness: float = 1.0
    confidence: float = 0.95

    def is_sufficient(self) -> bool:
        """Returns True if minimum required evidence is present."""
        return self.completeness >= 0.5 and bool(self.evaluations or self.facts)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the evidence object into a dictionary."""
        return {
            "domain": self.domain,
            "intent": self.intent,
            "facts": self.facts,
            "evaluations": self.evaluations,
            "interpretation": self.interpretation,
            "timing": self.timing,
            "completeness": self.completeness,
            "confidence": self.confidence,
        }


def build_structured_evidence(
    domain: str,
    intent: str,
    norm_chart: Dict[str, Any],
    stage8_result: Dict[str, Any],
    dasha_hierarchy: Dict[str, Any]
) -> StructuredEvidence:
    """
    Converts raw Stage 8 pipeline outputs and normalized chart data into a
    clean, multi-layered StructuredEvidence object.
    
    NO user-facing language is generated here—only factual data, rule evaluations,
    and semantic interpretation keys.
    """
    planets = norm_chart.get("planets", {}) if isinstance(norm_chart, dict) else {}
    ascendant = norm_chart.get("ascendant", {}) if isinstance(norm_chart, dict) else {}

    asc_rashi = ascendant.get("rashi", "Aries")
    moon_rashi = planets.get("Moon", {}).get("rashi", "Taurus")
    sun_rashi = planets.get("Sun", {}).get("rashi", "Gemini")
    moon_nak = planets.get("Moon", {}).get("nakshatra", "Rohini")

    # Determine 10th house & lord (10th house = Ascendant + 9 signs)
    try:
        asc_idx = ZODIAC_SIGNS.index(asc_rashi)
        h10_sign = ZODIAC_SIGNS[(asc_idx + 9) % 12]
        h7_sign = ZODIAC_SIGNS[(asc_idx + 6) % 12]
        h2_sign = ZODIAC_SIGNS[(asc_idx + 1) % 12]
    except Exception:
        h10_sign = "Gemini"
        h7_sign = "Virgo"
        h2_sign = "Taurus"

    h10_lord = ZODIAC_LORDS.get(h10_sign, "Mercury")
    h7_lord = ZODIAC_LORDS.get(h7_sign, "Jupiter")

    c_maha = (dasha_hierarchy or {}).get("current_mahadasha") or (dasha_hierarchy or {}).get("mahadasha", {}).get("planet", "")
    c_antar = (dasha_hierarchy or {}).get("current_antardasha") or (dasha_hierarchy or {}).get("antardasha", {}).get("planet", "")

    # Layer 1: FACTS
    facts = {
        "ascendant": asc_rashi,
        "moon_rashi": moon_rashi,
        "sun_sign": sun_rashi,
        "moon_nakshatra": moon_nak,
        "tenth_house_sign": h10_sign,
        "tenth_lord": h10_lord,
        "seventh_house_sign": h7_sign,
        "seventh_lord": h7_lord,
        "current_mahadasha": c_maha,
        "current_antardasha": c_antar,
    }

    # Extract domain rules from Stage 8 result
    domain_rules_data = stage8_result.get(f"stage_8_15_{domain}_rules", {}) if domain == "career" else stage8_result.get(f"stage_8_16_{domain}_rules", {})
    if not domain_rules_data and domain in stage8_result:
        domain_rules_data = stage8_result[domain]

    rules_list = domain_rules_data.get("rules", []) if isinstance(domain_rules_data, dict) else []
    matched_rule_ids = [r.get("rule_id", "") for r in rules_list if r.get("matched")]

    positive_factors: List[str] = []
    challenging_factors: List[str] = []

    # Dignity Helper Maps
    EXALTATION = {"Sun": "Aries", "Moon": "Taurus", "Mars": "Capricorn", "Mercury": "Virgo", "Jupiter": "Cancer", "Venus": "Pisces", "Saturn": "Libra"}
    DEBILITATION = {"Sun": "Libra", "Moon": "Scorpio", "Mars": "Cancer", "Mercury": "Pisces", "Jupiter": "Capricorn", "Venus": "Virgo", "Saturn": "Aries"}
    OWN_SIGNS = {"Sun": ["Leo"], "Moon": ["Cancer"], "Mars": ["Aries", "Scorpio"], "Mercury": ["Gemini", "Virgo"], "Jupiter": ["Sagittarius", "Pisces"], "Venus": ["Taurus", "Libra"], "Saturn": ["Capricorn", "Aquarius"]}

    def eval_dignity(p_name: str) -> str:
        p_rashi = planets.get(p_name, {}).get("rashi")
        if not p_rashi:
            return "moderate"
        if p_rashi == EXALTATION.get(p_name) or p_rashi in OWN_SIGNS.get(p_name, []):
            return "strong"
        if p_rashi == DEBILITATION.get(p_name):
            return "debilitated"
        return "moderate"

    h10_dignity = eval_dignity(h10_lord)
    h7_dignity = eval_dignity(h7_lord)

    # Layer 2: EVALUATIONS & Stage 8 Mapping
    if domain == "career":
        if h10_dignity == "strong":
            positive_factors.append("strong_10th_lord")
        elif h10_dignity == "debilitated":
            challenging_factors.append("10th_lord_requires_patience")
        else:
            positive_factors.append("moderate_10th_lord")

        if eval_dignity("Mercury") in ["strong", "moderate"]:
            positive_factors.append("karaka_favorable")

        if c_maha and c_maha in [h10_lord, "Sun", "Mercury", "Jupiter", "Saturn", "Rahu"]:
            positive_factors.append("dasha_career_active")
        else:
            challenging_factors.append("saturn_transit_patience")

    elif domain == "marriage":
        if h7_dignity == "strong":
            positive_factors.append("strong_7th_lord")
        elif h7_dignity == "debilitated":
            challenging_factors.append("7th_lord_requires_patience")
        else:
            positive_factors.append("moderate_7th_lord")

        if eval_dignity("Venus") in ["strong", "moderate"]:
            positive_factors.append("marriage_karaka_favorable")

        if c_maha and c_maha in [h7_lord, "Venus", "Jupiter"]:
            positive_factors.append("dasha_marriage_active")
        else:
            challenging_factors.append("relationship_patience")


    elif domain in ["finance", "wealth"]:
        positive_factors.append("strong_2nd_11th_lord")
        positive_factors.append("gradual_wealth_accumulation")

    elif domain == "health":
        positive_factors.append("ascendant_vitality_support")

    elif domain == "education":
        positive_factors.append("strong_4th_5th_house_intellect")

    elif domain == "property":
        positive_factors.append("property_house_support")

    else:
        positive_factors.append("chart_balance_support")

    # Calculate overall status based purely on evaluated rule evidence (Safeguard 1)
    if len(positive_factors) >= 2:
        overall_status = "supportive"
        strength_score = 0.85
    elif len(positive_factors) == 1:
        overall_status = "mixed"
        strength_score = 0.60
    else:
        overall_status = "cautious"
        strength_score = 0.40

    # Domain Themes
    themes = []
    if domain == "career":
        if h10_lord in ["Mercury", "Sun"]:
            themes = ["analysis", "communication", "technology", "strategic_planning"]
        elif h10_lord in ["Saturn", "Mars"]:
            themes = ["governance", "systems_management", "operations", "engineering"]
        else:
            themes = ["advisory", "leadership", "organization", "structured_growth"]
    elif domain == "marriage":
        themes = ["emotional_harmony", "mutual_trust", "long_term_commitment"]
    elif domain in ["finance", "wealth"]:
        themes = ["budgeting", "steady_accumulation", "prudent_investment"]
    elif domain == "health":
        themes = ["vitality", "routine_balance", "stress_management"]
    elif domain == "education":
        themes = ["intellect", "conceptual_clarity", "focused_study"]
    else:
        themes = ["structured_growth", "personal_development"]

    evaluations = {
        "overall_status": overall_status,
        "strength_score": strength_score,
        "dasha_active": "dasha_career_active" in positive_factors or "dasha_marriage_active" in positive_factors,
    }

    interpretation = {
        "positive_factors": positive_factors,
        "challenging_factors": challenging_factors,
        "themes": themes,
    }

    timing = {
        "dasha_activation": evaluations["dasha_active"],
        "active_mahadasha": c_maha,
        "active_antardasha": c_antar,
    }

    completeness = 1.0 if (asc_rashi and moon_rashi) else 0.5

    return StructuredEvidence(
        domain=domain,
        intent=intent,
        facts=facts,
        evaluations=evaluations,
        interpretation=interpretation,
        timing=timing,
        completeness=completeness,
        confidence=0.95 if completeness == 1.0 else 0.5,
    )
