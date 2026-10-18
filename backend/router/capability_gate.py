# -*- coding: utf-8 -*-
"""
backend/router/capability_gate.py
---------------------------------
Evaluates backend capability and determines the authoritative execution path
for user astrological inquiries across all 12 houses (1..12).

Routing Logic:
- Complete factual lookup & single-fact inquiries -> LOCAL_DIRECT_FACT (0 LLM Tokens)
- Exact Stage 8 deterministic rule matches -> LOCAL_RULE_SYNTHESIS (0 LLM Tokens)
- Compound queries (e.g., 4H+6H or 8H+12H), predictive queries, or advisory questions -> LLM_ASSISTED
- Timing prediction / window queries -> TIMING_ENGINE
- Missing chart evidence -> UNRESOLVED
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from backend.reasoning.lordship import get_house_details, RASHI_LORDS, CORE_PLANET_NAMES
from backend.router.semantic_parser import QueryIntentProfile, AstrologicalIntent, extract_all_houses


class ExecutionPath(str, Enum):
    LOCAL_DIRECT_FACT = "LOCAL_DIRECT_FACT"
    LOCAL_RULE_SYNTHESIS = "LOCAL_RULE_SYNTHESIS"
    LLM_ASSISTED = "LLM_ASSISTED"
    TIMING_ENGINE = "TIMING_ENGINE"
    UNRESOLVED = "UNRESOLVED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


def ordinal_house(n: int) -> str:
    """Returns the proper English ordinal representation of a house number (e.g. 1st, 2nd, 3rd, 10th)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


@dataclass
class CapabilityDecision:
    path: ExecutionPath
    intent_profile: QueryIntentProfile
    confidence: float
    reason: str
    direct_answer: Optional[Dict[str, Any]] = None
    target_houses: List[int] = field(default_factory=list)
    required_evidence_complete: bool = True
    debug_trace: Dict[str, Any] = field(default_factory=dict)


def format_house_direct_answer(
    intent: AstrologicalIntent,
    house_num: int,
    chart_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Shared, reusable direct fact formatter for all 12 houses (1..12).
    Uses canonical get_house_details without any house-specific hardcoding.
    """
    h_details = get_house_details(chart_data, house_num)
    ord_str = ordinal_house(house_num)

    if not h_details["data_available"]:
        return {
            "answer": f"Your {ord_str} house data is unavailable in the current birth chart data.",
            "fact_type": intent.value.lower(),
            "answer_mode": "UNRESOLVED",
            "gemini_calls": 0,
            "evidence_complete": False,
            "fact_sources": {"house_data": "MISSING"}
        }

    h_sign = h_details["sign"]
    h_lord = h_details["lord"]

    # 1. House Sign lookup (e.g. "Which sign occupies my 10th house?")
    if intent == AstrologicalIntent.HOUSE_SIGN:
        return {
            "answer": f"The zodiac sign occupying your **{ord_str} house** is **{h_sign}** (ruled by **{h_lord}**).",
            "fact_type": "house_sign",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True,
            "fact_sources": {"house_sign": "FreeAstrologyAPI", "ascendant": "FreeAstrologyAPI"}
        }

    # 2. House Lord lookup (e.g. "Who rules my 7th house?")
    elif intent == AstrologicalIntent.HOUSE_LORD:
        placement_clause = ""
        if h_details["lord_placement_available"]:
            lord_ord = ordinal_house(h_details["lord_house"])
            placement_clause = f" In your birth chart, {h_lord} is placed in your **{lord_ord} house** in **{h_details['lord_sign']}**."
        return {
            "answer": f"Your **{ord_str} house** is in **{h_sign}**, which is ruled by **{h_lord}**.{placement_clause}",
            "fact_type": "house_lord",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True,
            "fact_sources": {"house_lord": "FreeAstrologyAPI", "ascendant": "FreeAstrologyAPI"}
        }

    # 3. House Lord Placement lookup (e.g. "Where is my 7th house lord placed?")
    elif intent == AstrologicalIntent.LORD_PLACEMENT:
        if not h_details["lord_placement_available"]:
            return {
                "answer": f"The placement of your {ord_str} house lord is unavailable in the current birth chart data.",
                "fact_type": "house_lord_placement",
                "answer_mode": "UNRESOLVED",
                "gemini_calls": 0,
                "evidence_complete": False,
                "fact_sources": {"house_lord_placement": "MISSING"}
            }
        lord_ord = ordinal_house(h_details["lord_house"])
        return {
            "answer": f"Your **{ord_str} house lord ({h_lord})** is placed in your **{lord_ord} house** in **{h_details['lord_sign']}**.",
            "fact_type": "house_lord_placement",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True,
            "fact_sources": {"house_lord_placement": "FreeAstrologyAPI"}
        }

    # 4. House Occupants lookup (e.g. "Which planet is in my 1st house?")
    elif intent == AstrologicalIntent.HOUSE_OCCUPANTS:
        if not h_details["occupants_available"]:
            return {
                "answer": f"Planet occupancy data for your {ord_str} house is unavailable or incomplete in the current birth chart data.",
                "fact_type": "house_occupants",
                "answer_mode": "UNRESOLVED",
                "gemini_calls": 0,
                "evidence_complete": False,
                "fact_sources": {"house_occupants": "MISSING"}
            }
        if h_details["is_empty"]:
            return {
                "answer": f"There are no natal planets positioned in your **{ord_str} house**.",
                "fact_type": "house_occupants",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True,
                "fact_sources": {"house_occupants": "FreeAstrologyAPI"}
            }
        occ_str = ", ".join(f"**{p}**" for p in h_details["occupants"])
        return {
            "answer": f"The planets positioned in your **{ord_str} house** are: {occ_str}.",
            "fact_type": "house_occupants",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True,
            "fact_sources": {"house_occupants": "FreeAstrologyAPI"}
        }

    return None


def evaluate_capability(
    intent_profile: QueryIntentProfile,
    chart_data: Dict[str, Any],
    dasha_hierarchy: Optional[Dict[str, Any]] = None,
    stage8_result: Optional[Dict[str, Any]] = None
) -> CapabilityDecision:
    """
    Evaluates backend capability and determines the routing path.
    Enforces complete query answerability:
    - Pure single-house factual inquiries route to LOCAL_DIRECT_FACT if data is present.
    - Compound inquiries (multiple houses, multiple clauses) route to LLM_ASSISTED.
    - Predictive / timing queries route to TIMING_ENGINE or LLM_ASSISTED.
    - Missing chart data routes to UNRESOLVED.
    """
    houses = intent_profile.target_houses
    core = intent_profile.core_intent

    # Check 1: Compound / Multi-House questions
    # E.g. "What does my 8th house indicate and how does it relate to my 12th house?"
    # Or "Can I buy a house with a loan (4H + 6H)?"
    if intent_profile.is_compound or len(houses) > 1:
        return CapabilityDecision(
            path=ExecutionPath.LLM_ASSISTED,
            intent_profile=intent_profile,
            confidence=0.95,
            reason=f"Compound query involving multiple houses {houses} or multi-part questions requires LLM contextual synthesis.",
            target_houses=houses,
            required_evidence_complete=True,
            debug_trace={"houses": houses, "is_compound": True}
        )

    # Check 2: Predictive / Timing Modifiers
    # E.g. "Will my 10th house bring a job soon?"
    if intent_profile.has_timing_modifier and any(w in intent_profile.normalized_query for w in ["dasha", "transit", "when", "timing", "kab"]):
        return CapabilityDecision(
            path=ExecutionPath.TIMING_ENGINE,
            intent_profile=intent_profile,
            confidence=0.95,
            reason="Query contains specific timing prediction request requiring Vimshottari & Gochar engine.",
            target_houses=houses,
            required_evidence_complete=True
        )

    if intent_profile.has_predictive_modifier or intent_profile.has_evaluative_modifier:
        return CapabilityDecision(
            path=ExecutionPath.LLM_ASSISTED,
            intent_profile=intent_profile,
            confidence=0.90,
            reason="Query contains predictive or evaluative phrasing requiring contextual interpretation.",
            target_houses=houses,
            required_evidence_complete=True
        )

    # Check 3: Direct Single-House Factual Queries (1..12)
    if core in (
        AstrologicalIntent.HOUSE_SIGN,
        AstrologicalIntent.HOUSE_LORD,
        AstrologicalIntent.LORD_PLACEMENT,
        AstrologicalIntent.HOUSE_OCCUPANTS
    ):
        if len(houses) == 1:
            h = houses[0]
            ans_payload = format_house_direct_answer(core, h, chart_data)
            if ans_payload and ans_payload.get("answer_mode") == "DIRECT":
                return CapabilityDecision(
                    path=ExecutionPath.LOCAL_DIRECT_FACT,
                    intent_profile=intent_profile,
                    confidence=1.0,
                    reason=f"Pure direct factual query for house {h} ({core.value}) resolved locally (0 LLM tokens).",
                    direct_answer=ans_payload,
                    target_houses=[h],
                    required_evidence_complete=True
                )
            else:
                return CapabilityDecision(
                    path=ExecutionPath.UNRESOLVED,
                    intent_profile=intent_profile,
                    confidence=1.0,
                    reason=f"Direct factual query for house {h} cannot be answered because required chart evidence is missing.",
                    direct_answer=ans_payload,
                    target_houses=[h],
                    required_evidence_complete=False
                )

    # Check 4: Planet Placement Lookup (e.g. "Where is Mercury placed in my chart?")
    if core == AstrologicalIntent.PLANET_PLACEMENT:
        planets = chart_data.get("planets", {}) if isinstance(chart_data, dict) else {}
        for p in intent_profile.target_planets:
            p_info = planets.get(p)
            if isinstance(p_info, dict) and p_info.get("house") and p_info.get("rashi"):
                h_ord = ordinal_house(p_info["house"])
                ans_payload = {
                    "answer": f"**{p}** is placed in **{p_info['rashi']}** in your **{h_ord} house**.",
                    "fact_type": "planet_placement",
                    "answer_mode": "DIRECT",
                    "gemini_calls": 0,
                    "evidence_complete": True,
                    "fact_sources": {"planetary_positions": "FreeAstrologyAPI"}
                }
                return CapabilityDecision(
                    path=ExecutionPath.LOCAL_DIRECT_FACT,
                    intent_profile=intent_profile,
                    confidence=1.0,
                    reason=f"Planet placement for {p} resolved locally.",
                    direct_answer=ans_payload,
                    required_evidence_complete=True
                )

    # Check 5: All Placements Lookup (e.g. "Which planet is in which house?")
    if core == AstrologicalIntent.ALL_PLACEMENTS:
        planets = chart_data.get("planets", {}) if isinstance(chart_data, dict) else {}
        ascendant = chart_data.get("ascendant", {}) if isinstance(chart_data, dict) else {}
        lines = []
        if ascendant and ascendant.get("rashi"):
            asc_deg = ascendant.get("degree_in_rashi")
            deg_s = f" ({asc_deg:.1f}°)" if asc_deg is not None else ""
            lines.append(f"- **Ascendant (Lagna)**: **{ascendant.get('rashi')}**{deg_s} in **1st House**")
        for p_name in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]:
            p_data = planets.get(p_name)
            if isinstance(p_data, dict):
                h_num = p_data.get("house")
                r_name = p_data.get("rashi")
                p_deg = p_data.get("degree_in_rashi") or p_data.get("normDegree")
                deg_s = f" ({p_deg:.1f}°)" if p_deg is not None else ""
                if h_num:
                    lines.append(f"- **{p_name}**: **{r_name}**{deg_s} in **{ordinal_house(h_num)} House**")
        if lines:
            ans_payload = {
                "answer": "Here are your natal planetary house placements:\n\n" + "\n".join(lines),
                "fact_type": "all_planet_placements",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True,
                "fact_sources": {"planetary_positions": "FreeAstrologyAPI"}
            }
            return CapabilityDecision(
                path=ExecutionPath.LOCAL_DIRECT_FACT,
                intent_profile=intent_profile,
                confidence=1.0,
                reason="All planetary placements resolved directly from verified chart payload.",
                direct_answer=ans_payload,
                required_evidence_complete=True
            )

    # Check 6: House Interpretation (e.g. "What does my 10th house indicate?")
    # If single house interpretation with complete chart data, route to LLM_ASSISTED with house facts
    if core in (AstrologicalIntent.HOUSE_INTERPRETATION, AstrologicalIntent.HOUSE_SIGNIFICANCE):
        return CapabilityDecision(
            path=ExecutionPath.LLM_ASSISTED,
            intent_profile=intent_profile,
            confidence=0.85,
            reason=f"House interpretation for houses {houses} requires contextual synthesis with verified house evidence.",
            target_houses=houses,
            required_evidence_complete=True
        )

    # Default fallback to LLM_ASSISTED
    return CapabilityDecision(
        path=ExecutionPath.LLM_ASSISTED,
        intent_profile=intent_profile,
        confidence=0.75,
        reason="General astrological interpretation requiring contextual reasoning.",
        target_houses=houses,
        required_evidence_complete=True
    )
