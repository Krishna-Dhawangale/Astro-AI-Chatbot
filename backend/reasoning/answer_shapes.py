"""
Intent-Aware Answer Shapes (backend/reasoning/answer_shapes.py)
-----------------------------------------------------------------
Pillar 4 & Safeguard 3: Flexible, Intent-Specific Answer Structures.

Defines dynamic section shapes for different question intents.
Sections are allowed dynamic structures based on available evidence,
NOT mandatory rigid templates.
"""

from typing import Dict, Any, List

ANSWER_SHAPES: Dict[str, List[str]] = {
    # ------------------------------------------------------------------
    # CAREER INTENT SHAPES
    # ------------------------------------------------------------------
    "career_general": [
        "opening_chart_snapshot",
        "primary_strengths",
        "active_dasha_timing",
        "balanced_guidance"
    ],
    "career_suitability": [
        "opening_suitability",
        "career_themes",
        "suitable_domain_fields",
        "closing_guidance"
    ],
    "dasha_career_interaction": [
        "current_dasha_identity",
        "career_connection",
        "active_influence_effect",
        "timing_recommendation"
    ],
    "career_timing": [
        "timing_window_intro",
        "supporting_factors",
        "cautionary_note",
        "actionable_next_step"
    ],

    # ------------------------------------------------------------------
    # MARRIAGE INTENT SHAPES
    # ------------------------------------------------------------------
    "marriage_general": [
        "opening_relationship_snapshot",
        "relationship_strengths",
        "harmony_guidance"
    ],
    "marriage_timing": [
        "timing_window_intro",
        "supporting_factors",
        "astrological_caveat"
    ],

    # ------------------------------------------------------------------
    # WEALTH & FINANCE INTENT SHAPES
    # ------------------------------------------------------------------
    "finance_general": [
        "opening_financial_snapshot",
        "wealth_accumulation_factors",
        "prudent_budgeting_advice"
    ],

    # ------------------------------------------------------------------
    # HEALTH INTENT SHAPES
    # ------------------------------------------------------------------
    "health_general": [
        "opening_health_snapshot",
        "vitality_factors",
        "wellness_routine_advice"
    ],

    # ------------------------------------------------------------------
    # EDUCATION INTENT SHAPES
    # ------------------------------------------------------------------
    "education_general": [
        "opening_education_snapshot",
        "intellect_factors",
        "study_focus_advice"
    ],

    # ------------------------------------------------------------------
    # PROPERTY INTENT SHAPES
    # ------------------------------------------------------------------
    "property_general": [
        "opening_property_snapshot",
        "real_estate_alignment",
        "home_stability_guidance"
    ],

    # ------------------------------------------------------------------
    # DEFAULT FALLBACK SHAPE
    # ------------------------------------------------------------------
    "general": [
        "opening_chart_snapshot",
        "primary_strengths",
        "closing_guidance"
    ]
}


def get_answer_shape(intent: str, available_sections: List[str]) -> List[str]:
    """
    Safeguard 3: Selects answer shape sections based on intent and filters
    out sections for which underlying evidence is missing.
    
    Prevents empty or forced sections.
    """
    base_shape = ANSWER_SHAPES.get(intent, ANSWER_SHAPES["general"])
    # Return sections that are supported or present
    return [section for section in base_shape if section in available_sections or "opening" in section or "closing" in section or "guidance" in section or "snapshot" in section]
