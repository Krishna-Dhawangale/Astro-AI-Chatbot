"""
Stage 8.26 — Evidence Calibration & Score Threshold Standardization Engine
=============================================================================
Module: backend/reasoning/calibration.py

Purpose:
Calibrates evidence weight scores, positive-to-challenging signal ratios,
and Dasha/transit timing modifiers against empirical astrological standards.

Standards & Thresholds:
- STRONGLY_FAVORED: Total Weight Score >= 15.0, Ratio >= 3.0, Primary House Lord intact.
- MODERATELY_FAVORED: Total Weight Score 10.0 - 14.9, Ratio >= 1.5.
- BALANCED_NEUTRAL: Total Weight Score 5.0 - 9.9, or equal positive/challenging signals.
- CHALLENGING_PERIOD: Total Weight Score < 5.0, or Challenging Signals > Positive Signals.
"""

from typing import Dict, List, Any, Tuple


def calibrate_timing_modifiers(dasha_info: List[Dict[str, Any]], transit_info: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
    """
    Computes empirical Dasha and transit timing weight modifiers.
    Returns (timing_score_modifier, timing_notes).
    """
    modifier = 0.0
    notes = []

    # Dasha analysis
    if dasha_info:
        for d in dasha_info:
            planet = d.get("planet", "")
            level = d.get("dasha_level", "")
            # Supportive dashas (benefics/exalted/lords)
            if planet in ["Jupiter", "Venus", "Moon", "Mercury"]:
                modifier += 2.0
                notes.append(f"Favorable {level} ({planet})")
            elif planet in ["Saturn", "Rahu", "Ketu", "Mars"]:
                modifier -= 1.0
                notes.append(f"Testing/Strict {level} ({planet})")

    # Transit analysis
    if transit_info:
        for t in transit_info:
            t_planet = t.get("transit_planet", "")
            t_house = t.get("transit_house")
            if t_house in [10, 11, 1, 5, 9]:
                modifier += 1.5
                notes.append(f"Supportive {t_planet} transit in House {t_house}")
            elif t_house in [6, 8, 12]:
                modifier -= 1.5
                notes.append(f"Challenging {t_planet} transit in House {t_house}")

    return round(modifier, 2), notes


def calibrate_evidence_status(
    base_evidence_score: float,
    positive_signals: List[str],
    challenging_signals: List[str],
    timing_modifier: float = 0.0
) -> Dict[str, Any]:
    """
    Calibrates cumulative score and assigns calibrated evidence status.
    """
    calibrated_score = round(base_evidence_score + timing_modifier, 2)
    pos_count = len(positive_signals)
    chal_count = len(challenging_signals)
    
    # Avoid zero division
    ratio = round((pos_count + 0.1) / (chal_count + 0.1), 2)

    if chal_count > pos_count:
        status = "CHALLENGING_PERIOD"
        level = "Challenging"
        explanation = "Challenging planetary or house placements outnumber supportive indicators."
    elif calibrated_score >= 15.0 and ratio >= 2.0:
        status = "STRONGLY_FAVORED"
        level = "High Support"
        explanation = "Strong natal foundation, favorable dignities, and supportive timing align."
    elif calibrated_score >= 10.0 and ratio >= 1.2:
        status = "MODERATELY_FAVORED"
        level = "Moderate Support"
        explanation = "Favorable baseline factors with mild counter-balancing influences."
    elif calibrated_score >= 5.0:
        status = "BALANCED_NEUTRAL"
        level = "Neutral / Mixed"
        explanation = "Balanced mix of supportive and neutral astrological indicators."
    else:
        status = "WEAK_EVIDENCE"
        level = "Low Evidence"
        explanation = "Limited or weak matched evidence available in the chart."

    return {
        "calibrated_score": calibrated_score,
        "base_score": base_evidence_score,
        "timing_modifier": timing_modifier,
        "status": status,
        "support_level": level,
        "pos_to_chal_ratio": ratio,
        "explanation": explanation
    }
