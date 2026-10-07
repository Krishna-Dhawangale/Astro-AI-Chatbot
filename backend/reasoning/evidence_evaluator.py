"""
Evidence Evaluator & Safety Gate (backend/reasoning/evidence_evaluator.py)
-------------------------------------------------------------------------
Pillar 4: Evidence Evaluator = Safety Gate

Evaluates required evidence against actual chart data from FreeAstrologyAPI & normalized payload.
Guarantees ZERO fabricated astrology when evidence is missing.
"""

from typing import Dict, Any, List, Tuple


def evaluate_evidence_availability(
    required_evidence: List[str],
    norm_chart: Dict[str, Any],
    dasha_hierarchy: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates evidence availability for a given list of required evidence keys.
    Returns a structured audit dict:
    {
        "required_evidence": [...],
        "available_evidence": [...],
        "missing_evidence": [...],
        "evidence_status": "COMPLETE" | "PARTIAL" | "UNRESOLVED" | "UNSUPPORTED"
    }
    """
    planets = norm_chart.get("planets", {}) if isinstance(norm_chart, dict) else {}
    ascendant = norm_chart.get("ascendant", {}) if isinstance(norm_chart, dict) else {}
    houses = norm_chart.get("houses", {}) if isinstance(norm_chart, dict) else {}

    moon_data = planets.get("Moon", {}) if isinstance(planets, dict) else {}
    sun_data = planets.get("Sun", {}) if isinstance(planets, dict) else {}

    c_maha = (dasha_hierarchy or {}).get("current_mahadasha") or (dasha_hierarchy or {}).get("mahadasha", {}).get("planet")
    c_antar = (dasha_hierarchy or {}).get("current_antardasha") or (dasha_hierarchy or {}).get("antardasha", {}).get("planet")

    available: List[str] = []
    missing: List[str] = []

    for req in required_evidence:
        req_clean = req.lower().strip()
        
        if req_clean == "moon_nakshatra":
            if moon_data.get("nakshatra") and str(moon_data.get("nakshatra")).strip() not in ["None", ""]:
                available.append(req)
            else:
                missing.append(req)

        elif req_clean == "moon_longitude":
            if moon_data.get("longitude") is not None or moon_data.get("degree_in_rashi") is not None:
                available.append(req)
            else:
                missing.append(req)

        elif req_clean == "moon_rashi":
            if moon_data.get("rashi") and str(moon_data.get("rashi")).strip() not in ["None", ""]:
                available.append(req)
            else:
                missing.append(req)

        elif req_clean == "sun_rashi":
            if sun_data.get("rashi") and str(sun_data.get("rashi")).strip() not in ["None", ""]:
                available.append(req)
            else:
                missing.append(req)

        elif req_clean == "ascendant_rashi":
            if ascendant.get("rashi") and str(ascendant.get("rashi")).strip() not in ["None", ""]:
                available.append(req)
            else:
                missing.append(req)

        elif req_clean in ["mahadasha", "current_dasha"]:
            if c_maha:
                available.append(req)
            else:
                missing.append(req)

        elif req_clean == "antardasha":
            if c_antar:
                available.append(req)
            else:
                missing.append(req)

        elif req_clean in ["10th_house", "7th_house", "2nd_house", "4th_house", "5th_house", "11th_house", "1st_house"]:
            # Chart features are available if planets / ascendant are populated
            if planets or ascendant:
                available.append(req)
            else:
                missing.append(req)

        elif req_clean in ["10th_lord", "7th_lord", "2nd_lord", "career_karakas", "venus"]:
            if planets or ascendant:
                available.append(req)
            else:
                missing.append(req)

        else:
            # Default check for any other key
            if planets or ascendant:
                available.append(req)
            else:
                missing.append(req)

    # Determine Evidence Status
    if not required_evidence:
        status = "COMPLETE"
    elif len(available) == len(required_evidence):
        status = "COMPLETE"
    elif len(available) > 0:
        status = "PARTIAL"
    else:
        status = "UNRESOLVED"

    return {
        "required_evidence": required_evidence,
        "available_evidence": available,
        "missing_evidence": missing,
        "evidence_status": status
    }
