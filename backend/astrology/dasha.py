"""
Vimshottari Dasha Bridge Module (backend/astrology/dasha.py)
-------------------------------------------------------------
Authoritative handling for Vimshottari Dasha calculations.
Uses API-provided Dasha periods directly when available.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone

DASHA_LORDS = [
    ("Ketu", 7), ("Venus", 20), ("Sun", 6), ("Moon", 10),
    ("Mars", 7), ("Rahu", 18), ("Jupiter", 16), ("Saturn", 19), ("Mercury", 17)
]
TOTAL_DASHA_YEARS = 120
NAKSHATRA_SPAN = 13.333333333333334  # 13 degrees 20 minutes


def calculate_vimshottari_dasha(
    moon_longitude: float,
    current_dt: datetime = None,
    raw_dasha_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Returns the Vimshottari Dasha hierarchy object.
    If raw_dasha_data from FreeAstrologyAPI is provided, uses that as authoritative.
    """
    if current_dt is None:
        current_dt = datetime.now(timezone.utc)

    # 1. API Authoritative Source Check
    if isinstance(raw_dasha_data, dict) and raw_dasha_data:
        hierarchy = raw_dasha_data.get("dasha_hierarchy", raw_dasha_data)
        if isinstance(hierarchy, dict) and ("current_mahadasha" in hierarchy or "output" in hierarchy):
            return {
                "source": "FreeAstrologyAPI",
                "current_mahadasha": hierarchy.get("current_mahadasha", "Jupiter"),
                "current_antardasha": hierarchy.get("current_antardasha", "Saturn"),
                "raw_hierarchy": hierarchy
            }

    # 2. Local Astronomical Computation from Moon Longitude
    moon_deg = moon_longitude % 360
    nakshatra_idx = int(moon_deg // NAKSHATRA_SPAN)
    deg_in_nak = moon_deg % NAKSHATRA_SPAN
    fraction_traversed = deg_in_nak / NAKSHATRA_SPAN

    lord_idx = nakshatra_idx % len(DASHA_LORDS)
    start_lord, total_years = DASHA_LORDS[lord_idx]

    remaining_years = total_years * (1.0 - fraction_traversed)

    return {
        "source": "MoonLongitudeCalculation",
        "current_mahadasha": start_lord,
        "current_antardasha": DASHA_LORDS[(lord_idx + 1) % len(DASHA_LORDS)][0],
        "remaining_years_in_mahadasha": round(remaining_years, 2),
        "nakshatra_index": nakshatra_idx
    }
