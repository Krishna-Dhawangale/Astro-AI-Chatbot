"""
Transit Calculation Module (backend/astrology/transit.py)
-----------------------------------------------------------
Calculates current transit planetary longitudes and maps them
to the user's natal houses based on their natal Ascendant.
"""

from typing import Dict, Any
from datetime import datetime, timezone
from backend.ephemeris import get_astronomical_features

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]


def calculate_transit_planets(dt: datetime = None) -> Dict[str, Any]:
    """
    Calculates current planetary transit positions.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour + (dt.minute / 60.0)

    # Use Swiss Ephemeris (topocentric sidereal) for current transit calculation
    features = get_astronomical_features(year, month, day, hour, 20.5937, 78.9629)

    planet_keys = ["Ascendant", "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
    transits = {}

    for idx, key in enumerate(planet_keys):
        if idx < len(features):
            sign_idx = int(features[idx]) % 12
            transits[key] = {
                "rashi": ZODIAC_SIGNS[sign_idx],
                "longitude": sign_idx * 30.0 + 15.0  # Mid-sign estimate
            }

    return transits


def add_transit_houses(transit_data: Dict[str, Any], chart_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps transit planetary positions onto the natal house structure.
    """
    ascendant = chart_data.get("ascendant", {})
    asc_rashi = ascendant.get("rashi", "Aries")
    asc_idx = ZODIAC_SIGNS.index(asc_rashi) if asc_rashi in ZODIAC_SIGNS else 0

    results = {}
    for p_name, p_data in transit_data.items():
        p_rashi = p_data.get("rashi", "Aries")
        p_idx = ZODIAC_SIGNS.index(p_rashi) if p_rashi in ZODIAC_SIGNS else 0
        house_num = ((p_idx - asc_idx) % 12) + 1

        results[p_name] = {
            "rashi": p_rashi,
            "longitude": p_data.get("longitude", 0.0),
            "transit_house": house_num
        }

    return results
