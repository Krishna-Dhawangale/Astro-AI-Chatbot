# ============================================================
# STAGE 8.11.7 — UNIFIED NATAL LORDSHIP ANALYSIS
# ============================================================

from typing import Dict, Any, Optional, List


def build_natal_lordship_analysis(
    ascendant_longitude,
    ascendant_rashi,
    natal_house_lord_map,
    house_lord_placements,
    planet_owned_houses,
    house_lord_self_placements,
):
    """
    Build the unified natal lordship analysis object.

    The actual house-lord calculations are expected to have
    already been performed upstream.
    """

    return {
        "ascendant": {
            "longitude":
                ascendant_longitude,

            "rashi":
                ascendant_rashi,
        },

        "house_lord_map":
            natal_house_lord_map,

        "house_lord_placements":
            house_lord_placements,

        "planet_owned_houses":
            planet_owned_houses,

        "house_lord_self_placements":
            house_lord_self_placements,
    }


RASHI_LORDS = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
    "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
    "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"
}

RASHI_LIST = list(RASHI_LORDS.keys())
CORE_PLANET_NAMES = {"Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"}


def get_house_lord(rashi_name: str) -> str:
    return RASHI_LORDS.get(rashi_name, "Unknown")


def get_house_details(norm_chart: Dict[str, Any], house_num: int) -> Dict[str, Any]:
    """
    Canonical Whole-Sign House Resolver.
    Computes sign, lord, lord placement, and verified occupants strictly from FreeAstrologyAPI payload.
    Safely handles invalid house numbers, missing ascendants, unknown rashis, and incomplete planet data.
    """
    empty_result = {
        "house": house_num,
        "sign": None,
        "lord": None,
        "lord_house": None,
        "lord_sign": None,
        "lord_placement_available": False,
        "occupants": [],
        "occupants_available": False,
        "is_empty": False,
        "data_available": False,
    }

    if not isinstance(house_num, int) or house_num < 1 or house_num > 12:
        return empty_result

    if not isinstance(norm_chart, dict):
        return empty_result

    ascendant = norm_chart.get("ascendant") or {}
    asc_rashi = (ascendant.get("rashi") or ascendant.get("sign")) if isinstance(ascendant, dict) else None

    if not isinstance(asc_rashi, str) or asc_rashi not in RASHI_LORDS:
        return empty_result

    asc_idx = RASHI_LIST.index(asc_rashi)
    house_sign = RASHI_LIST[(asc_idx + house_num - 1) % 12]
    house_lord = RASHI_LORDS[house_sign]

    planets = norm_chart.get("planets") or {}
    if not isinstance(planets, dict):
        planets = {}

    lord_data = planets.get(house_lord) if isinstance(planets.get(house_lord), dict) else {}
    lord_house = lord_data.get("house")
    lord_sign = lord_data.get("rashi")
    lord_available = (
        isinstance(lord_house, int)
        and 1 <= lord_house <= 12
        and isinstance(lord_sign, str)
        and bool(lord_sign.strip())
    )

    # Occupancy validation:
    # An empty occupant list is valid ONLY when planet data is sufficiently complete
    # (i.e. at least core planets are populated with valid house numbers).
    valid_planets_with_houses = {
        p_name: p_info
        for p_name, p_info in planets.items()
        if isinstance(p_info, dict) and isinstance(p_info.get("house"), int) and 1 <= p_info.get("house") <= 12
    }

    has_sufficient_planet_data = len(valid_planets_with_houses.keys() & CORE_PLANET_NAMES) >= 7

    occupants = []
    occupants_available = False
    is_empty = False

    if has_sufficient_planet_data:
        occupants_available = True
        occupants = sorted(
            [
                p_name
                for p_name, p_info in valid_planets_with_houses.items()
                if p_info.get("house") == house_num
            ]
        )
        is_empty = len(occupants) == 0

    return {
        "house": house_num,
        "sign": house_sign,
        "lord": house_lord,
        "lord_house": lord_house if lord_available else None,
        "lord_sign": lord_sign if lord_available else None,
        "lord_placement_available": lord_available,
        "occupants": occupants,
        "occupants_available": occupants_available,
        "is_empty": is_empty,
        "data_available": True,
    }
