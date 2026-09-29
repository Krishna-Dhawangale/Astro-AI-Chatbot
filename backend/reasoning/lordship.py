# ============================================================
# STAGE 8.11.7 — UNIFIED NATAL LORDSHIP ANALYSIS
# ============================================================

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


def get_house_lord(rashi_name: str) -> str:
    return RASHI_LORDS.get(rashi_name, "Unknown")