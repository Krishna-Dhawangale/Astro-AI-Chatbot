# ============================================================
# STAGE 8.13 — PLANETARY DIGNITY / STRENGTH
# ============================================================

"""
Deterministic planetary dignity analysis.

IMPORTANT:
- Does NOT calculate planetary positions.
- Uses already calculated natal planet positions.
- Does NOT produce predictions.
- Does NOT assign arbitrary numerical scores.
- Produces structured dignity evidence for later rule engines.

This module is a newly designed VS Code implementation because
the original Colab dignity source was not available.
"""


# ============================================================
# 1. VEDIC DIGNITY CONFIGURATION
# ============================================================

EXALTATION_SIGNS = {
    "Sun": "Aries",
    "Moon": "Taurus",
    "Mars": "Capricorn",
    "Mercury": "Virgo",
    "Jupiter": "Cancer",
    "Venus": "Pisces",
    "Saturn": "Libra",
}


DEBILITATION_SIGNS = {
    "Sun": "Libra",
    "Moon": "Scorpio",
    "Mars": "Cancer",
    "Mercury": "Pisces",
    "Jupiter": "Capricorn",
    "Venus": "Virgo",
    "Saturn": "Aries",
}


OWN_SIGNS = {
    "Sun": ["Leo"],
    "Moon": ["Cancer"],
    "Mars": ["Aries", "Scorpio"],
    "Mercury": ["Gemini", "Virgo"],
    "Jupiter": ["Sagittarius", "Pisces"],
    "Venus": ["Taurus", "Libra"],
    "Saturn": ["Capricorn", "Aquarius"],
}


# Traditional sign lords are useful for deriving
# planetary relationships.
SIGN_LORDS = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": "Mars",
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}


# Natural planetary relationships.
#
# These are used only to classify the sign lord relationship
# when a planet is neither exalted, debilitated, nor in own sign.
#
# This is a newly introduced deterministic configuration,
# not recovered from your original Colab.
NATURAL_RELATIONSHIPS = {

    "Sun": {
        "friends": ["Moon", "Mars", "Jupiter"],
        "enemies": ["Venus", "Saturn"],
        "neutral": ["Mercury"],
    },

    "Moon": {
        "friends": ["Sun", "Mercury"],
        "enemies": [],
        "neutral": ["Mars", "Jupiter", "Venus", "Saturn"],
    },

    "Mars": {
        "friends": ["Sun", "Moon", "Jupiter"],
        "enemies": ["Mercury"],
        "neutral": ["Venus", "Saturn"],
    },

    "Mercury": {
        "friends": ["Sun", "Venus"],
        "enemies": ["Moon"],
        "neutral": ["Mars", "Jupiter", "Saturn"],
    },

    "Jupiter": {
        "friends": ["Sun", "Moon", "Mars"],
        "enemies": ["Mercury", "Venus"],
        "neutral": ["Saturn"],
    },

    "Venus": {
        "friends": ["Mercury", "Saturn"],
        "enemies": ["Sun", "Moon"],
        "neutral": ["Mars", "Jupiter"],
    },

    "Saturn": {
        "friends": ["Mercury", "Venus"],
        "enemies": ["Sun", "Moon", "Mars"],
        "neutral": ["Jupiter"],
    },
}


# ============================================================
# 2. DETERMINE BASIC DIGNITY
# ============================================================

def determine_planetary_dignity(
    planet: str,
    rashi: str,
) -> dict:
    """
    Determine the basic dignity of one planet from its Rashi.

    Priority:
        exalted
        debilitated
        own_sign
        friend_sign
        neutral_sign
        enemy_sign
        unknown
    """

    if planet in EXALTATION_SIGNS:

        if rashi == EXALTATION_SIGNS[planet]:
            return {
                "dignity": "exalted",
                "basis": "exaltation_sign",
            }

    if planet in DEBILITATION_SIGNS:

        if rashi == DEBILITATION_SIGNS[planet]:
            return {
                "dignity": "debilitated",
                "basis": "debilitation_sign",
            }

    if planet in OWN_SIGNS:

        if rashi in OWN_SIGNS[planet]:
            return {
                "dignity": "own_sign",
                "basis": "own_sign",
            }

    sign_lord = SIGN_LORDS.get(rashi)

    if sign_lord is None:
        return {
            "dignity": "unknown",
            "basis": "unknown_sign",
        }

    relationships = NATURAL_RELATIONSHIPS.get(
        planet,
        {}
    )

    if sign_lord in relationships.get(
        "friends",
        []
    ):
        return {
            "dignity": "friend_sign",
            "basis": "natural_friend",
            "sign_lord": sign_lord,
        }

    if sign_lord in relationships.get(
        "enemies",
        []
    ):
        return {
            "dignity": "enemy_sign",
            "basis": "natural_enemy",
            "sign_lord": sign_lord,
        }

    return {
        "dignity": "neutral_sign",
        "basis": "natural_neutral",
        "sign_lord": sign_lord,
    }


# ============================================================
# 3. ANALYZE ONE PLANET
# ============================================================

def analyze_planet_dignity(
    planet: str,
    planet_data: dict,
) -> dict:
    """
    Build structured dignity evidence for one natal planet.
    """

    rashi = planet_data.get("rashi")

    result = determine_planetary_dignity(
        planet=planet,
        rashi=rashi,
    )

    return {
        "planet": planet,

        "rashi": rashi,

        "house":
            planet_data.get("house"),

        "longitude":
            planet_data.get("longitude"),

        "nakshatra":
            planet_data.get("nakshatra"),

        "pada":
            planet_data.get("pada"),

        "dignity":
            result["dignity"],

        "basis":
            result["basis"],

        "sign_lord":
            result.get("sign_lord"),

        "exaltation_sign":
            EXALTATION_SIGNS.get(planet),

        "debilitation_sign":
            DEBILITATION_SIGNS.get(planet),

        "own_signs":
            OWN_SIGNS.get(planet, []),
    }


# ============================================================
# 4. ANALYZE ALL PLANETS
# ============================================================

def analyze_planetary_dignity(
    natal_planet_lookup: dict,
) -> dict:
    """
    Analyze dignity for all available natal planets.
    """

    planetary_dignity = {}

    for planet, planet_data in natal_planet_lookup.items():

        planetary_dignity[planet] = (
            analyze_planet_dignity(
                planet=planet,
                planet_data=planet_data,
            )
        )

    return planetary_dignity


# ============================================================
# 5. BUILD PLANETARY STRENGTH ANALYSIS
# ============================================================

def build_planetary_strength_analysis(
    natal_planet_lookup: dict,
) -> dict:
    """
    Build the structure consumed by Stage 8.14.
    """

    planetary_dignity = (
        analyze_planetary_dignity(
            natal_planet_lookup
        )
    )

    return {
        "planetary_dignity":
            planetary_dignity
    }


# ============================================================
# 6. VALIDATION
# ============================================================

def validate_planetary_strength_analysis(
    planetary_strength_analysis: dict,
) -> None:

    assert isinstance(
        planetary_strength_analysis,
        dict,
    )

    assert (
        "planetary_dignity"
        in planetary_strength_analysis
    )

    dignity_data = (
        planetary_strength_analysis[
            "planetary_dignity"
        ]
    )

    assert isinstance(
        dignity_data,
        dict,
    )

    for planet, data in dignity_data.items():

        assert "planet" in data
        assert "rashi" in data
        assert "dignity" in data
        assert "basis" in data

    return None


calculate_planetary_dignity = analyze_planetary_dignity