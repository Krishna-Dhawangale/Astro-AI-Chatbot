"""
Health-domain reasoning layer.

This module handles health-related questions in two ways:

1. Astrology / wellness questions
   -> provide structured astrology evidence.

2. Medical questions
   -> provide appropriate informational/safety routing.

Important:
This module must never use an astrology chart to diagnose,
confirm, or rule out a medical condition.
"""


# ============================================================
# HEALTH MODES
# ============================================================

HEALTH_MODES = {
    "astrology_wellness",
    "medical_information",
    "medical_diagnosis_request",
    "medical_treatment_request",
    "medical_emergency",
}


# ============================================================
# INTENT DETECTION
# ============================================================

ASTROLOGY_PATTERNS = [
    "my chart",
    "my horoscope",
    "birth chart",
    "kundli",
    "kundali",
    "astrology",
    "according to my chart",
    "according to my horoscope",
    "according to my kundli",
]


DIAGNOSIS_PATTERNS = [
    "do i have",
    "am i suffering from",
    "can i have",
    "does this mean i have",
    "can you diagnose",
    "diagnose me",
    "what disease do i have",
    "what condition do i have",
    "is this a sign of",
]


TREATMENT_PATTERNS = [
    "what medicine should i take",
    "which medicine should i take",
    "what medication should i take",
    "what treatment should i take",
    "how should i treat",
    "how can i cure",
    "what should i take for",
]


EMERGENCY_PATTERNS = [
    "emergency",
    "severe chest pain",
    "difficulty breathing",
    "can't breathe",
    "cannot breathe",
    "unconscious",
    "passed out",
    "heavy bleeding",
    "severe bleeding",
    "suicide",
    "suicidal",
    "overdose",
]


INFORMATION_PATTERNS = [
    "what is",
    "what are",
    "explain",
    "meaning of",
    "symptoms of",
    "causes of",
    "what does",
]


def normalize_question(question: str) -> str:
    """
    Normalize user input for lightweight intent detection.
    """

    if not question:
        return ""

    return " ".join(
        question.lower().strip().split()
    )


def contains_any_pattern(
    question: str,
    patterns: list[str],
) -> bool:
    """
    Check whether any phrase appears in the question.
    """

    return any(
        pattern in question
        for pattern in patterns
    )


# ============================================================
# HEALTH MODE CLASSIFIER
# ============================================================

def classify_health_mode(question: str) -> str:
    """
    Determine the high-level mode for a health question.

    Priority:

        emergency
            ↓
        treatment
            ↓
        diagnosis
            ↓
        astrology
            ↓
        medical information
    """

    q = normalize_question(question)

    if not q:
        return "medical_information"

    # Emergency takes highest priority.
    if contains_any_pattern(
        q,
        EMERGENCY_PATTERNS,
    ):
        return "medical_emergency"

    # Treatment requests.
    if contains_any_pattern(
        q,
        TREATMENT_PATTERNS,
    ):
        return "medical_treatment_request"

    # Diagnosis requests.
    if contains_any_pattern(
        q,
        DIAGNOSIS_PATTERNS,
    ):
        return "medical_diagnosis_request"

    # Explicit astrology context.
    if contains_any_pattern(
        q,
        ASTROLOGY_PATTERNS,
    ):
        return "astrology_wellness"

    # General medical information.
    if contains_any_pattern(
        q,
        INFORMATION_PATTERNS,
    ):
        return "medical_information"

    # Default for health-domain questions.
    return "astrology_wellness"


# ============================================================
# ASTROLOGY HEALTH EVIDENCE
# ============================================================

def build_health_evidence(
    natal_chart,
    current_dasha_hierarchy=None,
    transit_timing=None,
):
    """
    Build structured astrology-domain health evidence.

    This function calculates/collects evidence only.
    It does not diagnose medical conditions.
    """

    evidence = {
        "domain": "health",
        "mode": "astrology_wellness",
        "medical_diagnosis_supported": False,
        "natal_indicators": [],
        "dasha_indicators": [],
        "transit_indicators": [],
    }

    planets = natal_chart.get(
        "planets",
        {},
    )

    # --------------------------------------------------------
    # Natal evidence
    # --------------------------------------------------------

    health_related_houses = {
        1,
        6,
        8,
        12,
    }

    for planet, data in planets.items():

        house = data.get("house")

        if house in health_related_houses:

            evidence["natal_indicators"].append({
                "type": "planet_in_health_related_house",
                "planet": planet,
                "house": house,
                "rashi": data.get("rashi"),
                "longitude": data.get("longitude"),
                "nakshatra": data.get("nakshatra"),
                "pada": data.get("pada"),
            })

    # --------------------------------------------------------
    # Dasha evidence
    # --------------------------------------------------------

    if current_dasha_hierarchy:

        for level in (
            "mahadasha",
            "antardasha",
            "pratyantardasha",
        ):

            dasha = current_dasha_hierarchy.get(
                level
            )

            if not dasha:
                continue

            evidence["dasha_indicators"].append({
                "level": level,
                "planet": dasha.get("planet"),
                "start": dasha.get("start"),
                "end": dasha.get("end"),
            })

    # --------------------------------------------------------
    # Transit evidence
    # --------------------------------------------------------

    if transit_timing:

        for planet, data in transit_timing.get(
            "planets",
            {},
        ).items():

            evidence["transit_indicators"].append({
                "planet": planet,
                "transit_house": data.get(
                    "transit_house"
                ),
                "rashi": data.get("rashi"),
                "nakshatra": data.get(
                    "nakshatra"
                ),
            })

    return evidence


# ============================================================
# MAIN HEALTH REASONING FUNCTION
# ============================================================

def evaluate_health_question(
    question,
    natal_chart=None,
    current_dasha_hierarchy=None,
    transit_timing=None,
):
    """
    Main health-domain entry point.

    Returns structured routing + astrology evidence.
    """

    mode = classify_health_mode(question)

    result = {
        "domain": "health",
        "mode": mode,
        "medical_diagnosis_supported": False,
        "astrology_evidence": None,
    }

    # --------------------------------------------------------
    # Medical / emergency modes
    # --------------------------------------------------------

    if mode in {
        "medical_information",
        "medical_diagnosis_request",
        "medical_treatment_request",
        "medical_emergency",
    }:
        return result

    # --------------------------------------------------------
    # Astrology / wellness mode
    # --------------------------------------------------------

    if natal_chart is None:
        raise ValueError(
            "natal_chart is required for astrology health questions."
        )

    result["astrology_evidence"] = build_health_evidence(
        natal_chart=natal_chart,
        current_dasha_hierarchy=current_dasha_hierarchy,
        transit_timing=transit_timing,
    )

    return result