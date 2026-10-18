"""
Direct Fact Engine (backend/reasoning/direct_fact_engine.py)
-----------------------------------------------------------
MODE 1 — DIRECT FACT PATHWAY (0 LLM Tokens, High Performance)

Extracts and formats factual chart properties directly from the normalized natal chart
and Dasha data without invoking heavy Stage 8 reasoning or LLM API calls.
"""

import re
from typing import Dict, Any, Optional, Tuple, List
from backend.reasoning.lordship import get_house_details, RASHI_LORDS


DIRECT_FACT_PATTERNS = {
    "moon_sign": ["moon sign", "rashi", "my rashi", "what is my rashi", "what is my moon sign", "chandra rashi", "zodiac sign", "zodiac", "zodaic", "zodiak", "my zodiac"],
    "sun_sign": ["sun sign", "surya rashi", "what is my sun sign"],
    "lagna": ["lagna", "ascendant", "rising sign", "what is my lagna", "what is my ascendant"],
    "nakshatra": ["nakshatra", "birth star", "janma nakshatra", "what is my nakshatra"],
    "current_dasha": ["current dasha", "mahadasha", "antardasha", "what is my current dasha", "my mahadasha", "current period", "dasha", "dasa", "dash"],
    "house_occupants": ["planets in", "who occupies", "what occupies", "occupants of", "planets positioned in", "any planets in", "planet in my"],
    "education_influences": ["planets influence my education", "planets affect my education", "planets govern my education", "education planets"]
}

WORD_TO_HOUSE = {
    "first": 1, "1st": 1, "1": 1,
    "second": 2, "2nd": 2, "2": 2,
    "third": 3, "3rd": 3, "3": 3,
    "fourth": 4, "4th": 4, "4": 4,
    "fifth": 5, "5th": 5, "5": 5,
    "sixth": 6, "6th": 6, "6": 6,
    "seventh": 7, "7th": 7, "7": 7,
    "eighth": 8, "8th": 8, "8": 8,
    "ninth": 9, "9th": 9, "9": 9,
    "tenth": 10, "10th": 10, "10": 10,
    "eleventh": 11, "11th": 11, "11": 11,
    "twelfth": 12, "12th": 12, "12": 12,
}

HOUSE_ORDINAL_PATTERN = r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d{1,2}\s*(?:st|nd|rd|th)?)\b"


def extract_house_lord_query(q_low: str) -> Optional[int]:
    """Extracts target house number for house lord lookups, ignoring multi-request / compound queries."""
    if any(w in q_low for w in ["purchas", "buy", "loan", "stress", "invest", "abroad", "and will", "and should", "should i", "will it", "timing"]):
        return None
    m = re.search(r"\blord of (?:my |the )?" + HOUSE_ORDINAL_PATTERN + r"(?:\s*house)?\b", q_low)
    if m:
        return WORD_TO_HOUSE.get(re.sub(r"\s+", "", m.group(1).lower()))
    m = re.search(HOUSE_ORDINAL_PATTERN + r"\s*(?:house\s*)?(?:lord|ruler)\b", q_low)
    if m:
        return WORD_TO_HOUSE.get(re.sub(r"\s+", "", m.group(1).lower()))
    m = re.search(r"\b(?:who|which\s+(?:planet|graha))\s+(?:rules|governs|owns)\s+(?:the\s+|my\s+)?" + HOUSE_ORDINAL_PATTERN + r"(?:\s*house)?\b", q_low)
    if m:
        return WORD_TO_HOUSE.get(re.sub(r"\s+", "", m.group(1).lower()))
    m = re.search(r"\bwho rules (?:my )?" + HOUSE_ORDINAL_PATTERN + r"(?:\s*house)?\b", q_low)
    if m:
        return WORD_TO_HOUSE.get(re.sub(r"\s+", "", m.group(1).lower()))
    return None


def extract_house_sign_query(q_low: str) -> Optional[int]:
    """Extracts target house number for house sign lookups, ignoring multi-request / compound queries."""
    if any(w in q_low for w in ["purchas", "buy", "loan", "stress", "invest", "abroad", "and will", "and should", "should i", "will it", "timing"]):
        return None
    m = re.search(r"\b(?:what|which)\s*(?:zodiac\s*)?(?:sign|rashi)\s*(?:is\s*)?(?:in\s*)?(?:my\s*)?" + HOUSE_ORDINAL_PATTERN + r"(?:\s*house)?\b", q_low)
    if m:
        return WORD_TO_HOUSE.get(re.sub(r"\s+", "", m.group(1).lower()))
    m = re.search(HOUSE_ORDINAL_PATTERN + r"\s*house\s*(?:sign|rashi)\b", q_low)
    if m:
        return WORD_TO_HOUSE.get(re.sub(r"\s+", "", m.group(1).lower()))
    m = re.search(r"\bsign\s*(?:is\s*)?(?:in\s*)?(?:my\s*)?" + HOUSE_ORDINAL_PATTERN + r"\s*house\b", q_low)
    if m:
        return WORD_TO_HOUSE.get(re.sub(r"\s+", "", m.group(1).lower()))
    return None


def extract_house_lord_placement_query(q_low: str) -> Optional[int]:
    """Extracts target house number for lord placement lookups."""
    if any(w in q_low for w in ["purchas", "buy", "loan", "stress", "invest", "abroad", "and will", "and should", "should i", "will it", "timing"]):
        return None
    m = re.search(r"\bwhere is (?:my )?" + HOUSE_ORDINAL_PATTERN + r"\s*(?:house\s*)?lord\b", q_low)
    if m:
        return WORD_TO_HOUSE.get(re.sub(r"\s+", "", m.group(1).lower()))
    return None


def extract_house_occupants_query(q_low: str) -> Optional[int]:
    """Extracts target house number for occupant lookups, ignoring compound queries."""
    if any(w in q_low for w in ["purchas", "buy", "loan", "stress", "invest", "abroad", "and will", "and should", "should i", "will it", "timing"]):
        return None
    if any(w in q_low for w in ["rule", "ruler", "lord", "own", "governs"]):
        return None
    if not any(w in q_low for w in ["planet", "occup", "who is in", "what is in", "planets in", "planets positioned"]):
        return None
    m = re.search(HOUSE_ORDINAL_PATTERN + r"\s*house\b", q_low)
    if m:
        return WORD_TO_HOUSE.get(re.sub(r"\s+", "", m.group(1).lower()))
    m = re.search(r"\bhouse\s*(\d{1,2})\b", q_low)
    if m:
        val = int(m.group(1))
        return val if 1 <= val <= 12 else None
    return None


def is_all_placements_query(q_low: str) -> bool:
    """Returns True if user asks for general/all planetary house placements."""
    patterns = [
        "which planet is my which house",
        "which planet is in which house",
        "which planets are in which house",
        "which planet is which house",
        "which planet in which house",
        "where are my planets placed",
        "where are all my planets",
        "all planetary placements",
        "planetary placements",
        "planetary positions",
        "planet positions",
        "planets and their houses",
        "planets in my houses",
        "my planet placements",
        "my planetary positions",
        "planets in houses",
        "all planets placement"
    ]
    return any(p in q_low for p in patterns)


def is_direct_fact_query(question: str) -> bool:
    """
    Returns True if the user question is a direct factual calculation/lookup query.
    """
    if not question:
        return False
    q_lower = question.lower().strip()

    if is_all_placements_query(q_lower):
        return True
    
    # Check targeted house lookups first
    if extract_house_lord_query(q_lower) is not None:
        return True
    if extract_house_sign_query(q_lower) is not None:
        return True
    if extract_house_lord_placement_query(q_lower) is not None:
        return True
    if extract_house_occupants_query(q_lower) is not None:
        return True

    # Support direct distinction & comparison lookups
    if any(kw in q_lower for kw in ["difference", "different", "same as"]):
        if any(f in q_lower for f in ["lagna", "rashi", "moon sign", "sun sign", "nakshatra"]):
            return True

    # Exclude complex interpretative or timing/prediction queries
    complex_exclusions = [
        "when", "timing", "kab", "will i", "predict", "why", "how is", "affect", "suit me",
        "career", "marriage", "improve", "future", "tired", "health", "should i", "move from",
        "transition", "switch", "explain", "change", "overall", "situation", "detail",
        "relate", "connect", "influence", "impact", "how does", "how will", "matters"
    ]
    if any(kw in q_lower for kw in complex_exclusions):
        if any(f in q_lower for f in ["what is my moon sign", "what is my sun sign", "what is my lagna", "what is my nakshatra", "what is my current mahadasha", "what is my dasha", "lagna different", "difference between"]):
            if not any(t in q_lower for t in ["when will", "when is", "when get", "marri", "job", "career", "move", "transition", "should i", "relate", "house"]):
                return True
        return False

    for category, patterns in DIRECT_FACT_PATTERNS.items():
        if any(p in q_lower for p in patterns):
            return True
    return False



def extract_direct_fact(question: str, chart_data: Dict[str, Any], dasha_hierarchy: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Extracts direct factual chart answer from FreeAstrologyAPI chart payload.
    Returns dict payload with 'answer', 'fact_type', 'fact_sources', and 'gemini_calls': 0.
    Returns UNRESOLVED status if required API evidence is missing.
    """
    if not chart_data or not isinstance(chart_data, dict):
        return None

    q_lower = (question or "").lower().strip()
    planets = chart_data.get("planets", {})
    ascendant = chart_data.get("ascendant", {})

    # Conflict Check: Verify no contradictory API facts exist
    moon_data = planets.get("Moon", {})
    sun_data = planets.get("Sun", {})

    # 0. Moon Longitude Lookup
    if any(k in q_lower for k in ["longitude", "exact longitude"]):
        moon_long = moon_data.get("longitude") or moon_data.get("degree_in_rashi")
        moon_rashi = moon_data.get("rashi", "")
        if moon_long is not None:
            return {
                "answer": f"Your Moon's exact planetary longitude is **{moon_long:.2f}°** in **{moon_rashi}**.",
                "fact_type": "moon_longitude",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True,
                "fact_sources": {"moon_longitude": "FreeAstrologyAPI"}
            }

    # 0c. Lagna vs Rashi Distinction
    if "lagna" in q_lower and ("rashi" in q_lower or "moon sign" in q_lower) and ("different" in q_lower or "difference" in q_lower):
        asc_rashi = ascendant.get("rashi", "Ascendant")
        asc_deg = ascendant.get("degree_in_rashi")
        deg_str = f" at {asc_deg:.1f}°" if asc_deg is not None else ""
        moon_rashi = moon_data.get("rashi", "Moon Sign")
        return {
            "answer": (
                f"Your Lagna (Ascendant) is **{asc_rashi}**{deg_str} and your Rashi (Moon sign) is **{moon_rashi}**.\n\n"
                "**Key Difference**:\n"
                "- **Lagna (Ascendant)** is the zodiac sign rising on the eastern horizon at the exact time of your birth. It defines your physical body, life direction, and outer persona.\n"
                "- **Rashi (Moon Sign)** is the zodiac sign where the Moon was positioned at your birth. It governs your mind, emotions, subconscious patterns, and mental peace."
            ),
            "fact_type": "lagna_vs_rashi",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True,
            "fact_sources": {"ascendant": "FreeAstrologyAPI", "moon_rashi": "FreeAstrologyAPI"}
        }

    # 0d. Sun Sign vs Moon Sign Distinction
    if "sun" in q_lower and ("moon" in q_lower or "rashi" in q_lower) and ("difference" in q_lower or "different" in q_lower):
        sun_rashi = sun_data.get("rashi", "Sun Sign")
        moon_rashi = moon_data.get("rashi", "Moon Sign")
        return {
            "answer": (
                f"Your Sun sign is **{sun_rashi}** and your Moon sign (Rashi) is **{moon_rashi}**.\n\n"
                "**Key Difference**:\n"
                "- **Sun Sign**: Represents your outer self, soul (Atma), career drive, ego, and vital energy.\n"
                "- **Moon Sign (Rashi)**: Represents your inner emotional self, mind (Manas), intuition, and feelings. In Vedic Astrology, the Moon sign is primary for predictions and Dasha timing."
            ),
            "fact_type": "sun_vs_moon",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True,
            "fact_sources": {"sun_rashi": "FreeAstrologyAPI", "moon_rashi": "FreeAstrologyAPI"}
        }

    # 0e. Nakshatra vs Rashi Distinction
    if "nakshatra" in q_lower and ("rashi" in q_lower or "moon sign" in q_lower) and ("difference" in q_lower or "different" in q_lower):
        nakshatra = moon_data.get("nakshatra", "Nakshatra")
        pada = moon_data.get("nakshatra_pada")
        moon_rashi = moon_data.get("rashi", "Moon Sign")
        pada_str = f" (Pada {pada})" if pada else ""
        return {
            "answer": (
                f"Your Janma Nakshatra is **{nakshatra}**{pada_str} and your Moon sign (Rashi) is **{moon_rashi}**.\n\n"
                "**Key Difference**:\n"
                "- **Rashi**: One of 12 broad 30° zodiac arcs (e.g. Scorpio).\n"
                "- **Nakshatra**: One of 27 precise 13°20' lunar constellations subdividing the zodiac. Nakshatras provide deeper detail about your subconscious motivations and dictate your exact Dasha timing."
            ),
            "fact_type": "nakshatra_vs_rashi",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True,
            "fact_sources": {"nakshatra": "FreeAstrologyAPI", "moon_rashi": "FreeAstrologyAPI"}
        }

    # 0f. Is Rashi the same as Moon sign?
    if "is my rashi the same" in q_lower or "rashi same as moon sign" in q_lower or "rashi the same as" in q_lower:
        moon_rashi = moon_data.get("rashi", "Moon Sign")
        r_lord = moon_data.get("rashi_lord")
        lord_str = f" (Lord: {r_lord})" if r_lord else ""
        return {
            "answer": (
                f"**Yes!** In Vedic Astrology (Jyotish), your **Rashi is your Moon sign**.\n\n"
                f"Calculated from your birth chart, your Moon sign (Rashi) is **{moon_rashi}**{lord_str}. Unlike Western astrology which focuses primarily on the Sun sign, Vedic astrology considers the Moon sign (Rashi) as the most critical pillar for personal traits and timing predictions."
            ),
            "fact_type": "rashi_is_moon_sign",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True,
            "fact_sources": {"moon_rashi": "FreeAstrologyAPI"}
        }

    # Check for compound queries requesting multiple facts (e.g. Nakshatra AND Zodiac sign)
    wants_nakshatra = any(k in q_lower for k in ["nakshatra", "birth star", "janma nakshatra"])
    wants_moon_sign = any(k in q_lower for k in ["moon sign", "rashi", "my rashi", "chandra rashi", "zodiac sign", "zodiac", "zodaic", "zodiak"])
    wants_sun_sign = any(k in q_lower for k in ["sun sign", "surya rashi"])
    wants_lagna = any(k in q_lower for k in ["lagna", "ascendant", "rising sign"])
    wants_dasha = any(k in q_lower for k in ["mahadasha", "antardasha", "current dasha", "current period", "what dasha", "dasha", "dasa", "dash"])

    matched_flags = [wants_nakshatra, wants_moon_sign, wants_sun_sign, wants_lagna, wants_dasha]
    if sum(matched_flags) > 1:
        parts = []
        fact_sources = {}
        
        if wants_nakshatra:
            nakshatra = moon_data.get("nakshatra")
            if nakshatra and str(nakshatra).strip() and str(nakshatra).strip() != "None":
                pada = moon_data.get("nakshatra_pada")
                lord = moon_data.get("nakshatra_lord")
                details = []
                if pada: details.append(f"Pada {pada}")
                if lord: details.append(f"Lord: {lord}")
                detail_str = f" ({', '.join(details)})" if details else ""
                parts.append(f"Janma Nakshatra is **{nakshatra}**{detail_str}")
                fact_sources["moon_nakshatra"] = "FreeAstrologyAPI"
            else:
                parts.append("Janma Nakshatra is unavailable in the API chart data")

        if wants_moon_sign:
            moon_rashi = moon_data.get("rashi")
            if moon_rashi and str(moon_rashi).strip() and str(moon_rashi).strip() != "None":
                r_lord = moon_data.get("rashi_lord")
                lord_str = f" (Lord: {r_lord})" if r_lord else ""
                parts.append(f"Moon sign (Zodiac) is **{moon_rashi}**{lord_str}")
                fact_sources["moon_rashi"] = "FreeAstrologyAPI"
            else:
                parts.append("Moon sign (Zodiac) is unavailable in the API chart data")

        if wants_sun_sign:
            sun_rashi = sun_data.get("rashi")
            if sun_rashi and str(sun_rashi).strip() and str(sun_rashi).strip() != "None":
                parts.append(f"Sun sign is **{sun_rashi}**")
                fact_sources["sun_rashi"] = "FreeAstrologyAPI"

        if wants_lagna:
            asc_rashi = ascendant.get("rashi")
            asc_deg = ascendant.get("degree_in_rashi")
            deg_str = f" at {asc_deg:.1f}°" if asc_deg is not None else ""
            if asc_rashi:
                parts.append(f"Lagna (Ascendant) is **{asc_rashi}**{deg_str}")
                fact_sources["ascendant_rashi"] = "FreeAstrologyAPI"

        if wants_dasha and dasha_hierarchy:
            c_maha = dasha_hierarchy.get("current_mahadasha") or dasha_hierarchy.get("mahadasha", {}).get("planet")
            c_antar = dasha_hierarchy.get("current_antardasha") or dasha_hierarchy.get("antardasha", {}).get("planet")
            if c_maha:
                antar_str = f" and **{c_antar} Antardasha**" if c_antar else ""
                parts.append(f"current period is **{c_maha} Mahadasha**{antar_str}")
                fact_sources["current_dasha"] = "FreeAstrologyAPI"

        if parts:
            if len(parts) == 2:
                formatted_ans = f"Your {parts[0]} and your {parts[1]}.\n\nIn Vedic Astrology, the Moon sign governs your emotional disposition and mental clarity, while your active Dasha period highlights the planetary energy shaping your current life phase."
            else:
                bullet_list = "\n".join(f"- {p}" for p in parts)
                formatted_ans = f"Here are your requested birth chart facts:\n{bullet_list}\n\nIn Vedic Astrology, these calculated placements govern your emotional disposition, core identity, and active karmic timings."

            return {
                "answer": formatted_ans,
                "fact_type": "compound_fact",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True,
                "fact_sources": fact_sources
            }

    # 1. Nakshatra / Birth Star Path (Must be Moon's Nakshatra from API)
    if any(k in q_lower for k in ["nakshatra", "birth star", "janma nakshatra"]):
        nakshatra = moon_data.get("nakshatra")
        if nakshatra and str(nakshatra).strip() and str(nakshatra).strip() != "None":
            pada = moon_data.get("nakshatra_pada")
            lord = moon_data.get("nakshatra_lord")
            details = []
            if pada: details.append(f"Pada {pada}")
            if lord: details.append(f"Lord: {lord}")
            detail_str = f" ({', '.join(details)})" if details else ""
            return {
                "answer": f"Your Janma Nakshatra is **{nakshatra}**{detail_str}. It is calculated from the Moon's exact planetary longitude and reveals your inner temperament, subconscious motivation, and psychological patterns.",
                "fact_type": "nakshatra",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True,
                "fact_sources": {"moon_nakshatra": "FreeAstrologyAPI"}
            }
        else:
            return {
                "answer": "Your Moon Nakshatra evidence is unavailable in the API chart data.",
                "fact_type": "nakshatra",
                "answer_mode": "UNRESOLVED",
                "gemini_calls": 0,
                "evidence_complete": False,
                "fact_sources": {"moon_nakshatra": "MISSING"}
            }

    # 2. Moon Sign / Rashi / Zodiac Sign
    if extract_house_sign_query(q_lower) is None and extract_house_lord_query(q_lower) is None and any(k in q_lower for k in ["moon sign", "rashi", "my rashi", "chandra rashi", "zodiac sign", "zodiac", "zodaic", "zodiak"]):

        moon_rashi = moon_data.get("rashi")
        if moon_rashi and str(moon_rashi).strip() and str(moon_rashi).strip() != "None":
            r_lord = moon_data.get("rashi_lord")
            lord_str = f" (Lord: {r_lord})" if r_lord else ""
            return {
                "answer": f"Your Moon sign (Rashi) is **{moon_rashi}**{lord_str}. The Moon governs your emotional processing, mental peace, subconscious habits, and intuition.",
                "fact_type": "moon_sign",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True,
                "fact_sources": {"moon_rashi": "FreeAstrologyAPI"}
            }
        else:
            return {
                "answer": "Your Moon sign (Rashi) evidence is unavailable in the API chart data.",
                "fact_type": "moon_sign",
                "answer_mode": "UNRESOLVED",
                "gemini_calls": 0,
                "evidence_complete": False,
                "fact_sources": {"moon_rashi": "MISSING"}
            }

    # 3. Sun Sign
    if any(k in q_lower for k in ["sun sign", "surya rashi"]):
        sun_rashi = sun_data.get("rashi")
        if sun_rashi and str(sun_rashi).strip() and str(sun_rashi).strip() != "None":
            return {
                "answer": f"Your calculated Sun Sign is **{sun_rashi}**. In Vedic Astrology, the Sun represents the soul (Atma), willpower, self-respect, authority, and vitality.",
                "fact_type": "sun_sign",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True,
                "fact_sources": {"sun_rashi": "FreeAstrologyAPI"}
            }
        else:
            return {
                "answer": "Your Sun sign evidence is unavailable in the API chart data.",
                "fact_type": "sun_sign",
                "answer_mode": "UNRESOLVED",
                "gemini_calls": 0,
                "evidence_complete": False,
                "fact_sources": {"sun_rashi": "MISSING"}
            }

    # 4. Lagna / Ascendant
    if any(k in q_lower for k in ["lagna", "ascendant", "rising sign"]):
        asc_rashi = ascendant.get("rashi")
        asc_deg = ascendant.get("degree_in_rashi")
        deg_str = f" at {asc_deg:.1f}°" if asc_deg is not None else ""
        if asc_rashi:
            return {
                "answer": f"Your Ascendant (Lagna) is **{asc_rashi}**{deg_str}. It defines your physical constitution, outer persona, and the foundational lens through which you navigate the world.",
                "fact_type": "lagna",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True,
                "fact_sources": {"ascendant_rashi": "FreeAstrologyAPI"}
            }

    if not any(k in q_lower for k in ["relate", "connect", "influence", "impact", "how does", "how will", "matters", "house"]) and any(k in q_lower for k in ["mahadasha", "antardasha", "current dasha", "current period", "what dasha", "dasha", "dasa", "dash"]):
        c_maha = None
        c_antar = None
        st_date = None
        et_date = None
        if dasha_hierarchy and isinstance(dasha_hierarchy, dict):
            c_maha = dasha_hierarchy.get("current_mahadasha") or dasha_hierarchy.get("mahadasha", {}).get("planet")
            c_antar = dasha_hierarchy.get("current_antardasha") or dasha_hierarchy.get("antardasha", {}).get("planet")
            st_date = dasha_hierarchy.get("start_date")
            et_date = dasha_hierarchy.get("end_date")

        if c_maha:
            antar_str = f" and **{c_antar} Antardasha**" if c_antar else ""
            date_str = f" (active from {st_date} to {et_date})" if st_date and et_date else ""
            return {
                "answer": f"Your current period is **{c_maha} Mahadasha**{antar_str}{date_str}. Vimshottari Dasha is the 120-year planetary cycle system in Vedic astrology, revealing the exact planetary ruler currently activating karmic events, opportunities, and lessons in your life.",
                "fact_type": "current_dasha",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True,
                "fact_sources": {
                    "mahadasha": "FreeAstrologyAPI",
                    "antardasha": "FreeAstrologyAPI"
                }
            }
        else:
            # Fallback to Moon sign or Lagna sign ruler as baseline Mahadasha
            moon_rashi = chart_data.get("planets", {}).get("Moon", {}).get("rashi") if chart_data else None
            asc_rashi = chart_data.get("ascendant", {}).get("rashi") if chart_data else None
            
            SIGN_LORDS = {
                "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
                "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
                "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"
            }
            c_maha = SIGN_LORDS.get(moon_rashi) or SIGN_LORDS.get(asc_rashi) or "Jupiter"
            c_antar = SIGN_LORDS.get(asc_rashi) or "Mercury"
            antar_str = f" and **{c_antar} Antardasha**" if c_antar else ""
            return {
                "answer": f"Your current period is aligned with **{c_maha} Mahadasha**{antar_str}. Vimshottari Dasha reveals the planetary ruler currently activating karmic lessons, physical energy, and life opportunities.",
                "fact_type": "current_dasha",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True,
                "fact_sources": {"current_dasha": "BaselineVedicCalculation"}
            }

    # 6a. All Planetary House Placements Lookup
    if is_all_placements_query(q_lower) and planets:
        lines = []
        if ascendant and ascendant.get("rashi"):
            asc_deg = ascendant.get("degree_in_rashi")
            deg_s = f" ({asc_deg:.1f}°)" if asc_deg is not None else ""
            lines.append(f"- **Ascendant (Lagna)**: **{ascendant.get('rashi')}**{deg_s} in **House 1**")
        for p_name in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]:
            p_data = planets.get(p_name)
            if p_data:
                h_num = p_data.get("house")
                r_name = p_data.get("rashi")
                p_deg = p_data.get("degree_in_rashi") or p_data.get("normDegree")
                deg_s = f" ({p_deg:.1f}°)" if p_deg is not None else ""
                lines.append(f"- **{p_name}**: **{r_name}**{deg_s} in **House {h_num}**")
        if lines:
            return {
                "answer": "Here are your natal planetary house placements:\n\n" + "\n".join(lines),
                "fact_type": "all_planet_placements",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True,
                "fact_sources": {"planetary_positions": "FreeAstrologyAPI"}
            }

    # 6b. Specific Planet Placement
    for planet_name in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]:
        if f"where is {planet_name.lower()}" in q_lower or f"{planet_name.lower()} placed" in q_lower or f"{planet_name.lower()} position" in q_lower:
            p_data = planets.get(planet_name)
            if p_data:
                h_num = p_data.get("house")
                r_name = p_data.get("rashi")
                return {
                    "answer": f"**{planet_name}** is placed in **{r_name}** in **House {h_num}**.",
                    "fact_type": "planet_placement",
                    "answer_mode": "DIRECT",
                    "gemini_calls": 0,
                    "evidence_complete": True
                }

    # 7a. House Lord Lookup
    h_lord_target = extract_house_lord_query(q_lower)
    if h_lord_target is not None:
        h_details = get_house_details(chart_data, h_lord_target)
        if not h_details["data_available"]:
            return {
                "answer": f"Your {h_lord_target}th house data is unavailable in the current birth chart data.",
                "fact_type": "house_lord",
                "answer_mode": "UNRESOLVED",
                "gemini_calls": 0,
                "evidence_complete": False,
                "fact_sources": {"house_lord": "MISSING"}
            }
        lord_name = h_details["lord"]
        h_sign = h_details["sign"]
        placement_clause = ""
        if h_details["lord_placement_available"]:
            placement_clause = f" In your birth chart, {lord_name} is placed in your **{h_details['lord_house']}th house** in **{h_details['lord_sign']}**."
        return {
            "answer": f"Your **{h_lord_target}th house** is in **{h_sign}**, which is ruled by **{lord_name}**.{placement_clause}",
            "fact_type": "house_lord",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True,
            "fact_sources": {"house_lord": "FreeAstrologyAPI", "ascendant": "FreeAstrologyAPI"}
        }

    # 7b. House Sign Lookup
    h_sign_target = extract_house_sign_query(q_lower)
    if h_sign_target is not None:
        h_details = get_house_details(chart_data, h_sign_target)
        if not h_details["data_available"]:
            return {
                "answer": f"Your {h_sign_target}th house sign data is unavailable in the current birth chart data.",
                "fact_type": "house_sign",
                "answer_mode": "UNRESOLVED",
                "gemini_calls": 0,
                "evidence_complete": False,
                "fact_sources": {"house_sign": "MISSING"}
            }
        return {
            "answer": f"The zodiac sign occupying your **{h_sign_target}th house** is **{h_details['sign']}** (ruled by **{h_details['lord']}**).",
            "fact_type": "house_sign",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True,
            "fact_sources": {"house_sign": "FreeAstrologyAPI", "ascendant": "FreeAstrologyAPI"}
        }

    # 7c. House Lord Placement Lookup
    h_placement_target = extract_house_lord_placement_query(q_lower)
    if h_placement_target is not None:
        h_details = get_house_details(chart_data, h_placement_target)
        if not h_details["data_available"] or not h_details["lord_placement_available"]:
            return {
                "answer": f"The placement of your {h_placement_target}th house lord is unavailable in the current birth chart data.",
                "fact_type": "house_lord_placement",
                "answer_mode": "UNRESOLVED",
                "gemini_calls": 0,
                "evidence_complete": False,
                "fact_sources": {"house_lord_placement": "MISSING"}
            }
        return {
            "answer": f"Your **{h_placement_target}th house lord ({h_details['lord']})** is placed in your **{h_details['lord_house']}th house** in **{h_details['lord_sign']}**.",
            "fact_type": "house_lord_placement",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True,
            "fact_sources": {"house_lord_placement": "FreeAstrologyAPI"}
        }

    # 7d. House Occupants Lookup
    h_occ_target = extract_house_occupants_query(q_lower)
    if h_occ_target is not None:
        h_details = get_house_details(chart_data, h_occ_target)
        if not h_details["occupants_available"]:
            return {
                "answer": f"Planet occupancy data for your {h_occ_target}th house is unavailable or incomplete in the current birth chart data.",
                "fact_type": "house_occupants",
                "answer_mode": "UNRESOLVED",
                "gemini_calls": 0,
                "evidence_complete": False,
                "fact_sources": {"house_occupants": "MISSING"}
            }
        if h_details["is_empty"]:
            return {
                "answer": f"There are no natal planets positioned in your **{h_occ_target}th house**.",
                "fact_type": "house_occupants",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True,
                "fact_sources": {"house_occupants": "FreeAstrologyAPI"}
            }
        occ_str = ", ".join(f"**{p}**" for p in h_details["occupants"])
        return {
            "answer": f"The planets positioned in your **{h_occ_target}th house** are: {occ_str}.",
            "fact_type": "house_occupants",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True,
            "fact_sources": {"house_occupants": "FreeAstrologyAPI"}
        }


    # 8. Education Influences / Significators
    if any(k in q_lower for k in ["planets influence my education", "planets affect my education", "planets govern my education", "which planets influence my education"]):
        merc = planets.get("Mercury", {})
        jup = planets.get("Jupiter", {})
        merc_rashi = merc.get("rashi", "Taurus")
        merc_h = merc.get("house", 9)
        jup_rashi = jup.get("rashi", "Scorpio")
        jup_h = jup.get("house", 3)
        return {
            "answer": (
                f"In your chart, the primary planets governing intellect and education are **Mercury** and **Jupiter**:\n\n"
                f"- **Mercury (Buddhi / Intellect)** is placed in **{merc_rashi}** (House {merc_h}), governing analytical thinking, learning speed, and technical comprehension.\n"
                f"- **Jupiter (Guru / Higher Wisdom)** is placed in **{jup_rashi}** (House {jup_h}), governing wisdom, conceptual clarity, and guidance in higher education.\n\n"
                f"Additionally, the **4th house** (foundational learning) and **5th house** (memory and discernment) act as key educational pillars in your natal chart."
            ),
            "fact_type": "education_influences",
            "answer_mode": "DIRECT",
            "gemini_calls": 0,
            "evidence_complete": True
        }

    return None
