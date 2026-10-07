"""
Direct Fact Engine (backend/reasoning/direct_fact_engine.py)
-----------------------------------------------------------
MODE 1 — DIRECT FACT PATHWAY (0 LLM Tokens, High Performance)

Extracts and formats factual chart properties directly from the normalized natal chart
and Dasha data without invoking heavy Stage 8 reasoning or LLM API calls.
"""

from typing import Dict, Any, Optional, Tuple


DIRECT_FACT_PATTERNS = {
    "moon_sign": ["moon sign", "rashi", "my rashi", "what is my rashi", "what is my moon sign", "chandra rashi", "zodiac sign", "zodiac", "zodaic", "zodiak", "my zodiac"],
    "sun_sign": ["sun sign", "surya rashi", "what is my sun sign"],
    "lagna": ["lagna", "ascendant", "rising sign", "what is my lagna", "what is my ascendant"],
    "nakshatra": ["nakshatra", "birth star", "janma nakshatra", "what is my nakshatra"],
    "current_dasha": ["current dasha", "mahadasha", "antardasha", "what is my current dasha", "my mahadasha", "current period", "dasha", "dasa", "dash"],
    "planet_placement": ["where is jupiter", "where is saturn", "where is mars", "where is venus", "where is mercury", "where is sun", "where is moon", "where is rahu", "where is ketu"],
    "house_occupants": ["7th house", "10th house", "1st house", "2nd house", "4th house", "5th house", "6th house", "8th house", "9th house", "11th house", "12th house"]
}


def is_direct_fact_query(question: str) -> bool:
    """
    Returns True if the user question is a direct factual calculation/lookup query.
    """
    if not question:
        return False
    q_lower = question.lower().strip()
    
    # Support direct distinction & comparison lookups
    if any(kw in q_lower for kw in ["difference", "different", "same as"]):
        if any(f in q_lower for f in ["lagna", "rashi", "moon sign", "sun sign", "nakshatra"]):
            return True

    # Exclude complex interpretative or timing/prediction queries
    complex_exclusions = [
        "when", "timing", "kab", "will i", "predict", "why", "how is", "affect", "suit me",
        "career", "marriage", "improve", "future", "tired", "health", "should i", "move from",
        "transition", "switch", "explain", "change", "overall", "situation", "detail"
    ]
    if any(kw in q_lower for kw in complex_exclusions):
        if any(f in q_lower for f in ["what is my moon sign", "what is my sun sign", "what is my lagna", "what is my nakshatra", "what is my current mahadasha", "what is my dasha", "lagna different", "difference between"]):
            if not any(t in q_lower for t in ["when will", "when is", "when get", "marri", "job", "career", "move", "transition", "should i"]):
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
    if any(k in q_lower for k in ["moon sign", "rashi", "my rashi", "chandra rashi", "zodiac sign", "zodiac", "zodaic", "zodiak"]):
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

    # 5. Current Mahadasha / Antardasha Path
    if any(k in q_lower for k in ["mahadasha", "antardasha", "current dasha", "current period", "what dasha", "dasha", "dasa", "dash"]):
        if dasha_hierarchy:
            c_maha = dasha_hierarchy.get("current_mahadasha") or dasha_hierarchy.get("mahadasha", {}).get("planet")
            c_antar = dasha_hierarchy.get("current_antardasha") or dasha_hierarchy.get("antardasha", {}).get("planet")
            if c_maha:
                antar_str = f" and **{c_antar} Antardasha**" if c_antar else ""
                return {
                    "answer": f"Your current period is **{c_maha} Mahadasha**{antar_str}. Vimshottari Dasha is the 120-year planetary cycle system in Vedic astrology, revealing the exact planetary ruler currently activating karmic events, opportunities, and lessons in your life.",
                    "fact_type": "current_dasha",
                    "answer_mode": "DIRECT",
                    "gemini_calls": 0,
                    "evidence_complete": True,
                    "fact_sources": {"current_dasha": "FreeAstrologyAPI"}
                }
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
                return {
                    "answer": "Your Dasha timing evidence is unavailable in the API chart data.",
                    "fact_type": "current_dasha",
                    "answer_mode": "UNRESOLVED",
                    "gemini_calls": 0,
                    "evidence_complete": False,
                    "fact_sources": {"current_dasha": "MISSING"}
                }

    # 6. Specific Planet Placement
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

    # 7. House Occupants
    for h in range(1, 13):
        h_words = [f"{h}th house", f"{h}st house", f"{h}nd house", f"{h}rd house"]
        if any(w in q_lower for w in h_words) and any(q in q_lower for q in ["planet", "occupy", "occupants", "in my"]):
            occupants = [p_name for p_name, p_info in planets.items() if p_info.get("house") == h]
            if occupants:
                occ_str = ", ".join(f"**{p}**" for p in occupants)
                return {
                    "answer": f"The planets positioned in your **{h}th house** are: {occ_str}.",
                    "fact_type": "house_occupants",
                    "answer_mode": "DIRECT",
                    "gemini_calls": 0,
                    "evidence_complete": True
                }
            else:
                return {
                    "answer": f"There are no natal planets positioned in your **{h}th house**.",
                    "fact_type": "house_occupants",
                    "answer_mode": "DIRECT",
                    "gemini_calls": 0,
                    "evidence_complete": True
                }

    return None
