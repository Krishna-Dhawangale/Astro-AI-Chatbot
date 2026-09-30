"""
Direct Fact Engine (backend/reasoning/direct_fact_engine.py)
-----------------------------------------------------------
MODE 1 — DIRECT FACT PATHWAY (0 LLM Tokens, High Performance)

Extracts and formats factual chart properties directly from the normalized natal chart
and Dasha data without invoking heavy Stage 8 reasoning or LLM API calls.
"""

from typing import Dict, Any, Optional, Tuple


DIRECT_FACT_PATTERNS = {
    "moon_sign": ["moon sign", "rashi", "my rashi", "what is my rashi", "what is my moon sign", "chandra rashi"],
    "sun_sign": ["sun sign", "surya rashi", "what is my sun sign"],
    "lagna": ["lagna", "ascendant", "rising sign", "what is my lagna", "what is my ascendant"],
    "nakshatra": ["nakshatra", "birth star", "janma nakshatra", "what is my nakshatra"],
    "current_dasha": ["current dasha", "mahadasha", "antardasha", "what is my current dasha", "my mahadasha", "current period"],
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
    
    # Exclude complex interpretative queries containing 'why', 'how', 'affect', 'suit me'
    if any(kw in q_lower for kw in ["why", "how is", "affect", "suit me", "career", "marriage", "improve", "future", "tired", "health"]):
        # Check if query is explicitly asking a factual question despite keywords
        if any(f in q_lower for f in ["what is my moon sign", "what is my sun sign", "what is my lagna", "what is my nakshatra", "what is my current mahadasha"]):
            return True
        return False

    for category, patterns in DIRECT_FACT_PATTERNS.items():
        if any(p in q_lower for p in patterns):
            return True
    return False


def extract_direct_fact(question: str, chart_data: Dict[str, Any], dasha_hierarchy: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Extracts direct factual chart answer from chart payload.
    Returns dict payload with 'answer', 'fact_type', and 'gemini_calls': 0.
    """
    if not chart_data or not isinstance(chart_data, dict):
        return None

    q_lower = (question or "").lower().strip()
    planets = chart_data.get("planets", {})
    ascendant = chart_data.get("ascendant", {})

    # 1. Moon Sign / Rashi
    if any(k in q_lower for k in ["moon sign", "rashi", "chandra rashi"]):
        moon_data = planets.get("Moon", {})
        moon_rashi = moon_data.get("rashi")
        if moon_rashi:
            return {
                "answer": f"Your Moon sign (Rashi) is **{moon_rashi}**.",
                "fact_type": "moon_sign",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True
            }

    # 2. Sun Sign
    if any(k in q_lower for k in ["sun sign", "surya rashi"]):
        sun_data = planets.get("Sun", {})
        sun_rashi = sun_data.get("rashi")
        if sun_rashi:
            return {
                "answer": f"Your Sun sign is **{sun_rashi}**.",
                "fact_type": "sun_sign",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True
            }

    # 3. Lagna / Ascendant
    if any(k in q_lower for k in ["lagna", "ascendant", "rising sign"]):
        asc_rashi = ascendant.get("rashi")
        asc_deg = ascendant.get("degree_in_rashi")
        deg_str = f" at {asc_deg:.1f}°" if asc_deg is not None else ""
        if asc_rashi:
            return {
                "answer": f"Your Lagna (Ascendant) is **{asc_rashi}**{deg_str}.",
                "fact_type": "lagna",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True
            }

    # 4. Nakshatra
    if any(k in q_lower for k in ["nakshatra", "birth star", "janma nakshatra"]):
        moon_data = planets.get("Moon", {})
        nakshatra = moon_data.get("nakshatra")
        pada = moon_data.get("pada")
        pada_str = f" (Pada {pada})" if pada else ""
        if nakshatra:
            return {
                "answer": f"Your Janma Nakshatra is **{nakshatra}**{pada_str}.",
                "fact_type": "nakshatra",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True
            }

    # 5. Current Mahadasha / Antardasha
    if any(k in q_lower for k in ["mahadasha", "antardasha", "current dasha", "current period"]):
        if dasha_hierarchy:
            maha = dasha_hierarchy.get("mahadasha", {})
            antar = dasha_hierarchy.get("antardasha", {})
            m_planet = maha.get("planet", "Unknown")
            a_planet = antar.get("planet", "Unknown")
            m_end = maha.get("end", "")
            end_str = f" (active until {m_end})" if m_end else ""
            return {
                "answer": f"Your current period is **{m_planet} Mahadasha** and **{a_planet} Antardasha**{end_str}.",
                "fact_type": "current_dasha",
                "answer_mode": "DIRECT",
                "gemini_calls": 0,
                "evidence_complete": True
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
