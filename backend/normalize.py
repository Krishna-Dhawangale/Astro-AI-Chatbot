"""
normalize.py — Standardized Adapter Layer
------------------------------------------
Adapts raw API response payloads (FreeAstrologyAPI / Prokerala API)
into the unified InternalAstroSchema expected by Your Reasoning Engine.
"""

from typing import Dict, Any, List

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]


def normalize_api_response(api_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes FreeAstrologyAPI planet extended payload into the
    Common Internal Astro Schema required by stage8_pipeline and pipeline_helper.
    """
    if not isinstance(api_response, dict):
        raise ValueError("Invalid API response: expected dictionary.")

    output = api_response.get("output", {})
    if not isinstance(output, dict):
        output = api_response  # Fallback if unnested

    planets: Dict[str, Dict[str, Any]] = {}
    ascendant = {"longitude": 0.0, "rashi": "Aries", "degree_in_rashi": 0.0}

    planet_names = [
        "Ascendant", "Sun", "Moon", "Mars", "Mercury",
        "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"
    ]

    for p_name in planet_names:
        p_data = output.get(p_name, {})
        if not isinstance(p_data, dict):
            continue

        # Extract Sign Name & Lord
        rashi = p_data.get("zodiac_sign_name")
        if not isinstance(rashi, str):
            curr_sign = p_data.get("current_sign")
            if isinstance(curr_sign, int) and 1 <= curr_sign <= 12:
                rashi = ZODIAC_SIGNS[curr_sign - 1]
            else:
                rashi = None
        else:
            rashi = rashi.strip().title()

        rashi_lord = p_data.get("zodiac_sign_lord")

        # Extract Longitude / Degrees
        full_deg = p_data.get("fullDegree", p_data.get("longitude"))
        norm_deg = p_data.get("normDegree", full_deg % 30 if isinstance(full_deg, (int, float)) else None)
        house = p_data.get("house_number", 1)

        nakshatra = p_data.get("nakshatra_name")
        nakshatra_pada = p_data.get("nakshatra_pada")
        nakshatra_lord = p_data.get("nakshatra_vimsottari_lord") or p_data.get("nakshatra_lord_name")

        planet_obj = {
            "longitude": float(full_deg) if isinstance(full_deg, (int, float)) else 0.0,
            "degree_in_rashi": float(norm_deg) if isinstance(norm_deg, (int, float)) else 0.0,
            "rashi": rashi,
            "rashi_lord": rashi_lord,
            "house": int(house) if isinstance(house, int) else 1,
            "nakshatra": nakshatra,
            "nakshatra_pada": nakshatra_pada,
            "nakshatra_lord": nakshatra_lord
        }

        if p_name == "Ascendant":
            ascendant = {
                "longitude": planet_obj["longitude"],
                "rashi": planet_obj["rashi"],
                "degree_in_rashi": planet_obj["degree_in_rashi"],
                "nakshatra": nakshatra
            }
        else:
            planets[p_name] = planet_obj

    return {
        "ascendant": ascendant,
        "planets": planets,
        "raw_source": "FreeAstrologyAPI"
    }


def normalize_dasha_response(dasha_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes Vimshottari Dasha payload into standardized Dasha hierarchy.
    """
    if not isinstance(dasha_response, dict):
        return {"current_dasha": {}, "dasha_hierarchy": {}, "raw_source": "FreeAstrologyAPI"}

    output = dasha_response.get("output", dasha_response)
    if isinstance(output, str):
        import json
        try:
            output = json.loads(output)
        except Exception:
            output = {}

    current_mahadasha = None
    current_antardasha = None
    start_date = None
    end_date = None

    if isinstance(output, dict):
        from datetime import datetime
        now = datetime.now()
        for maha, antars in output.items():
            if isinstance(antars, dict):
                for antar, dates in antars.items():
                    if isinstance(dates, dict):
                        st_str = dates.get("start_time") or dates.get("start")
                        et_str = dates.get("end_time") or dates.get("end")
                        if st_str and et_str:
                            try:
                                st = datetime.strptime(st_str[:19], "%Y-%m-%d %H:%M:%S")
                                et = datetime.strptime(et_str[:19], "%Y-%m-%d %H:%M:%S")
                                if st <= now <= et:
                                    current_mahadasha = maha
                                    current_antardasha = antar
                                    start_date = st_str
                                    end_date = et_str
                                    break
                            except Exception:
                                pass
                if current_mahadasha:
                    break

    return {
        "current_mahadasha": current_mahadasha,
        "current_antardasha": current_antardasha,
        "start_date": start_date,
        "end_date": end_date,
        "dasha_hierarchy": output if isinstance(output, (dict, list)) else {},
        "raw_source": "FreeAstrologyAPI"
    }


def normalize_prokerala_response(chart_svg: str) -> Dict[str, Any]:
    """
    Normalizes Prokerala SVG chart payload.
    """
    return {
        "chart_svg": chart_svg,
        "format": "svg+xml",
        "raw_source": "Prokerala"
    }
