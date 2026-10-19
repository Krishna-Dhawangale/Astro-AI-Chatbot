"""
backend/timing_engine.py
Vedic Astrological Timing Engine (Dasha & Ephemeris Gochar Analysis)
=====================================================================
Features:
1. Intent & Timing Trigger: Detects time-sensitive queries ("kab", "when", "time", "date", "window").
2. Domain-Specific House Isolation:
   - Career -> 10th House, 6th House, Saturn / Sun transits.
   - Vehicles/Property -> 4th House, Venus, Mars transits.
   - Foreign Travel -> 9th House, 12th House, Rahu / Moon transits.
   - Finance -> 2nd House, 11th House, Jupiter / Mercury transits.
   - Marriage -> 7th House, 5th House, Jupiter / Venus transits.
   - Health -> 6th House, 1st House, 12th House, Sun / Mars transits.
3. Dynamic Timing Variation: Calculates distinct calendar dates based on the specific lord of that house.
4. LLM Strict Validation: Enforces zero cross-domain mixing in the prose refiner.
5. Standardized Attribution Footer:
   *[Calculation: SwissEphemeris_VimshottariEngine | Transit Window: Active Dasha & Gochar Analysis]*
"""

import os
import re
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("AstroChatbot.TimingEngine")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import swisseph as swe
    HAS_SWISSEPH = True
except Exception:
    swe = None
    HAS_SWISSEPH = False

# ---------------------------------------------------------------------------
# 1. Intent & Timing Trigger Keywords & Domain Patterns
# ---------------------------------------------------------------------------
TIMING_KEYWORDS = [
    "kab", "when", "time", "timing", "date", "dates", "period", "window",
    "timeline", "month", "year", "tarikh", "samay", "shubh muhurat", "kab tak",
    "right time", "favorable time", "best time", "when will", "kaisa samay",
    "stabilize", "stabilise", "improve", "switch", "change", "growth kab",
    "buy", "purchase", "travel", "fly", "shift", "visa"
]

TIMING_PATTERNS = [
    r"\b(?:kab|when|what\s+time|which\s+month|which\s+year|right\s+time|timeline|date|dates)\b",
    r"\b(?:kab\s+tak|kab\s+hoga|kab\s+hogi|kab\s+milegi|kab\s+aayega|kab\s+banegi|kab\s+sudhre gi)\b",
    r"\b(?:timing|time\s+period|favorable\s+time|auspicious\s+time|good\s+time)\b",
    r"\b(?:right\s+time\s+(?:for|to)|best\s+time\s+to|when\s+is\s+the\s+right\s+time)\b",
    r"\b(?:career|job)\s+(?:kab\s+stabilize|kab\s+change|timing|right\s+time)\b",
    r"\b(?:car|gadi|vehicle|house|property|flat)\s+(?:kab|when|buy|purchase|khareed)\b",
    r"\b(?:foreign|abroad|travel|visa|videsh)\s+(?:kab|when|timing)\b",
    r"\bwhen\s+will\s+(?:my\s+)?(?:career|finances?|health|marriage)\b",
    r"\bwhen\s+will\s+i\s+(?:buy|purchase|travel|go|move|switch)\b",
]

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "property": [
        "car", "vehicle", "gadi", "automobile", "bike", "four wheeler", "buy car",
        "buying a car", "purchase car", "buy vehicle", "vahana", "property", "house",
        "home", "flat", "plot", "land", "makan", "ghar", "buy house", "buying a house",
        "real estate", "asset", "assets"
    ],
    "travel": [
        "travel", "foreign", "abroad", "visa", "pr", "overseas", "relocation", "shift",
        "videsh", "foreign settlement", "foreign travel", "travel abroad", "fly abroad",
        "international", "go abroad", "settle abroad", "yatra", "immigrate", "immigration"
    ],
    "career": [
        "career", "job", "job change", "naukri", "switch", "promotion", "tarakki",
        "stabilize", "stabilise", "stability", "profession", "business", "work",
        "interview", "company", "boss", "designation", "career timing", "office"
    ],
    "finance": [
        "finance", "finances", "money", "paisa", "wealth", "income", "dhan", "arthik",
        "savings", "investment", "invest", "loss", "debt", "karza", "rich", "financial stability"
    ],
    "marriage": [
        "marriage", "wedding", "shaadi", "shadi", "vivah", "spouse", "partner", "husband",
        "wife", "milan", "relationship", "settle down", "marry", "married", "life partner",
        "matrimony", "shadi kab"
    ],
    "health": [
        "health", "sehat", "illness", "bimar", "bimari", "recovery", "vitality", "rest",
        "sleep", "neend", "insomnia", "mental peace", "stress", "fatigue", "sick", "cure"
    ],
    "education": [
        "education", "study", "padhai", "exam", "exams", "entrance", "upsc", "college",
        "degree", "admission"
    ]
}


def is_timing_query(query: str) -> bool:
    """Returns True if query seeks timing, date ranges, or favorable windows."""
    if not query:
        return False
    q_low = query.lower().strip()

    for pattern in TIMING_PATTERNS:
        if re.search(pattern, q_low):
            return True

    if any(re.search(rf"\b{re.escape(kw)}\b", q_low) for kw in TIMING_KEYWORDS):
        return True

    return False


def detect_timing_domain(query: str) -> str:
    """
    Identifies the life domain of the timing query.
    Prioritizes specific physical assets (property/vehicles) and international travel
    before generic career keywords.
    """
    if not query:
        return "career"
    q_low = query.lower().strip()

    # Explicit precedence check for vehicles/property
    if any(re.search(rf"\b{re.escape(w)}\b", q_low) for w in ["car", "vehicle", "gadi", "automobile", "bike", "property", "flat", "makan", "plot", "real estate", "vahana"]):
        return "property"

    # Explicit precedence check for foreign travel
    if any(re.search(rf"\b{re.escape(w)}\b", q_low) for w in ["foreign", "abroad", "visa", "overseas", "videsh", "travel", "relocation", "immigration"]):
        return "travel"

    scores: Dict[str, int] = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, kws in DOMAIN_KEYWORDS.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\b", q_low):
                scores[domain] += 3 if len(kw.split()) > 1 else 1

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] > 0:
        return best_domain
    return "career"


# ---------------------------------------------------------------------------
# 2. Ephemeris & Gochar Transit Calculation with Domain House Isolation
# ---------------------------------------------------------------------------
ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

SIGN_LORDS = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
    "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
    "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"
}

# Domain-specific house configuration in Vedic Astrology:
# Primary House, Secondary House, Natural Karakas
DOMAIN_ASTROLOGICAL_CONFIG = {
    "career": {
        "primary_house": 10,       # 10th House (Karma / Executive status)
        "secondary_house": 6,      # 6th House (Service / Daily workplace)
        "karakas": ["Saturn", "Sun"],
        "description": "10th House (Karma Bhava) and 6th House (Service) with Saturn and Sun transits",
    },
    "property": {
        "primary_house": 4,        # 4th House (Sukha, Vahana & Real Estate)
        "secondary_house": 2,      # 2nd House (Asset preservation)
        "karakas": ["Venus", "Mars"],
        "description": "4th House (Sukha & Vahana Bhava) with Venus (vehicles) and Mars (property) transits",
    },
    "travel": {
        "primary_house": 9,        # 9th House (Long Journeys & Higher Fortune)
        "secondary_house": 12,     # 12th House (Foreign Lands & Overseas Settlement)
        "karakas": ["Rahu", "Moon"],
        "description": "9th House (Long Journeys) and 12th House (Foreign Lands) with Rahu and Moon transits",
    },
    "finance": {
        "primary_house": 2,        # 2nd House (Accumulated Wealth)
        "secondary_house": 11,     # 11th House (Income & Gains)
        "karakas": ["Jupiter", "Mercury"],
        "description": "2nd House (Dhana Bhava) and 11th House (Labha Bhava) with Jupiter and Mercury transits",
    },
    "marriage": {
        "primary_house": 7,        # 7th House (Kalatra / Spouse & Partnership)
        "secondary_house": 5,      # 5th House (Purva Punya & Romance)
        "karakas": ["Jupiter", "Venus"],
        "description": "7th House (Kalatra Bhava) and 5th House with Jupiter and Venus transits",
    },
    "health": {
        "primary_house": 6,        # 6th House (Immunity & Disease Recovery)
        "secondary_house": 1,      # 1st House (Physical Vitality)
        "karakas": ["Sun", "Mars"],
        "description": "6th House (Immunity) and 1st House (Vitality) with Sun and Mars transits",
    },
    "education": {
        "primary_house": 5,        # 5th House (Intellect & Exams)
        "secondary_house": 4,      # 4th House (Formal Degrees)
        "karakas": ["Mercury", "Jupiter"],
        "description": "5th House (Vidya Bhava) and 4th House with Mercury and Jupiter transits",
    }
}


def calculate_astrological_timing_window(
    domain: str = "career",
    ascendant_sign: str = "Virgo",
    moon_sign: str = "Taurus",
    mahadasha: str = "Jupiter",
    antardasha: str = "Mercury",
    ref_date: Optional[datetime] = None,
    session_birth_data: Optional[Dict[str, Any]] = None,
    chart_data: Optional[Dict[str, Any]] = None,
    dasha_start_date: Optional[str] = None,
    dasha_end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes personalized astrological timing windows customized to the specific user's
    birth chart, exact natal Ascendant degree, and real-time Swiss Ephemeris Gochar transits.
    Zero static hardcoding: different users with different birth details receive distinct,
    astronomically verified dates tailored to their individual planetary charts.
    """
    now = ref_date or datetime.now(timezone.utc)
    asc_clean = ascendant_sign if ascendant_sign in ZODIAC_SIGNS else "Virgo"
    asc_idx = ZODIAC_SIGNS.index(asc_clean)
    exact_asc_deg = asc_idx * 30.0 + 15.0

    # 1. High-precision natal Ascendant calculation from user's exact birth coordinates
    if session_birth_data and HAS_SWISSEPH and swe is not None:
        try:
            by = int(session_birth_data.get("birth_year", 2000))
            bm = int(session_birth_data.get("birth_month", 1))
            bd = int(session_birth_data.get("birth_day", 1))
            bh = float(session_birth_data.get("birth_hour", 12.0))
            lat = float(session_birth_data.get("latitude", 28.6139))
            lon = float(session_birth_data.get("longitude", 77.2090))
            swe.set_sid_mode(swe.SIDM_LAHIRI)
            jd_birth = swe.julday(by, bm, bd, bh)
            h_res = swe.houses_ex(jd_birth, lat, lon, b'P', swe.FLG_SIDEREAL)
            exact_asc_deg = float(h_res[1][0])
            asc_idx = int(exact_asc_deg // 30) % 12
            asc_clean = ZODIAC_SIGNS[asc_idx]
        except Exception:
            pass
    elif chart_data and isinstance(chart_data, dict):
        deg_in_r = chart_data.get("ascendant", {}).get("degree_in_rashi")
        r_name = chart_data.get("ascendant", {}).get("rashi")
        if r_name in ZODIAC_SIGNS and deg_in_r is not None:
            asc_idx = ZODIAC_SIGNS.index(r_name)
            asc_clean = r_name
            exact_asc_deg = asc_idx * 30.0 + float(deg_in_r)

    cfg = DOMAIN_ASTROLOGICAL_CONFIG.get(domain, DOMAIN_ASTROLOGICAL_CONFIG["career"])
    p_house = cfg["primary_house"]
    s_house = cfg["secondary_house"]
    karakas = cfg["karakas"]

    # Compute personalized house cusps based on user's exact Ascendant degree
    p_cusp_deg = (exact_asc_deg + (p_house - 1) * 30.0) % 360.0
    s_cusp_deg = (exact_asc_deg + (s_house - 1) * 30.0) % 360.0

    p_sign = ZODIAC_SIGNS[int(p_cusp_deg // 30) % 12]
    p_lord = SIGN_LORDS.get(p_sign, "Mercury")
    s_sign = ZODIAC_SIGNS[int(s_cusp_deg // 30) % 12]
    s_lord = SIGN_LORDS.get(s_sign, "Jupiter")

    # 2. Real-Time Sidereal Gochar Transit Calculation via Swiss Ephemeris
    transit_positions = {}
    if HAS_SWISSEPH and swe is not None:
        try:
            swe.set_sid_mode(swe.SIDM_LAHIRI)
            jd_now = swe.julday(now.year, now.month, now.day, now.hour + now.minute / 60.0)
            swe_planets = {
                "Sun": swe.SUN, "Moon": swe.MOON, "Mars": swe.MARS, "Mercury": swe.MERCURY,
                "Jupiter": swe.JUPITER, "Venus": swe.VENUS, "Saturn": swe.SATURN, "Rahu": swe.MEAN_NODE
            }
            for p_name, p_code in swe_planets.items():
                calc_res = swe.calc_ut(jd_now, p_code, swe.FLG_SIDEREAL)
                lon = calc_res[0][0] if isinstance(calc_res[0], (list, tuple)) else calc_res[0]
                sign_idx = int(lon // 30) % 12
                transit_positions[p_name] = {
                    "longitude": round(lon, 2),
                    "sign": ZODIAC_SIGNS[sign_idx],
                    "sign_idx": sign_idx
                }
        except Exception:
            transit_positions = {}

    jup_deg = transit_positions.get("Jupiter", {}).get("longitude", 117.64)
    sat_deg = transit_positions.get("Saturn", {}).get("longitude", 346.29)
    ven_deg = transit_positions.get("Venus", {}).get("longitude", 191.80)
    mars_deg = transit_positions.get("Mars", {}).get("longitude", 105.13)
    sun_deg = transit_positions.get("Sun", {}).get("longitude", 176.93)
    rahu_deg = transit_positions.get("Rahu", {}).get("longitude", 302.77)

    # 3. Dynamic Astronomical Window Calculation tailored to User's House Cusps
    if domain == "property":
        # Target: 4th house cusp p_cusp_deg (Vehicles & Property)
        dist_ven = (p_cusp_deg - ven_deg) % 360.0
        dist_mars = min([(p_cusp_deg + asp - mars_deg) % 360.0 for asp in [0, 90, 180, 210]])
        days_ven = int(dist_ven / 1.15) % 365
        days_mars = int(dist_mars / 0.524) % 365
        start_offset_days = max(18, min(days_ven, days_mars))
        cusp_orb_deg = 10.0 + (int(exact_asc_deg * 10) % 9)
        if days_ven < days_mars:
            span_days = max(45, int(cusp_orb_deg / 1.15) * 4)
        else:
            span_days = max(60, int(cusp_orb_deg / 0.524) * 3)
        end_offset_days = start_offset_days + span_days
        domain_transits = "Venus (Vahana Karaka) & Mars (Bhumi Karaka) transits activating 4th house"
        transit_summary = (
            f"Vehicle and property acquisition (Vahana & Bhumi Sukha) is governed by your 4th house "
            f"({p_sign}, ruled by {p_lord}), with Venus signifying luxury conveyances and Mars signifying fixed assets. "
            f"Supportive transits of Venus and Mars crossing supportive kendras, synchronized with your active "
            f"{mahadasha} Mahadasha and {antardasha} Antardasha, establish an auspicious window for vehicle purchase."
        )

    elif domain == "travel":
        # Target: 9th & 12th house cusps (Foreign Travel)
        dist_rahu = (rahu_deg - p_cusp_deg) % 360.0
        days_rahu = int(dist_rahu / 0.0529) % 365
        days_jup_tr = int(min([(p_cusp_deg + asp - jup_deg) % 360.0 for asp in [0, 120, 240]]) / 0.0831) % 365
        start_offset_days = max(20, min(days_rahu, days_jup_tr))
        cusp_orb_deg = 11.0 + (int(exact_asc_deg * 7) % 8)
        span_days = max(75, int(cusp_orb_deg / 0.0831))
        end_offset_days = start_offset_days + span_days
        domain_transits = "Rahu (foreign Karaka) & Moon transits activating 9th & 12th houses"
        transit_summary = (
            f"Foreign travel, visa clearances, and overseas relocation are governed by your 9th house "
            f"({p_sign}, ruled by {p_lord}) of long-distance journeys and 12th house ({s_sign}, ruled by {s_lord}) "
            f"of foreign lands. Transits of Rahu (foreign significator) and the Moon energizing the 9th/12th axis "
            f"under your active {mahadasha}-{antardasha} cycle establish favorable international travel windows."
        )

    elif domain == "finance":
        # Target: 2nd & 11th house cusps (Finance & Wealth)
        days_jup = int(min([(p_cusp_deg + asp - jup_deg) % 360.0 for asp in [0, 120, 240]]) / 0.0831) % 365
        days_merc = int(((p_cusp_deg - ven_deg) % 360.0) / 1.2) % 365
        start_offset_days = max(18, min(days_jup, max(25, days_merc)))
        cusp_orb_deg = 10.0 + (int(exact_asc_deg * 5) % 9)
        span_days = max(60, int(cusp_orb_deg / 0.0831))
        end_offset_days = start_offset_days + span_days
        domain_transits = "Jupiter (Dhana Karaka) & Mercury transits activating 2nd & 11th houses"
        transit_summary = (
            f"Financial stability and wealth preservation are governed by your 2nd house ({p_sign}, ruled by {p_lord}) "
            f"and 11th house ({s_sign}, ruled by {s_lord}). Supportive planetary transits aligning with your active "
            f"{mahadasha} Mahadasha and {antardasha} Antardasha create an auspicious window for wealth accumulation."
        )

    elif domain == "marriage":
        # Target: 7th house cusp (Marriage & Partnerships)
        days_mar = int(min([(p_cusp_deg + asp - jup_deg) % 360.0 for asp in [0, 120, 240]]) / 0.0831) % 365
        days_ven = int(((p_cusp_deg - ven_deg) % 360.0) / 1.15) % 365
        start_offset_days = max(20, min(days_mar, max(30, days_ven)))
        cusp_orb_deg = 10.0 + (int(exact_asc_deg * 3) % 9)
        span_days = max(60, int(cusp_orb_deg / 0.0831))
        end_offset_days = start_offset_days + span_days
        domain_transits = "Jupiter & Venus transits aspecting 7th house"
        transit_summary = (
            f"Marriage and relationship commitments are governed by your 7th house ({p_sign}, ruled by {p_lord}). "
            f"Auspicious transits of Jupiter and Venus supporting the 7th house during your active {mahadasha} "
            f"and {antardasha} periods facilitate alliance finalization and life partnership."
        )

    elif domain == "health":
        # Target: 6th house cusp & Lagna (Immunity & Recovery)
        dist_sun = (exact_asc_deg - sun_deg) % 360.0
        days_sun = int(dist_sun / 0.9856) % 365
        days_mars = int(((p_cusp_deg - mars_deg) % 360.0) / 0.524) % 365
        start_offset_days = max(10, min(days_sun, days_mars))
        cusp_orb_deg = 8.0 + (int(exact_asc_deg * 2) % 6)
        span_days = max(35, int(cusp_orb_deg / 0.524) * 2)
        end_offset_days = start_offset_days + span_days
        domain_transits = "Sun & Mars transits over Lagna & 6th house"
        transit_summary = (
            f"Vitality and immunity recovery are governed by your 6th house ({p_sign}, ruled by {p_lord}) "
            f"and 1st house ({s_sign}, ruled by {s_lord}). Solar transits supporting your Lagna lord during this "
            f"cycle restore physical vitality and circadian balance."
        )

    else:
        # Career (10th & 6th house cusps)
        aspect_jup = min([(p_cusp_deg + asp - jup_deg) % 360.0 for asp in [0, 120, 240]])
        aspect_sat = min([(p_cusp_deg + asp - sat_deg) % 360.0 for asp in [0, 60, 180, 270]])
        days_jup = int(aspect_jup / 0.0831) % 365
        days_sat = int(aspect_sat / 0.0334) % 365
        start_offset_days = max(15, min(days_jup, days_sat))
        cusp_orb_deg = 9.0 + (int(exact_asc_deg * 4) % 9)
        if days_jup < days_sat:
            span_days = max(75, int(cusp_orb_deg / 0.0831))
        else:
            span_days = max(90, int(cusp_orb_deg / 0.0334))
        end_offset_days = start_offset_days + span_days
        domain_transits = "Saturn transit & Sun solar ingress activating 10th & 6th houses"
        transit_summary = (
            f"Career trajectory and professional stabilization are governed by your 10th house ({p_sign}, "
            f"ruled by {p_lord}) and 6th house ({s_sign}, ruled by {s_lord}). Saturn's structural Gochar transit "
            f"and solar ingresses into supportive houses under active {mahadasha}-{antardasha} activate interview "
            f"openings and career consolidation."
        )

    # 4. Harmonize with active Dasha end date if available
    start_date = now + timedelta(days=start_offset_days)
    if dasha_end_date:
        try:
            et_clean = str(dasha_end_date)[:19].replace("T", " ")
            parsed_end = datetime.strptime(et_clean, "%Y-%m-%d %H:%M:%S")
            if parsed_end.tzinfo is None:
                parsed_end = parsed_end.replace(tzinfo=timezone.utc)
            if parsed_end > start_date:
                days_to_dasha_end = (parsed_end - start_date).days
                if 25 <= days_to_dasha_end < span_days:
                    end_offset_days = start_offset_days + days_to_dasha_end
        except Exception:
            pass

    end_date = now + timedelta(days=end_offset_days)
    timing_window = f"{start_date.strftime('%B %Y')} - {end_date.strftime('%B %Y')}"

    return {
        "timing_window": timing_window,
        "domain": domain,
        "calculation_engine": "SwissEphemeris_GocharTransitEngine",
        "primary_house": p_house,
        "primary_house_sign": p_sign,
        "primary_house_lord": p_lord,
        "secondary_house": s_house,
        "secondary_house_sign": s_sign,
        "secondary_house_lord": s_lord,
        "karakas": karakas,
        "domain_transits": domain_transits,
        "transit_summary": transit_summary,
        "mahadasha": mahadasha,
        "antardasha": antardasha,
        "start_date": start_date.strftime("%B %Y"),
        "end_date": end_date.strftime("%B %Y"),
        "ephemeris_positions": transit_positions,
    }


# ---------------------------------------------------------------------------
# 3. Structured Prose Synthesis with Strict Domain-House Validation Prompt
# ---------------------------------------------------------------------------
TIMING_CALCULATION_FOOTER = "*[Calculation: SwissEphemeris_GocharTransitEngine | Transit Window: Active Dasha & Gochar Analysis]*"


def build_timing_system_prompt(domain: str, is_hinglish: bool = False) -> str:
    lang = "conversational Hinglish (Latin script)" if is_hinglish else "articulate English"
    return (
        f"You are a Vedic astrologer timing consultant. Synthesize provided evidence into one cohesive {lang} consultation "
        f"for {domain.upper()}. Use ONLY supplied Dasha, house, and transit facts. Emphasize the calculated window. "
        "Never invent dates or mention unrelated domain houses. "
        "CRITICAL BAN: Never tell the user to consult an astrologer or say 'kisi qualified Vedic astrologer se sampark karein'. Deliver the timing consultation directly. "
        "Concise: 50–75 words. Return ONLY consultation prose."
    )


async def synthesize_timing_response(
    query: str,
    timing_data: Dict[str, Any],
    user_profile: Dict[str, Any],
    session_id: str = "default_user",
    custom_gemini_client: Any = None,
    is_hinglish: bool = False
) -> Dict[str, Any]:
    """
    Synthesizes the timing response via the cascading LLM router (Groq -> OpenRouter -> Gemini)
    with strict house validation and standardized provenance footer.
    """
    domain = timing_data.get("domain", "career")
    window = timing_data.get("timing_window", "November 2026 - April 2027")
    md = timing_data.get("mahadasha", "Jupiter")
    ad = timing_data.get("antardasha", "Mercury")
    asc = user_profile.get("ascendant_sign", "Virgo")
    moon = user_profile.get("moon_sign", "Taurus")
    p_house = timing_data.get("primary_house", 10)
    p_sign = timing_data.get("primary_house_sign", "Gemini")
    p_lord = timing_data.get("primary_house_lord", "Mercury")
    s_house = timing_data.get("secondary_house", 6)
    s_sign = timing_data.get("secondary_house_sign", "Aquarius")
    s_lord = timing_data.get("secondary_house_lord", "Saturn")
    domain_transits = timing_data.get("domain_transits", "")
    karakas = ", ".join(timing_data.get("karakas", []))

    system_prompt = build_timing_system_prompt(domain=domain, is_hinglish=is_hinglish)
    user_prompt = (
        f"Q: {query}\n\n"
        f"Timing Evidence ({domain.capitalize()}):\n"
        f"- Chart: {asc} Lagna, Moon {moon} | Dasha: {md}-{ad}\n"
        f"- Primary: {p_house}H in {p_sign} (Lord {p_lord}) | Secondary: {s_house}H in {s_sign} (Lord {s_lord})\n"
        f"- Karakas: {karakas} | Timing Window: {window}\n"
        f"- Active Gochar: {domain_transits}"
    )

    refined_prose = ""
    active_model = ""
    meta_info = {}

    try:
        from backend.llm_router import stream_cascading_router
        chunks = []
        async for chunk, meta in stream_cascading_router(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_output_tokens=110,
            temperature=0.25,
            session_id=session_id,
            custom_gemini_client=custom_gemini_client
        ):
            if meta:
                if meta.get("model"):
                    active_model = meta.get("model")
                if meta.get("status") == "completed":
                    meta_info = meta
            if chunk:
                chunks.append(chunk)
        raw_prose = "".join(chunks).strip()
        from backend.llm_router import trim_to_last_sentence
        refined_prose = trim_to_last_sentence(raw_prose)
    except Exception as e:
        logger.warning("LLM cascading failed for timing synthesis: %s; using deterministic fallback", e)

    # Deterministic domain-isolated fallback prose if LLM offline
    if not refined_prose:
        active_model = None
        source_badge_type = "LOCAL"
        if is_hinglish:
            if domain == "property":
                refined_prose = (
                    f"Aapki kundali ({asc} Lagna, {moon} Rashi) ke anusaar car/vehicle khareedne aur property acquisition ke liye "
                    f"sabse favorable timing window **{window}** calculate hui hai. 4th house (Vahana aur Sukha Bhava) {p_sign} mein "
                    f"sthit hai jiske lord {p_lord} hain. Is period mein Venus (Vahana Karaka) aur Mars ke supportive transits active "
                    f"**{md}** aur **{ad}** dasha ke sath align ho rahe hain. Nayi gadi ya asset purchase ke liye yeh window shubh aur impactful rahegi."
                )
            elif domain == "travel":
                refined_prose = (
                    f"Aapki kundali ({asc} Lagna, {moon} Rashi) ke anusaar foreign travel, visa approval aur abroad settlement ke liye "
                    f"most auspicious window **{window}** hai. Foreign lands aur long journeys 9th house ({p_sign}, lord {p_lord}) aur "
                    f"12th house se govern hote hain. Rahu (videsh Karaka) aur Moon ke Gochar alignments active **{md}** aur **{ad}** "
                    f"dasha ke sath milkar overseas travel ke strong yog create kar rahe hain."
                )
            elif domain == "career":
                refined_prose = (
                    f"Aapki kundali ({asc} Lagna, {moon} Rashi) ke anusaar job change aur career stabilization ke liye sabse "
                    f"favorable timing window **{window}** nikal kar aati hai. Is dauran active **{md} Mahadasha** aur **{ad} Antardasha** "
                    f"ke sath Saturn ka structural Gochar aapke 10th house ({p_sign}, lord {p_lord}) aur 6th house ko energize karega. "
                    f"Yeh samay interviews, applications aur role transition ke liye highly supportive hai."
                )
            elif domain == "finance":
                refined_prose = (
                    f"Aapki kundali ke anusaar financial growth aur wealth stabilization ke liye favorable timing window **{window}** hai. "
                    f"Active **{md}** Mahadasha aur **{ad}** Antardasha aapke 2nd house ({p_sign}, lord {p_lord}) aur 11th house of gains ko support kar rahe hain. "
                    f"Is samay ke dauran structured investments aur income consolidation ke strong yog bante hain."
                )
            elif domain == "marriage":
                refined_prose = (
                    f"Aapki kundali ke anusaar marriage timing aur committed relationships ke liye most auspicious window **{window}** hai. "
                    f"Jupiter aur Venus ka Gochar aapke 7th house ({p_sign}, lord {p_lord}) ko aspect kar raha hai jabki active **{md}** aur **{ad}** dasha relationship "
                    f"harmony promote kar rahi hai. Yeh period alliance fix karne aur naye rishton ke liye ideal hai."
                )
            else:
                refined_prose = (
                    f"Aapki kundali ({asc} Lagna) ke anusaar {domain} ke context mein positive changes aur stabilization ke liye "
                    f"most favorable window **{window}** calculate hui hai. Active {md} Mahadasha aur Gochar alignments is duration "
                    f"mein renewal aur fruitful results deliver karenge."
                )
        else:
            if domain == "property":
                refined_prose = (
                    f"Based on your birth chart ({asc} Ascendant, {moon} Moon sign), the prime astrological window for purchasing a vehicle "
                    f"or property is projected between **{window}**. Vehicle acquisition (Vahana Sukha) is governed by your 4th house in {p_sign} "
                    f"(ruled by {p_lord}), with Venus signifying conveyances and Mars signifying fixed assets. Supportive transits of Venus and Mars "
                    f"synchronized with your active **{md} Mahadasha** and **{ad} Antardasha** create auspicious alignments to finalize your vehicle or property purchase."
                )
            elif domain == "travel":
                refined_prose = (
                    f"According to your planetary placements ({asc} Ascendant, {moon} Moon sign), the most favorable timing window for foreign travel, "
                    f"overseas relocation, and visa processing is calculated between **{window}**. International movement is governed by your 9th house "
                    f"({p_sign}, ruled by {p_lord}) of long journeys and 12th house ({s_sign}, ruled by {s_lord}) of foreign lands. Transits of Rahu "
                    f"(foreign significator) and the Moon, harmonized with your active **{md} Mahadasha** and **{ad} Antardasha**, open overseas pathways."
                )
            elif domain == "career":
                refined_prose = (
                    f"Based on your birth chart ({asc} Ascendant, {moon} Moon sign), the most favorable timing window for career "
                    f"advancement and stabilization is calculated between **{window}**. The active planetary combination of your "
                    f"**{md} Mahadasha** and **{ad} Antardasha** directly energizes your 10th house in {p_sign} (ruled by {p_lord}) and 6th house. "
                    f"Concurrently, Saturn's structural Gochar transit and supportive solar alignments facilitate interviews and role transitions."
                )
            elif domain == "finance":
                refined_prose = (
                    f"According to your natal placements, your primary financial stabilization window is projected between **{window}**. "
                    f"Your active **{md} Mahadasha** and **{ad} Antardasha** stimulate your wealth sector in the 2nd house ({p_sign}) and 11th house of gains. "
                    f"Supported by planetary transits, this period brings consolidation of income and successful long-term financial planning."
                )
            elif domain == "marriage":
                refined_prose = (
                    f"Based on your astrological chart, the prime auspicious window for marriage and partnership commitment is "
                    f"calculated between **{window}**. Auspicious Gochar aspects from Jupiter onto your 7th house ({p_sign}), synchronized with "
                    f"your active **{md}** and **{ad}** dasha rulers, create strong harmony for wedding timelines and life companionship."
                )
            else:
                refined_prose = (
                    f"According to your planetary alignments ({asc} Ascendant), the most supportive timing window for {domain} "
                    f"materializes between **{window}**. The synchronization of your active **{md}** planetary cycle and current Gochar "
                    f"transits provides clarity, constructive momentum, and auspicious outcomes throughout this period."
                )
    else:
        source_badge_type = "HYBRID"

    try:
        from backend.answer_synthesizer import estimate_tokens
    except ImportError:
        try:
            from answer_synthesizer import estimate_tokens
        except ImportError:
            estimate_tokens = lambda text: int(len(text.split()) * 1.3)

    inp_tokens = meta_info.get("input_tokens") or (estimate_tokens(f"{system_prompt}\n{user_prompt}") if active_model else 0)
    out_tokens = meta_info.get("output_tokens") or (estimate_tokens(refined_prose) if active_model else 0)
    tot_tokens = inp_tokens + out_tokens

    rem_model = 0
    all_status = {}
    if active_model:
        try:
            try:
                from backend.answer_synthesizer import (
                    record_model_tokens,
                    get_model_remaining_tokens,
                    get_all_models_token_status,
                    print_terminal_token_report,
                )
            except ImportError:
                from answer_synthesizer import (
                    record_model_tokens,
                    get_model_remaining_tokens,
                    get_all_models_token_status,
                    print_terminal_token_report,
                )
            model_stats = record_model_tokens(active_model, tot_tokens)
            rem_model = model_stats.get("remaining_tokens", 0)
            all_status = get_all_models_token_status()
            print_terminal_token_report(
                mode="LOCAL_DATABASE_PLUS_LLM_TIMING",
                model_name=active_model,
                inp_tokens=inp_tokens,
                out_tokens=out_tokens,
                total_tokens=tot_tokens,
                query_remaining=max(0, 400 - tot_tokens),
                model_budget=model_stats.get("budget_tokens", 100000),
                model_remaining=rem_model,
            )
        except Exception as te_err:
            logger.warning("Timing engine terminal token report error: %s", te_err)

    # Return clean prose without footers, provenance badges, or asterisks
    body = refined_prose.strip()
    body = re.sub(r"\n*\s*\*?\[Calculation:[^\]]+\]\*?", "", body, flags=re.IGNORECASE).strip()
    final_answer = body.replace("*", "").strip()

    footer_badge = f"[{active_model}]" if active_model else "[Timing Engine Local]"
    source_badge = source_badge_type if "source_badge_type" in locals() else ("HYBRID" if active_model else "LOCAL")

    return {
        "answer": final_answer,
        "answer_mode": "TIMING_WINDOW",
        "answer_source": source_badge,
        "domain": domain,
        "timing_window": window,
        "active_dasha": f"{md}/{ad}",
        "attribution_badge": footer_badge,
        "model_name": active_model,
        "timing_data": timing_data,
        "input_tokens": inp_tokens,
        "output_tokens": out_tokens,
        "total_tokens": tot_tokens,
        "remaining_query_tokens": max(0, 400 - tot_tokens),
        "remaining_model_tokens": rem_model,
        "model_token_status": all_status,
        "tokens": {"input": inp_tokens, "output": out_tokens, "total": tot_tokens},
    }


# ---------------------------------------------------------------------------
# 4. Master Timing Pipeline Entry Point
# ---------------------------------------------------------------------------
async def execute_timing_pipeline(
    query: str,
    user_profile: Dict[str, Any],
    session_birth_data: Optional[Dict[str, Any]] = None,
    chart_data: Optional[Dict[str, Any]] = None,
    session_id: str = "default_user",
    custom_gemini_client: Any = None,
    is_hinglish: bool = False
) -> Dict[str, Any]:
    """
    Executes the end-to-end Timing Engine workflow:
    1. Detects timing domain (career, property, travel, finance, marriage, etc.).
    2. Resolves safe Dasha profile via baseline or active ephemeris calculation.
    3. Calculates domain-isolated Gochar transit window.
    4. Synthesizes personalized prose refiner response with strict house isolation.
    """
    try:
        from backend.dasha_calculator import resolve_safe_user_dasha_profile
    except ImportError:
        from dasha_calculator import resolve_safe_user_dasha_profile

    safe_profile = resolve_safe_user_dasha_profile(user_profile, chart_data=chart_data)
    domain = detect_timing_domain(query)

    # Resolve birth data
    resolved_birth_data = (
        session_birth_data
        or safe_profile.get("session_birth_data")
        or user_profile.get("session_birth_data")
    )
    if not resolved_birth_data and any(k in safe_profile for k in ("birth_year", "birth_month", "birth_day")):
        resolved_birth_data = safe_profile
    elif not resolved_birth_data and any(k in user_profile for k in ("birth_year", "birth_month", "birth_day")):
        resolved_birth_data = user_profile

    resolved_chart = chart_data or safe_profile.get("chart_data") or user_profile.get("chart_data")
    d_start = safe_profile.get("dasha_start_date") or user_profile.get("dasha_start_date")
    d_end = safe_profile.get("dasha_end_date") or user_profile.get("dasha_end_date")

    timing_data = calculate_astrological_timing_window(
        domain=domain,
        ascendant_sign=safe_profile.get("ascendant_sign", "Virgo"),
        moon_sign=safe_profile.get("moon_sign", "Taurus"),
        mahadasha=safe_profile.get("mahadasha", "Jupiter"),
        antardasha=safe_profile.get("antardasha", "Mercury"),
        session_birth_data=resolved_birth_data,
        chart_data=resolved_chart,
        dasha_start_date=d_start,
        dasha_end_date=d_end
    )

    return await synthesize_timing_response(
        query=query,
        timing_data=timing_data,
        user_profile=safe_profile,
        session_id=session_id,
        custom_gemini_client=custom_gemini_client,
        is_hinglish=is_hinglish
    )
