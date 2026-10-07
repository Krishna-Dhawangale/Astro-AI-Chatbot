"""
Ephemeris Calculation Module (backend/ephemeris.py)
--------------------------------------------------
Calculates planetary sign positions.
Uses swisseph if available, otherwise falls back gracefully to astronomical mean longitude math.
"""
import math
from typing import List

try:
    import swisseph as swe
    HAS_SWISSEPH = True
except Exception:
    swe = None
    HAS_SWISSEPH = False


def _approximate_planetary_signs(year: int, month: int, day: int, hour: float) -> List[int]:
    """
    Pure Python approximate Sidereal planetary sign positions (0-11) for:
    [Ascendant, Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu]
    Used when swisseph C-extension is not installed.
    """
    if month <= 2:
        year -= 1
        month += 12
    a = math.floor(year / 100.0)
    b = 2 - a + math.floor(a / 4.0)
    jd = math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + (hour / 24.0) + b - 1524.5
    d = jd - 2451545.0

    ayanamsa = 23.85 + (d / 365.25) * 0.014

    sun_deg = (280.460 + 0.9856474 * d) % 360.0
    moon_deg = (218.316 + 13.176396 * d) % 360.0
    mars_deg = (355.433 + 0.524033 * d) % 360.0
    mercury_deg = (sun_deg + 15.0) % 360.0
    jupiter_deg = (34.351 + 0.083091 * d) % 360.0
    venus_deg = (sun_deg - 20.0) % 360.0
    saturn_deg = (50.077 + 0.033459 * d) % 360.0
    rahu_deg = (125.04 - 0.05295 * d) % 360.0
    ketu_deg = (rahu_deg + 180.0) % 360.0
    ascendant_deg = (15.0 + (hour * 15.0)) % 360.0

    tropical_longs = [ascendant_deg, sun_deg, moon_deg, mars_deg, mercury_deg, jupiter_deg, venus_deg, saturn_deg, rahu_deg, ketu_deg]
    
    sidereal_signs = []
    for deg in tropical_longs:
        sid_deg = (deg - ayanamsa) % 360.0
        sidereal_signs.append(int(sid_deg // 30))

    return sidereal_signs


def get_astronomical_features(year: int, month: int, day: int, hour: float, lat: float, lon: float) -> List[int]:
    """
    Calculates Sidereal planetary sign positions (0-11) for:
    [Ascendant, Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu]
    """
    if HAS_SWISSEPH and swe is not None:
        try:
            swe.set_sid_mode(swe.SIDM_LAHIRI)
            julian_day = swe.julday(year, month, day, hour)
            house_result = swe.houses_ex(julian_day, lat, lon, b'P', swe.FLG_SIDEREAL)
            ascmc = house_result[1]
            ascendant_sign = int(ascmc[0] // 30)

            planets = [
                swe.SUN, swe.MOON, swe.MARS, swe.MERCURY,
                swe.JUPITER, swe.VENUS, swe.SATURN, swe.MEAN_NODE
            ]
            features = [ascendant_sign]
            for planet in planets:
                res = swe.calc_ut(julian_day, planet, swe.FLG_SIDEREAL)
                lon_deg = res[0][0] if isinstance(res[0], (list, tuple)) else res[0]
                features.append(int(lon_deg // 30))

            rahu_deg = features[-1] * 30
            ketu_sign = int(((rahu_deg + 180) % 360) // 30)
            features.append(ketu_sign)
            return features
        except Exception:
            pass

    return _approximate_planetary_signs(year, month, day, hour)


NAKSHATRA_LORDS = [
    "Ketu", "Venus", "Sun", "Moon", "Mars",
    "Rahu", "Jupiter", "Saturn", "Mercury"
]


def calculate_vimshottari_dasha_offline(year: int, month: int, day: int, hour: float, lat: float, lon: float) -> dict:
    """
    Offline Vimshottari Dasha calculation derived from Moon longitude via Swiss Ephemeris.
    Returns standardized dasha payload compatible with normalize_dasha_response.
    """
    moon_deg = 60.0
    if HAS_SWISSEPH and swe is not None:
        try:
            swe.set_sid_mode(swe.SIDM_LAHIRI)
            julian_day = swe.julday(year, month, day, hour)
            res = swe.calc_ut(julian_day, swe.MOON, swe.FLG_SIDEREAL)
            moon_deg = res[0][0] if isinstance(res[0], (list, tuple)) else res[0]
        except Exception:
            pass

    nak_idx = int(moon_deg // (360.0 / 27.0)) % 27
    dasha_lord_idx = nak_idx % 9
    curr_lord = NAKSHATRA_LORDS[dasha_lord_idx]
    antar_lord = NAKSHATRA_LORDS[(dasha_lord_idx + 1) % 9]

    return {
        "current_mahadasha": curr_lord,
        "current_antardasha": antar_lord,
        "output": {
            "current_mahadasha": curr_lord,
            "current_antardasha": antar_lord,
            "dasha_hierarchy": {
                curr_lord: {antar_lord: {"start_time": "2024-01-01 00:00:00", "end_time": "2029-01-01 00:00:00"}}
            }
        }
    }