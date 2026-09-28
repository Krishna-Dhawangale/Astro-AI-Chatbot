import swisseph as swe

def get_astronomical_features(year: int, month: int, day: int, hour: float, lat: float, lon: float) -> list:
    """
    Calculates Sidereal planetary sign positions (0-11) for:
    [Ascendant, Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu]
    """
    # Set Sidereal (Lahiri) Ayanamsa
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    
    # Calculate Julian Day
    julian_day = swe.julday(year, month, day, hour)
    
    # Safe unpack for swe.houses_ex (handles both 2 and 3 return values)
    house_result = swe.houses_ex(julian_day, lat, lon, b'P', swe.FLG_SIDEREAL)
    houses = house_result[0]
    ascmc = house_result[1]
    
    ascendant_deg = ascmc[0]
    ascendant_sign = int(ascendant_deg // 30)

    # Planets to calculate
    planets = [
        swe.SUN, swe.MOON, swe.MARS, swe.MERCURY,
        swe.JUPITER, swe.VENUS, swe.SATURN, swe.MEAN_NODE
    ]

    features = [ascendant_sign]

    # Calculate planet signs
    for planet in planets:
        res = swe.calc_ut(julian_day, planet, swe.FLG_SIDEREAL)
        # res[0] is tuple of coordinates (longitude, latitude, distance...)
        lon_deg = res[0][0] if isinstance(res[0], (list, tuple)) else res[0]
        features.append(int(lon_deg // 30))

    # Calculate Ketu (180 degrees opposite Rahu)
    rahu_deg = features[-1] * 30
    ketu_sign = int(((rahu_deg + 180) % 360) // 30)
    features.append(ketu_sign)

    return features