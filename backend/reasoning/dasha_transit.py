def build_dasha_transit_timing(
    dasha_timing,
    transit_timing,
):
    """
    Combine active Dasha planets with current transit evidence.

    This function does not make an astrological prediction.
    It only identifies structural overlap between the two
    timing systems.
    """

    active_dasha_planets = set(
        dasha_timing.get("active_planets", [])
    )

    transit_planets = transit_timing.get(
        "planets",
        {}
    )

    dasha_transit_overlap = []

    for planet in active_dasha_planets:

        if planet not in transit_planets:
            continue

        transit_data = transit_planets[planet]

        dasha_transit_overlap.append({
            "planet": planet,
            "dasha_active": True,
            "transit_active": True,
            "transit_house": transit_data.get(
                "transit_house"
            ),
            "transit_rashi": transit_data.get(
                "rashi"
            ),
            "transit_nakshatra": transit_data.get(
                "nakshatra"
            ),
        })

    aspect_overlap = []

    for aspect in transit_timing.get(
        "natal_aspects",
        transit_timing.get("aspects", [])
    ):

        transit_planet = aspect["transit_planet"]

        if transit_planet in active_dasha_planets:

            aspect_overlap.append({
                "dasha_planet": transit_planet,
                "natal_planet": aspect["natal_planet"],
                "transit_house": aspect["transit_house"],
                "natal_house": aspect["natal_house"],
                "aspect": aspect["aspect"],
            })

    return {
        "active_dasha_planets": list(
            active_dasha_planets
        ),
        "dasha_transit_overlap": dasha_transit_overlap,
        "dasha_transit_aspects": aspect_overlap,
    }