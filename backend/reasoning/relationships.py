# ============================================================
# STEP 7.22 — TRANSIT → NATAL RELATIONSHIPS
# ============================================================

def calculate_transit_natal_relationships(
    transit_result,
    natal_chart,
    orb=5.0,
):
    """
    Calculate geometric relationships between
    current transit planets and natal planets.
    """

    relationships = []

    transit_planets = transit_result["planets"]
    natal_planets = natal_chart["planets"]

    for transit_planet, transit_data in transit_planets.items():

        transit_longitude = transit_data["longitude"]

        for natal_planet, natal_data in natal_planets.items():

            natal_longitude = natal_data["longitude"]

            separation = calculate_angular_separation(
                transit_longitude,
                natal_longitude,
            )

            relationship = classify_angular_relationship(
                separation,
                orb=orb,
            )

            if relationship is not None:

                relationships.append({
                    "transit_planet":
                        transit_planet,

                    "natal_planet":
                        natal_planet,

                    "separation":
                        separation,

                    "aspect":
                        relationship["aspect"],

                    "exact_angle":
                        relationship["exact_angle"],

                    "orb":
                        relationship["orb"],
                })

    return relationships