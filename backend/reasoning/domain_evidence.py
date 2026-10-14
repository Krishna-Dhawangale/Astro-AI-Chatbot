# ============================================================
# STAGE 8.10.7 — DOMAIN EVIDENCE
# ============================================================

def build_domain_evidence(
    domain,
    domain_config,
    domain_natal_planets,
    dasha_connections,
    transit_connections,
    dasha_transit_connections,
):
    """
    Combine deterministic evidence for one domain.

    No prediction or interpretation is performed.
    """

    primary_dasha = [
        item
        for item in dasha_connections
        if item["house_type"] == "primary"
    ]

    supporting_dasha = [
        item
        for item in dasha_connections
        if item["house_type"] == "supporting"
    ]

    primary_transits = [
        item
        for item in transit_connections
        if item["house_type"] == "primary"
    ]

    supporting_transits = [
        item
        for item in transit_connections
        if item["house_type"] == "supporting"
    ]

    return {
        "domain": domain,

        "primary_houses":
            domain_config["primary_houses"],

        "supporting_houses":
            domain_config["supporting_houses"],

        "karaka_planets":
            domain_config["karaka_planets"],

        "natal_planets_by_house":
            domain_natal_planets,

        "dasha_connections": {
            "primary": primary_dasha,
            "supporting": supporting_dasha,
        },

        "transit_connections": {
            "primary": primary_transits,
            "supporting": supporting_transits,
        },

        "dasha_transit_connections":
            dasha_transit_connections,
    }