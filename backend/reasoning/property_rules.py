# ============================================================
# STAGE 8.19 — PROPERTY RULE ENGINE
# ============================================================


def evaluate_property_rules(
    domain_evidence,
    natal_lordship_analysis,
    planetary_strength_analysis,
    generic_rule_results,
    natal_planet_lookup,
):
    """
    Property-specific astrology rule evaluation.

    Uses existing calculated evidence.
    Does not recalculate planetary positions.
    Does not assign arbitrary probabilities.
    Does not produce the final prediction.
    """

    rules = []

    property_evidence = domain_evidence["property"]

    house_lord_map = natal_lordship_analysis[
        "house_lord_map"
    ]

    house_lord_placements = natal_lordship_analysis[
        "house_lord_placements"
    ]

    # --------------------------------------------------------
    # RULE 1 — 4TH HOUSE LORD
    # --------------------------------------------------------

    fourth_lord_data = house_lord_map.get(4)

    fourth_lord_placement = house_lord_placements.get(
        4,
        {}
    )

    fourth_lord_exists = (
        fourth_lord_data is not None
        and isinstance(
            fourth_lord_data.get("lord"),
            str
        )
    )

    fourth_lord_placement_exists = (
        fourth_lord_placement.get(
            "lord_data_available",
            False
        )
    )

    rules.append({
        "rule_id": "PROPERTY_4TH_LORD_PLACEMENT",
        "domain": "property",
        "matched": (
            fourth_lord_exists
            and fourth_lord_placement_exists
        ),
        "conditions": [
            {
                "condition": "4th_house_lord_exists",
                "matched": fourth_lord_exists
            },
            {
                "condition":
                    "4th_lord_natal_placement_exists",
                "matched":
                    fourth_lord_placement_exists
            }
        ],
        "evidence": {
            "house": 4,
            "lord": (
                fourth_lord_data.get("lord")
                if fourth_lord_data
                else None
            ),
            "house_rashi": (
                fourth_lord_data.get("rashi")
                if fourth_lord_data
                else None
            ),
            "lord_natal_house":
                fourth_lord_placement.get(
                    "lord_natal_house"
                ),
            "lord_natal_rashi":
                fourth_lord_placement.get(
                    "lord_natal_rashi"
                )
        },
        "interpretation_key":
            "property_4th_lord"
    })

    # --------------------------------------------------------
    # RULE 2 — PROPERTY KARAKA PRESENCE
    # --------------------------------------------------------

    property_karakas = [
        "Mars",
        "Moon"
    ]

    karaka_evidence = []

    for planet in property_karakas:

        if planet in natal_planet_lookup:

            data = natal_planet_lookup[planet]

            karaka_evidence.append({
                "planet": planet,
                "house": data["house"],
                "rashi": data["rashi"],
                "nakshatra": data["nakshatra"],
                "pada": data["pada"]
            })

    rules.append({
        "rule_id": "PROPERTY_KARAKA_PRESENCE",
        "domain": "property",
        "matched": len(karaka_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "property_karaka_available",
                "matched":
                    len(karaka_evidence) > 0
            }
        ],
        "evidence": karaka_evidence,
        "interpretation_key":
            "property_karaka_presence"
    })

    # --------------------------------------------------------
    # RULE 3 — PROPERTY KARAKA DIGNITY
    # --------------------------------------------------------

    dignity_data = planetary_strength_analysis[
        "planetary_dignity"
    ]

    dignity_evidence = []

    for planet in property_karakas:

        if planet in dignity_data:

            data = dignity_data[planet]

            dignity_evidence.append({
                "planet": planet,
                "rashi": data.get("rashi"),
                "dignity": data.get("dignity")
            })

    rules.append({
        "rule_id": "PROPERTY_KARAKA_DIGNITY",
        "domain": "property",
        "matched": len(dignity_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "property_karaka_dignity_available",
                "matched":
                    len(dignity_evidence) > 0
            }
        ],
        "evidence": dignity_evidence,
        "interpretation_key":
            "property_karaka_dignity"
    })

    # --------------------------------------------------------
    # RULE 4 — PROPERTY HOUSE EVIDENCE
    # --------------------------------------------------------

    natal_planets_by_house = property_evidence[
        "natal_planets_by_house"
    ]

    property_house_evidence = {}

    for house in [4, 2, 11]:

        planets = natal_planets_by_house.get(
            house,
            []
        )

        if planets:
            property_house_evidence[house] = planets

    rules.append({
        "rule_id":
            "PROPERTY_RELEVANT_HOUSE_EVIDENCE",
        "domain": "property",
        "matched":
            len(property_house_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "property_relevant_house_evidence",
                "matched":
                    len(property_house_evidence) > 0
            }
        ],
        "evidence":
            property_house_evidence,
        "interpretation_key":
            "property_relevant_houses"
    })

    # --------------------------------------------------------
    # RULE 5 — DASHA PROPERTY ACTIVATION
    # --------------------------------------------------------

    dasha_connections = property_evidence[
        "dasha_connections"
    ]

    dasha_primary = dasha_connections.get(
        "primary",
        []
    )

    dasha_supporting = dasha_connections.get(
        "supporting",
        []
    )

    dasha_evidence = (
        dasha_primary
        + dasha_supporting
    )

    rules.append({
        "rule_id":
            "PROPERTY_DASHA_ACTIVATION",
        "domain": "property",
        "matched":
            len(dasha_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "property_dasha_connection_exists",
                "matched":
                    len(dasha_evidence) > 0
            }
        ],
        "evidence":
            dasha_evidence,
        "interpretation_key":
            "property_dasha_activation"
    })

    # --------------------------------------------------------
    # RULE 6 — TRANSIT PROPERTY ACTIVATION
    # --------------------------------------------------------

    transit_connections = property_evidence[
        "transit_connections"
    ]

    transit_primary = transit_connections.get(
        "primary",
        []
    )

    transit_supporting = transit_connections.get(
        "supporting",
        []
    )

    transit_evidence = (
        transit_primary
        + transit_supporting
    )

    rules.append({
        "rule_id":
            "PROPERTY_TRANSIT_ACTIVATION",
        "domain": "property",
        "matched":
            len(transit_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "property_transit_connection_exists",
                "matched":
                    len(transit_evidence) > 0
            }
        ],
        "evidence":
            transit_evidence,
        "interpretation_key":
            "property_transit_activation"
    })

    # --------------------------------------------------------
    # RULE 7 — DASHA + TRANSIT
    # --------------------------------------------------------

    dasha_transit_connections = property_evidence[
        "dasha_transit_connections"
    ]

    rules.append({
        "rule_id":
            "PROPERTY_DASHA_TRANSIT_COMBINATION",
        "domain": "property",
        "matched":
            len(dasha_transit_connections) > 0,
        "conditions": [
            {
                "condition":
                    "property_dasha_transit_connection_exists",
                "matched":
                    len(dasha_transit_connections) > 0
            }
        ],
        "evidence":
            dasha_transit_connections,
        "interpretation_key":
            "property_dasha_transit_activation"
    })

    # --------------------------------------------------------
    # RULE 8 — MULTI-FACTOR PROPERTY ACTIVATION
    # --------------------------------------------------------

    evidence_categories = {

        "house_lord":
            (
                fourth_lord_exists
                and fourth_lord_placement_exists
            ),

        "karaka":
            len(karaka_evidence) > 0,

        "dasha":
            len(dasha_evidence) > 0,

        "transit":
            len(transit_evidence) > 0,

        "dasha_transit":
            len(dasha_transit_connections) > 0
    }

    active_categories = [
        category
        for category, matched
        in evidence_categories.items()
        if matched
    ]

    rules.append({
        "rule_id":
            "PROPERTY_MULTI_FACTOR_ACTIVATION",
        "domain": "property",
        "matched":
            len(active_categories) >= 2,
        "conditions": [
            {
                "condition":
                    "multiple_independent_property_evidence_categories",
                "required_minimum": 2,
                "actual":
                    len(active_categories),
                "matched":
                    len(active_categories) >= 2
            }
        ],
        "evidence": {
            "active_categories":
                active_categories,
            "category_status":
                evidence_categories
        },
        "interpretation_key":
            "property_multi_factor_activation"
    })

    return {
        "domain": "property",
        "rules": rules,
        "matched_rule_count": sum(
            1
            for rule in rules
            if rule["matched"]
        ),
        "total_rule_count": len(rules)
    }


# ============================================================
# VALIDATION HELPER
# ============================================================

def validate_property_rule_analysis(
    property_rule_analysis
):
    assert property_rule_analysis["domain"] == "property"

    assert (
        property_rule_analysis["total_rule_count"]
        == 8
    )

    for rule in property_rule_analysis["rules"]:

        assert "rule_id" in rule
        assert "domain" in rule
        assert "matched" in rule
        assert "conditions" in rule
        assert "evidence" in rule
        assert "interpretation_key" in rule

        assert rule["domain"] == "property"

        assert isinstance(
            rule["matched"],
            bool
        )

        assert isinstance(
            rule["conditions"],
            list
        )

    return True