def evaluate_marriage(evidence: list[str]) -> list[str]:
    return list(evidence)# ============================================================
# STAGE 8.16 — MARRIAGE RULE ENGINE
# ============================================================


def evaluate_marriage_rules(
    domain_evidence,
    natal_lordship_analysis,
    planetary_strength_analysis,
    generic_rule_results,
    natal_planet_lookup,
):
    """
    Marriage-specific astrology rule evaluation.

    Uses existing calculated evidence.

    Does not:
        - recalculate planetary positions
        - assign arbitrary probabilities
        - produce the final prediction
    """

    rules = []

    marriage_evidence = domain_evidence["marriage"]

    house_lord_map = natal_lordship_analysis[
        "house_lord_map"
    ]

    house_lord_placements = natal_lordship_analysis[
        "house_lord_placements"
    ]

    # --------------------------------------------------------
    # RULE 1 — 7TH HOUSE LORD
    # --------------------------------------------------------

    seventh_lord_data = house_lord_map.get(7)

    seventh_lord_placement = house_lord_placements.get(
        7,
        {}
    )

    seventh_lord_exists = (
        seventh_lord_data is not None
        and isinstance(
            seventh_lord_data.get("lord"),
            str
        )
    )

    seventh_lord_placement_exists = (
        seventh_lord_placement.get(
            "lord_data_available",
            False
        )
    )

    rules.append({
        "rule_id": "MARRIAGE_7TH_LORD_PLACEMENT",
        "domain": "marriage",
        "matched": (
            seventh_lord_exists
            and seventh_lord_placement_exists
        ),
        "conditions": [
            {
                "condition": "7th_house_lord_exists",
                "matched": seventh_lord_exists
            },
            {
                "condition":
                    "7th_lord_natal_placement_exists",
                "matched":
                    seventh_lord_placement_exists
            }
        ],
        "evidence": {
            "house": 7,
            "lord": (
                seventh_lord_data.get("lord")
                if seventh_lord_data
                else None
            ),
            "house_rashi": (
                seventh_lord_data.get("rashi")
                if seventh_lord_data
                else None
            ),
            "lord_natal_house":
                seventh_lord_placement.get(
                    "lord_natal_house"
                ),
            "lord_natal_rashi":
                seventh_lord_placement.get(
                    "lord_natal_rashi"
                )
        },
        "interpretation_key":
            "marriage_7th_lord"
    })


    # --------------------------------------------------------
    # RULE 2 — MARRIAGE KARAKA PRESENCE
    # --------------------------------------------------------

    marriage_karakas = [
        "Venus",
        "Jupiter",
        "Moon"
    ]

    karaka_evidence = []

    for planet in marriage_karakas:

        if planet in natal_planet_lookup:

            data = natal_planet_lookup[planet]

            karaka_evidence.append({
                "planet": planet,
                "house": data.get("house"),
                "rashi": data.get("rashi"),
                "nakshatra": data.get("nakshatra"),
                "pada": data.get("pada")
            })

    rules.append({
        "rule_id": "MARRIAGE_KARAKA_PRESENCE",
        "domain": "marriage",
        "matched":
            len(karaka_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "marriage_karaka_available",
                "matched":
                    len(karaka_evidence) > 0
            }
        ],
        "evidence":
            karaka_evidence,
        "interpretation_key":
            "marriage_karaka_presence"
    })


    # --------------------------------------------------------
    # RULE 3 — MARRIAGE KARAKA DIGNITY
    # --------------------------------------------------------

    dignity_data = planetary_strength_analysis[
        "planetary_dignity"
    ]

    dignity_evidence = []

    for planet in marriage_karakas:

        if planet in dignity_data:

            data = dignity_data[planet]

            dignity_evidence.append({
                "planet": planet,
                "rashi": data.get("rashi"),
                "dignity": data.get("dignity")
            })

    rules.append({
        "rule_id": "MARRIAGE_KARAKA_DIGNITY",
        "domain": "marriage",
        "matched":
            len(dignity_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "marriage_karaka_dignity_available",
                "matched":
                    len(dignity_evidence) > 0
            }
        ],
        "evidence":
            dignity_evidence,
        "interpretation_key":
            "marriage_karaka_dignity"
    })


    # --------------------------------------------------------
    # RULE 4 — MARRIAGE HOUSE EVIDENCE
    # --------------------------------------------------------

    natal_planets_by_house = marriage_evidence[
        "natal_planets_by_house"
    ]

    marriage_house_evidence = {}

    for house in [7, 2, 5, 8, 11]:

        planets = natal_planets_by_house.get(
            house,
            []
        )

        if planets:

            marriage_house_evidence[house] = planets

    rules.append({
        "rule_id":
            "MARRIAGE_RELEVANT_HOUSE_EVIDENCE",
        "domain": "marriage",
        "matched":
            len(marriage_house_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "marriage_relevant_house_evidence",
                "matched":
                    len(marriage_house_evidence) > 0
            }
        ],
        "evidence":
            marriage_house_evidence,
        "interpretation_key":
            "marriage_relevant_houses"
    })


    # --------------------------------------------------------
    # RULE 5 — DASHA MARRIAGE ACTIVATION
    # --------------------------------------------------------

    dasha_connections = marriage_evidence[
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
            "MARRIAGE_DASHA_ACTIVATION",
        "domain": "marriage",
        "matched":
            len(dasha_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "marriage_dasha_connection_exists",
                "matched":
                    len(dasha_evidence) > 0
            }
        ],
        "evidence":
            dasha_evidence,
        "interpretation_key":
            "marriage_dasha_activation"
    })


    # --------------------------------------------------------
    # RULE 6 — TRANSIT MARRIAGE ACTIVATION
    # --------------------------------------------------------

    transit_connections = marriage_evidence[
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
            "MARRIAGE_TRANSIT_ACTIVATION",
        "domain": "marriage",
        "matched":
            len(transit_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "marriage_transit_connection_exists",
                "matched":
                    len(transit_evidence) > 0
            }
        ],
        "evidence":
            transit_evidence,
        "interpretation_key":
            "marriage_transit_activation"
    })


    # --------------------------------------------------------
    # RULE 7 — DASHA + TRANSIT
    # --------------------------------------------------------

    dasha_transit_connections = marriage_evidence[
        "dasha_transit_connections"
    ]

    rules.append({
        "rule_id":
            "MARRIAGE_DASHA_TRANSIT_COMBINATION",
        "domain": "marriage",
        "matched":
            len(dasha_transit_connections) > 0,
        "conditions": [
            {
                "condition":
                    "marriage_dasha_transit_connection_exists",
                "matched":
                    len(dasha_transit_connections) > 0
            }
        ],
        "evidence":
            dasha_transit_connections,
        "interpretation_key":
            "marriage_dasha_transit_activation"
    })


    # --------------------------------------------------------
    # RULE 8 — MULTI-FACTOR MARRIAGE ACTIVATION
    # --------------------------------------------------------

    evidence_categories = {

        "house_lord":
            (
                seventh_lord_exists
                and seventh_lord_placement_exists
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
            "MARRIAGE_MULTI_FACTOR_ACTIVATION",
        "domain": "marriage",
        "matched":
            len(active_categories) >= 2,
        "conditions": [
            {
                "condition":
                    "multiple_independent_marriage_evidence_categories",
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
            "marriage_multi_factor_activation"
    })


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {
        "domain": "marriage",
        "rules": rules,
        "matched_rule_count": sum(
            1
            for rule in rules
            if rule["matched"]
        ),
        "total_rule_count": len(rules)
    }


# ============================================================
# OPTIONAL VALIDATION HELPER
# ============================================================

def validate_marriage_rule_analysis(
    marriage_rule_analysis
):
    """
    Validate the Stage 8.16 output structure.
    """

    assert isinstance(
        marriage_rule_analysis,
        dict
    )

    assert (
        marriage_rule_analysis["domain"]
        == "marriage"
    )

    assert (
        marriage_rule_analysis["total_rule_count"]
        == 8
    )

    assert isinstance(
        marriage_rule_analysis["rules"],
        list
    )

    for rule in marriage_rule_analysis["rules"]:

        assert "rule_id" in rule
        assert "domain" in rule
        assert "matched" in rule
        assert "conditions" in rule
        assert "evidence" in rule
        assert "interpretation_key" in rule

        assert (
            rule["domain"]
            == "marriage"
        )

        assert isinstance(
            rule["matched"],
            bool
        )

        assert isinstance(
            rule["conditions"],
            list
        )

    return True