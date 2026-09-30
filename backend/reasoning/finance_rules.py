def evaluate_finance(evidence: list[str]) -> list[str]:
    return list(evidence)# ============================================================
# STAGE 8.17 — FINANCE RULE ENGINE
# ============================================================


def evaluate_finance_rules(
    domain_evidence,
    natal_lordship_analysis,
    planetary_strength_analysis,
    generic_rule_results,
    natal_planet_lookup,
):
    """
    Finance-specific astrology rule evaluation.

    Uses existing calculated evidence.

    Does not:
        - recalculate planetary positions
        - assign arbitrary probabilities
        - produce the final prediction
    """

    rules = []

    finance_evidence = domain_evidence["finance"]

    house_lord_map = natal_lordship_analysis[
        "house_lord_map"
    ]

    house_lord_placements = natal_lordship_analysis[
        "house_lord_placements"
    ]

    # --------------------------------------------------------
    # RULE 1 — 2ND HOUSE LORD
    # --------------------------------------------------------

    second_lord_data = house_lord_map.get(2)

    second_lord_placement = house_lord_placements.get(
        2,
        {}
    )

    second_lord_exists = (
        second_lord_data is not None
        and isinstance(
            second_lord_data.get("lord"),
            str
        )
    )

    second_lord_placement_exists = (
        second_lord_placement.get(
            "lord_data_available",
            False
        )
    )

    rules.append({
        "rule_id": "FINANCE_2ND_LORD_PLACEMENT",
        "domain": "finance",
        "matched": (
            second_lord_exists
            and second_lord_placement_exists
        ),
        "conditions": [
            {
                "condition":
                    "2nd_house_lord_exists",
                "matched":
                    second_lord_exists
            },
            {
                "condition":
                    "2nd_lord_natal_placement_exists",
                "matched":
                    second_lord_placement_exists
            }
        ],
        "evidence": {
            "house": 2,
            "lord": (
                second_lord_data.get("lord")
                if second_lord_data
                else None
            ),
            "house_rashi": (
                second_lord_data.get("rashi")
                if second_lord_data
                else None
            ),
            "lord_natal_house":
                second_lord_placement.get(
                    "lord_natal_house"
                ),
            "lord_natal_rashi":
                second_lord_placement.get(
                    "lord_natal_rashi"
                )
        },
        "interpretation_key":
            "finance_2nd_lord"
    })


    # --------------------------------------------------------
    # RULE 2 — 11TH HOUSE LORD
    # --------------------------------------------------------

    eleventh_lord_data = house_lord_map.get(11)

    eleventh_lord_placement = house_lord_placements.get(
        11,
        {}
    )

    eleventh_lord_exists = (
        eleventh_lord_data is not None
        and isinstance(
            eleventh_lord_data.get("lord"),
            str
        )
    )

    eleventh_lord_placement_exists = (
        eleventh_lord_placement.get(
            "lord_data_available",
            False
        )
    )

    rules.append({
        "rule_id":
            "FINANCE_11TH_LORD_PLACEMENT",
        "domain": "finance",
        "matched": (
            eleventh_lord_exists
            and eleventh_lord_placement_exists
        ),
        "conditions": [
            {
                "condition":
                    "11th_house_lord_exists",
                "matched":
                    eleventh_lord_exists
            },
            {
                "condition":
                    "11th_lord_natal_placement_exists",
                "matched":
                    eleventh_lord_placement_exists
            }
        ],
        "evidence": {
            "house": 11,
            "lord": (
                eleventh_lord_data.get("lord")
                if eleventh_lord_data
                else None
            ),
            "house_rashi": (
                eleventh_lord_data.get("rashi")
                if eleventh_lord_data
                else None
            ),
            "lord_natal_house":
                eleventh_lord_placement.get(
                    "lord_natal_house"
                ),
            "lord_natal_rashi":
                eleventh_lord_placement.get(
                    "lord_natal_rashi"
                )
        },
        "interpretation_key":
            "finance_11th_lord"
    })


    # --------------------------------------------------------
    # RULE 3 — FINANCE KARAKA PRESENCE
    # --------------------------------------------------------

    finance_karakas = [
        "Jupiter",
        "Venus"
    ]

    karaka_evidence = []

    for planet in finance_karakas:

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
        "rule_id":
            "FINANCE_KARAKA_PRESENCE",
        "domain": "finance",
        "matched":
            len(karaka_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "finance_karaka_available",
                "matched":
                    len(karaka_evidence) > 0
            }
        ],
        "evidence":
            karaka_evidence,
        "interpretation_key":
            "finance_karaka_presence"
    })


    # --------------------------------------------------------
    # RULE 4 — FINANCE KARAKA DIGNITY
    # --------------------------------------------------------

    dignity_data = planetary_strength_analysis[
        "planetary_dignity"
    ]

    dignity_evidence = []

    for planet in finance_karakas:

        if planet in dignity_data:

            data = dignity_data[planet]

            dignity_evidence.append({
                "planet": planet,
                "rashi": data.get("rashi"),
                "dignity": data.get("dignity")
            })

    rules.append({
        "rule_id":
            "FINANCE_KARAKA_DIGNITY",
        "domain": "finance",
        "matched":
            len(dignity_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "finance_karaka_dignity_available",
                "matched":
                    len(dignity_evidence) > 0
            }
        ],
        "evidence":
            dignity_evidence,
        "interpretation_key":
            "finance_karaka_dignity"
    })


    # --------------------------------------------------------
    # RULE 5 — FINANCE HOUSE EVIDENCE
    # --------------------------------------------------------

    natal_planets_by_house = finance_evidence[
        "natal_planets_by_house"
    ]

    finance_house_evidence = {}

    for house in [2, 11, 5, 9]:

        planets = natal_planets_by_house.get(
            house,
            []
        )

        if planets:

            finance_house_evidence[house] = planets

    rules.append({
        "rule_id":
            "FINANCE_RELEVANT_HOUSE_EVIDENCE",
        "domain": "finance",
        "matched":
            len(finance_house_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "finance_relevant_house_evidence",
                "matched":
                    len(finance_house_evidence) > 0
            }
        ],
        "evidence":
            finance_house_evidence,
        "interpretation_key":
            "finance_relevant_houses"
    })


    # --------------------------------------------------------
    # RULE 6 — DASHA FINANCE ACTIVATION
    # --------------------------------------------------------

    dasha_connections = finance_evidence[
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
            "FINANCE_DASHA_ACTIVATION",
        "domain": "finance",
        "matched":
            len(dasha_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "finance_dasha_connection_exists",
                "matched":
                    len(dasha_evidence) > 0
            }
        ],
        "evidence":
            dasha_evidence,
        "interpretation_key":
            "finance_dasha_activation"
    })


    # --------------------------------------------------------
    # RULE 7 — TRANSIT FINANCE ACTIVATION
    # --------------------------------------------------------

    transit_connections = finance_evidence[
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
            "FINANCE_TRANSIT_ACTIVATION",
        "domain": "finance",
        "matched":
            len(transit_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "finance_transit_connection_exists",
                "matched":
                    len(transit_evidence) > 0
            }
        ],
        "evidence":
            transit_evidence,
        "interpretation_key":
            "finance_transit_activation"
    })


    # --------------------------------------------------------
    # RULE 8 — DASHA + TRANSIT
    # --------------------------------------------------------

    dasha_transit_connections = finance_evidence[
        "dasha_transit_connections"
    ]

    rules.append({
        "rule_id":
            "FINANCE_DASHA_TRANSIT_COMBINATION",
        "domain": "finance",
        "matched":
            len(dasha_transit_connections) > 0,
        "conditions": [
            {
                "condition":
                    "finance_dasha_transit_connection_exists",
                "matched":
                    len(dasha_transit_connections) > 0
            }
        ],
        "evidence":
            dasha_transit_connections,
        "interpretation_key":
            "finance_dasha_transit_activation"
    })


    # --------------------------------------------------------
    # RULE 9 — MULTI-FACTOR FINANCE ACTIVATION
    # --------------------------------------------------------

    evidence_categories = {

        "house_lords":
            (
                second_lord_exists
                and second_lord_placement_exists
            )
            or
            (
                eleventh_lord_exists
                and eleventh_lord_placement_exists
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
            "FINANCE_MULTI_FACTOR_ACTIVATION",
        "domain": "finance",
        "matched":
            len(active_categories) >= 2,
        "conditions": [
            {
                "condition":
                    "multiple_independent_finance_evidence_categories",
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
            "finance_multi_factor_activation"
    })


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {
        "domain": "finance",
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

def validate_finance_rule_analysis(
    finance_rule_analysis
):
    """
    Validate the Stage 8.17 output structure.
    """

    assert isinstance(
        finance_rule_analysis,
        dict
    )

    assert (
        finance_rule_analysis["domain"]
        == "finance"
    )

    assert (
        finance_rule_analysis["total_rule_count"]
        == 9
    )

    assert isinstance(
        finance_rule_analysis["rules"],
        list
    )

    for rule in finance_rule_analysis["rules"]:

        assert "rule_id" in rule
        assert "domain" in rule
        assert "matched" in rule
        assert "conditions" in rule
        assert "evidence" in rule
        assert "interpretation_key" in rule

        assert (
            rule["domain"]
            == "finance"
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