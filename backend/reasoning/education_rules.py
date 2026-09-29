# ============================================================
# STAGE 8.18 — EDUCATION RULE ENGINE
# ============================================================


def evaluate_education_rules(
    domain_evidence,
    natal_lordship_analysis,
    planetary_strength_analysis,
    generic_rule_results,
    natal_planet_lookup,
):
    """
    Education-specific astrology rule evaluation.

    Uses existing calculated evidence.

    Does not:
        - recalculate planetary positions
        - assign arbitrary probabilities
        - produce the final prediction
    """

    rules = []

    education_evidence = domain_evidence["education"]

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
        "rule_id":
            "EDUCATION_4TH_LORD_PLACEMENT",
        "domain": "education",
        "matched": (
            fourth_lord_exists
            and fourth_lord_placement_exists
        ),
        "conditions": [
            {
                "condition":
                    "4th_house_lord_exists",
                "matched":
                    fourth_lord_exists
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
            "education_4th_lord"
    })


    # --------------------------------------------------------
    # RULE 2 — 5TH HOUSE LORD
    # --------------------------------------------------------

    fifth_lord_data = house_lord_map.get(5)

    fifth_lord_placement = house_lord_placements.get(
        5,
        {}
    )

    fifth_lord_exists = (
        fifth_lord_data is not None
        and isinstance(
            fifth_lord_data.get("lord"),
            str
        )
    )

    fifth_lord_placement_exists = (
        fifth_lord_placement.get(
            "lord_data_available",
            False
        )
    )

    rules.append({
        "rule_id":
            "EDUCATION_5TH_LORD_PLACEMENT",
        "domain": "education",
        "matched": (
            fifth_lord_exists
            and fifth_lord_placement_exists
        ),
        "conditions": [
            {
                "condition":
                    "5th_house_lord_exists",
                "matched":
                    fifth_lord_exists
            },
            {
                "condition":
                    "5th_lord_natal_placement_exists",
                "matched":
                    fifth_lord_placement_exists
            }
        ],
        "evidence": {
            "house": 5,
            "lord": (
                fifth_lord_data.get("lord")
                if fifth_lord_data
                else None
            ),
            "house_rashi": (
                fifth_lord_data.get("rashi")
                if fifth_lord_data
                else None
            ),
            "lord_natal_house":
                fifth_lord_placement.get(
                    "lord_natal_house"
                ),
            "lord_natal_rashi":
                fifth_lord_placement.get(
                    "lord_natal_rashi"
                )
        },
        "interpretation_key":
            "education_5th_lord"
    })


    # --------------------------------------------------------
    # RULE 3 — EDUCATION KARAKA PRESENCE
    # --------------------------------------------------------

    education_karakas = [
        "Jupiter",
        "Mercury"
    ]

    karaka_evidence = []

    for planet in education_karakas:

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
        "rule_id":
            "EDUCATION_KARAKA_PRESENCE",
        "domain": "education",
        "matched":
            len(karaka_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "education_karaka_available",
                "matched":
                    len(karaka_evidence) > 0
            }
        ],
        "evidence":
            karaka_evidence,
        "interpretation_key":
            "education_karaka_presence"
    })


    # --------------------------------------------------------
    # RULE 4 — EDUCATION KARAKA DIGNITY
    # --------------------------------------------------------

    dignity_data = planetary_strength_analysis[
        "planetary_dignity"
    ]

    dignity_evidence = []

    for planet in education_karakas:

        if planet in dignity_data:

            data = dignity_data[planet]

            dignity_evidence.append({
                "planet": planet,
                "rashi": data.get("rashi"),
                "dignity": data.get("dignity")
            })

    rules.append({
        "rule_id":
            "EDUCATION_KARAKA_DIGNITY",
        "domain": "education",
        "matched":
            len(dignity_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "education_karaka_dignity_available",
                "matched":
                    len(dignity_evidence) > 0
            }
        ],
        "evidence":
            dignity_evidence,
        "interpretation_key":
            "education_karaka_dignity"
    })


    # --------------------------------------------------------
    # RULE 5 — EDUCATION HOUSE EVIDENCE
    # --------------------------------------------------------

    natal_planets_by_house = education_evidence[
        "natal_planets_by_house"
    ]

    education_house_evidence = {}

    for house in [4, 5, 9]:

        planets = natal_planets_by_house.get(
            house,
            []
        )

        if planets:

            education_house_evidence[house] = planets

    rules.append({
        "rule_id":
            "EDUCATION_RELEVANT_HOUSE_EVIDENCE",
        "domain": "education",
        "matched":
            len(education_house_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "education_relevant_house_evidence",
                "matched":
                    len(education_house_evidence) > 0
            }
        ],
        "evidence":
            education_house_evidence,
        "interpretation_key":
            "education_relevant_houses"
    })


    # --------------------------------------------------------
    # RULE 6 — DASHA EDUCATION ACTIVATION
    # --------------------------------------------------------

    dasha_connections = education_evidence[
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
            "EDUCATION_DASHA_ACTIVATION",
        "domain": "education",
        "matched":
            len(dasha_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "education_dasha_connection_exists",
                "matched":
                    len(dasha_evidence) > 0
            }
        ],
        "evidence":
            dasha_evidence,
        "interpretation_key":
            "education_dasha_activation"
    })


    # --------------------------------------------------------
    # RULE 7 — TRANSIT EDUCATION ACTIVATION
    # --------------------------------------------------------

    transit_connections = education_evidence[
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
            "EDUCATION_TRANSIT_ACTIVATION",
        "domain": "education",
        "matched":
            len(transit_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "education_transit_connection_exists",
                "matched":
                    len(transit_evidence) > 0
            }
        ],
        "evidence":
            transit_evidence,
        "interpretation_key":
            "education_transit_activation"
    })


    # --------------------------------------------------------
    # RULE 8 — DASHA + TRANSIT
    # --------------------------------------------------------

    dasha_transit_connections = education_evidence[
        "dasha_transit_connections"
    ]

    rules.append({
        "rule_id":
            "EDUCATION_DASHA_TRANSIT_COMBINATION",
        "domain": "education",
        "matched":
            len(dasha_transit_connections) > 0,
        "conditions": [
            {
                "condition":
                    "education_dasha_transit_connection_exists",
                "matched":
                    len(dasha_transit_connections) > 0
            }
        ],
        "evidence":
            dasha_transit_connections,
        "interpretation_key":
            "education_dasha_transit_activation"
    })


    # --------------------------------------------------------
    # RULE 9 — MULTI-FACTOR EDUCATION ACTIVATION
    # --------------------------------------------------------

    evidence_categories = {

        "house_lords":
            (
                fourth_lord_exists
                and fourth_lord_placement_exists
            )
            or
            (
                fifth_lord_exists
                and fifth_lord_placement_exists
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
            "EDUCATION_MULTI_FACTOR_ACTIVATION",
        "domain": "education",
        "matched":
            len(active_categories) >= 2,
        "conditions": [
            {
                "condition":
                    "multiple_independent_education_evidence_categories",
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
            "education_multi_factor_activation"
    })


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {
        "domain": "education",
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

def validate_education_rule_analysis(
    education_rule_analysis
):
    """
    Validate the Stage 8.18 output structure.
    """

    assert isinstance(
        education_rule_analysis,
        dict
    )

    assert (
        education_rule_analysis["domain"]
        == "education"
    )

    assert (
        education_rule_analysis["total_rule_count"]
        == 9
    )

    assert isinstance(
        education_rule_analysis["rules"],
        list
    )

    for rule in education_rule_analysis["rules"]:

        assert "rule_id" in rule
        assert "domain" in rule
        assert "matched" in rule
        assert "conditions" in rule
        assert "evidence" in rule
        assert "interpretation_key" in rule

        assert (
            rule["domain"]
            == "education"
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