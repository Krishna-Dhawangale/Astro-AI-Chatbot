# ============================================================
# STAGE 8.15 — CAREER RULE ENGINE
# ============================================================
#
# Purpose:
#   Career-specific astrology rule evaluation.
#
# IMPORTANT:
#   - No planetary calculation here
#   - No ML here
#   - No arbitrary prediction score
#   - No probability
#   - Uses previously calculated evidence
# ============================================================


def evaluate_career_rules(
    domain_evidence,
    natal_lordship_analysis,
    planetary_strength_analysis,
    generic_rule_results,
    natal_planet_lookup
):
    """
    Career-specific astrology rule evaluation.

    This layer combines previously calculated evidence.
    It does NOT calculate planetary positions.
    It does NOT assign arbitrary prediction scores.
    """

    rules = []

    career_evidence = domain_evidence["career"]

    # --------------------------------------------------------
    # RULE 1 — 10TH HOUSE LORD
    # --------------------------------------------------------

    house_lord_map = natal_lordship_analysis[
        "house_lord_map"
    ]

    house_lord_placements = natal_lordship_analysis[
        "house_lord_placements"
    ]

    tenth_lord_data = house_lord_map.get(10)

    tenth_lord_placement = house_lord_placements.get(
        10,
        {}
    )

    tenth_lord_exists = (
        tenth_lord_data is not None
        and isinstance(
            tenth_lord_data.get("lord"),
            str
        )
    )

    tenth_lord_placement_exists = (
        tenth_lord_placement.get(
            "lord_data_available",
            False
        )
    )

    rules.append({
        "rule_id": "CAREER_10TH_LORD_PLACEMENT",
        "domain": "career",
        "matched": (
            tenth_lord_exists
            and tenth_lord_placement_exists
        ),
        "conditions": [
            {
                "condition": "10th_house_lord_exists",
                "matched": tenth_lord_exists
            },
            {
                "condition": "10th_lord_natal_placement_exists",
                "matched": tenth_lord_placement_exists
            }
        ],
        "evidence": {
            "house": 10,
            "lord": (
                tenth_lord_data.get("lord")
                if tenth_lord_data
                else None
            ),
            "house_rashi": (
                tenth_lord_data.get("rashi")
                if tenth_lord_data
                else None
            ),
            "lord_natal_house":
                tenth_lord_placement.get(
                    "lord_natal_house"
                ),
            "lord_natal_rashi":
                tenth_lord_placement.get(
                    "lord_natal_rashi"
                )
        },
        "interpretation_key":
            "career_10th_lord"
    })

    # --------------------------------------------------------
    # RULE 2 — CAREER KARAKA PRESENCE
    # --------------------------------------------------------

    career_karakas = ["Sun", "Saturn"]

    karaka_evidence = []

    for planet in career_karakas:

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
        "rule_id": "CAREER_KARAKA_PRESENCE",
        "domain": "career",
        "matched": len(karaka_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "career_karaka_available",
                "matched":
                    len(karaka_evidence) > 0
            }
        ],
        "evidence": karaka_evidence,
        "interpretation_key":
            "career_karaka_presence"
    })

    # --------------------------------------------------------
    # RULE 3 — CAREER KARAKA DIGNITY
    # --------------------------------------------------------

    dignity_data = planetary_strength_analysis[
        "planetary_dignity"
    ]

    dignity_evidence = []

    for planet in career_karakas:

        if planet in dignity_data:

            data = dignity_data[planet]

            dignity_evidence.append({
                "planet": planet,
                "rashi": data.get("rashi"),
                "dignity": data.get("dignity")
            })

    rules.append({
        "rule_id": "CAREER_KARAKA_DIGNITY",
        "domain": "career",
        "matched": len(dignity_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "career_karaka_dignity_available",
                "matched":
                    len(dignity_evidence) > 0
            }
        ],
        "evidence": dignity_evidence,
        "interpretation_key":
            "career_karaka_dignity"
    })

    # --------------------------------------------------------
    # RULE 4 — CAREER HOUSE EVIDENCE
    # --------------------------------------------------------

    natal_planets_by_house = career_evidence[
        "natal_planets_by_house"
    ]

    career_house_evidence = {}

    for house in [10, 2, 6, 11]:

        planets = natal_planets_by_house.get(
            house,
            []
        )

        if planets:
            career_house_evidence[house] = planets

    rules.append({
        "rule_id": "CAREER_RELEVANT_HOUSE_EVIDENCE",
        "domain": "career",
        "matched": (
            len(career_house_evidence) > 0
        ),
        "conditions": [
            {
                "condition":
                    "career_relevant_house_evidence",
                "matched":
                    len(career_house_evidence) > 0
            }
        ],
        "evidence": career_house_evidence,
        "interpretation_key":
            "career_relevant_houses"
    })

    # --------------------------------------------------------
    # RULE 5 — DASHA CAREER ACTIVATION
    # --------------------------------------------------------

    dasha_connections = career_evidence[
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
        "rule_id": "CAREER_DASHA_ACTIVATION",
        "domain": "career",
        "matched": len(dasha_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "career_dasha_connection_exists",
                "matched":
                    len(dasha_evidence) > 0
            }
        ],
        "evidence": dasha_evidence,
        "interpretation_key":
            "career_dasha_activation"
    })

    # --------------------------------------------------------
    # RULE 6 — TRANSIT CAREER ACTIVATION
    # --------------------------------------------------------

    transit_connections = career_evidence[
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
        "rule_id": "CAREER_TRANSIT_ACTIVATION",
        "domain": "career",
        "matched": len(transit_evidence) > 0,
        "conditions": [
            {
                "condition":
                    "career_transit_connection_exists",
                "matched":
                    len(transit_evidence) > 0
            }
        ],
        "evidence": transit_evidence,
        "interpretation_key":
            "career_transit_activation"
    })

    # --------------------------------------------------------
    # RULE 7 — DASHA + TRANSIT COMBINATION
    # --------------------------------------------------------

    dasha_transit_connections = career_evidence[
        "dasha_transit_connections"
    ]

    rules.append({
        "rule_id":
            "CAREER_DASHA_TRANSIT_COMBINATION",
        "domain": "career",
        "matched":
            len(dasha_transit_connections) > 0,
        "conditions": [
            {
                "condition":
                    "career_dasha_transit_connection_exists",
                "matched":
                    len(dasha_transit_connections) > 0
            }
        ],
        "evidence":
            dasha_transit_connections,
        "interpretation_key":
            "career_dasha_transit_activation"
    })

    # --------------------------------------------------------
    # RULE 8 — MULTI-FACTOR CAREER ACTIVATION
    # --------------------------------------------------------
    #
    # This is NOT a probability score.
    #
    # It simply checks whether several independent
    # evidence categories are simultaneously present.

    evidence_categories = {

        "house_lord":
            tenth_lord_exists
            and tenth_lord_placement_exists,

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
            "CAREER_MULTI_FACTOR_ACTIVATION",
        "domain": "career",
        "matched":
            len(active_categories) >= 2,
        "conditions": [
            {
                "condition":
                    "multiple_independent_career_evidence_categories",
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
            "career_multi_factor_activation"
    })

    return {
        "domain": "career",
        "rules": rules,
        "matched_rule_count": sum(
            1
            for rule in rules
            if rule["matched"]
        ),
        "total_rule_count": len(rules)
    }


# ============================================================
# STAGE 8.15 BUILDER
# ============================================================

def build_career_rule_analysis(
    domain_evidence,
    natal_lordship_analysis,
    planetary_strength_analysis,
    generic_rule_results,
    natal_planet_lookup
):
    """
    Builds the complete Stage 8.15 object.
    """

    career_rule_analysis = evaluate_career_rules(
        domain_evidence=domain_evidence,
        natal_lordship_analysis=natal_lordship_analysis,
        planetary_strength_analysis=planetary_strength_analysis,
        generic_rule_results=generic_rule_results,
        natal_planet_lookup=natal_planet_lookup
    )

    stage_8_15_career_rules = {
        "career_rule_analysis":
            career_rule_analysis,

        "generic_career_rules":
            generic_rule_results["career"]
    }

    return stage_8_15_career_rules


# ============================================================
# VALIDATION
# ============================================================

def validate_career_rule_analysis(
    stage_8_15_career_rules
):
    """
    Validate Stage 8.15 structure.
    """

    career_rule_analysis = (
        stage_8_15_career_rules[
            "career_rule_analysis"
        ]
    )

    assert career_rule_analysis["domain"] == "career"

    assert (
        career_rule_analysis["total_rule_count"]
        == 8
    )

    assert (
        career_rule_analysis["matched_rule_count"]
        >= 0
    )

    for rule in career_rule_analysis["rules"]:

        assert "rule_id" in rule
        assert "domain" in rule
        assert "matched" in rule
        assert "conditions" in rule
        assert "evidence" in rule
        assert "interpretation_key" in rule

        assert rule["domain"] == "career"

        assert isinstance(
            rule["matched"],
            bool
        )

    assert "generic_career_rules" in (
        stage_8_15_career_rules
    )

    return True


# ============================================================
# MODULE SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STAGE 8.15 — CAREER RULE ENGINE")
    print("=" * 80)
    print()
    print("✓ career_rules.py loaded successfully")
    print("✓ 8 career rules defined")
    print("✓ VS Code-compatible")
    print("✓ natal_planet_lookup passed explicitly")
    print()
    print("Stage 8.15 requires Stage 8.10–8.14 inputs to execute.")