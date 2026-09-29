# ============================================================
# STAGE 8.24 — STRUCTURED INTERPRETATION
# ============================================================

"""
Convert validated Stage 8 evidence into structured
interpretation components.

IMPORTANT:
- Does NOT calculate planets.
- Does NOT calculate Dasha.
- Does NOT calculate transits.
- Does NOT create astrology evidence.
- Does NOT assign probabilities.
- Does NOT override rule results.

This is a newly designed VS Code implementation because
the original Colab interpretation source was not available.
"""


# ============================================================
# 1. INTERPRETATION TEMPLATES
# ============================================================

INTERPRETATION_TEMPLATES = {

    # --------------------------------------------------------
    # Career
    # --------------------------------------------------------

    "career_10th_lord":
        "The 10th-house lord has available natal placement evidence.",

    "career_karaka_presence":
        "Career-related karaka planets have available natal placement evidence.",

    "career_karaka_dignity":
        "Career-related karaka planets have available dignity evidence.",

    "career_relevant_houses":
        "Natal planetary evidence is present in houses relevant to career.",

    "career_dasha_activation":
        "Current Dasha evidence is connected with the career domain.",

    "career_transit_activation":
        "Current transit evidence is connected with the career domain.",

    "career_dasha_transit_activation":
        "Dasha and transit evidence are connected with the career domain.",

    "career_multi_factor_activation":
        "Multiple independent career evidence categories are simultaneously active.",


    # --------------------------------------------------------
    # Marriage
    # --------------------------------------------------------

    "marriage_house_lord":
        "The 7th-house lord has available natal placement evidence.",

    "marriage_karaka_presence":
        "Marriage-related karaka planets have available natal placement evidence.",

    "marriage_karaka_dignity":
        "Marriage-related karaka planets have available dignity evidence.",

    "marriage_relevant_houses":
        "Natal planetary evidence is present in houses relevant to marriage.",

    "marriage_dasha_activation":
        "Current Dasha evidence is connected with the marriage domain.",

    "marriage_transit_activation":
        "Current transit evidence is connected with the marriage domain.",

    "marriage_dasha_transit_activation":
        "Dasha and transit evidence are connected with the marriage domain.",

    "marriage_multi_factor_activation":
        "Multiple independent marriage evidence categories are simultaneously active.",


    # --------------------------------------------------------
    # Finance
    # --------------------------------------------------------

    "finance_2nd_house_lord":
        "The 2nd-house lord has available natal placement evidence.",

    "finance_11th_house_lord":
        "The 11th-house lord has available natal placement evidence.",

    "finance_karaka":
        "Finance-related karaka planets have available natal placement evidence.",

    "finance_karaka_dignity":
        "Finance-related karaka planets have available dignity evidence.",

    "finance_relevant_houses":
        "Natal planetary evidence is present in houses relevant to finance.",

    "finance_dasha_activation":
        "Current Dasha evidence is connected with the finance domain.",

    "finance_transit_activation":
        "Current transit evidence is connected with the finance domain.",

    "finance_dasha_transit_activation":
        "Dasha and transit evidence are connected with the finance domain.",

    "finance_multi_factor_activation":
        "Multiple independent finance evidence categories are simultaneously active.",


    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    "education_4th_house_lord":
        "The 4th-house lord has available natal placement evidence.",

    "education_5th_house_lord":
        "The 5th-house lord has available natal placement evidence.",

    "education_karaka":
        "Education-related karaka planets have available natal placement evidence.",

    "education_karaka_dignity":
        "Education-related karaka planets have available dignity evidence.",

    "education_relevant_houses":
        "Natal planetary evidence is present in houses relevant to education.",

    "education_dasha_activation":
        "Current Dasha evidence is connected with the education domain.",

    "education_transit_activation":
        "Current transit evidence is connected with the education domain.",

    "education_dasha_transit_activation":
        "Dasha and transit evidence are connected with the education domain.",

    "education_multi_factor_activation":
        "Multiple independent education evidence categories are simultaneously active.",


    # --------------------------------------------------------
    # Property
    # --------------------------------------------------------

    "property_4th_house_lord":
        "The 4th-house lord has available natal placement evidence.",

    "property_karaka":
        "Property-related karaka planets have available natal placement evidence.",

    "property_karaka_dignity":
        "Property-related karaka planets have available dignity evidence.",

    "property_relevant_houses":
        "Natal planetary evidence is present in houses relevant to property.",

    "property_dasha_activation":
        "Current Dasha evidence is connected with the property domain.",

    "property_transit_activation":
        "Current transit evidence is connected with the property domain.",

    "property_dasha_transit_activation":
        "Dasha and transit evidence are connected with the property domain.",

    "property_multi_factor_activation":
        "Multiple independent property evidence categories are simultaneously active.",
}


# ============================================================
# 2. FALLBACK INTERPRETATION
# ============================================================

def get_interpretation_text(
    interpretation_key: str,
) -> str:
    """
    Return a human-readable explanation for a rule key.

    Unknown keys receive a neutral fallback instead of
    inventing an interpretation.
    """

    return INTERPRETATION_TEMPLATES.get(
        interpretation_key,
        f"Structured evidence is available for rule '{interpretation_key}'.",
    )


# ============================================================
# 3. INTERPRET ONE RULE
# ============================================================

def interpret_rule(
    rule: dict,
) -> dict:
    """
    Convert one rule result into a structured interpretation.
    """

    interpretation_key = (
        rule.get("interpretation_key")
    )

    matched = bool(
        rule.get("matched", False)
    )

    return {
        "rule_id":
            rule.get("rule_id"),

        "domain":
            rule.get("domain"),

        "matched":
            matched,

        "interpretation_key":
            interpretation_key,

        "interpretation":
            (
                get_interpretation_text(
                    interpretation_key
                )
                if matched
                else None
            ),

        "evidence":
            rule.get("evidence", []),

        "conditions":
            rule.get("conditions", []),
    }


# ============================================================
# 4. INTERPRET ONE DOMAIN
# ============================================================

def interpret_domain_rules(
    domain: str,
    rule_analysis: dict,
) -> dict:
    """
    Interpret all matched rules for one domain.
    """

    rules = rule_analysis.get(
        "rules",
        [],
    )

    interpreted_rules = []

    for rule in rules:

        interpreted_rules.append(
            interpret_rule(rule)
        )

    matched_interpretations = [
        item
        for item in interpreted_rules
        if item["matched"]
    ]

    return {
        "domain":
            domain,

        "matched_rule_count":
            len(matched_interpretations),

        "total_rule_count":
            len(interpreted_rules),

        "rules":
            interpreted_rules,

        "matched_interpretations":
            matched_interpretations,
    }


# ============================================================
# 5. INTERPRET ALL DOMAINS
# ============================================================

def build_structured_interpretation(
    domain_rule_analyses: dict,
) -> dict:
    """
    Build structured interpretation for all domains.
    """

    domain_interpretations = {}

    for domain, rule_analysis in (
        domain_rule_analyses.items()
    ):

        domain_interpretations[domain] = (
            interpret_domain_rules(
                domain=domain,
                rule_analysis=rule_analysis,
            )
        )

    return {
        "domains":
            domain_interpretations
    }


# ============================================================
# 6. ADD TIMING CONTEXT
# ============================================================

def attach_timing_context(
    structured_interpretation: dict,
    stage_8_20_dasha_timing: dict,
    stage_8_21_transit_timing: dict,
) -> dict:
    """
    Attach previously calculated timing evidence.

    This function does not interpret the timing itself.
    """

    domain_interpretations = (
        structured_interpretation[
            "domains"
        ]
    )

    dasha_timing = (
        stage_8_20_dasha_timing.get(
            "domain_timing",
            {}
        )
    )

    transit_timing = (
        stage_8_21_transit_timing.get(
            "domain_transit_timing",
            {}
        )
    )

    dasha_transit_timing = (
        stage_8_21_transit_timing.get(
            "dasha_transit_timing",
            {}
        )
    )

    for domain in domain_interpretations:

        domain_interpretations[
            domain
        ][
            "timing"
        ] = {

            "dasha":
                dasha_timing.get(
                    domain,
                    []
                ),

            "transit":
                transit_timing.get(
                    domain,
                    []
                ),

            "dasha_transit":
                dasha_transit_timing.get(
                    domain,
                    []
                ),
        }

    return structured_interpretation


# ============================================================
# 7. BUILD FINAL STAGE 8.24 OBJECT
# ============================================================

def build_interpretation_analysis(
    domain_rule_analyses: dict,
    stage_8_20_dasha_timing: dict,
    stage_8_21_transit_timing: dict,
) -> dict:
    """
    Build the complete structured interpretation layer.
    """

    interpretation = (
        build_structured_interpretation(
            domain_rule_analyses=
                domain_rule_analyses,
        )
    )

    interpretation = attach_timing_context(
        structured_interpretation=
            interpretation,

        stage_8_20_dasha_timing=
            stage_8_20_dasha_timing,

        stage_8_21_transit_timing=
            stage_8_21_transit_timing,
    )

    return interpretation


# ============================================================
# 8. VALIDATION
# ============================================================

def validate_interpretation_analysis(
    interpretation_analysis: dict,
) -> None:

    assert isinstance(
        interpretation_analysis,
        dict,
    )

    assert "domains" in (
        interpretation_analysis
    )

    domains = (
        interpretation_analysis[
            "domains"
        ]
    )

    expected_domains = {
        "career",
        "marriage",
        "finance",
        "education",
        "property",
    }

    assert set(domains.keys()) == (
        expected_domains
    )

    for domain, data in domains.items():

        assert (
            data["domain"]
            == domain
        )

        assert isinstance(
            data["rules"],
            list,
        )

        assert isinstance(
            data["matched_interpretations"],
            list,
        )

        assert isinstance(
            data["matched_rule_count"],
            int,
        )

        assert isinstance(
            data["total_rule_count"],
            int,
        )

        assert "timing" in data

        assert "dasha" in (
            data["timing"]
        )

        assert "transit" in (
            data["timing"]
        )

        assert "dasha_transit" in (
            data["timing"]
        )

    return None