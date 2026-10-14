# ============================================================
# STAGE 8.22 — MULTI-DOMAIN RULE ENGINE
# ============================================================

"""
Multi-domain astrology rule integration.

This module combines the already calculated domain-specific
rule engines with Dasha and transit timing.

IMPORTANT:
- Does NOT calculate planets.
- Does NOT recalculate Dasha.
- Does NOT recalculate transits.
- Does NOT assign arbitrary probabilities.
- Does NOT rank or select a "winning" domain.
- Preserves each domain independently.
"""


EXPECTED_DOMAINS = {
    "career",
    "marriage",
    "finance",
    "education",
    "property",
}


# ============================================================
# 1. DOMAIN RELEVANCE
# ============================================================

def evaluate_domain_relevance(
    domain: str,
    rule_analysis: dict,
) -> dict:
    """
    Determine whether a domain has meaningful rule evidence.

    A domain is considered relevant when at least one
    domain-specific rule is matched.

    This is NOT a prediction score.
    """

    rules = rule_analysis["rules"]

    matched_rules = [
        rule
        for rule in rules
        if rule["matched"]
    ]

    unmatched_rules = [
        rule
        for rule in rules
        if not rule["matched"]
    ]

    return {
        "domain": domain,
        "relevant": len(matched_rules) > 0,
        "matched_rule_count": len(matched_rules),
        "total_rule_count": len(rules),
        "matched_rules": matched_rules,
        "unmatched_rules": unmatched_rules,
    }


# ============================================================
# 2. EVALUATE ALL DOMAINS
# ============================================================

def evaluate_all_domains(
    domain_rule_analyses: dict,
) -> dict:
    """
    Evaluate relevance for every configured domain.
    """

    if set(domain_rule_analyses.keys()) != EXPECTED_DOMAINS:
        raise ValueError(
            f"Expected domains {EXPECTED_DOMAINS}, "
            f"got {set(domain_rule_analyses.keys())}"
        )

    domain_relevance = {}

    for domain, rule_analysis in domain_rule_analyses.items():
        domain_relevance[domain] = evaluate_domain_relevance(
            domain=domain,
            rule_analysis=rule_analysis,
        )

    return domain_relevance


# ============================================================
# 3. EXTRACT RELEVANT DOMAINS
# ============================================================

def extract_relevant_domains(
    domain_relevance: dict,
) -> list:
    """
    Return domains with at least one matched rule.
    """

    return [
        domain
        for domain, result in domain_relevance.items()
        if result["relevant"]
    ]


# ============================================================
# 4. EXTRACT INACTIVE DOMAINS
# ============================================================

def extract_inactive_domains(
    domain_relevance: dict,
) -> list:
    """
    Return domains with no matched rules.
    """

    return [
        domain
        for domain, result in domain_relevance.items()
        if not result["relevant"]
    ]


# ============================================================
# 5. BUILD CROSS-DOMAIN COMBINATIONS
# ============================================================

def build_cross_domain_combinations(
    relevant_domains: list,
    domain_relevance: dict,
) -> list:
    """
    Build pairwise combinations of relevant domains.

    This does not rank domains.
    It only records that multiple domains are active.
    """

    cross_domain_combinations = []

    for i in range(len(relevant_domains)):

        for j in range(i + 1, len(relevant_domains)):

            domain_a = relevant_domains[i]
            domain_b = relevant_domains[j]

            result_a = domain_relevance[domain_a]
            result_b = domain_relevance[domain_b]

            cross_domain_combinations.append({
                "domains": [
                    domain_a,
                    domain_b,
                ],
                "domain_a": domain_a,
                "domain_b": domain_b,
                "domain_a_matched_rules":
                    result_a["matched_rule_count"],
                "domain_b_matched_rules":
                    result_b["matched_rule_count"],
                "relationship_type":
                    "multi_domain_activation",
            })

    return cross_domain_combinations


# ============================================================
# 6. BUILD TIMING SUMMARY
# ============================================================

def build_domain_timing_summary(
    domain_rule_analyses: dict,
    stage_8_20_dasha_timing: dict,
    stage_8_21_transit_timing: dict,
) -> dict:
    """
    Attach Dasha and transit timing information to each domain.

    Timing information is already calculated by earlier stages.
    """

    dasha_timing = stage_8_20_dasha_timing[
        "domain_timing"
    ]

    transit_timing = stage_8_21_transit_timing[
        "domain_transit_timing"
    ]

    dasha_transit_timing = stage_8_21_transit_timing[
        "dasha_transit_timing"
    ]

    domain_timing_summary = {}

    for domain in domain_rule_analyses:

        domain_timing_summary[domain] = {
            "dasha_records":
                dasha_timing.get(domain, []),

            "transit_records":
                transit_timing.get(domain, []),

            "dasha_transit_records":
                dasha_transit_timing.get(domain, []),
        }

    return domain_timing_summary


# ============================================================
# 7. BUILD MULTI-DOMAIN EVIDENCE
# ============================================================

def build_multi_domain_evidence(
    relevant_domains: list,
    domain_relevance: dict,
    domain_timing_summary: dict,
) -> dict:
    """
    Combine matched rules and timing information
    for each relevant domain.
    """

    multi_domain_evidence = {}

    for domain in relevant_domains:

        multi_domain_evidence[domain] = {
            "matched_rules":
                domain_relevance[
                    domain
                ][
                    "matched_rules"
                ],

            "dasha_timing":
                domain_timing_summary[
                    domain
                ][
                    "dasha_records"
                ],

            "transit_timing":
                domain_timing_summary[
                    domain
                ][
                    "transit_records"
                ],

            "dasha_transit_timing":
                domain_timing_summary[
                    domain
                ][
                    "dasha_transit_records"
                ],
        }

    return multi_domain_evidence


# ============================================================
# 8. BUILD QUESTION-READY CONTEXT
# ============================================================

def build_multi_domain_context(
    relevant_domains: list,
    inactive_domains: list,
    multi_domain_evidence: dict,
    cross_domain_combinations: list,
) -> dict:
    """
    Build structured context that can later be consumed
    by the interpretation layer.
    """

    return {
        "relevant_domains": relevant_domains,
        "inactive_domains": inactive_domains,
        "domain_count": len(relevant_domains),
        "multi_domain_question":
            len(relevant_domains) > 1,
        "domains": multi_domain_evidence,
        "cross_domain_combinations":
            cross_domain_combinations,
    }


# ============================================================
# 9. MAIN STAGE 8.22 ENGINE
# ============================================================

def build_multi_domain_analysis(
    generic_rule_results: dict,
    career_rule_analysis: dict,
    marriage_rule_analysis: dict,
    finance_rule_analysis: dict,
    education_rule_analysis: dict,
    property_rule_analysis: dict,
    stage_8_20_dasha_timing: dict,
    stage_8_21_transit_timing: dict,
) -> dict:
    """
    Run the complete Stage 8.22 multi-domain engine.
    """

    domain_rule_analyses = {
        "career": career_rule_analysis,
        "marriage": marriage_rule_analysis,
        "finance": finance_rule_analysis,
        "education": education_rule_analysis,
        "property": property_rule_analysis,
    }

    # --------------------------------------------------------
    # Domain relevance
    # --------------------------------------------------------

    domain_relevance = evaluate_all_domains(
        domain_rule_analyses
    )

    relevant_domains = extract_relevant_domains(
        domain_relevance
    )

    inactive_domains = extract_inactive_domains(
        domain_relevance
    )

    # --------------------------------------------------------
    # Cross-domain combinations
    # --------------------------------------------------------

    cross_domain_combinations = (
        build_cross_domain_combinations(
            relevant_domains=relevant_domains,
            domain_relevance=domain_relevance,
        )
    )

    # --------------------------------------------------------
    # Timing summary
    # --------------------------------------------------------

    domain_timing_summary = build_domain_timing_summary(
        domain_rule_analyses=domain_rule_analyses,
        stage_8_20_dasha_timing=stage_8_20_dasha_timing,
        stage_8_21_transit_timing=stage_8_21_transit_timing,
    )

    # --------------------------------------------------------
    # Multi-domain evidence
    # --------------------------------------------------------

    multi_domain_evidence = build_multi_domain_evidence(
        relevant_domains=relevant_domains,
        domain_relevance=domain_relevance,
        domain_timing_summary=domain_timing_summary,
    )

    # --------------------------------------------------------
    # Question-ready context
    # --------------------------------------------------------

    multi_domain_context = build_multi_domain_context(
        relevant_domains=relevant_domains,
        inactive_domains=inactive_domains,
        multi_domain_evidence=multi_domain_evidence,
        cross_domain_combinations=cross_domain_combinations,
    )

    # --------------------------------------------------------
    # Unified Stage 8.22 object
    # --------------------------------------------------------

    return {
        "domain_relevance": domain_relevance,
        "relevant_domains": relevant_domains,
        "inactive_domains": inactive_domains,
        "cross_domain_combinations":
            cross_domain_combinations,
        "timing_summary":
            domain_timing_summary,
        "multi_domain_evidence":
            multi_domain_evidence,
        "context":
            multi_domain_context,
    }


# ============================================================
# 10. VALIDATION
# ============================================================

def validate_multi_domain_analysis(
    stage_8_22_multi_domain: dict,
) -> None:
    """
    Validate the Stage 8.22 output structure.
    """

    domain_relevance = (
        stage_8_22_multi_domain[
            "domain_relevance"
        ]
    )

    assert set(
        domain_relevance.keys()
    ) == EXPECTED_DOMAINS

    timing_summary = (
        stage_8_22_multi_domain[
            "timing_summary"
        ]
    )

    assert set(
        timing_summary.keys()
    ) == EXPECTED_DOMAINS

    relevant_domains = (
        stage_8_22_multi_domain[
            "relevant_domains"
        ]
    )

    inactive_domains = (
        stage_8_22_multi_domain[
            "inactive_domains"
        ]
    )

    assert isinstance(
        relevant_domains,
        list,
    )

    assert isinstance(
        inactive_domains,
        list,
    )

    assert set(relevant_domains).isdisjoint(
        set(inactive_domains)
    )

    assert set(
        relevant_domains + inactive_domains
    ) == EXPECTED_DOMAINS

    # --------------------------------------------------------
    # Domain relevance validation
    # --------------------------------------------------------

    for domain, result in domain_relevance.items():

        assert result["domain"] == domain

        assert isinstance(
            result["relevant"],
            bool,
        )

        assert isinstance(
            result["matched_rules"],
            list,
        )

        assert isinstance(
            result["unmatched_rules"],
            list,
        )

    # --------------------------------------------------------
    # Cross-domain validation
    # --------------------------------------------------------

    cross_domain_combinations = (
        stage_8_22_multi_domain[
            "cross_domain_combinations"
        ]
    )

    for combination in cross_domain_combinations:

        assert len(
            combination["domains"]
        ) == 2

        assert (
            combination["domain_a"]
            != combination["domain_b"]
        )

        assert (
            combination["relationship_type"]
            == "multi_domain_activation"
        )

    # --------------------------------------------------------
    # Context validation
    # --------------------------------------------------------

    context = (
        stage_8_22_multi_domain[
            "context"
        ]
    )

    assert (
        context["domain_count"]
        == len(relevant_domains)
    )

    assert (
        context["multi_domain_question"]
        ==
        (
            len(relevant_domains) > 1
        )
    )


# ============================================================
# 11. CONVENIENCE RUNNER
# ============================================================

def run_multi_domain_engine(
    generic_rule_results: dict,
    career_rule_analysis: dict,
    marriage_rule_analysis: dict,
    finance_rule_analysis: dict,
    education_rule_analysis: dict,
    property_rule_analysis: dict,
    stage_8_20_dasha_timing: dict,
    stage_8_21_transit_timing: dict,
) -> dict:
    """
    Execute and validate Stage 8.22.
    """

    result = build_multi_domain_analysis(
        generic_rule_results=
            generic_rule_results,

        career_rule_analysis=
            career_rule_analysis,

        marriage_rule_analysis=
            marriage_rule_analysis,

        finance_rule_analysis=
            finance_rule_analysis,

        education_rule_analysis=
            education_rule_analysis,

        property_rule_analysis=
            property_rule_analysis,

        stage_8_20_dasha_timing=
            stage_8_20_dasha_timing,

        stage_8_21_transit_timing=
            stage_8_21_transit_timing,
    )

    validate_multi_domain_analysis(result)

    return result