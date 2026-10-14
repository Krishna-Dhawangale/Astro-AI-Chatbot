# ============================================================
# STAGE 8 — REASONING PIPELINE
# ============================================================

from .multi_domain import (
    run_multi_domain_engine,
)


def run_stage8_reasoning(
    generic_rule_results,
    career_rule_analysis,
    marriage_rule_analysis,
    finance_rule_analysis,
    education_rule_analysis,
    property_rule_analysis,
    stage_8_20_dasha_timing,
    stage_8_21_transit_timing,
):
    """
    Connect the validated Stage 8 reasoning components.

    This function does not calculate astrology itself.
    It only connects outputs from the previously validated
    reasoning stages.
    """

    # --------------------------------------------------------
    # Stage 8.22 — Multi-Domain Integration
    # --------------------------------------------------------

    stage_8_22_multi_domain = run_multi_domain_engine(
        generic_rule_results=generic_rule_results,

        career_rule_analysis=career_rule_analysis,

        marriage_rule_analysis=marriage_rule_analysis,

        finance_rule_analysis=finance_rule_analysis,

        education_rule_analysis=education_rule_analysis,

        property_rule_analysis=property_rule_analysis,

        stage_8_20_dasha_timing=stage_8_20_dasha_timing,

        stage_8_21_transit_timing=stage_8_21_transit_timing,
    )

    # --------------------------------------------------------
    # Final Stage 8 object
    # --------------------------------------------------------

    return {
        "stage_8_14_generic_rules":
            generic_rule_results,

        "stage_8_15_career_rules":
            career_rule_analysis,

        "stage_8_16_marriage_rules":
            marriage_rule_analysis,

        "stage_8_17_finance_rules":
            finance_rule_analysis,

        "stage_8_18_education_rules":
            education_rule_analysis,

        "stage_8_19_property_rules":
            property_rule_analysis,

        "stage_8_20_dasha_timing":
            stage_8_20_dasha_timing,

        "stage_8_21_transit_timing":
            stage_8_21_transit_timing,

        "stage_8_22_multi_domain":
            stage_8_22_multi_domain,
    }

def validate_stage8_pipeline(stage8_result):
    """
    Validate that all Stage 8 components are present.
    """

    expected_keys = {
        "stage_8_14_generic_rules",
        "stage_8_15_career_rules",
        "stage_8_16_marriage_rules",
        "stage_8_17_finance_rules",
        "stage_8_18_education_rules",
        "stage_8_19_property_rules",
        "stage_8_20_dasha_timing",
        "stage_8_21_transit_timing",
        "stage_8_22_multi_domain",
    }

    assert set(stage8_result.keys()) == expected_keys

    # --------------------------------------------------------
    # Validate Stage 8.22
    # --------------------------------------------------------

    multi_domain = stage8_result[
        "stage_8_22_multi_domain"
    ]

    assert "domain_relevance" in multi_domain
    assert "relevant_domains" in multi_domain
    assert "inactive_domains" in multi_domain
    assert "cross_domain_combinations" in multi_domain
    assert "timing_summary" in multi_domain
    assert "multi_domain_evidence" in multi_domain
    assert "context" in multi_domain

    # --------------------------------------------------------
    # Validate expected domains
    # --------------------------------------------------------

    expected_domains = {
        "career",
        "marriage",
        "finance",
        "education",
        "property",
    }

    assert set(
        multi_domain["domain_relevance"].keys()
    ) == expected_domains

    assert set(
        multi_domain["timing_summary"].keys()
    ) == expected_domains

    # --------------------------------------------------------
    # Validate domain separation
    # --------------------------------------------------------

    relevant = set(
        multi_domain["relevant_domains"]
    )

    inactive = set(
        multi_domain["inactive_domains"]
    )

    assert relevant.isdisjoint(inactive)

    assert relevant | inactive == expected_domains

    return True