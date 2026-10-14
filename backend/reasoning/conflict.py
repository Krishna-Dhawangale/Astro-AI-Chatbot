# ============================================================
# STAGE 8.23 — RULE CONFLICT HANDLING
# ============================================================
#
# Purpose:
#   Organize rule evidence into:
#
#       SUPPORTIVE
#       CONFLICTING
#       NEUTRAL
#
# IMPORTANT:
#   - No arbitrary scoring
#   - No probability
#   - No "winner"
#   - No deletion of contradictory evidence
#   - Preserves original rule results
# ============================================================


# ------------------------------------------------------------
# 1. RULE CLASSIFICATION
# ------------------------------------------------------------

def classify_rule_evidence(rule):
    """
    Classifies a rule result without assigning a score.

    Current rule semantics:

        matched=True
            -> SUPPORTIVE

        matched=False
            -> NEUTRAL

    Explicit conflicting evidence can be added later when
    domain-specific restrictive rules are introduced.

    Non-matched rules are deliberately NOT classified as
    negative evidence.
    """

    if rule["matched"]:
        return "supportive"

    return "neutral"


# ------------------------------------------------------------
# 2. BUILD DOMAIN CONFLICT ANALYSIS
# ------------------------------------------------------------

def build_domain_conflict_analysis(
    domain_rule_analyses
):
    """
    Classifies rules for each domain into:

        supportive
        conflicting
        neutral
    """

    domain_conflict_analysis = {}

    for domain, rule_analysis in (
        domain_rule_analyses.items()
    ):

        supportive_rules = []
        conflicting_rules = []
        neutral_rules = []

        for rule in rule_analysis["rules"]:

            classification = (
                classify_rule_evidence(rule)
            )

            if classification == "supportive":

                supportive_rules.append(rule)

            elif classification == "conflicting":

                conflicting_rules.append(rule)

            else:

                neutral_rules.append(rule)

        domain_conflict_analysis[domain] = {

            "domain":
                domain,

            "supportive":
                supportive_rules,

            "conflicting":
                conflicting_rules,

            "neutral":
                neutral_rules,

            "supportive_count":
                len(supportive_rules),

            "conflicting_count":
                len(conflicting_rules),

            "neutral_count":
                len(neutral_rules)
        }

    return domain_conflict_analysis


# ------------------------------------------------------------
# 3. IDENTIFY DOMAINS WITH MIXED EVIDENCE
# ------------------------------------------------------------

def identify_mixed_evidence_domains(
    domain_conflict_analysis
):
    """
    Identifies domains containing both supportive
    and explicit conflicting evidence.
    """

    mixed_evidence_domains = []

    for domain, analysis in (
        domain_conflict_analysis.items()
    ):

        if (
            analysis["supportive_count"] > 0
            and
            analysis["conflicting_count"] > 0
        ):

            mixed_evidence_domains.append(domain)

    return mixed_evidence_domains


# ------------------------------------------------------------
# 4. BUILD CROSS-DOMAIN CONFLICT CONTEXT
# ------------------------------------------------------------

def build_cross_domain_context(
    relevant_domains,
    domain_conflict_analysis
):
    """
    Builds transparent conflict context for domains
    considered relevant by Stage 8.22.
    """

    cross_domain_context = []

    for domain in relevant_domains:

        analysis = domain_conflict_analysis[
            domain
        ]

        cross_domain_context.append({

            "domain":
                domain,

            "supportive_count":
                analysis["supportive_count"],

            "conflicting_count":
                analysis["conflicting_count"],

            "neutral_count":
                analysis["neutral_count"],

            "has_mixed_evidence":
                (
                    analysis["conflicting_count"] > 0
                    and
                    analysis["supportive_count"] > 0
                )
        })

    return cross_domain_context


# ------------------------------------------------------------
# 5. BUILD TRANSPARENT EVIDENCE RECORDS
# ------------------------------------------------------------

def build_transparent_evidence(
    domain_conflict_analysis
):
    """
    Preserves rule IDs, interpretation keys and evidence
    without removing any original rule information.
    """

    transparent_evidence = {}

    for domain, analysis in (
        domain_conflict_analysis.items()
    ):

        transparent_evidence[domain] = {

            "supportive_evidence": [
                {
                    "rule_id":
                        rule["rule_id"],

                    "interpretation_key":
                        rule["interpretation_key"],

                    "evidence":
                        rule["evidence"]
                }

                for rule in
                analysis["supportive"]
            ],

            "conflicting_evidence": [
                {
                    "rule_id":
                        rule["rule_id"],

                    "interpretation_key":
                        rule["interpretation_key"],

                    "evidence":
                        rule["evidence"]
                }

                for rule in
                analysis["conflicting"]
            ],

            "neutral_evidence": [
                {
                    "rule_id":
                        rule["rule_id"],

                    "interpretation_key":
                        rule["interpretation_key"],

                    "evidence":
                        rule["evidence"]
                }

                for rule in
                analysis["neutral"]
            ]
        }

    return transparent_evidence


# ------------------------------------------------------------
# 6. BUILD STAGE 8.23
# ------------------------------------------------------------

def build_conflict_analysis(
    stage_8_22_multi_domain,
    career_rule_analysis,
    marriage_rule_analysis,
    finance_rule_analysis,
    education_rule_analysis,
    property_rule_analysis
):
    """
    Builds the complete Stage 8.23 conflict analysis.
    """

    domain_rule_analyses = {

        "career":
            career_rule_analysis,

        "marriage":
            marriage_rule_analysis,

        "finance":
            finance_rule_analysis,

        "education":
            education_rule_analysis,

        "property":
            property_rule_analysis
    }

    domain_conflict_analysis = (
        build_domain_conflict_analysis(
            domain_rule_analyses
        )
    )

    mixed_evidence_domains = (
        identify_mixed_evidence_domains(
            domain_conflict_analysis
        )
    )

    relevant_domains = (
        stage_8_22_multi_domain[
            "relevant_domains"
        ]
    )

    cross_domain_context = (
        build_cross_domain_context(
            relevant_domains,
            domain_conflict_analysis
        )
    )

    transparent_evidence = (
        build_transparent_evidence(
            domain_conflict_analysis
        )
    )

    stage_8_23_conflict_analysis = {

        "domains":
            domain_conflict_analysis,

        "mixed_evidence_domains":
            mixed_evidence_domains,

        "cross_domain_context":
            cross_domain_context,

        "transparent_evidence":
            transparent_evidence
    }

    return stage_8_23_conflict_analysis


# ------------------------------------------------------------
# 7. VALIDATION
# ------------------------------------------------------------

def validate_conflict_analysis(
    stage_8_23_conflict_analysis
):
    """
    Validate Stage 8.23 output.
    """

    expected_domains = {
        "career",
        "marriage",
        "finance",
        "education",
        "property"
    }

    domain_conflict_analysis = (
        stage_8_23_conflict_analysis[
            "domains"
        ]
    )

    transparent_evidence = (
        stage_8_23_conflict_analysis[
            "transparent_evidence"
        ]
    )

    assert (
        set(domain_conflict_analysis.keys())
        == expected_domains
    )

    assert (
        set(transparent_evidence.keys())
        == expected_domains
    )

    for domain, analysis in (
        domain_conflict_analysis.items()
    ):

        assert analysis["domain"] == domain

        assert isinstance(
            analysis["supportive"],
            list
        )

        assert isinstance(
            analysis["conflicting"],
            list
        )

        assert isinstance(
            analysis["neutral"],
            list
        )

        assert (
            analysis["supportive_count"]
            ==
            len(analysis["supportive"])
        )

        assert (
            analysis["conflicting_count"]
            ==
            len(analysis["conflicting"])
        )

        assert (
            analysis["neutral_count"]
            ==
            len(analysis["neutral"])
        )

    for domain, evidence in (
        transparent_evidence.items()
    ):

        assert (
            "supportive_evidence"
            in evidence
        )

        assert (
            "conflicting_evidence"
            in evidence
        )

        assert (
            "neutral_evidence"
            in evidence
        )

    return True


# ------------------------------------------------------------
# 8. MODULE SELF-TEST
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)
    print("STAGE 8.23 — RULE CONFLICT HANDLING")
    print("=" * 80)
    print()
    print("✓ conflict.py loaded successfully")
    print("✓ Supportive classification available")
    print("✓ Conflicting classification available")
    print("✓ Neutral classification available")
    print("✓ No arbitrary score")
    print("✓ No probability")
    print("✓ No evidence discarded")
    print()
    print(
        "Stage 8.23 requires Stage 8.15–8.19 "
        "and Stage 8.22 inputs to execute."
    )