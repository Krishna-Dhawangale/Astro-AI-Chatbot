"""
tests/test_phase3_multidomain_synthesis.py
==========================================
Comprehensive Test Suite for Phase 3: Multi-Domain Evidence Gathering & Unified Synthesis

Enforces:
1. Support for combinations of core domains (Career, Finance, Marriage, Education, Property, Health, Dasha).
2. Direct reuse of Stage 8 evidence without re-running planetary or astrological calculations.
3. Non-causal independence between life domains (no fabricated causal links between unrelated domains).
4. Direct opening addressing all requested domains, followed by structured domain evidence and strategic conclusion.
5. Zero fabricated calendar dates or ungrounded predictions.
6. Guaranteed local execution (gemini_calls == 0).
7. Hinglish fluency for multi-domain queries.
"""

import pytest
from backend.reasoning.decomposer import decompose_and_synthesize_multidomain


def _make_dummy_stage8(lord_10="Mercury", lord_7="Jupiter"):
    return {
        "stage_8_14_generic_rules": {"rules": []},
        "stage_8_15_career_rules": {
            "rules": [
                {
                    "rule_id": "CAREER_10TH_LORD_PLACEMENT",
                    "matched": True,
                    "evidence": {"lord": lord_10, "sign": "Gemini"}
                }
            ]
        },
        "stage_8_16_marriage_rules": {
            "rules": [
                {
                    "rule_id": "MARRIAGE_7TH_LORD_PLACEMENT",
                    "matched": True,
                    "evidence": {"lord": lord_7, "sign": "Pisces"}
                }
            ]
        },
        "stage_8_17_finance_rules": {
            "rules": [
                {
                    "rule_id": "FINANCE_DHANA_YOGA",
                    "matched": True,
                    "evidence": {"strength": "supportive"}
                }
            ]
        },
        "stage_8_18_education_rules": {
            "rules": [
                {
                    "rule_id": "EDUCATION_5TH_LORD_PLACEMENT",
                    "matched": True,
                    "evidence": {"intellect": "analytical"}
                }
            ]
        },
        "stage_8_19_property_rules": {
            "rules": [
                {
                    "rule_id": "PROPERTY_4TH_HOUSE_SECURITY",
                    "matched": True,
                    "evidence": {"asset_type": "fixed"}
                }
            ]
        },
        "stage_8_20_dasha_timing": {
            "current_mahadasha": "Mercury",
            "current_antardasha": "Venus"
        },
        "stage_8_21_transit_timing": {},
        "stage_8_22_multi_domain": {
            "relevant_domains": ["career", "finance"]
        }
    }


# ---------------------------------------------------------------------------
# 1. Multi-Domain Combinations & Direct Opening Sentence
# ---------------------------------------------------------------------------
def test_career_plus_finance_combination():
    q = "Should I switch jobs and invest my savings?"
    stage8 = _make_dummy_stage8(lord_10="Mercury")
    res = decompose_and_synthesize_multidomain(
        query=q,
        stage8_result=stage8,
        asc_sign="Virgo",
        moon_sign="Taurus",
        dasha_hierarchy={"current_mahadasha": "Mercury", "current_antardasha": "Venus"}
    )

    ans = res["answer"]
    first_sentence = ans.split("\n")[0]

    # Direct opening addressing both domains
    assert "professional development" in first_sentence or "career" in first_sentence.lower()
    assert "financial" in first_sentence.lower()
    # Distinct Stage 8 evidence for Career and Finance
    assert "**Career Indicators**:" in ans
    assert "**Financial Outlook**:" in ans or "**Financial Indicators**:" in ans
    assert "Mercury" in ans  # Uses lord from stage8
    # Dasha integration
    assert "Mercury/Venus" in ans
    # Strategic conclusion
    assert "**Strategic Recommendation**:" in ans
    # Guaranteed 0 gemini calls
    assert res["gemini_calls"] == 0
    assert set(res["detected_domains"]) == {"career", "finance"}


def test_career_plus_marriage_combination():
    q = "How will my job change and marriage be affected?"
    stage8 = _make_dummy_stage8(lord_10="Mercury", lord_7="Jupiter")
    res = decompose_and_synthesize_multidomain(
        query=q,
        stage8_result=stage8,
        asc_sign="Virgo",
        moon_sign="Taurus",
        dasha_hierarchy={"current_mahadasha": "Jupiter", "current_antardasha": "Moon"}
    )

    ans = res["answer"]
    first_sentence = ans.split("\n")[0]

    assert "professional" in first_sentence.lower() or "career" in first_sentence.lower()
    assert "partnerships" in first_sentence.lower() or "marriage" in first_sentence.lower()
    assert "**Career Indicators**:" in ans
    assert "**Relationship Indicators**:" in ans
    assert "Jupiter/Moon" in ans
    # Verify no fabricated dates
    assert "2027–2028" not in ans
    assert res["gemini_calls"] == 0


def test_finance_plus_property_combination():
    q = "Can I invest in commercial property or real estate?"
    stage8 = _make_dummy_stage8()
    res = decompose_and_synthesize_multidomain(
        query=q,
        stage8_result=stage8,
        asc_sign="Virgo",
        moon_sign="Taurus",
        dasha_hierarchy={"current_mahadasha": "Mercury"}
    )

    ans = res["answer"]
    first_sentence = ans.split("\n")[0]

    assert "wealth accumulation" in first_sentence.lower() or "financial" in first_sentence.lower()
    assert "property" in first_sentence.lower()
    assert "**Financial Indicators**:" in ans or "**Financial Outlook**:" in ans
    assert "**Property & Assets**:" in ans
    assert res["gemini_calls"] == 0


def test_education_plus_career_combination():
    q = "Will my college degree help my career growth?"
    stage8 = _make_dummy_stage8()
    res = decompose_and_synthesize_multidomain(
        query=q,
        stage8_result=stage8,
        asc_sign="Virgo",
        moon_sign="Taurus",
        dasha_hierarchy={"current_mahadasha": "Sun"}
    )

    ans = res["answer"]
    first_sentence = ans.split("\n")[0]

    assert "academic" in first_sentence.lower() or "education" in first_sentence.lower()
    assert "career" in first_sentence.lower() or "professional" in first_sentence.lower()
    assert "**Education & Learning**:" in ans
    assert "**Career Indicators**:" in ans
    assert res["gemini_calls"] == 0


# ---------------------------------------------------------------------------
# 2. Stage 8 Direct Evidence Reuse (No Recalculation)
# ---------------------------------------------------------------------------
def test_stage8_evidence_custom_lord_reuse():
    # Test that lord passed in Stage 8 is dynamically used without recalculation
    stage8_sun_lord = _make_dummy_stage8(lord_10="Sun", lord_7="Saturn")
    res = decompose_and_synthesize_multidomain(
        query="Tell me about my career and marriage",
        stage8_result=stage8_sun_lord,
        asc_sign="Scorpio",
        moon_sign="Leo",
    )

    ans = res["answer"]
    assert "Sun" in ans
    assert "Saturn" in ans
    assert res["gemini_calls"] == 0


# ---------------------------------------------------------------------------
# 3. Non-Causal Independence
# ---------------------------------------------------------------------------
def test_non_causal_independence_career_marriage():
    # Does not claim that getting promoted will automatically find a spouse
    q = "Will switching jobs get me married?"
    stage8 = _make_dummy_stage8()
    res = decompose_and_synthesize_multidomain(
        query=q,
        stage8_result=stage8,
        asc_sign="Virgo",
        moon_sign="Taurus",
    )

    ans = res["answer"].lower()
    # Both domains are addressed with distinct astrological indicators
    assert "career indicators" in ans
    assert "relationship indicators" in ans
    # Non-causal wording
    assert "independently" in ans or "advancing your primary goals while" in ans or "balancing both" in ans


# ---------------------------------------------------------------------------
# 4. Hinglish Multi-Domain Support
# ---------------------------------------------------------------------------
def test_hinglish_career_and_finance():
    q = "kya mujhe job switch karke paise invest karne chahiye?"
    stage8 = _make_dummy_stage8()
    res = decompose_and_synthesize_multidomain(
        query=q,
        stage8_result=stage8,
        asc_sign="Virgo",
        moon_sign="Taurus",
        dasha_hierarchy={"current_mahadasha": "Mercury", "current_antardasha": "Venus"},
        is_hinglish=True
    )

    ans = res["answer"]
    first_sentence = ans.split("\n")[0]

    assert "career" in first_sentence.lower()
    assert "financial" in first_sentence.lower() or "paise" in first_sentence.lower()
    assert "**Timing & Dasha Alignment (Mercury/Venus)**:" in ans
    assert "**Strategic Conclusion**:" in ans
    assert res["gemini_calls"] == 0


def test_hinglish_career_and_marriage():
    q = "naukri aur shaadi dono ka kya outlook hai?"
    stage8 = _make_dummy_stage8()
    res = decompose_and_synthesize_multidomain(
        query=q,
        stage8_result=stage8,
        asc_sign="Virgo",
        moon_sign="Taurus",
        dasha_hierarchy={"current_mahadasha": "Jupiter"},
        is_hinglish=True
    )

    ans = res["answer"]
    first_sentence = ans.split("\n")[0]

    assert "career" in first_sentence.lower()
    assert "marriage" in first_sentence.lower() or "relationship" in first_sentence.lower()
    assert "**Relationship & Marriage**:" in ans
    assert res["gemini_calls"] == 0
