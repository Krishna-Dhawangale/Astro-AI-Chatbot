"""
tests/test_dasha_evidence_integration.py
=========================================
Regression Test Suite for Dasha Evidence Parameter Flow and Integration.

Covers:
1. Populated Dasha data (direct dict, norm_dasha hierarchy, and nested output).
2. Fallback to Stage 8 dasha_timing when dasha_hierarchy is empty or omitted.
3. Missing Dasha data: strictly preserves anti-hallucination timing safeguard.
4. Malformed dasha_hierarchy inputs (None, strings, lists, arbitrary invalid objects).
5. Two-year career question with populated Dasha and incomplete transit evidence.
6. Two-year career question with genuinely missing Dasha evidence.
7. Hinglish paraphrase handling with populated Dasha evidence.
"""

import pytest
from backend.normalize import normalize_dasha_response
from backend.reasoning.structured_evidence import build_structured_evidence
from backend.reasoning.answer_synthesizer import synthesize_local_answer
from backend.main import format_local_interpretation_answer


@pytest.fixture
def base_chart():
    return {
        "ascendant": {"rashi": "Virgo", "longitude": 165.0},
        "planets": {
            "Moon": {"rashi": "Taurus", "longitude": 45.0, "nakshatra": "Rohini"},
            "Sun": {"rashi": "Gemini", "longitude": 75.0},
            "Mercury": {"rashi": "Gemini", "house": 10},
            "Jupiter": {"rashi": "Pisces", "house": 7},
            "Venus": {"rashi": "Taurus", "house": 9},
            "Saturn": {"rashi": "Aquarius", "house": 6},
        }
    }


@pytest.fixture
def career_stage8_with_incomplete_transit():
    return {
        "stage_8_14_generic_rules": {"rules": []},
        "stage_8_15_career_rules": {
            "rules": [
                {
                    "rule_id": "CAREER_10TH_LORD_PLACEMENT",
                    "matched": True,
                    "evidence": {
                        "house": 10,
                        "house_rashi": "Gemini",
                        "lord": "Mercury",
                        "lord_natal_house": 10,
                        "lord_natal_rashi": "Gemini"
                    }
                }
            ]
        },
        "stage_8_16_marriage_rules": {"rules": []},
        "stage_8_17_finance_rules": {"rules": []},
        "stage_8_20_dasha_timing": {
            "current_dasha_windows": [
                {"dasha_level": "mahadasha", "planet": "Mercury", "start": "2020-01-01", "end": "2037-01-01"},
                {"dasha_level": "antardasha", "planet": "Venus", "start": "2024-05-01", "end": "2027-03-01"}
            ]
        },
        # Transit evidence is explicitly incomplete / empty
        "stage_8_21_transit_timing": {}
    }


# ---------------------------------------------------------------------------
# 1. Populated Dasha Data Verification
# ---------------------------------------------------------------------------

def test_populated_dasha_data_direct(base_chart, career_stage8_with_incomplete_transit):
    """Direct dictionary with current_mahadasha and current_antardasha."""
    dasha_hier = {"current_mahadasha": "Mercury", "current_antardasha": "Venus"}
    ev = build_structured_evidence(
        domain="career",
        intent="career_timeline",
        norm_chart=base_chart,
        stage8_result=career_stage8_with_incomplete_transit,
        dasha_hierarchy=dasha_hier
    )

    assert ev.timing["active_mahadasha"] == "Mercury"
    assert ev.timing["active_antardasha"] == "Venus"
    assert ev.facts["current_mahadasha"] == "Mercury"
    assert ev.facts["current_antardasha"] == "Venus"


def test_populated_dasha_via_norm_dasha_hierarchy():
    """normalize_dasha_response propagates active Dasha into dasha_hierarchy."""
    raw_payload = {
        "current_mahadasha": "Saturn",
        "current_antardasha": "Mercury",
        "output": {
            "current_mahadasha": "Saturn",
            "current_antardasha": "Mercury",
            "dasha_hierarchy": {
                "Saturn": {
                    "Mercury": {
                        "start_time": "2024-01-01 00:00:00",
                        "end_time": "2029-01-01 00:00:00"
                    }
                }
            }
        }
    }
    norm = normalize_dasha_response(raw_payload)
    assert norm["current_mahadasha"] == "Saturn"
    assert norm["current_antardasha"] == "Mercury"
    
    dh = norm["dasha_hierarchy"]
    assert isinstance(dh, dict)
    assert dh["current_mahadasha"] == "Saturn"
    assert dh["current_antardasha"] == "Mercury"


# ---------------------------------------------------------------------------
# 2. Fallback to Stage 8 Dasha Timing Evidence
# ---------------------------------------------------------------------------

def test_populated_dasha_fallback_to_stage8(base_chart, career_stage8_with_incomplete_transit):
    """When dasha_hierarchy is empty, structured_evidence cleanly falls back to Stage 8."""
    ev = build_structured_evidence(
        domain="career",
        intent="career_timeline",
        norm_chart=base_chart,
        stage8_result=career_stage8_with_incomplete_transit,
        dasha_hierarchy={}
    )

    assert ev.timing["active_mahadasha"] == "Mercury"
    assert ev.timing["active_antardasha"] == "Venus"


# ---------------------------------------------------------------------------
# 3. Missing Dasha Data: Strictly Preserves Anti-Hallucination Safeguard
# ---------------------------------------------------------------------------

def test_missing_dasha_data_preserves_safeguard(base_chart):
    """When Dasha data is genuinely missing from all sources, timing questions invoke safeguard."""
    empty_stage8 = {
        "stage_8_15_career_rules": {
            "rules": [
                {
                    "rule_id": "CAREER_10TH_LORD_PLACEMENT",
                    "matched": True,
                    "evidence": {"house": 10, "house_rashi": "Gemini", "lord": "Mercury"}
                }
            ]
        },
        "stage_8_20_dasha_timing": {}
    }
    ev = build_structured_evidence(
        domain="career",
        intent="career_timeline",
        norm_chart=base_chart,
        stage8_result=empty_stage8,
        dasha_hierarchy={}
    )

    ans = synthesize_local_answer(ev, is_hinglish=False, question="what will happen with my career in next 2 years")

    # Must preserve the exact safeguard statement
    assert "While your birth chart indicates general foundational tendencies for your career" in ans
    assert "specific timing windows cannot be confirmed without verified active Dasha and transit data" in ans
    assert "The system avoids generating ungrounded calendar dates when timing evidence is uninitialized" in ans
    assert "**Foundational Alignment**:" in ans


# ---------------------------------------------------------------------------
# 4. Malformed Dasha Hierarchy Handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("malformed_input", [
    None,
    "corrupt_string_value",
    [{"some": "list"}],
    12345,
    {"unexpected_key": [1, 2, 3]},
])
def test_malformed_dasha_hierarchy_safe(base_chart, malformed_input):
    """Malformed hierarchy types must not cause AttributeError or TypeError."""
    empty_stage8 = {
        "stage_8_15_career_rules": {
            "rules": [
                {
                    "rule_id": "CAREER_10TH_LORD_PLACEMENT",
                    "matched": True,
                    "evidence": {"house": 10, "house_rashi": "Gemini", "lord": "Mercury"}
                }
            ]
        }
    }
    ev = build_structured_evidence(
        domain="career",
        intent="career_timeline",
        norm_chart=base_chart,
        stage8_result=empty_stage8,
        dasha_hierarchy=malformed_input
    )

    ans = synthesize_local_answer(ev, is_hinglish=False, question="what will happen with my career in next 2 years")
    assert "While your birth chart indicates general foundational tendencies for your career" in ans


# ---------------------------------------------------------------------------
# 5. Two-Year Career Question with Incomplete Transit Evidence
# ---------------------------------------------------------------------------

def test_two_year_career_question_with_incomplete_transit_evidence(base_chart, career_stage8_with_incomplete_transit):
    """
    Test the exact user query: 'what will happen with my career in next 2 years'
    when transit evidence is incomplete, but Dasha evidence is verified.
    """
    dasha_hier = {"current_mahadasha": "Mercury", "current_antardasha": "Venus"}
    q = "what will happen with my career in next 2 years"

    ans = format_local_interpretation_answer(
        domain="career",
        intent="career_timeline",
        evidence={},
        asc_sign="Virgo",
        sun_sign="Gemini",
        moon_sign="Taurus",
        is_hinglish=False,
        question=q,
        norm_chart=base_chart,
        stage8_result=career_stage8_with_incomplete_transit,
        dasha_hierarchy=dasha_hier
    )

    # 1. Direct opening addressing stated 2-year timeline
    assert "2 years" in ans or "query for the 2 years" in ans
    # 2. Active Dasha integrated
    assert "Mercury/Venus" in ans or "Mercury" in ans
    # 3. Grounded Astrological Evidence
    assert "**Astrological Evidence (Virgo Ascendant, Taurus Moon sign)**:" in ans
    assert "10th-house" in ans or "analytical" in ans
    # 4. Actionable Conclusion
    assert "**Actionable Conclusion**:" in ans
    # 5. Never invents unsupported calendar dates
    assert "2039" not in ans and "2045" not in ans


def test_two_year_career_question_with_missing_dasha(base_chart):
    """Exact user query with missing Dasha preserves the timing safeguard."""
    empty_stage8 = {
        "stage_8_15_career_rules": {
            "rules": [
                {
                    "rule_id": "CAREER_10TH_LORD_PLACEMENT",
                    "matched": True,
                    "evidence": {"house": 10, "house_rashi": "Gemini", "lord": "Mercury"}
                }
            ]
        }
    }
    q = "what will happen with my career in next 2 years"

    ans = format_local_interpretation_answer(
        domain="career",
        intent="career_timeline",
        evidence={},
        asc_sign="Virgo",
        sun_sign="Gemini",
        moon_sign="Taurus",
        is_hinglish=False,
        question=q,
        norm_chart=base_chart,
        stage8_result=empty_stage8,
        dasha_hierarchy={}
    )

    assert "specific timing windows cannot be confirmed without verified active Dasha and transit data" in ans
    assert "**Foundational Alignment**:" in ans


def test_hinglish_two_year_career_question(base_chart, career_stage8_with_incomplete_transit):
    """Hinglish variant with verified Dasha synthesizes naturally."""
    dasha_hier = {"current_mahadasha": "Mercury", "current_antardasha": "Venus"}
    q = "agle 2 saal me career kaisa rahega"

    ans = format_local_interpretation_answer(
        domain="career",
        intent="career_timeline",
        evidence={},
        asc_sign="Virgo",
        sun_sign="Gemini",
        moon_sign="Taurus",
        is_hinglish=True,
        question=q,
        norm_chart=base_chart,
        stage8_result=career_stage8_with_incomplete_transit,
        dasha_hierarchy=dasha_hier
    )

    assert "Aapke stated timeline" in ans or "active Mercury/Venus Dasha" in ans
    assert "**Astrological Context (Virgo Lagna, Taurus Moon)**:" in ans
    assert "**Conclusion**:" in ans
