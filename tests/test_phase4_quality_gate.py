"""
tests/test_phase4_quality_gate.py
=================================
Comprehensive Test Suite for Phase 4: Quality Gate & Gemini Fallback Policy

Enforces:
1. Quality Gate evaluates required entities, domain coverage, evidence completeness,
   and question responsiveness.
2. Strips out or flags unsupported deterministic claims and guarantees.
3. Clean separation of the two failure modes:
   - Evidence incomplete -> SERVE_UNRESOLVED (0 LLM calls, zero fabricated claims).
   - Evidence complete but wording inadequate / creative request -> FALLBACK_GEMINI.
4. Seamless integration with production decision traces.
"""

import pytest
from backend.reasoning.quality_gate import (
    evaluate_answer_quality,
    QualityGateResult
)


def test_quality_gate_passes_complete_local_answer():
    q = "How does my 10th lord influence my career?"
    ans = (
        "Your 10th lord Mercury placed in Gemini directly anchors your career in methodical discipline, "
        "organizational integrity, and long-term authority.\n\n"
        "**Astrological Mechanics (Virgo Ascendant, Taurus Moon sign)**:\n"
        "In Vedic astrology, the 10th house governs Karma Bhava.\n\n"
        "**Core Takeaway**: Channeling this placement into analytical planning maximizes fulfillment."
    )
    res = evaluate_answer_quality(
        question=q,
        domain="career",
        intent="career_10th_lord",
        answer_text=ans,
        matched_rules=[{"rule_id": "CAREER_10TH_LORD_PLACEMENT", "matched": True}]
    )

    assert res.is_pass is True
    assert res.has_required_entities is True
    assert res.has_domain_coverage is True
    assert res.is_evidence_complete is True
    assert res.addresses_question is True
    assert res.has_no_unsupported_claims is True
    assert res.recommended_action == "SERVE_LOCAL"


def test_quality_gate_fails_on_unsupported_guarantee():
    q = "Will I get promoted?"
    ans = "A career promotion is guaranteed and you will definitely achieve it with 100% certainty."
    res = evaluate_answer_quality(
        question=q,
        domain="career",
        intent="career_promotion",
        answer_text=ans,
        matched_rules=[{"rule_id": "CAREER_10TH_LORD_PLACEMENT", "matched": True}]
    )

    assert res.has_no_unsupported_claims is False
    assert res.is_pass is False
    assert "guarantee" in (res.failure_reason or "").lower()


def test_quality_gate_detects_missing_domain_coverage_in_multidomain():
    q = "Should I switch jobs and invest my savings?"
    # Answer covers career, but omits finance completely
    ans = (
        "Your birth chart indicates steady career growth under active 10th house indicators. "
        "Analytical problem-solving and structured responsibilities favor professional progress."
    )
    res = evaluate_answer_quality(
        question=q,
        domain="career",
        intent="multi_domain",
        answer_text=ans,
        detected_domains=["career", "finance"]
    )

    assert res.has_domain_coverage is False
    assert "finance" in (res.failure_reason or "").lower()


def test_quality_gate_passes_full_multidomain_coverage():
    q = "Should I switch jobs and invest my savings?"
    ans = (
        "Your birth chart (Virgo Ascendant, Taurus Moon sign) supports strategic professional development "
        "alongside disciplined, long-term financial consolidation.\n\n"
        "**Multi-Domain Astrological Analysis**:\n"
        "- **Career Indicators**: Your 10th house indicators emphasize analytical precision.\n"
        "- **Financial Indicators**: Your 2nd and 11th house factors support steady accumulation.\n\n"
        "**Timing Dynamics (Mercury/Venus)**:\n"
        "Current period emphasizes systematic pacing.\n\n"
        "**Strategic Recommendation**: Balance both domains with measured pacing."
    )
    res = evaluate_answer_quality(
        question=q,
        domain="career",
        intent="multi_domain",
        answer_text=ans,
        detected_domains=["career", "finance"]
    )

    assert res.is_pass is True
    assert res.has_domain_coverage is True
    assert res.recommended_action == "SERVE_LOCAL"


def test_quality_gate_detects_unaddressed_comparison_question():
    q = "Job vs business, which one is better for me?"
    # Answer talks about personality, but doesn't address the comparison between job and business
    ans = "Your natal chart shows high creativity, artistic flair, and deep empathy in daily life."
    res = evaluate_answer_quality(
        question=q,
        domain="career",
        intent="career_guidance",
        answer_text=ans
    )

    assert res.addresses_question is False


def test_failure_mode_evidence_incomplete():
    # When evidence is incomplete, recommended action must be SERVE_UNRESOLVED (0 Gemini calls, no hallucinations)
    q = "When will I get married?"
    ans = "Details cannot be determined from incomplete backend facts."
    evidence_incomplete = {"is_sufficient": False, "missing_evidence": ["7th_house"]}

    res = evaluate_answer_quality(
        question=q,
        domain="marriage",
        intent="marriage_timing",
        answer_text=ans,
        evidence=evidence_incomplete
    )

    assert res.is_evidence_complete is False
    assert res.recommended_action == "SERVE_UNRESOLVED"


def test_failure_mode_evidence_complete_creative_request_fallback():
    # When user explicitly asks for a poem/creative story, evidence-grounded Gemini is appropriate
    q = "Write a poetic story about my career trajectory based on my chart."
    ans = "Your career involves analytical work and methodical discipline."
    res = evaluate_answer_quality(
        question=q,
        domain="career",
        intent="career_general",
        answer_text=ans,
        matched_rules=[{"rule_id": "CAREER_10TH_LORD_PLACEMENT", "matched": True}],
        is_creative_request=True
    )

    assert res.recommended_action == "FALLBACK_GEMINI"


def test_quality_gate_result_to_dict():
    res = QualityGateResult(
        is_pass=True,
        score=95,
        has_required_entities=True,
        has_domain_coverage=True,
        is_evidence_complete=True,
        addresses_question=True,
        has_no_unsupported_claims=True,
        failure_reason=None,
        recommended_action="SERVE_LOCAL"
    )
    d = res.to_dict()
    assert d["is_pass"] is True
    assert d["quality_score"] == 95
    assert d["recommended_action"] == "SERVE_LOCAL"
