"""
tests/test_phase2_local_synthesis.py
====================================
Comprehensive Test Suite for Phase 2: Local Answer Fluency & Synthesis

Enforces:
1. Question-aware answer shapes for timing, feasibility/advisory, comparison,
   growth outlook, and direct placement questions.
2. Direct answer in the opening sentence, followed by verified chart/Dasha evidence
   and a concise conclusion.
3. Evidence-aware discourse connectors and repetition reduction without deleting
   distinct or conflicting evidence.
4. Clear handling of missing evidence: zero invented dates, planetary facts,
   or actionable claims.
5. Unsupported claim guarding (deterministic words like 'guaranteed', '100% certainty').
6. Hinglish fluency and preserve existing fallback behaviors.
"""

import pytest
from backend.reasoning.structured_evidence import StructuredEvidence
from backend.reasoning.answer_synthesizer import (
    synthesize_local_answer,
    classify_question_type,
    guard_deterministic_claims,
    _connect_evidence_phrases
)


def _make_dummy_evidence(
    domain="career",
    intent="career_general",
    asc="Virgo",
    moon="Taurus",
    maha="Mercury",
    antar="Venus",
    positive_factors=None,
    challenging_factors=None,
    timing_window=None,
    completeness=1.0
):
    positives = positive_factors if positive_factors is not None else ["analytical precision in technical execution"]
    challenges = challenging_factors if challenging_factors is not None else []
    
    return StructuredEvidence(
        domain=domain,
        intent=intent,
        facts={
            "ascendant": asc,
            "moon_rashi": moon,
            "sun_sign": "Gemini",
            "tenth_house_sign": "Gemini",
            "tenth_lord": "Mercury",
            "seventh_house_sign": "Pisces",
            "seventh_lord": "Jupiter",
        },
        evaluations={
            "overall_status": "supportive",
            "strength_score": 0.85,
            "dasha_active": bool(maha),
        },
        interpretation={
            "positive_factors": positives,
            "challenging_factors": challenges,
            "themes": ["analytical leadership", "structured execution"],
        },
        timing={
            "dasha_activation": bool(maha),
            "active_mahadasha": maha,
            "active_antardasha": antar,
            "timing_window": timing_window,
        },
        completeness=completeness,
        confidence=0.95 if completeness >= 0.5 else 0.4
    )


# ---------------------------------------------------------------------------
# 1. Question-Type Classification Tests
# ---------------------------------------------------------------------------
def test_classify_question_type_timing():
    assert classify_question_type("When will I get a job promotion?", "career_timing") == "timing"
    assert classify_question_type("kab tak shaadi hogi?", "marriage_timing") == "timing"
    assert classify_question_type("What is the timing window for buying a home?") == "timing"


def test_classify_question_type_feasibility():
    assert classify_question_type("Should I switch jobs right now?", "career_advisory") == "feasibility_advisory"
    assert classify_question_type("Can I invest in mutual funds?", "wealth_growth") == "feasibility_advisory"
    assert classify_question_type("kya mujhe business start karna chahiye?") == "feasibility_advisory"


def test_classify_question_type_comparison():
    assert classify_question_type("Should I do job or business?", "career_guidance") == "comparison"
    assert classify_question_type("job vs business which is better for me?") == "comparison"
    assert classify_question_type("business ya job mein se kya sahi rahega?") == "comparison"


def test_classify_question_type_direct_placement():
    assert classify_question_type("What does my 10th lord say about my career?", "career_10th_lord") == "direct_placement"
    assert classify_question_type("Tell me about 7th house placement", "marriage_placement") == "direct_placement"
    assert classify_question_type("Saturn in 10th house impact?") == "direct_placement"


def test_classify_question_type_growth_outlook():
    assert classify_question_type("How is my career looking overall?", "career_general") == "growth_outlook"
    assert classify_question_type("What is the future outlook for my finances?") == "growth_outlook"


# ---------------------------------------------------------------------------
# 2. Direct Answer in Opening Sentence & Question-Aware Shapes
# ---------------------------------------------------------------------------
def test_timing_answer_shape_with_explicit_horizon():
    ev = _make_dummy_evidence(domain="career", maha="Mercury", antar="Venus")
    q = "Will I get a promotion in the next 6 months?"
    ans = synthesize_local_answer(ev, question=q)

    # Direct answer opening
    first_sentence = ans.split("\n")[0]
    assert "Regarding your query for the 6 months" in first_sentence
    assert "Mercury/Venus" in first_sentence
    # Chart context & conclusion
    assert "**Astrological Evidence (Virgo Ascendant, Taurus Moon sign)**:" in ans
    assert "**Actionable Conclusion**:" in ans


def test_timing_answer_shape_with_timing_window():
    ev = _make_dummy_evidence(domain="career", maha="Jupiter", timing_window="Q3 2026 to Q1 2027")
    q = "When is the right time for career advancement?"
    ans = synthesize_local_answer(ev, question=q)

    first_sentence = ans.split("\n")[0]
    assert "Q3 2026 to Q1 2027" in first_sentence
    assert "Jupiter" in first_sentence
    assert "**Actionable Conclusion**:" in ans


def test_feasibility_answer_shape():
    ev = _make_dummy_evidence(domain="career")
    q = "Should I switch jobs right now?"
    ans = synthesize_local_answer(ev, question=q)

    first_sentence = ans.split("\n")[0]
    assert first_sentence.startswith("Yes, your birth chart supports")
    assert "**Astrological Analysis (Virgo Ascendant, Taurus Moon sign)**:" in ans
    assert "**Key Recommendation**:" in ans


def test_comparison_answer_shape():
    ev = _make_dummy_evidence(domain="career")
    q = "Job vs business, which one is better?"
    ans = synthesize_local_answer(ev, question=q)

    first_sentence = ans.split("\n")[0]
    assert "Between structured employment and an independent business" in first_sentence
    assert "- **Structured Employment**:" in ans
    assert "- **Independent Business**:" in ans
    assert "**Strategic Recommendation**:" in ans


def test_direct_placement_answer_shape():
    ev = _make_dummy_evidence(domain="career")
    q = "What does my 10th lord tell about my career path?"
    ans = synthesize_local_answer(ev, question=q)

    first_sentence = ans.split("\n")[0]
    assert "Your 10th lord Mercury placed in Gemini" in first_sentence
    assert "**Astrological Mechanics" in ans
    assert "**Core Takeaway**:" in ans


def test_growth_outlook_answer_shape():
    ev = _make_dummy_evidence(domain="career")
    q = "What is my general career growth trajectory?"
    ans = synthesize_local_answer(ev, question=q)

    first_sentence = ans.split("\n")[0]
    assert "Your overall career outlook is characterized by steady, compounding advancement" in first_sentence
    assert "**Astrological Baseline (Virgo Ascendant, Taurus Moon sign)**:" in ans
    assert "**Strategic Outlook**:" in ans


# ---------------------------------------------------------------------------
# 3. Discourse Connectors & Conflicting Evidence Preservation
# ---------------------------------------------------------------------------
def test_conflicting_evidence_connectors():
    pos = ["Strong 10th lord supports administrative authority"]
    chal = ["Saturn transit demands strict patience and meticulous pacing"]
    connected = _connect_evidence_phrases(pos, chal, is_hinglish=False)

    # Both must be preserved
    assert "Strong 10th lord supports administrative authority" in connected
    assert "Saturn transit demands strict patience and meticulous pacing" in connected
    # Contrastive connector
    assert "While these foundational placements are supportive, conversely," in connected


def test_multiple_positive_connectors():
    pos = [
        "Analytical intelligence",
        "Strategic clarity in technical execution",
        "Supportive professional mentors"
    ]
    chal = []
    connected = _connect_evidence_phrases(pos, chal, is_hinglish=False)

    assert "Analytical intelligence" in connected
    assert "Furthermore, Strategic clarity in technical execution" in connected
    assert "Additionally, Supportive professional mentors" in connected


def test_repetition_reduction_in_evidence():
    # Duplicate phrases should be pruned seamlessly
    pos = [
        "Consistent technical focus",
        "consistent technical focus",  # case-insensitive duplicate
        "Strategic clarity"
    ]
    chal = ["Patience required during transit"]
    connected = _connect_evidence_phrases(pos, chal, is_hinglish=False)

    # Must only appear once
    count = connected.lower().count("consistent technical focus")
    assert count == 1
    assert "Strategic clarity" in connected
    assert "Patience required during transit" in connected


# ---------------------------------------------------------------------------
# 4. Missing Timing Evidence Handling (Zero Fabricated Dates/Claims)
# ---------------------------------------------------------------------------
def test_missing_timing_evidence_does_not_fabricate_dates():
    # maha="" simulates uninitialized Dasha
    ev = _make_dummy_evidence(domain="career", maha="")
    q = "When will I change jobs?"
    ans = synthesize_local_answer(ev, question=q)

    # Must not contain fabricated years or dates
    assert "2026" not in ans
    assert "2027" not in ans
    assert "2028" not in ans
    assert "specific timing windows cannot be confirmed without verified active Dasha and transit data" in ans
    assert "The system avoids generating ungrounded calendar dates" in ans


def test_missing_evidence_insufficient_fallback():
    ev = _make_dummy_evidence(domain="career", completeness=0.2)
    ev.evaluations = {}
    ev.facts = {}
    ans = synthesize_local_answer(ev, question="How is my career?")

    assert "cannot be fully determined from the backend facts" in ans
    assert "avoids generating ungrounded claims" in ans


# ---------------------------------------------------------------------------
# 5. Unsupported Claims Guarding
# ---------------------------------------------------------------------------
def test_guard_deterministic_claims():
    text = "A job promotion is guaranteed and you will definitely get exact date with 100% certainty."
    guarded = guard_deterministic_claims(text)

    assert "guaranteed" not in guarded.lower()
    assert "will definitely" not in guarded.lower()
    assert "exact date" not in guarded.lower()
    assert "100% certainty" not in guarded.lower()
    assert "supportive" in guarded.lower()


# ---------------------------------------------------------------------------
# 6. Hinglish Fluency across Question Types
# ---------------------------------------------------------------------------
def test_hinglish_timing_answer():
    ev = _make_dummy_evidence(domain="career", maha="Mercury", antar="Venus")
    q = "kab tak meri job change hogi?"
    ans = synthesize_local_answer(ev, is_hinglish=True, question=q)

    first_sentence = ans.split("\n")[0]
    assert "active Mercury/Venus Dasha" in first_sentence
    assert "supportive activation window" in first_sentence or "timing window" in first_sentence
    assert "**Astrological Context (Virgo Lagna, Taurus Moon)**:" in ans
    assert "**Conclusion**:" in ans


def test_hinglish_feasibility_answer():
    ev = _make_dummy_evidence(domain="career")
    q = "kya mujhe abhi switch karna chahiye?"
    ans = synthesize_local_answer(ev, is_hinglish=True, question=q)

    first_sentence = ans.split("\n")[0]
    assert "Haan, aapka birth chart independent initiatives" in first_sentence
    assert "**Chart Indicators (Virgo Lagna, Taurus Moon)**:" in ans
    assert "**Key Advisory**:" in ans


def test_hinglish_comparison_answer():
    ev = _make_dummy_evidence(domain="career")
    q = "job ya business kya sahi rahega?"
    ans = synthesize_local_answer(ev, is_hinglish=True, question=q)

    first_sentence = ans.split("\n")[0]
    assert "Job aur business ke comparison mein" in first_sentence
    assert "- **Job / Structured Employment**:" in ans
    assert "- **Business / Independent Venture**:" in ans
    assert "**Strategic Conclusion**:" in ans
