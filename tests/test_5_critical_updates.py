"""
tests/test_5_critical_updates.py
================================
Validation Test Suite for the 5 Critical Backend Updates:
1. Hinglish Language Support
2. Input Sanitization & Garbage / Special Character Filtering
3. Profanity & Vulgarity Guardrail (Hindi / Hinglish / English)
4. Astrological House-Related Query Handling (Bhava Karakas & Interpretations)
5. Intent-First Execution Pipeline (Intent identification & routing)
"""

import os
import sys
from pathlib import Path

# Add project root and backend to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

try:
    import pytest
except ImportError:
    pytest = None
from backend.intent_router import (
    classify_query_intent,
    is_garbage_or_special_char_input,
    check_profanity_guardrail,
    is_hinglish_query,
    GARBAGE_REFUSAL_MESSAGE,
    PROFANITY_REFUSAL_MESSAGE,
)
from backend.astrology.house_engine import (
    is_house_query,
    extract_house_number,
    synthesize_house_analysis,
    BHAVA_KNOWLEDGE_BASE,
)


# ===========================================================================
# 1. TEST HINGLISH LANGUAGE SUPPORT
# ===========================================================================
def test_hinglish_language_detection():
    hinglish_queries = [
        "mera 7th house kaisa hai",
        "meri kundali me kya likha hai",
        "job kab lagegi meri",
        "shaadi kab tak hogi",
        "kya mujhe business karna chahiye ya job",
        "paisa kab aayega mere paas",
        "kaunsa graha mere career ko support karta hai",
        "kripya meri shani dasha ke baare mein batao",
        "health kaisi rahegi aage",
        "videsh yatra ka yog hai kya"
    ]
    for q in hinglish_queries:
        assert is_hinglish_query(q) is True, f"Failed to detect Hinglish for: '{q}'"

    english_queries = [
        "What does my 7th house mean?",
        "When will I get a job promotion?",
        "Will I move abroad for higher studies?",
        "What are the significations of Saturn in 10th house?",
        "How is my financial stability?"
    ]
    for q in english_queries:
        assert is_hinglish_query(q) is False, f"Erroneously flagged English as Hinglish: '{q}'"


# ===========================================================================
# 2. TEST INPUT SANITIZATION & GARBAGE/SPECIAL CHARACTERS FILTERING
# ===========================================================================
def test_garbage_and_special_character_filtering():
    garbage_samples = [
        "a",
        "x",
        "z",
        "?",
        "!",
        ".",
        "???",
        "@@@",
        "!!!",
        "?#$%",
        "123",
        "99999",
        "aaaaa",
        "zzzz",
        "asdfghjk",
        "....",
        "&&&"
    ]
    for g in garbage_samples:
        is_garbage, reason = is_garbage_or_special_char_input(g)
        assert is_garbage is True, f"Failed to identify garbage for: '{g}' (reason: {reason})"
        res = classify_query_intent(g)
        assert res.primary_intent == "garbage_input"
        assert res.is_refusal is True
        assert res.refusal_message == GARBAGE_REFUSAL_MESSAGE
        assert res.refusal_source == "INPUT_SANITIZER"
        assert "garbage_input" in res.detected_intents

    valid_samples = [
        "What is my Moon sign?",
        "When will I get a job?",
        "What does the 7th house signify?",
        "Tell me about my active Rahu Mahadasha"
    ]
    for v in valid_samples:
        is_garbage, _ = is_garbage_or_special_char_input(v)
        assert is_garbage is False, f"Valid query wrongly flagged as garbage: '{v}'"


# ===========================================================================
# 3. TEST PROFANITY & VULGARITY GUARDRAIL (HINDI / HINGLISH / ENGLISH)
# ===========================================================================
def test_profanity_guardrail_detection():
    profanity_samples = [
        "bsdk",
        "chutiya",
        "gandu",
        "bhosdike",
        "lauda",
        "laude",
        "lund",
        "randi",
        "harami",
        "kamina",
        "kamine",
        "jhaat",
        "madarchod",
        "behenchod",
        "fuck",
        "fuck you",
        "bitch",
        "asshole",
        "bastard",
        "teri maa ki choot",
        "b.s.d.k"
    ]
    for p in profanity_samples:
        is_profane, term = check_profanity_guardrail(p)
        assert is_profane is True, f"Failed to catch profanity for: '{p}'"
        res = classify_query_intent(p)
        assert res.primary_intent == "profanity"
        assert res.is_refusal is True
        assert res.refusal_message == PROFANITY_REFUSAL_MESSAGE
        assert res.refusal_source == "PROFANITY_GUARDRAIL"
        assert "profanity" in res.detected_intents

    # Ensure historical 'BC' and astrological 'MC' are NOT falsely flagged
    safe_samples = [
        "What happened in 1500 BC?",
        "Tell me about my natal MC and Midheaven",
        "What does the 7th house mean?"
    ]
    for s in safe_samples:
        is_profane, _ = check_profanity_guardrail(s)
        assert is_profane is False, f"Safe query wrongly flagged as profanity: '{s}'"


def test_profanity_strike_escalation():
    from backend.abuse_guardrail import abuse_tracker
    test_sid = "test_strike_session_123"
    abuse_tracker.reset_session(test_sid)

    # Strike 1: "bsdk" -> Warning (1/3)
    d1 = abuse_tracker.check_message(test_sid, "bsdk")
    assert d1 is not None and d1.get("blocked") is True
    assert d1.get("strikes") == 1
    assert d1.get("message") == "⚠️ Warning (1/3): Inappropriate or vulgar language is not permitted."

    # Strike 2: "chutiya" -> Warning (2/3)
    d2 = abuse_tracker.check_message(test_sid, "chutiya")
    assert d2 is not None and d2.get("blocked") is True
    assert d2.get("strikes") == 2
    assert d2.get("message") == "⚠️ Warning (2/3): Inappropriate or vulgar language is not permitted."

    # Strike 3: "gandu" -> Termination
    d3 = abuse_tracker.check_message(test_sid, "gandu")
    assert d3 is not None and d3.get("blocked") is True
    assert d3.get("terminated") is True
    assert d3.get("strikes") == 3
    assert d3.get("message") == "🚫 Session Terminated: You have exceeded the limit for inappropriate language."

    abuse_tracker.reset_session(test_sid)


# ===========================================================================
# 4. TEST ASTROLOGICAL HOUSE-RELATED QUERY HANDLING
# ===========================================================================
def test_house_query_identification_and_synthesis():
    house_queries = [
        ("What does the 7th house mean?", 7),
        ("Planets in 10th house", 10),
        ("What is the 1st house?", 1),
        ("Tell me about my 4th house", 4),
        ("What does 8th house represent?", 8),
        ("Mera 7th house kaisa hai", 7),
        ("10th house mein kaunse graha hain", 10),
    ]
    for q, expected_h in house_queries:
        assert is_house_query(q) is True, f"Failed to identify house query: '{q}'"
        extracted_h = extract_house_number(q)
        assert extracted_h == expected_h, f"Expected house {expected_h}, got {extracted_h} for: '{q}'"

    # Test synthesis in English
    sample_chart = {
        "ascendant": {"rashi": "Leo", "sign": "Leo"},
        "planets": {
            "Sun": {"house": 10, "rashi": "Taurus"},
            "Mercury": {"house": 10, "rashi": "Taurus"},
            "Venus": {"house": 7, "rashi": "Aquarius"},
            "Moon": {"house": 9, "rashi": "Aries"}
        }
    }
    synth_en = synthesize_house_analysis("Planets in 10th house", chart_data=sample_chart, is_hinglish=False, asc_sign="Leo")
    assert synth_en["house_num"] == 10
    assert synth_en["sanskrit_name"] == "Karma Bhava & Rajyashrita Bhava"
    assert synth_en["sign"] == "Taurus"  # Leo ascendant -> 10th house is Taurus
    assert synth_en["lord"] == "Venus"
    assert "Sun" in synth_en["occupants"]
    assert "Mercury" in synth_en["occupants"]
    assert "Natural Karakas" in synth_en["answer"]
    assert "In Your Personal Horoscope (Leo Ascendant):" in synth_en["answer"]
    assert "Sun" in synth_en["answer"]

    # Test synthesis in Conversational Hinglish
    synth_hi = synthesize_house_analysis("mera 7th house kaisa hai", chart_data=sample_chart, is_hinglish=True, asc_sign="Leo")
    assert synth_hi["house_num"] == 7
    assert synth_hi["sign"] == "Aquarius"  # Leo ascendant -> 7th house is Aquarius
    assert synth_hi["lord"] == "Saturn"
    assert "Venus" in synth_hi["occupants"]
    assert "Vedic Astrology ke anusaar" in synth_hi["answer"]
    assert "Kalatra Bhava" in synth_hi["answer"]
    assert "Aapki Kundali (Leo Lagna) ke Anusaar:" in synth_hi["answer"]


# ===========================================================================
# 5. TEST INTENT-FIRST EXECUTION PIPELINE
# ===========================================================================
def test_intent_first_execution_pipeline():
    test_cases = [
        # (Query, Expected Primary Intent)
        ("???", "garbage_input"),
        ("x", "garbage_input"),
        ("123", "garbage_input"),
        ("bsdk", "profanity"),
        ("chutiya bot", "profanity"),
        ("What does the 7th house mean?", "house_analysis"),
        ("Planets in 10th house", "house_analysis"),
        ("When will I get a job promotion?", "career_timing"),
        ("When will I get married?", "marriage_timing"),
        ("What is my current dasha?", "dasha"),
        ("What is a Lagna?", "educational_faq"),
        ("Tell me about my career and marriage", "multi_domain"),
    ]
    for q, exp_intent in test_cases:
        res = classify_query_intent(q)
        assert res.primary_intent == exp_intent, f"Query '{q}' expected '{exp_intent}', got '{res.primary_intent}'"
        assert exp_intent in res.detected_intents, f"Query '{q}' missing '{exp_intent}' in detected_intents: {res.detected_intents}"


def test_disclaimer_removal_and_astrologer_fallback():
    from backend.response_formatter import clean_astrology_response, strip_astrologer_disclaimers

    raw_en = (
        "Your 10th house is governed by Saturn. "
        "Please consult a professional astrologer for a detailed, personalized analysis of your chart. "
        "Maintain disciplined daily focus."
    )
    cleaned_en = clean_astrology_response(raw_en)
    assert "Please consult a professional astrologer" not in cleaned_en
    assert "detailed, personalized analysis" not in cleaned_en
    assert "Your 10th house is governed by Saturn" in cleaned_en
    assert "Maintain disciplined daily focus" in cleaned_en

    raw_hi = (
        "Aapki kundali mein 7th house strong hai. "
        "Kripya kisi qualified Vedic astrologer se sampark karein."
    )
    cleaned_hi = clean_astrology_response(raw_hi, is_hinglish=True)
    assert "Kripya kisi qualified Vedic astrologer se sampark karein" not in cleaned_hi
    assert "Aapki kundali mein 7th house strong hai" in cleaned_hi

    # Only disclaimer present -> falls back to chart-grounded astrological synthesis
    only_disclaimer_en = "Please consult a professional astrologer for a detailed, personalized analysis of your chart."
    fallback_en = clean_astrology_response(only_disclaimer_en, is_hinglish=False)
    assert len(fallback_en) > 20
    assert "consult a professional astrologer" not in fallback_en.lower()
    assert "birth chart" in fallback_en.lower()

    only_disclaimer_hi = "Kripya kisi qualified Vedic astrologer se sampark karein."
    fallback_hi = clean_astrology_response(only_disclaimer_hi, is_hinglish=True)
    assert len(fallback_hi) > 20
    assert "sampark karein" not in fallback_hi.lower()
    assert "kundali" in fallback_hi.lower()


if __name__ == "__main__":
    if pytest:
        pytest.main(["-v", __file__])
    else:
        test_hinglish_language_detection()
        test_garbage_and_special_character_filtering()
        test_house_query_identification_and_synthesis()
        test_intent_first_execution_pipeline()
        test_disclaimer_removal_and_astrologer_fallback()
        print("ALL 5 CRITICAL TESTS PASSED!")
