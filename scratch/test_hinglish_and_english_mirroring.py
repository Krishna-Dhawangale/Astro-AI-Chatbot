import asyncio
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, "backend", ".env"), override=True)

from backend.llm_router import (
    is_hinglish_query,
    synthesize_hybrid_multidomain_rag,
    stream_hybrid_multidomain_rag,
)
from backend.reasoning.direct_fact_engine import extract_direct_fact, is_direct_fact_query

MOCK_CHART = {
    "ascendant": {"rashi": "Leo", "sign": "Leo", "degree_in_rashi": 14.2},
    "planets": {
        "Moon": {"rashi": "Scorpio", "sign": "Scorpio", "nakshatra": "Anuradha", "nakshatra_pada": 3, "rashi_lord": "Mars"},
        "Sun": {"rashi": "Aries", "sign": "Aries", "rashi_lord": "Mars"},
        "Mercury": {"rashi": "Gemini", "sign": "Gemini", "house": 10, "rashi_lord": "Mercury"},
        "Venus": {"rashi": "Pisces", "sign": "Pisces", "house": 8, "rashi_lord": "Jupiter"},
        "Jupiter": {"rashi": "Libra", "sign": "Libra", "house": 3, "rashi_lord": "Venus"}
    }
}
MOCK_DASHA = {
    "current_mahadasha": "Mars",
    "current_antardasha": "Rahu"
}

def test_language_detection():
    print("=" * 60)
    print("1. TESTING LANGUAGE DETECTION (is_hinglish_query)")
    print("=" * 60)
    
    test_cases = [
        # English queries
        ("What is my moon sign?", False),
        ("What is my rashi?", False),
        ("Can you analyze my career, finance, health, and recommended gemstones all at once?", False),
        ("Can you analyze my career, finances, health, and recommended gemstones all at once?", False),
        ("When will I get married?", False),
        ("Which gemstone is suitable for my 10th house?", False),
        ("Explain the difference between Lagna and Rashi", False),
        
        # Hinglish queries
        ("Mera moon sign kya hai?", True),
        ("Mera rashi kya hai?", True),
        ("Meri lagna aur rashi me kya antar hai?", True),
        ("Mera career, finance, health aur recommended gemstones kaisa rahega?", True),
        ("Shaadi kab hogi?", True),
        ("Naukri kab milegi?", True),
        ("10th house ke baare me batao", True),
        ("Meri kundali analyze karo", True),
    ]

    all_passed = True
    for q, expected in test_cases:
        res = is_hinglish_query(q)
        status = "PASS" if res == expected else "FAIL"
        if res != expected:
            all_passed = False
        print(f"[{status}] Query: '{q}' -> detected Hinglish: {res} (expected: {expected})")

    assert all_passed, "Some language detection tests failed!"
    print(">>> Language detection: ALL PASSED!\n")


def test_direct_fact_mirroring():
    print("=" * 60)
    print("2. TESTING DIRECT FACT ENGINE LANGUAGE MIRRORING")
    print("=" * 60)

    pairs = [
        # (English Query, Hinglish Query)
        ("What is my moon sign?", "Mera moon sign kya hai?"),
        ("What is my lagna?", "Mera lagna konsa hai?"),
        ("What is my nakshatra?", "Mera janma nakshatra kya hai?"),
        ("What is my current mahadasha?", "Mera current mahadasha konsa hai?"),
        ("Difference between Lagna and Rashi", "Lagna aur Rashi me kya difference hai?"),
    ]

    for en_q, hi_q in pairs:
        en_res = extract_direct_fact(en_q, MOCK_CHART, MOCK_DASHA, is_hinglish=False)
        hi_res = extract_direct_fact(hi_q, MOCK_CHART, MOCK_DASHA, is_hinglish=True)

        print(f"EN Query: {en_q}")
        print(f"EN Answer: {en_res['answer'][:90]}...")
        assert "Your " in en_res["answer"] or "The " in en_res["answer"] or "is" in en_res["answer"]

        print(f"HI Query: {hi_q}")
        print(f"HI Answer: {hi_res['answer'][:90]}...")
        assert "Aapka " in hi_res["answer"] or "Aapke " in hi_res["answer"] or "hai" in hi_res["answer"]
        print("-" * 50)

    print(">>> Direct fact engine language mirroring: ALL PASSED!\n")


async def test_hybrid_multidomain_mirroring():
    print("=" * 60)
    print("3. TESTING HYBRID MULTI-DOMAIN RAG LANGUAGE MIRRORING")
    print("=" * 60)

    # English query
    en_query = "Can you analyze my career, finance, health, and recommended gemstones all at once?"
    print(f"Testing English Hybrid RAG:\nQuery: {en_query}")
    en_output, en_meta = await synthesize_hybrid_multidomain_rag(en_query, is_hinglish=False)
    print(f"English Output:\n{en_output}\n")
    assert "**Astrological Insight:**" in en_output
    assert "**Actionable Advice:**" in en_output
    assert "[Local Database + LLM -" in en_output
    # English response should contain typical English words and not Hindi words
    assert not any(hi_word in en_output.lower() for hi_word in ["aapka", "aapke", "hai", "karein", "ke anusaar"])

    # Hinglish query
    hi_query = "Mera career, finance, health aur recommended gemstones kaisa rahega?"
    print(f"Testing Hinglish Hybrid RAG:\nQuery: {hi_query}")
    hi_output, hi_meta = await synthesize_hybrid_multidomain_rag(hi_query, is_hinglish=True)
    print(f"Hinglish Output:\n{hi_output}\n")
    assert "**Astrological Insight:**" in hi_output
    assert "**Actionable Advice:**" in hi_output
    assert "[Local Database + LLM -" in hi_output
    # Hinglish response should contain Hindi words (e.g. aapke / aapka / hai / ke anusaar / karein)
    has_hindi = any(hi_word in hi_output.lower() for hi_word in ["aap", "aapka", "aapke", "hai", "karein", "anusaar", "rahega", "aur"])
    print(f"Hinglish markers found in output: {has_hindi}")
    assert has_hindi, "Expected Hinglish words in Hinglish synthesis output!"

    print(">>> Hybrid Multi-Domain RAG language mirroring: ALL PASSED!\n")


async def main():
    test_language_detection()
    test_direct_fact_mirroring()
    await test_hybrid_multidomain_mirroring()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
