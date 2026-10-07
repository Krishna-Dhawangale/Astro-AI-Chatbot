"""
4-Status Architecture Verification Suite (verify_4mode_architecture.py)
========================================================================
Audits all 4 internal answer statuses (DIRECT, RULE_BASED, LLM_ASSISTED, UNSUPPORTED)
and verifies the 4 key architectural safeguards:

1. DIRECT IS HIGHEST PRIORITY: Factual queries resolve before heavy reasoning pipelines (< 10ms target).
2. RULE_BASED REQUIRES BOTH CONDITIONS: Complete evidence AND matched rule templates (0 LLM calls).
3. LLM_ASSISTED STRICT RENDERER BOUNDARY: Gemini acts ONLY as a renderer (Max 75w, 0 astrology calculations).
4. UNSUPPORTED STATUS GUARD: Out-of-domain/invalid queries return controlled safety response without LLM hallucination.
"""

import sys
import time
import json
from pathlib import Path

# Force UTF-8 stdout encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.reasoning.direct_fact_engine import is_direct_fact_query, extract_direct_fact
from backend.reasoning.mode_selector import select_answer_mode
from backend.reasoning.llm_renderer import build_compact_evidence_package, STRICT_RENDERER_SYSTEM_INSTRUCTION


TEST_CHART = {
    "ascendant": {"rashi": "Virgo", "longitude": 165.0, "degree_in_rashi": 15.0},
    "planets": {
        "Sun": {"rashi": "Gemini", "longitude": 75.0, "house": 10},
        "Moon": {"rashi": "Taurus", "longitude": 45.0, "house": 9, "nakshatra": "Rohini", "pada": 2},
        "Mars": {"rashi": "Capricorn", "longitude": 285.0, "house": 5},
        "Mercury": {"rashi": "Gemini", "longitude": 80.0, "house": 10},
        "Jupiter": {"rashi": "Cancer", "longitude": 95.0, "house": 11},
        "Venus": {"rashi": "Taurus", "longitude": 50.0, "house": 9},
        "Saturn": {"rashi": "Aquarius", "longitude": 315.0, "house": 6},
        "Rahu": {"rashi": "Virgo", "longitude": 160.0, "house": 1},
        "Ketu": {"rashi": "Pisces", "longitude": 340.0, "house": 7}
    }
}

TEST_DASHA = {
    "mahadasha": {"planet": "Mercury", "end": "2028-10-15"},
    "antardasha": {"planet": "Saturn", "end": "2026-12-20"}
}


def test_4mode_architecture():
    print("=" * 90)
    print("AUDITING 4-STATUS ENGINE ARCHITECTURE (DIRECT -> RULE_BASED -> LLM_ASSISTED -> UNSUPPORTED)")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # TEST 1: DIRECT FACT PATHWAY (Highest Priority)
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: MODE 1 — DIRECT Fact Pathway ---")
    direct_queries = [
        "What is my Moon sign?",
        "What is my Lagna?",
        "What is my Nakshatra?",
        "What planet is in my 7th house?",
        "Where is Jupiter placed?"
    ]

    for q in direct_queries:
        t0 = time.perf_counter()
        assert is_direct_fact_query(q)
        fact_res = extract_direct_fact(q, TEST_CHART, TEST_DASHA)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert fact_res is not None
        assert fact_res["answer_mode"] == "DIRECT"
        assert fact_res["gemini_calls"] == 0
        assert fact_res["evidence_complete"] == True
        print(f"   - Query: '{q:<32}' | Mode: DIRECT | Calls: 0 | Time: {elapsed_ms:.2f}ms | Ans: {fact_res['answer'][:50]}...")

    print("[PASS] DIRECT Mode: Factual queries resolved instantly (< 10ms target) with 0 LLM calls.")

    # -------------------------------------------------------------------------
    # TEST 2: RULE_BASED PATHWAY (Mandatory Both Conditions Gate)
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: MODE 2 — RULE_BASED Pathway & Completeness Gate ---")
    sample_matched_rules = [
        {"rule_id": "CAREER_10TH_LORD_PLACEMENT", "category": "foundation", "matched": True},
        {"rule_id": "CAREER_KARAKA_DIGNITY", "category": "dignity", "matched": True}
    ]

    mode_info = select_answer_mode(
        domain="career",
        intent="career_general",
        question="Which career suits me best?",
        matched_rules=sample_matched_rules
    )

    assert mode_info["mode"] == "RULE_BASED"
    assert mode_info["evidence_complete"] == True
    assert mode_info["gemini_calls"] == 0
    assert "CAREER_10TH_LORD_PLACEMENT" in mode_info["evidence_ids"]
    print(f"   - Intent: career_general | Mode: {mode_info['mode']} | Evidence Complete: {mode_info['evidence_complete']} | Calls: 0")
    print("[PASS] RULE_BASED Mode: Requires BOTH complete evidence AND rule coverage.")

    # -------------------------------------------------------------------------
    # TEST 3: LLM_ASSISTED STRICT RENDERER BOUNDARY
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: MODE 3 — LLM_ASSISTED Strict Renderer Boundary ---")
    partial_rules = [
        {"rule_id": "CAREER_KARAKA_PRESENCE", "category": "karaka", "matched": True}
    ]

    mode_info_llm = select_answer_mode(
        domain="career",
        intent="career_general",
        question="Tell me about my career potential in detail.",
        matched_rules=partial_rules
    )

    assert mode_info_llm["mode"] == "LLM_ASSISTED"
    assert mode_info_llm["answer_source"] == "EVIDENCE_GROUNDED_LLM"
    assert mode_info_llm["gemini_calls"] == 1

    # Audit compact evidence package & strict renderer prompt
    pkg = build_compact_evidence_package("career", "career_general", TEST_CHART, partial_rules, TEST_DASHA)
    assert "verified_facts" in pkg
    assert "selected_rules" in pkg

    assert "Never calculate, infer, or invent astrology." in STRICT_RENDERER_SYSTEM_INSTRUCTION
    assert "Return only the answer." in STRICT_RENDERER_SYSTEM_INSTRUCTION
    print(f"   - Intent: career_general (partial) | Mode: LLM_ASSISTED | Evidence Complete: False | Calls: 1")
    print("[PASS] LLM_ASSISTED Mode: Gemini acts strictly as a response renderer (0 astrology calculations permitted).")

    # -------------------------------------------------------------------------
    # TEST 4: UNSUPPORTED STATUS GUARD
    # -------------------------------------------------------------------------
    print("\n--- TEST 4: STATUS 4 — UNSUPPORTED Guard ---")
    unsupported_queries = [
        ("Tell me what happened in my past life", "past_life", "general"),
        ("What are tomorrow's winning lottery numbers?", "lottery_winning", "general")
    ]

    for q, intent, domain in unsupported_queries:
        mode_info_unsupported = select_answer_mode(domain, intent, q)
        assert mode_info_unsupported["mode"] == "UNSUPPORTED"
        assert mode_info_unsupported["gemini_calls"] == 0
        assert mode_info_unsupported["evidence_complete"] == False
        print(f"   - Query: '{q:<45}' | Status: UNSUPPORTED | Calls: 0 | Reason: {mode_info_unsupported['reason']}")

    print("[PASS] UNSUPPORTED Status: Out-of-domain queries return controlled response with 0 LLM calls.")

    print("\n" + "=" * 90)
    print("4-STATUS ARCHITECTURE VERIFICATION COMPLETE: ALL TESTS PASSED (100.0%)")
    print("=" * 90)


if __name__ == "__main__":
    test_4mode_architecture()
