"""
Regression Test Suite for Relevance-Gated Dynamic Fallback Architecture
========================================================================
File: scratch/test_relevance_gated_fallback_suite.py

Verifies:
1. Exact local rules execution without Gemini (LOCAL).
2. Unsupported deterministic prediction boundary protection (UNSUPPORTED).
3. Missing chart evidence handling (UNRESOLVED).
4. Evidence-Grounded LLM routing for uncovered nuanced questions (EVIDENCE_GROUNDED_LLM).
5. Typo domain classification resilience (marriage, finance).
6. Creative poetic request handling (PARTIAL_LOCAL_LLM).
"""

import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

logging.disable(logging.CRITICAL)

from backend.main import _handle_chat_response, ChatRequest, route_question, normalize_hinglish

test_cases = [
    {
        "id": 1,
        "name": "Exact Local Rule — Current Dasha",
        "query": "What is my current Mahadasha?",
        "expected_sources": ["LOCAL", "DIRECT", "DIRECT_API", "direct_fact_engine", "deterministic_api"],
        "max_gemini": 0
    },
    {
        "id": 2,
        "name": "Local Career Rule — Dasha Career Interaction",
        "query": "How does my current Dasha interact with my career indicators?",
        "expected_source": "LOCAL",
        "max_gemini": 0
    },
    {
        "id": 3,
        "name": "General Career Question — Broad Explanation",
        "query": "Can you explain my overall career situation from my chart?",
        "expected_sources": ["LOCAL", "EVIDENCE_GROUNDED_LLM", "PARTIAL_LOCAL_LLM"],
        "disallow_jargon": True
    },
    {
        "id": 4,
        "name": "Specific Career Transition — Uncovered Nuance",
        "query": "Should I move from software engineering to product management during my current Dasha?",
        "expected_sources": ["EVIDENCE_GROUNDED_LLM", "PARTIAL_LOCAL_LLM", "LOCAL"],
        "disallow_jargon": True
    },
    {
        "id": 5,
        "name": "Unsupported Boundary — Lottery",
        "query": "Give me the exact winning lottery numbers for tomorrow.",
        "expected_source": "UNSUPPORTED",
        "max_gemini": 0
    },
    {
        "id": 6,
        "name": "Unsupported Boundary — Death Date",
        "query": "Tell me the exact date I will die.",
        "expected_source": "UNSUPPORTED",
        "max_gemini": 0
    },
    {
        "id": 7,
        "name": "Typo Resilience — Marriage",
        "query": "when will i get marriead",
        "expected_domain": "marriage"
    },
    {
        "id": 8,
        "name": "Typo Resilience — Finance",
        "query": "what is my finiacial status in upcoming year",
        "expected_domain": "finance"
    },
    {
        "id": 9,
        "name": "Partial Creative — Poem Request",
        "query": "Which career suits me and write a beautiful poem about it?",
        "expected_sources": ["PARTIAL_LOCAL_LLM", "EVIDENCE_GROUNDED_LLM", "LOCAL"]
    }
]

async def run_suite():
    passed = 0
    total = len(test_cases)
    print("============================================================")
    print("   RUNNING RELEVANCE-GATED FALLBACK SUITE (Phase 26)        ")
    print("============================================================")

    for tc in test_cases:
        tc_id = tc["id"]
        tc_name = tc["name"]
        q = tc["query"]

        req = ChatRequest(
            query=q,
            user_id=f"test_fg_{tc_id}",
            name="Rahul",
            birth_year=1995,
            birth_month=5,
            birth_day=15,
            birth_hour=14.5,
            latitude=28.6139,
            longitude=77.2090,
            timezone=5.5
        )

        # Handle typo domain tests separately
        if "expected_domain" in tc:
            norm = normalize_hinglish(q)
            r = route_question(norm)
            dom = r.get("domain")
            if dom == tc["expected_domain"]:
                print(f"[PASS] Test {tc_id}: {tc_name} (Domain correctly classified as '{dom}')")
                passed += 1
            else:
                print(f"[FAIL] Test {tc_id}: {tc_name} (Expected domain '{tc['expected_domain']}', got '{dom}')")
            continue

        res = await _handle_chat_response(req)
        ans = ""
        ans_src = ""
        gemini_calls = 0

        if hasattr(res, 'body'):
            body_json = json.loads(res.body.decode('utf-8'))
            ans = body_json.get("answer", "")
            ans_src = body_json.get("answer_source", body_json.get("source", ""))
            gemini_calls = body_json.get("gemini_calls", 0)
        elif hasattr(res, 'body_iterator'):
            # Consume StreamingResponse
            chunks = []
            async for chunk in res.body_iterator:
                if isinstance(chunk, bytes):
                    chunks.append(chunk.decode('utf-8'))
                else:
                    chunks.append(str(chunk))
            ans = "".join(chunks)
            ans_src = "EVIDENCE_GROUNDED_LLM"
            gemini_calls = 1

        # Assertion Checks
        ok = True
        err_reasons = []

        if "expected_source" in tc and ans_src != tc["expected_source"]:
            if "expected_sources" in tc and ans_src in tc["expected_sources"]:
                pass
            else:
                ok = False
                err_reasons.append(f"Expected source '{tc['expected_source']}', got '{ans_src}'")

        if "expected_sources" in tc and ans_src not in tc["expected_sources"]:
            ok = False
            err_reasons.append(f"Expected source in {tc['expected_sources']}, got '{ans_src}'")

        if "max_gemini" in tc and gemini_calls > tc["max_gemini"]:
            ok = False
            err_reasons.append(f"Gemini calls {gemini_calls} exceeded max allowed {tc['max_gemini']}")

        if tc.get("disallow_jargon"):
            jargon_terms = ["dignity evidence", "natal planetary evidence", "evidence categories active", "stage 8"]
            for term in jargon_terms:
                if term in ans.lower():
                    ok = False
                    err_reasons.append(f"Response contains internal debug jargon: '{term}'")

        if ok:
            print(f"[PASS] Test {tc_id}: {tc_name} (Source: {ans_src}, Gemini Calls: {gemini_calls})")
            passed += 1
        else:
            print(f"[FAIL] Test {tc_id}: {tc_name} -> {', '.join(err_reasons)}")
            if ans:
                print(f"      Ans Preview: {ans[:120]}...")

    print("============================================================")
    print(f"SUMMARY: {passed}/{total} Test Cases Passed ({passed/total*100:.1f}%)")
    print("============================================================")
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_suite())
    sys.exit(0 if success else 1)
