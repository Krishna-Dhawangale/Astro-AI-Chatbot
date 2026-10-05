"""
LLM Token Observability & Trace Verification Script
===================================================
File: scratch/verify_llm_token_observability.py

Verifies:
1. Terminal formatting for LLM calls (Input, Output, Total, Evidence, Latency).
2. Terminal formatting for LOCAL (0 tokens) and UNSUPPORTED (0 tokens) responses.
3. Multi-call individual logging + REQUEST TOTAL output.
4. answer_trace schema extension (llm_usage & llm_calls).
5. production_decision_trace.jsonl schema expansion (llm_total_tokens & llm_call_details).
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.reasoning.token_logger import (
    print_llm_call_report,
    print_request_total_report,
    print_no_llm_call_report
)
from backend.reasoning.answer_synthesizer import build_auditable_answer_trace
from backend.router.decision_trace import record_production_decision_trace, TRACE_LOG_FILE


def test_token_observability():
    print("================================================================================")
    print("STARTING LLM TOKEN OBSERVABILITY & TRACE VERIFICATION")
    print("================================================================================\n")

    # 1. Test Single LLM Call Logging
    print("--- TEST 1: Single LLM Call Logging ---")
    c1 = print_llm_call_report(
        call_number=1,
        purpose="creative_interpretation",
        model_name="gemini-3.5-flash-lite",
        input_tokens=115,
        output_tokens=48,
        evidence_sent=["10th_house", "10th_lord", "career_karakas"],
        latency_ms=842.5
    )

    assert c1["input_tokens"] == 115
    assert c1["output_tokens"] == 48
    assert c1["total_tokens"] == 163

    # 2. Test Multi-LLM Call Request Summary Logging
    print("--- TEST 2: Multi-LLM Call Request Summary ---")
    c2 = print_llm_call_report(
        call_number=2,
        purpose="fallback_retry",
        model_name="gemini-3.5-flash-lite",
        input_tokens=90,
        output_tokens=25,
        evidence_sent=["dasha_hierarchy"],
        latency_ms=450.0
    )
    req_tot = print_request_total_report([c1, c2], total_latency_ms=1292.5)

    assert req_tot["gemini_calls"] == 2
    assert req_tot["input_tokens"] == 205
    assert req_tot["output_tokens"] == 73
    assert req_tot["total_tokens"] == 278

    # 3. Test 0-Token LOCAL & UNSUPPORTED Reports
    print("--- TEST 3: Zero-Token Reports (LOCAL & UNSUPPORTED) ---")
    print_no_llm_call_report("LOCAL", "Rule-based / Direct Fact calculation complete")
    print_no_llm_call_report("UNSUPPORTED", "Out of domain / unsupported query boundary enforced")

    # 4. Test answer_trace Schema Extension
    print("--- TEST 4: answer_trace Schema Extension ---")
    trace_llm = build_auditable_answer_trace(
        domain="career",
        intent="career_change",
        answer_source="PARTIAL_LOCAL_LLM",
        matched_rules=[{"rule_id": "CAREER_10TH_LORD_PLACEMENT"}],
        theme_lineage=[{"planet": "Mercury", "theme": "Data Analytics"}],
        gemini_calls=1,
        llm_input_tokens=115,
        llm_output_tokens=48,
        llm_calls_list=[c1]
    )

    print("Answer Trace LLM Usage:", json.dumps(trace_llm["llm_usage"], indent=2))
    print("Answer Trace LLM Calls:", json.dumps(trace_llm["llm_calls"], indent=2))

    assert "llm_usage" in trace_llm
    assert trace_llm["llm_usage"]["input_tokens"] == 115
    assert trace_llm["llm_usage"]["output_tokens"] == 48
    assert trace_llm["llm_usage"]["total_tokens"] == 163
    assert "llm_calls" in trace_llm
    assert len(trace_llm["llm_calls"]) == 1

    trace_local = build_auditable_answer_trace(
        domain="career",
        intent="career_general",
        answer_source="LOCAL",
        matched_rules=[{"rule_id": "CAREER_10TH_LORD_PLACEMENT"}],
        theme_lineage=[{"planet": "Mercury", "theme": "Data Analytics"}],
        gemini_calls=0,
        llm_input_tokens=0,
        llm_output_tokens=0
    )

    assert trace_local["llm_usage"]["total_tokens"] == 0
    assert trace_local["llm_calls"] == []

    # 5. Test production_decision_trace.jsonl Recording
    print("--- TEST 5: production_decision_trace.jsonl Schema ---")
    rec = record_production_decision_trace(
        question="Will my salary increase after switching companies?",
        domain="career",
        selected_model="V3",
        domain_confidence=0.91,
        intent="career_change",
        resolved_intent="career_change",
        complexity="moderate",
        chart_required=True,
        chart_evidence=[],
        matched_rules=["CAREER_10TH_LORD_PLACEMENT"],
        evidence_score=1.0,
        evidence_status="complete",
        answer_source="PARTIAL_LOCAL_LLM",
        gemini_calls=1,
        llm_input_tokens=115,
        llm_output_tokens=48,
        llm_total_tokens=163,
        llm_call_details=[c1],
        latency_ms=842.5
    )

    assert rec["llm_input_tokens"] == 115
    assert rec["llm_output_tokens"] == 48
    assert rec["llm_total_tokens"] == 163
    assert "llm_call_details" in rec
    assert len(rec["llm_call_details"]) == 1

    # Verify JSONL log file contains valid JSON entry
    assert TRACE_LOG_FILE.exists()
    with open(TRACE_LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        last_entry = json.loads(lines[-1])
        assert last_entry["question"] == "Will my salary increase after switching companies?"
        assert last_entry["llm_total_tokens"] == 163

    print("\n[SUCCESS] ALL LLM TOKEN OBSERVABILITY & TRACE TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    test_token_observability()
