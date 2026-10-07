"""
LLM Token Logger & Terminal Observability Module
=================================================
Module: backend/reasoning/token_logger.py

Purpose:
Provides real-time terminal observability for LLM token consumption.
Extracts EXACT token counts from Gemini API usage metadata (`usage_metadata`).
Logs individual LLM calls, total request summaries, and 0-token reports for LOCAL / UNSUPPORTED queries.
"""

import sys
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def print_llm_call_report(
    call_number: int,
    purpose: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    evidence_sent: Optional[List[str]] = None,
    latency_ms: float = 0.0
) -> Dict[str, Any]:
    """
    Prints a clearly formatted token usage block for a single Gemini API call.
    Uses exact API token counts (never character estimations).
    """
    total_tokens = input_tokens + output_tokens
    evidence_str = ", ".join(evidence_sent) if evidence_sent else "Standard chart evidence package"

    report_lines = [
        "==================================================",
        "                 LLM TOKEN REPORT                 ",
        "==================================================",
        f"LLM Call #     : {call_number}",
        f"Purpose        : {purpose}",
        f"Model Used     : {model_name}",
        f"Input Tokens   : {input_tokens}",
        f"Output Tokens  : {output_tokens}",
        f"Total Tokens   : {total_tokens}",
        f"Evidence Sent  : {evidence_str}",
        f"Latency        : {latency_ms:.2f} ms",
        "=================================================="
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text + "\n", flush=True)

    return {
        "call_number": call_number,
        "purpose": purpose,
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "evidence_sent": evidence_sent or [],
        "latency_ms": round(latency_ms, 2)
    }


def print_request_total_report(call_details: List[Dict[str, Any]], total_latency_ms: float = 0.0) -> Dict[str, Any]:
    """
    Prints a summary of total LLM token usage across all calls in a single request.
    """
    total_calls = len(call_details)
    total_input = sum(c.get("input_tokens", 0) for c in call_details)
    total_output = sum(c.get("output_tokens", 0) for c in call_details)
    total_tokens = total_input + total_output

    if total_calls > 1:
        report_lines = [
            "==================================================",
            "               REQUEST TOTAL (LLM)               ",
            "==================================================",
            f"Total LLM Calls: {total_calls}",
            f"Input Tokens   : {total_input}",
            f"Output Tokens  : {total_output}",
            f"Total Tokens   : {total_tokens}",
            f"Total Latency  : {total_latency_ms:.2f} ms",
            "=================================================="
        ]
        print("\n" + "\n".join(report_lines) + "\n", flush=True)

    return {
        "gemini_calls": total_calls,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_tokens,
        "latency_ms": round(total_latency_ms, 2)
    }


def print_no_llm_call_report(answer_source: str, reason: str) -> Dict[str, Any]:
    """
    Prints an explicit 0-token report when Gemini is NOT called (LOCAL or UNSUPPORTED).
    """
    is_unsupported = (answer_source == "UNSUPPORTED")
    title = "              UNSUPPORTED QUERY (NO LLM)          " if is_unsupported else "                 LOCAL RESPONSE (NO LLM)          "

    report_lines = [
        "==================================================",
        title,
        "==================================================",
        f"Answer Source  : {answer_source}",
        f"LLM Calls      : 0",
        f"Input Tokens   : 0",
        f"Output Tokens  : 0",
        f"Total Tokens   : 0",
        f"Reason         : {reason}",
        "=================================================="
    ]
    print("\n" + "\n".join(report_lines) + "\n", flush=True)

    return {
        "gemini_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "answer_source": answer_source,
        "reason": reason
    }


def print_evidence_observability_report(
    question: str,
    domain: str,
    intent: str,
    answer_source: str,
    api_calls: int,
    local_rules: int,
    gemini_calls: int,
    input_tokens: int,
    output_tokens: int,
    required_evidence: List[str],
    available_evidence: List[str],
    missing_evidence: List[str],
    evidence_status: str,
    domain_match: bool = True,
    intent_match: bool = True,
    exact_rule_match: bool = True,
    rule_coverage_score: float = 1.0,
    answer_mode: Optional[str] = None
):
    """
    Prints a terminal cost observability trace for every request as specified in Prompt Section 12.
    """
    mode_label = answer_mode or ("DIRECT_API" if answer_source == "DIRECT_API" else ("LOCAL_SYNTHESIS" if answer_source in ["LOCAL", "RULE_BASED"] else ("LLM_ASSISTED" if answer_source in ["EVIDENCE_GROUNDED_LLM", "GEMINI_GROUNDED", "PARTIAL_LOCAL_LLM"] else answer_source)))
    src_label = "GEMINI_GROUNDED" if answer_source in ["EVIDENCE_GROUNDED_LLM", "PARTIAL_LOCAL_LLM"] else answer_source
    coverage_label = "COMPLETE" if exact_rule_match else "INSUFFICIENT"

    total_tokens = input_tokens + output_tokens

    report_lines = [
        "============================================================",
        f"QUESTION       : {question}",
        "",
        f"DOMAIN         : {domain}",
        f"INTENT         : {intent}",
        "",
        f"ANSWER MODE    : {mode_label}",
        f"ANSWER SOURCE  : {src_label}",
        "",
        f"LOCAL RULES    : {local_rules}",
        f"API CALLS      : {api_calls}",
        f"GEMINI CALLS   : {gemini_calls}",
        "",
        f"LLM INPUT      : {input_tokens} tokens",
        f"LLM OUTPUT     : {output_tokens} tokens",
        f"LLM TOTAL      : {total_tokens} tokens",
        "",
        f"EVIDENCE       : {evidence_status}",
        f"LOCAL COVERAGE : {coverage_label}",
        "============================================================"
    ]
    print("\n" + "\n".join(report_lines) + "\n", flush=True)


