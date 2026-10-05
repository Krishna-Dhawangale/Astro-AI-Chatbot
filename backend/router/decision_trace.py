"""
Production Decision Trace Logger
=================================
Module: backend/router/decision_trace.py

Purpose:
Logs complete production decision traces for every user request,
recording classification model decisions, intent overlays, Stage 8 rules,
evidence scores, final routing sources, and Gemini call counts.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

TRACE_LOG_FILE = LOG_DIR / "production_decision_trace.jsonl"


def record_production_decision_trace(
    question: str,
    domain: str,
    selected_model: str,
    domain_confidence: float,
    intent: str,
    resolved_intent: str,
    complexity: str,
    chart_required: bool,
    chart_evidence: List[Dict[str, Any]],
    matched_rules: List[str],
    evidence_score: float,
    evidence_status: str,
    answer_source: str,
    gemini_calls: int,
    domain_margin: float = 0.0,
    llm_input_tokens: int = 0,
    llm_output_tokens: int = 0,
    latency_ms: float = 0.0,
    errors: Optional[List[str]] = None,
    local_answer_quality: Optional[Dict[str, bool]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Constructs and records an explicit Production Decision Trace entry for Phase 19 observation.
    """
    trace_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "domain": domain,
        "selected_model": selected_model,
        "domain_confidence": round(domain_confidence, 4),
        "domain_margin": round(domain_margin, 4),
        "intent": intent,
        "resolved_intent": resolved_intent,
        "complexity": complexity,
        "chart_required": chart_required,
        "chart_evidence": chart_evidence,
        "matched_rules": matched_rules,
        "evidence_score": round(evidence_score, 2),
        "evidence_status": evidence_status,
        "answer_source": answer_source,
        "gemini_calls": gemini_calls,
        "llm_input_tokens": llm_input_tokens,
        "llm_output_tokens": llm_output_tokens,
        "latency_ms": round(latency_ms, 2),
        "errors": errors or [],
        "local_answer_quality": local_answer_quality or {
            "correct_domain": True,
            "correct_intent": True,
            "correct_chart_data": True,
            "correct_rule": True,
            "correct_interpretation": True,
            "no_unsupported_claim": True,
            "addresses_question": True
        },
        "metadata": additional_metadata or {}
    }

    try:
        with open(TRACE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_record) + "\n")
    except Exception as e:
        logging.error(f"Failed to record production decision trace: {e}")

    return trace_record
