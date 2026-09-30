"""
Stage 8.27 — Interpretation Audit Trail Logger
==============================================
Module: backend/reasoning/audit_trail.py

Purpose:
Generates structured audit logs (JSON & Markdown) for every executed query,
recording complete evidence traceability from raw question down to final interpretation.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

AUDIT_LOG_FILE = LOG_DIR / "interpretation_audit_trail.jsonl"


def record_interpretation_audit(
    question: str,
    domain: str,
    intent: str,
    chart_id: str,
    matched_rules: List[Dict[str, Any]],
    rule_weights: List[Dict[str, Any]],
    evidence_score: float,
    evidence_status: str,
    dasha_summary: List[Dict[str, Any]],
    transit_summary: List[Dict[str, Any]],
    final_interpretation: str,
    gemini_calls: int
) -> Dict[str, Any]:
    """
    Creates and records a complete interpretation audit trail entry.
    """
    audit_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "domain": domain,
        "intent": intent,
        "chart_id": chart_id,
        "matched_rules": [r.get("rule_id") for r in matched_rules],
        "matched_rule_count": len(matched_rules),
        "rule_weights": [
            {"rule_id": w["rule_id"], "weight": w["weight"], "category": w["category"]}
            for w in rule_weights
        ],
        "evidence_score": round(evidence_score, 2),
        "evidence_status": evidence_status,
        "dasha": dasha_summary,
        "transits": transit_summary,
        "final_interpretation_length": len(final_interpretation),
        "final_interpretation": final_interpretation,
        "gemini_calls": gemini_calls
    }

    # Append to JSONL audit log file
    try:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_record) + "\n")
    except Exception as e:
        logging.error(f"Failed to append interpretation audit log: {e}")

    return audit_record
