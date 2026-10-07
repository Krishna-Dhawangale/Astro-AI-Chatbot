"""
Mode Selector (backend/reasoning/mode_selector.py)
---------------------------------------------------
Evaluates intent, chart evidence availability, rule coverage, and domain boundaries
to assign one of 5 standardized answer sources:

1. LOCAL                 : Exact deterministic rule exists and evidence complete (0 LLM calls)
2. PARTIAL_LOCAL_LLM     : Core facts answered locally + LLM for creative/poetic rendering (1 LLM call)
3. EVIDENCE_GROUNDED_LLM : Valid domain question, chart evidence complete, but local rule coverage insufficient (1 LLM call)
4. UNSUPPORTED           : Inherently unsupported deterministic claim / out-of-domain (0 LLM calls)
5. UNRESOLVED            : Required chart/factual evidence is missing and cannot be obtained (0 LLM calls)
"""

from typing import Dict, Any, List, Optional, Tuple
from backend.reasoning.direct_fact_engine import is_direct_fact_query
from backend.reasoning.rule_coverage import evaluate_rule_coverage, is_unsupported_boundary_query

ANSWER_SOURCES = {
    "LOCAL",
    "PARTIAL_LOCAL_LLM",
    "EVIDENCE_GROUNDED_LLM",
    "UNSUPPORTED",
    "UNRESOLVED"
}

UNSUPPORTED_INTENTS = {
    "past_life",
    "lottery_winning",
    "sports_prediction",
    "stock_picking",
    "medical_diagnosis",
    "medical_treatment",
    "medical_emergency",
    "gambling"
}

SUPPORTED_DOMAINS = {
    "career",
    "marriage",
    "finance",
    "education",
    "property",
    "health"
}

LLM_FALLBACK_KEYWORDS = [
    "philosophical",
    "psychological",
    "poetic",
    "storytelling",
    "creative interpretation",
    "life story",
    "poem",
    "write a poem",
    "short story",
    "creative story",
    "inspirational message"
]


def select_answer_mode(
    domain: str,
    intent: str,
    question: str,
    matched_rules: List[Dict[str, Any]] = None,
    is_faq: bool = False,
    structured_evidence: Any = None,
    evidence_status: str = "COMPLETE"
) -> Dict[str, Any]:
    """
    Master Mode Selector implementing Section 3 Routing Decision Hierarchy:

    if boundary_detected:
        answer_source = "UNSUPPORTED"
    elif evidence_status == "UNRESOLVED":
        answer_source = "UNRESOLVED"
    elif exact_rule_match and required_evidence_complete:
        answer_source = "LOCAL"
    elif valid_domain_question and required_evidence_complete:
        answer_source = "EVIDENCE_GROUNDED_LLM"
    else:
        answer_source = "UNRESOLVED"
    """
    q_lower = (question or "").lower().strip()
    target_dom = (domain or "general").lower()
    target_intent = (intent or "general").lower()

    # 1. Tier 1 — UNSUPPORTED BOUNDARY
    boundary_detected = (
        target_intent in UNSUPPORTED_INTENTS
        or is_unsupported_boundary_query(question)
    )
    if boundary_detected:
        return {
            "mode": "UNSUPPORTED",
            "answer_source": "UNSUPPORTED",
            "reason": "unsupported_or_out_of_domain_intent",
            "domain_match": False,
            "intent_match": False,
            "exact_rule_match": False,
            "rule_coverage_score": 0.0,
            "evidence_complete": False,
            "gemini_calls": 0,
            "llm_tokens": 0,
            "local_answered": False,
            "llm_answered": False,
            "evidence_ids": [],
            "fallback_reason": "Query requests insights outside Vedic astrology chart boundaries."
        }

    # 2. Tier 4 — MISSING EVIDENCE (UNRESOLVED)
    is_sufficient_ev = True
    if structured_evidence is not None:
        if hasattr(structured_evidence, "is_sufficient"):
            is_sufficient_ev = structured_evidence.is_sufficient()
        elif isinstance(structured_evidence, dict):
            is_sufficient_ev = structured_evidence.get("is_sufficient", True)

    if evidence_status == "UNRESOLVED" or (structured_evidence and not is_sufficient_ev and not matched_rules):
        return {
            "mode": "UNRESOLVED",
            "answer_source": "UNRESOLVED",
            "reason": "required_evidence_missing",
            "domain_match": True,
            "intent_match": True,
            "exact_rule_match": False,
            "rule_coverage_score": 0.0,
            "evidence_complete": False,
            "gemini_calls": 0,
            "llm_tokens": 0,
            "local_answered": False,
            "llm_answered": False,
            "evidence_ids": [],
            "fallback_reason": "Required natal chart evidence is missing."
        }

    # 3. Direct Fact Path (HIGHEST PRIORITY FACT PATH)
    if is_faq or is_direct_fact_query(question):
        return {
            "mode": "DIRECT",
            "answer_source": "LOCAL",
            "reason": "direct_factual_lookup",
            "domain_match": True,
            "intent_match": True,
            "exact_rule_match": True,
            "rule_coverage_score": 1.0,
            "evidence_complete": True,
            "gemini_calls": 0,
            "llm_tokens": 0,
            "local_answered": True,
            "llm_answered": False,
            "evidence_ids": ["DIRECT_FACT_LOOKUP"],
            "fallback_reason": None
        }

    # 4. Creative Request (Partial Local + Gemini)
    has_creative_req = any(k in q_lower for k in LLM_FALLBACK_KEYWORDS)
    if has_creative_req:
        return {
            "mode": "LLM_ASSISTED",
            "answer_source": "PARTIAL_LOCAL_LLM",
            "reason": "local_core_matched_with_creative_llm_rendering",
            "domain_match": True,
            "intent_match": True,
            "exact_rule_match": False,
            "rule_coverage_score": 0.5,
            "evidence_complete": True,
            "gemini_calls": 1,
            "llm_tokens": 95,
            "local_answered": True,
            "llm_answered": True,
            "evidence_ids": [r.get("rule_id", "") for r in (matched_rules or []) if r.get("matched", True)],
            "fallback_reason": "Local rules answered core facts; Gemini invoked for creative/poetic rendering."
        }

    # 5. Evaluate Rule Coverage (Tier 2 vs Tier 3)
    cov_eval = evaluate_rule_coverage(
        question=question,
        domain=domain,
        intent=intent,
        structured_evidence=structured_evidence,
        matched_rules=matched_rules
    )

    exact_rule_match = cov_eval.get("exact_rule_match", False)
    req_ev_complete = cov_eval.get("required_evidence_complete", True)
    domain_match = cov_eval.get("domain_match", True)

    # Tier 2 — LOCAL DETERMINISTIC ANSWER
    if exact_rule_match and req_ev_complete:
        return {
            "mode": "RULE_BASED",
            "answer_source": "LOCAL",
            "reason": "exact_deterministic_rule_coverage_available",
            "domain_match": True,
            "intent_match": True,
            "exact_rule_match": True,
            "rule_coverage_score": 1.0,
            "evidence_complete": True,
            "gemini_calls": 0,
            "llm_tokens": 0,
            "local_answered": True,
            "llm_answered": False,
            "evidence_ids": cov_eval.get("matching_rules", ["EXACT_LOCAL_RULE"]),
            "fallback_reason": None
        }

    # Tier 3 — EVIDENCE-GROUNDED LLM
    if domain_match and req_ev_complete:
        return {
            "mode": "LLM_ASSISTED",
            "answer_source": "EVIDENCE_GROUNDED_LLM",
            "reason": "valid_domain_question_insufficient_local_rule_coverage",
            "domain_match": True,
            "intent_match": cov_eval.get("intent_match", True),
            "exact_rule_match": False,
            "rule_coverage_score": 0.0,
            "evidence_complete": True,
            "gemini_calls": 1,
            "llm_tokens": 140,
            "local_answered": False,
            "llm_answered": True,
            "evidence_ids": cov_eval.get("matching_rules", []),
            "fallback_reason": cov_eval.get("reason", "Valid domain question requiring evidence-grounded LLM synthesis.")
        }

    # Default Tier 4 — UNRESOLVED
    return {
        "mode": "UNRESOLVED",
        "answer_source": "UNRESOLVED",
        "reason": "unresolved_or_unmatched_query",
        "domain_match": False,
        "intent_match": False,
        "exact_rule_match": False,
        "rule_coverage_score": 0.0,
        "evidence_complete": False,
        "gemini_calls": 0,
        "llm_tokens": 0,
        "local_answered": False,
        "llm_answered": False,
        "evidence_ids": [],
        "fallback_reason": "Query cannot be safely resolved from available evidence or rules."
    }
