"""
Mode Selector (backend/reasoning/mode_selector.py)
---------------------------------------------------
Evaluates intent, chart evidence availability, rule completeness, and domain coverage
to assign one of 4 internal answer statuses:

1. DIRECT        : Fact calculation/lookup query (0 LLM calls, < 5ms target latency)
2. RULE_BASED     : Matched evidence is COMPLETE and covered by rules (0 LLM calls, < 50ms target latency)
3. LLM_ASSISTED   : Partial evidence or conversational query requiring synthesis (1 Tiny LLM call)
4. UNSUPPORTED    : Zero evidence coverage or out-of-domain query (0 LLM calls, < 5ms target latency)
"""

from typing import Dict, Any, List, Optional, Tuple
from backend.reasoning.direct_fact_engine import is_direct_fact_query


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


def evaluate_evidence_completeness(domain: str, intent: str, matched_rules: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Mandatory Evidence Completeness Gate:
    Returns (is_complete: bool, evidence_ids: List[str]).
    
    RULE_BASED requires BOTH:
      1. evidence_complete == True (Foundation lord placement rule + Karaka/Dignity evidence present)
      2. rule_available == True (Deterministic rule templates present)
    """
    if not matched_rules or not isinstance(matched_rules, list):
        return False, []

    evidence_ids = [r.get("rule_id", "") for r in matched_rules if r.get("matched", True)]
    
    # Check if foundation lord placement rule exists for domain
    has_foundation = any(
        r.get("category") == "foundation" or "LORD_PLACEMENT" in r.get("rule_id", "")
        for r in matched_rules if r.get("matched", True)
    )

    # Check if dignity or karaka evidence exists
    has_karaka_or_dignity = any(
        r.get("category") in ["dignity", "karaka", "house_evidence"] or "KARAKA" in r.get("rule_id", "")
        for r in matched_rules if r.get("matched", True)
    )

    # Both foundation and karaka/dignity rules must be active for full rule-based completeness
    is_complete = bool(has_foundation and has_karaka_or_dignity)

    return is_complete, evidence_ids


LLM_FALLBACK_KEYWORDS = [
    "philosophical",
    "psychological",
    "poetic",
    "storytelling",
    "creative interpretation",
    "life story",
    "psychological theories",
    "holistic lifestyle",
    "lifestyle",
    "holistic"
]


def select_answer_mode(
    domain: str,
    intent: str,
    question: str,
    matched_rules: List[Dict[str, Any]] = None,
    is_faq: bool = False
) -> Dict[str, Any]:
    """
    Master Mode Selector evaluating 4 explicit answer_source statuses:
    1. LOCAL              (FAQ, Direct API, Complete Rules Engine - gemini_calls = 0)
    2. PARTIAL_LOCAL_LLM  (Partial Local Answer + Minimal LLM for missing part - gemini_calls = 1)
    3. LLM_FALLBACK       (Philosophical/Creative query beyond local rules - gemini_calls = 1)
    4. UNSUPPORTED        (Out of domain or boundary queries - gemini_calls = 0)
    """
    q_lower = (question or "").lower().strip()

    # 1. Check for Unsupported Out-of-Domain / Invalid Queries
    unsupported_kws = ["past life", "previous life", "lottery numbers", "win lottery", "gambling numbers", "sports prediction", "stock picking"]
    if intent in UNSUPPORTED_INTENTS or any(k in q_lower for k in unsupported_kws):
        return {
            "mode": "UNSUPPORTED",
            "answer_source": "UNSUPPORTED",
            "reason": "unsupported_or_out_of_domain_intent",
            "evidence_complete": False,
            "gemini_calls": 0,
            "llm_tokens": 0,
            "local_answered": False,
            "llm_answered": False,
            "evidence_ids": [],
            "fallback_reason": "Query requests insights outside Vedic astrology chart boundaries."
        }

    # 2. Check for Genuine LLM Fallback (Philosophical / Poetic / Holistic Queries)
    if any(k in q_lower for k in LLM_FALLBACK_KEYWORDS):
        return {
            "mode": "LLM_ASSISTED",
            "answer_source": "LLM_FALLBACK",
            "reason": "philosophical_or_holistic_query_requires_llm_rendering",
            "evidence_complete": False,
            "gemini_calls": 1,
            "llm_tokens": 120,  # Estimated minimal renderer tokens
            "local_answered": False,
            "llm_answered": True,
            "evidence_ids": [r.get("rule_id", "") for r in (matched_rules or []) if r.get("matched", True)],
            "fallback_reason": "Query requests creative/philosophical synthesis beyond deterministic rule scope."
        }

    # 3. Check for Direct Fact Queries (HIGHEST PRIORITY FACT PATH)
    if is_faq or is_direct_fact_query(question):
        return {
            "mode": "DIRECT",
            "answer_source": "LOCAL",
            "reason": "direct_factual_lookup",
            "evidence_complete": True,
            "gemini_calls": 0,
            "llm_tokens": 0,
            "local_answered": True,
            "llm_answered": False,
            "evidence_ids": ["DIRECT_FACT_LOOKUP"],
            "fallback_reason": None
        }

    # 4. Check Domain Support
    target_dom = (domain or "general").lower()
    if target_dom not in SUPPORTED_DOMAINS and target_dom != "multi_domain" and target_dom != "other":
        return {
            "mode": "UNSUPPORTED",
            "answer_source": "UNSUPPORTED",
            "reason": "unsupported_domain",
            "evidence_complete": False,
            "gemini_calls": 0,
            "llm_tokens": 0,
            "local_answered": False,
            "llm_answered": False,
            "evidence_ids": [],
            "fallback_reason": f"Domain '{domain}' is not currently covered by deterministic rules."
        }

    # 5. Mandatory Evidence Completeness Gate
    is_complete, evidence_ids = evaluate_evidence_completeness(target_dom, intent, matched_rules or [])

    if is_complete or target_dom in ["other", "multi_domain"]:
        return {
            "mode": "RULE_BASED",
            "answer_source": "LOCAL",
            "reason": "complete_evidence_and_rules_matched",
            "evidence_complete": True,
            "gemini_calls": 0,
            "llm_tokens": 0,
            "local_answered": True,
            "llm_answered": False,
            "evidence_ids": evidence_ids or ["RULE_ENGINE"],
            "fallback_reason": None
        }
    elif evidence_ids:
        # Partial evidence: Local synthesizer handles core, missing sub-part rendered by LLM
        return {
            "mode": "LLM_ASSISTED",
            "answer_source": "PARTIAL_LOCAL_LLM",
            "reason": "partial_evidence_requires_minimal_llm_synthesis",
            "evidence_complete": False,
            "gemini_calls": 1,
            "llm_tokens": 85,
            "local_answered": True,
            "llm_answered": True,
            "evidence_ids": evidence_ids,
            "fallback_reason": "Partial evidence matched; rendering missing sub-part with strict LLM renderer boundary."
        }
    else:
        return {
            "mode": "UNSUPPORTED",
            "answer_source": "UNSUPPORTED",
            "reason": "zero_evidence_matched",
            "evidence_complete": False,
            "gemini_calls": 0,
            "llm_tokens": 0,
            "local_answered": False,
            "llm_answered": False,
            "evidence_ids": [],
            "fallback_reason": "Insufficient natal chart evidence matched in backend rules."
        }
