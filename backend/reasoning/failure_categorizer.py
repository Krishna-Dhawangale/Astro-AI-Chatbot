"""
Failure Categorizer (backend/reasoning/failure_categorizer.py)
================================================================
Phase 23 — Automatic Failure Categorization Loop

Categorizes production execution traces into 11 precise categories:
1. PASS — High-quality, production-ready output
2. ROUTING_ERROR — Model selector misclassified domain
3. INTENT_ERROR — Intent router misclassified question intent
4. MISSING_EVIDENCE — Required chart evidence absent
5. WRONG_RULE — Stage 8 rule mismatched or missing
6. INTERPRETATION_ERROR — Flawed local text grounding
7. UNSUPPORTED_CLAIM — Unfounded claim or hallucination detected
8. INCOMPLETE_ANSWER — Output failed to address user query
9. UNNECESSARY_GEMINI — Gemini called on fully locally answerable query
10. EXCESSIVE_GEMINI_PAYLOAD — Gemini input token payload exceeds 200 tokens
11. UNSUPPORTED_QUERY_ERROR — Failed to enforce OOD / unsupported query boundary
"""

from typing import Dict, Any, Optional


FAILURE_CATEGORIES = [
    "PASS",
    "ROUTING_ERROR",
    "INTENT_ERROR",
    "MISSING_EVIDENCE",
    "WRONG_RULE",
    "INTERPRETATION_ERROR",
    "UNSUPPORTED_CLAIM",
    "INCOMPLETE_ANSWER",
    "UNNECESSARY_GEMINI",
    "EXCESSIVE_GEMINI_PAYLOAD",
    "UNSUPPORTED_QUERY_ERROR",
]


def categorize_answer_failure(
    question: str,
    predicted_domain: str,
    expected_domain: Optional[str],
    predicted_intent: str,
    expected_intent: Optional[str],
    answer_source: str,
    expected_source: Optional[str],
    gemini_calls: int,
    llm_tokens: int,
    actual_evidence: list,
    matched_rules: list,
    answer_text: str,
    quality_score: int,
    is_ood: bool = False
) -> Dict[str, Any]:
    """
    Evaluates an answer execution trace against deterministic quality rules
    and returns a failure category payload.
    """
    category = "PASS"
    reason = "Answer meets all quality and architectural standards."

    # Rule 11: OOD / Unsupported Boundary Enforcement
    if is_ood and answer_source != "UNSUPPORTED":
        category = "UNSUPPORTED_QUERY_ERROR"
        reason = "System attempted chart prediction on out-of-domain / unsupported question."

    # Rule 2: Domain Routing Error
    elif expected_domain and predicted_domain != expected_domain:
        category = "ROUTING_ERROR"
        reason = f"Domain mismatch: predicted '{predicted_domain}', expected '{expected_domain}'."

    # Rule 3: Intent Classification Error
    elif expected_intent and predicted_intent != expected_intent:
        category = "INTENT_ERROR"
        reason = f"Intent mismatch: predicted '{predicted_intent}', expected '{expected_intent}'."

    # Rule 9: Unnecessary Gemini Call
    elif expected_source == "LOCAL" and gemini_calls > 0:
        category = "UNNECESSARY_GEMINI"
        reason = "Gemini was called for a query marked as 100% locally answerable."

    # Rule 10: Excessive Gemini Payload
    elif gemini_calls > 0 and llm_tokens > 200:
        category = "EXCESSIVE_GEMINI_PAYLOAD"
        reason = f"Gemini payload ({llm_tokens} tokens) exceeded the 200 token budget limit."

    # Rule 4: Missing Evidence
    elif answer_source in ("LOCAL", "PARTIAL_LOCAL_LLM") and not actual_evidence:
        category = "MISSING_EVIDENCE"
        reason = "No actual chart evidence was passed to answer synthesizer."

    # Rule 5: Wrong Rule Matching
    elif answer_source in ("LOCAL", "PARTIAL_LOCAL_LLM") and not matched_rules:
        category = "WRONG_RULE"
        reason = "No Stage 8 rules matched for chart interpretation."

    # Rule 7: Unsupported Claim / Hallucination
    elif "You will definitely" in answer_text or "Guaranteed" in answer_text:
        category = "UNSUPPORTED_CLAIM"
        reason = "Prescriptive / deterministic absolute statement detected in text output."

    # Rule 8: Incomplete Answer
    elif len(answer_text.strip()) < 15 or quality_score < 70:
        category = "INCOMPLETE_ANSWER"
        reason = f"Answer output is incomplete or has low quality score ({quality_score}/100)."

    # Rule 6: Interpretation Error
    elif quality_score < 90:
        category = "INTERPRETATION_ERROR"
        reason = f"Sub-optimal interpretation quality score: {quality_score}/100."

    return {
        "category": category,
        "is_pass": category == "PASS",
        "reason": reason,
        "quality_score": quality_score
    }
