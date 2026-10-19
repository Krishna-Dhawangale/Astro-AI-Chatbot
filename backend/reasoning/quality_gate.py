"""
Quality Gate & Gemini Fallback Policy (backend/reasoning/quality_gate.py)
========================================================================
Pillar 6: Authoritative Quality Gate & Deterministic Decision Point

Evaluates synthesized answers against 5 core criteria:
1. Required Entities: Ascendant, Moon sign, house, or planetary references present when required.
2. Domain Coverage: All requested domains covered (especially in multi-domain queries).
3. Evidence Completeness: Verified chart/Stage 8 facts backing the answer.
4. Question Responsiveness: Output directly answers the user's specific question type.
5. Unsupported Claim & Safety Check: Zero deterministic guarantees, fabricated dates, or clinical claims.

Separates failure modes cleanly:
- Evidence Incomplete: Returns qualified limitation without calling Gemini or inventing facts.
- Evidence Sufficient but Wording Inadequate: Allows evidence-grounded Gemini synthesis.
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class QualityGateResult:
    is_pass: bool
    score: int
    has_required_entities: bool
    has_domain_coverage: bool
    is_evidence_complete: bool
    addresses_question: bool
    has_no_unsupported_claims: bool
    failure_reason: Optional[str] = None
    recommended_action: str = "SERVE_LOCAL"  # SERVE_LOCAL, SERVE_UNRESOLVED, FALLBACK_GEMINI, SERVE_UNSUPPORTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_pass": self.is_pass,
            "quality_score": self.score,
            "has_required_entities": self.has_required_entities,
            "has_domain_coverage": self.has_domain_coverage,
            "is_evidence_complete": self.is_evidence_complete,
            "addresses_question": self.addresses_question,
            "has_no_unsupported_claims": self.has_no_unsupported_claims,
            "failure_reason": self.failure_reason,
            "recommended_action": self.recommended_action,
        }


def evaluate_answer_quality(
    question: str,
    domain: str,
    intent: str,
    answer_text: str,
    evidence: Optional[Any] = None,
    matched_rules: Optional[List[Dict[str, Any]]] = None,
    detected_domains: Optional[List[str]] = None,
    is_creative_request: bool = False
) -> QualityGateResult:
    """
    Authoritative Quality Gate evaluating a candidate local answer.
    """
    q_low = (question or "").lower().strip()
    ans_low = (answer_text or "").lower().strip()

    # 1. Unsupported Deterministic Claims & Absolute Guarantees Check
    unsupported_patterns = [
        r"\bguaranteed\b",
        r"\bwill definitely\b",
        r"\b100% certainty\b",
        r"\bexact date\b",
        r"\bcertain to happen\b",
    ]
    has_unsupported_claims = any(re.search(p, ans_low) for p in unsupported_patterns)

    # 2. Check Evidence Completeness
    evidence_complete = True
    if evidence is not None:
        if hasattr(evidence, "is_sufficient"):
            evidence_complete = evidence.is_sufficient()
        elif isinstance(evidence, dict):
            evidence_complete = evidence.get("is_sufficient", True) and not evidence.get("missing_evidence")
    elif matched_rules is not None:
        evidence_complete = len(matched_rules) > 0

    # 3. Required Entities Check
    # For chart-based answers, verify chart entities or lord/sign/dasha references exist
    has_required_entities = True
    if len(answer_text) > 30 and "✨" not in answer_text:
        entity_keywords = [
            "lagna", "ascendant", "moon", "rashi", "sun", "house", "lord",
            "dasha", "transit", "planet", "indicator", "chart"
        ]
        has_required_entities = any(k in ans_low for k in entity_keywords)

    # 4. Domain Coverage Check
    # For multi-domain queries, verify EVERY requested domain has a presence in the answer
    has_domain_coverage = True
    domains_to_check = detected_domains or ([domain] if domain not in ("general", "multi_domain") else [])
    for d in domains_to_check:
        d_clean = d.lower().strip()
        if d_clean == "career" and not any(k in ans_low for k in ["career", "job", "work", "profession", "10th"]):
            has_domain_coverage = False
        elif d_clean in ["finance", "wealth"] and not any(k in ans_low for k in ["finance", "wealth", "income", "money", "invest", "2nd", "11th"]):
            has_domain_coverage = False
        elif d_clean == "marriage" and not any(k in ans_low for k in ["marriage", "relationship", "partner", "spouse", "7th"]):
            has_domain_coverage = False
        elif d_clean == "education" and not any(k in ans_low for k in ["education", "study", "academic", "learning", "degree", "5th"]):
            has_domain_coverage = False
        elif d_clean == "property" and not any(k in ans_low for k in ["property", "asset", "real estate", "house", "vehicle", "4th"]):
            has_domain_coverage = False

    # 5. Question Responsiveness Check
    addresses_question = True
    # If comparison: check that both choices or structured options are mentioned
    if any(k in q_low for k in ["job vs business", "business vs job", "job or business", "vs", "versus"]):
        if not ("job" in ans_low and "business" in ans_low):
            addresses_question = False
    # If timing: check that dasha or timing window or explicit horizon is addressed
    elif any(k in q_low for k in ["when", "kab", "timing", "which year", "which month"]):
        if not any(k in ans_low for k in ["dasha", "window", "timing", "period", "alignment", "cannot be confirmed"]):
            addresses_question = False

    # Scoring & Action Recommendation
    score = 100
    failure_reasons = []

    if has_unsupported_claims:
        score -= 30
        failure_reasons.append("Contains unsupported deterministic guarantee.")
    if not has_domain_coverage:
        score -= 25
        failure_reasons.append(f"Missing full domain coverage for requested domains ({domains_to_check}).")
    if not has_required_entities:
        score -= 15
        failure_reasons.append("Missing required astrological chart entities.")
    if not addresses_question:
        score -= 20
        failure_reasons.append("Answer does not fully address the specific question type.")
    if not evidence_complete:
        score -= 20
        failure_reasons.append("Underlying chart evidence is incomplete.")

    is_pass = score >= 75 and not has_unsupported_claims

    # Separate failure modes:
    # 1. Creative request where user explicitly asked for poem/story -> Gemini synthesis
    if is_creative_request:
        action = "FALLBACK_GEMINI"
        reason = "User explicitly requested creative/poetic interpretation."
    # 2. Local answer passed quality gate -> Serve local
    elif is_pass and evidence_complete:
        action = "SERVE_LOCAL"
        reason = "Local answer passed all quality gate checks."
    # 3. Evidence is incomplete -> Return unresolved without calling Gemini or inventing facts
    elif not evidence_complete:
        action = "SERVE_UNRESOLVED"
        reason = "Chart evidence incomplete; returning qualified limitation without LLM hallucination."
    # 4. Evidence is complete but local wording is inadequate for complex custom question -> Gemini fallback
    elif evidence_complete and not is_pass:
        action = "FALLBACK_GEMINI"
        reason = "Evidence is complete but query requires advanced language synthesis; routing to evidence-grounded Gemini."
    else:
        action = "SERVE_UNRESOLVED"
        reason = "Query cannot be safely resolved from available evidence."

    return QualityGateResult(
        is_pass=is_pass,
        score=max(0, score),
        has_required_entities=has_required_entities,
        has_domain_coverage=has_domain_coverage,
        is_evidence_complete=evidence_complete,
        addresses_question=addresses_question,
        has_no_unsupported_claims=not has_unsupported_claims,
        failure_reason="; ".join(failure_reasons) if failure_reasons else None,
        recommended_action=action
    )
