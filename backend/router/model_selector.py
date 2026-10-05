"""
Evaluation-Driven Multi-Model Domain Selector
=============================================
Module: backend/router/model_selector.py

Responsibility:
Loads all 3 domain model generations (OLD, CONTEXT_V1, V3) independently with
their paired vectorizers. Evaluates incoming questions across multiple empirical
signals (validation reliability, class probability, margin separation, inter-model
agreement, domain-specific calibration, and query-level semantic evidence).

Strict Principles:
- Models & vectorizers are read-only (no retraining, no artifact mutation).
- Complexity and Intent classifiers remain separate downstream pipeline stages.
- Preserves raw model outputs in the evaluation payload for auditing.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "question_classifier"

DOMAIN_CLASSES = ["career", "finance", "health", "marriage", "other"]

# ------------------------------------------------------------------
# EMPIRICAL DOMAIN RELIABILITY MATRIX P(correct | model, domain)
# Derived from baseline evaluation dataset checks:
# - OLD: Strong career/finance knowledge, weak health/marriage recall
# - CONTEXT_V1: Strong marriage knowledge, weak health recall
# - V3: High overall accuracy, 1.00 health recall, 1.00 marriage recall
# ------------------------------------------------------------------
DOMAIN_RELIABILITY_SCORES: Dict[str, Dict[str, float]] = {
    "OLD": {
        "career": 0.85,
        "finance": 0.85,
        "health": 0.22,
        "marriage": 0.60,
        "other": 0.73,
    },
    "CONTEXT_V1": {
        "career": 0.70,
        "finance": 0.67,
        "health": 0.22,
        "marriage": 0.82,
        "other": 0.86,
    },
    "V3": {
        "career": 0.86,
        "finance": 0.75,
        "health": 1.00,
        "marriage": 0.93,
        "other": 0.86,
    },
}

# Key domain evidence vocabulary for auxiliary signal calculation
DOMAIN_SIGNALS: Dict[str, List[str]] = {
    "health": [
        "health", "tired", "exhausted", "energy", "sleep", "weak",
        "sleeping", "fatigue", "sick", "illness", "pain", "stress", "anxiety",
        "depression", "wellness", "recovery", "bimaar", "sehat", "swasthya",
        "tanav", "chinta", "bimari", "dawai", "neend", "thakaan", "stamina",
        "vitality", "wellbeing", "lethargic", "drained", "body pain"
    ],
    "career": [
        "career", "job", "work", "promotion", "business", "naukri", "naukari",
        "profession", "salary", "interview", "office", "boss", "resign",
        "employment", "workplace", "company", "hire", "hired", "vyapar",
        "kaam", "kaamkaj"
    ],
    "marriage": [
        "marriage", "married", "wedding", "relationship", "spouse", "partner",
        "crush", "shaadi", "love", "couple", "7th house", "vivah", "husband", "wife",
        "rishta", "patni", "pati", "prem"
    ],
    "finance": [
        "money", "finance", "wealth", "income", "paisa", "invest", "investment",
        "investments", "rich", "loan", "debt", "debts", "savings", "saving", "financial", "profit",
        "assets", "earning", "earnings", "dhan", "karz", "kamai", "rupaye",
        "lucrative", "monetary", "economic", "capital", "bonus", "funds"
    ],
    "other": [
        "nakshatra", "astrology", "birth chart", "kundali", "ascendant", "rashi", "graha"
    ]
}

_model_pool: Optional[Dict[str, Tuple[Any, Any]]] = None

def load_domain_model_pool() -> Dict[str, Tuple[Any, Any]]:
    """Load all 3 domain model & vectorizer pairs into memory safely."""
    global _model_pool
    if _model_pool is not None:
        return _model_pool

    _model_pool = {}
    pairs = [
        ("OLD", MODEL_DIR / "domain_model.pkl", MODEL_DIR / "domain_vectorizer.pkl"),
        ("CONTEXT_V1", MODEL_DIR / "domain_model_context_v1.pkl", MODEL_DIR / "domain_vectorizer_context_v1.pkl"),
        ("V3", MODEL_DIR / "domain_model_v3.pkl", MODEL_DIR / "domain_vectorizer_v3.pkl"),
    ]

    for name, m_path, v_path in pairs:
        if m_path.exists() and v_path.exists():
            try:
                m = joblib.load(m_path)
                v = joblib.load(v_path)
                _model_pool[name] = (m, v)
                logger.info(f"Loaded domain model pair '{name}' successfully.")
            except Exception as e:
                logger.error(f"Error loading domain pair '{name}': {e}")
        else:
            logger.warning(f"Missing file paths for domain pair '{name}'")

    return _model_pool

def predict_single_pair(model: Any, vectorizer: Any, question: str) -> Dict[str, Any]:
    """Execute prediction for a single model + vectorizer pair."""
    X = vectorizer.transform([question.strip()])
    has_vocab_match = (getattr(X, "nnz", 0) > 0)
    pred_label = str(model.predict(X)[0])

    probs_map = {cls: 0.0 for cls in DOMAIN_CLASSES}
    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probs = model.predict_proba(X)[0]
        for cls, prob in zip(list(model.classes_), probs):
            probs_map[str(cls)] = float(prob)

    sorted_probs = sorted(probs_map.items(), key=lambda x: x[1], reverse=True)
    top_prob = sorted_probs[0][1]
    second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    margin = top_prob - second_prob

    return {
        "domain": pred_label,
        "confidence": top_prob,
        "margin": margin,
        "probabilities": probs_map,
        "has_vocab_match": has_vocab_match
    }

def detect_query_evidence(question: str) -> Dict[str, float]:
    """Calculate query-level semantic keyword evidence weights per domain."""
    q_lower = question.lower().strip()
    evidence_scores = {d: 0.0 for d in DOMAIN_CLASSES}

    # Explicit OOD signal detection
    ood_kws = ["lottery", "sports match", "sports prediction", "past life", "gambling", "stock picking", "crypto prediction"]
    if any(kw in q_lower for kw in ood_kws):
        evidence_scores["other"] = 0.75
        return evidence_scores

    for domain, keywords in DOMAIN_SIGNALS.items():
        count = sum(1 for kw in keywords if kw in q_lower)
        if count > 0:
            evidence_scores[domain] = min(0.60, 0.25 * count)

    return evidence_scores

def evaluate_and_select_domain(question: str) -> Dict[str, Any]:
    """
    Evaluates predictions from OLD, CONTEXT_V1, and V3 using multi-signal scoring:
    Score(m, d) = Reliability(m, d) * Prob(m, d) * (1 + Margin) * (1 + AgreementBonus) * (1 + Evidence) * DomainMismatchPenalty
    """
    pool = load_domain_model_pool()
    if not pool:
        return {
            "model_predictions": {},
            "selected_domain": "other",
            "selected_model": "none",
            "selection_method": "fallback_empty_pool",
            "confidence": 0.0
        }

    raw_predictions: Dict[str, Dict[str, Any]] = {}
    for name, (model, vectorizer) in pool.items():
        raw_predictions[name] = predict_single_pair(model, vectorizer, question)

    # 1. Inter-model agreement check
    predicted_domains = [res["domain"] for res in raw_predictions.values()]
    unique_domains = set(predicted_domains)
    is_unanimous = len(unique_domains) == 1

    # 2. Query evidence calculation
    evidence = detect_query_evidence(question)

    # 3. Multi-signal scoring for candidate predictions across all domains
    scored_candidates = []
    for name, pred in raw_predictions.items():
        probs_map = pred.get("probabilities", {})
        top_domain = pred["domain"]

        for target_domain in DOMAIN_CLASSES:
            domain_prob = probs_map.get(target_domain, pred["confidence"] if target_domain == top_domain else 0.0)
            if domain_prob <= 0.01:
                continue

            # Signal A: Historical validation reliability P(correct | model, domain)
            reliability = DOMAIN_RELIABILITY_SCORES.get(name, {}).get(target_domain, 0.50)

            # Signal B: Agreement multiplier
            agreement_count = predicted_domains.count(target_domain)
            agreement_bonus = 0.20 if agreement_count >= 2 else 0.0
            if is_unanimous and target_domain == top_domain:
                agreement_bonus = 0.35

            # Signal C: Semantic evidence boost
            evidence_boost = evidence.get(target_domain, 0.0)

            # Signal D: Domain Mismatch Penalty
            has_own_evidence = (evidence_boost > 0)
            has_other_evidence = any(ev > 0 for d, ev in evidence.items() if d != target_domain)
            mismatch_penalty = 0.25 if (not has_own_evidence and has_other_evidence) else 1.0

            # Signal E: Combined Score calculation
            composite_score = (
                reliability *
                domain_prob *
                (1.0 + min(0.5, pred["margin"])) *
                (1.0 + agreement_bonus) *
                (1.0 + (2.0 * evidence_boost)) *
                mismatch_penalty
            )

            scored_candidates.append({
                "model": name,
                "domain": target_domain,
                "score": composite_score,
                "raw_confidence": domain_prob,
                "margin": pred["margin"],
                "reliability": reliability,
                "agreement_count": agreement_count,
                "evidence_boost": evidence_boost,
                "mismatch_penalty": mismatch_penalty
            })

    # Sort scored candidates by composite score descending
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    winner = scored_candidates[0]

    # Check if a domain with strong evidence was penalized due to model misclassification
    strongest_evidence_domain = max(evidence.items(), key=lambda x: x[1])[0] if any(v > 0 for v in evidence.values()) else None
    max_ev = evidence.get(strongest_evidence_domain, 0.0) if strongest_evidence_domain else 0.0

    if strongest_evidence_domain and max_ev >= 0.25 and winner["domain"] != strongest_evidence_domain and winner["raw_confidence"] < 0.75:
        # Override winner domain with strongest evidence domain
        selected_domain = strongest_evidence_domain
        selected_model = winner["model"]
        selection_method = f"semantic_evidence_override_{selected_domain}"
        final_conf = max_ev
    else:
        selected_domain = winner["domain"]
        selected_model = winner["model"]
        final_conf = winner["raw_confidence"]
        if is_unanimous:
            selection_method = f"unanimous_agreement_{selected_model}"
        else:
            selection_method = f"evaluation_weighted_selector_{selected_model}"

    # 4. Out-of-distribution / Uncertainty Abstention Check
    has_any_vocab = any(res.get("has_vocab_match", False) for res in raw_predictions.values())
    has_any_evidence = any(ev > 0 for ev in evidence.values())
    max_raw_conf = max(res["confidence"] for res in raw_predictions.values())
    word_count = len(question.strip().split())

    if not has_any_vocab and not has_any_evidence and (max_raw_conf < 0.35 or word_count < 3):
        selected_domain = "other"
        selection_method = "abstain_due_to_zero_vocabulary_match"
        selected_model = "none"
        final_conf = 0.0
    elif len(unique_domains) == 3 and max_raw_conf < 0.35 and winner["score"] < 0.30:
        selected_domain = "uncertain"
        selection_method = "abstain_due_to_high_disagreement"
        selected_model = "none"
        final_conf = max_raw_conf
    else:
        selected_domain = winner["domain"]
        selected_model = winner["model"]
        final_conf = winner["raw_confidence"]
        if is_unanimous:
            selection_method = f"unanimous_agreement_{selected_model}"
        else:
            selection_method = f"evaluation_weighted_selector_{selected_model}"

    return {
        "model_predictions": raw_predictions,
        "selected_domain": selected_domain,
        "selected_model": selected_model,
        "selection_method": selection_method,
        "confidence": final_conf,
        "score_details": scored_candidates
    }
