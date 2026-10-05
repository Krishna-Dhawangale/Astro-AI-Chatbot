from pathlib import Path
import joblib
from typing import Dict, Any, Tuple

from .predictor import predict_with_confidence


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "question_classifier"

INTENT_MODEL_PATH = MODEL_DIR / "intent_model.pkl"
INTENT_VECTORIZER_PATH = MODEL_DIR / "intent_vectorizer.pkl"

_intent_model = None
_intent_vectorizer = None


def get_intent_model():
    global _intent_model, _intent_vectorizer
    if _intent_model is None:
        if INTENT_MODEL_PATH.exists() and INTENT_VECTORIZER_PATH.exists():
            _intent_model = joblib.load(INTENT_MODEL_PATH)
            _intent_vectorizer = joblib.load(INTENT_VECTORIZER_PATH)
    return _intent_model, _intent_vectorizer


def resolve_intent_overlay(question: str, raw_intent: str, raw_confidence: float, probabilities: Dict[str, float]) -> Tuple[str, str]:
    """
    Applies multi-domain signal detection and structural intent resolution without overwriting raw_intent.
    Returns (resolved_intent, resolution_reason).
    """
    q_lower = question.lower().strip()

    # 1. Multi-domain & Temporal signal detection
    dasha_signal = any(k in q_lower for k in ["dasha", "mahadasha", "antardasha", "dasas"])
    timing_signal = any(k in q_lower for k in ["right now", "now", "currently", "this year", "presently", "current period", "near future"])

    career_signal = any(k in q_lower for k in ["career", "job", "work", "profession", "business", "promotion", "naukri", "office"])
    marriage_signal = any(k in q_lower for k in ["marri", "spouse", "partner", "relationship", "shaadi", "love", "crush", "couple"])
    finance_signal = any(k in q_lower for k in ["money", "finance", "wealth", "income", "paisa", "invest", "dhan"])
    health_signal = any(k in q_lower for k in ["health", "disease", "stress", "sehat", "energy", "sleep", "tired"])

    domain_signals = []
    if career_signal: domain_signals.append("career_signal")
    if marriage_signal: domain_signals.append("marriage_signal")
    if finance_signal: domain_signals.append("finance_signal")
    if health_signal: domain_signals.append("health_signal")

    if dasha_signal and domain_signals:
        return "multi_domain", f"dasha_signal + {domain_signals[0]}"

    if timing_signal and career_signal:
        matched_kw = [k for k in ["right now", "now", "currently", "this year", "presently"] if k in q_lower]
        return "career_timing", f"temporal_timing_signal ({matched_kw[0] if matched_kw else 'timing'}) + career_signal"

    if timing_signal and marriage_signal:
        return "marriage_timing", "temporal_timing_signal + marriage_signal"

    if len(domain_signals) >= 2:
        return "multi_domain", f"multi_domain_signals ({'+'.join(domain_signals)})"

    # 2. Structural explicit pattern resolution for misclassified cases
    if dasha_signal and not domain_signals:
        if raw_intent != "dasha":
            return "dasha", "explicit_dasha_keyword_match"

    if "promot" in q_lower and raw_intent != "career_promotion":
        return "career_promotion", "explicit_promotion_keyword_match"

    if any(k in q_lower for k in ["when will i get married", "wedding date", "marriage timing"]) and raw_intent != "marriage_timing":
        return "marriage_timing", "explicit_marriage_timing_keyword_match"

    if "income increase" in q_lower or "salary increase" in q_lower:
        return "income", "explicit_income_keyword_match"

    if health_signal and any(k in q_lower for k in ["tired", "fatigue", "exhausted", "weakness", "low energy", "draining", "lethargic"]):
        return "health_period", "explicit_health_fatigue_keyword_match"

    if any(k in q_lower for k in ["crush", "like them", "confess", "propose", "good couple"]) and raw_intent not in ["relationship", "love_marriage"]:
        return "relationship", "explicit_relationship_crush_keyword_match"

    if "what is a nakshatra" in q_lower or "what is a birth chart" in q_lower:
        return "definition", "explicit_definition_keyword_match"

    # 3. Default: keep raw ML intent
    return raw_intent, "ml_model_prediction"


def predict_intent_with_confidence(question: str) -> Dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    model, vectorizer = get_intent_model()
    if model is None or vectorizer is None:
        return {
            "raw_intent": "general",
            "raw_confidence": 0.0,
            "resolved_intent": "general",
            "resolution_reason": "model_files_missing",
            "probabilities": {},
        }

    res = predict_with_confidence(model, vectorizer, question.strip())
    raw_intent = res["label"]
    raw_confidence = res["confidence"]
    probabilities = res["probabilities"]

    resolved_intent, resolution_reason = resolve_intent_overlay(
        question, raw_intent, raw_confidence, probabilities
    )

    return {
        "raw_intent": raw_intent,
        "raw_confidence": raw_confidence,
        "resolved_intent": resolved_intent,
        "resolution_reason": resolution_reason,
        "probabilities": probabilities,
    }


def predict_intent(question: str) -> str:
    res = predict_intent_with_confidence(question)
    return res["resolved_intent"]


classify_intent = predict_intent
classify_intent_with_confidence = predict_intent_with_confidence