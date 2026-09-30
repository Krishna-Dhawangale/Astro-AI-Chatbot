"""
Domain Classifier Module with Selector Audit Mode
=================================================
Module: backend/router/domain.py

Routes domain queries through the Evaluation-Driven Multi-Model Domain Selector
(backend/router/model_selector.py) combining predictions from OLD, CONTEXT_V1, and V3.

Logs detailed [DOMAIN SELECTOR] audit trails for full auditing.
"""

from pathlib import Path
import joblib

from .model_selector import evaluate_and_select_domain, load_domain_model_pool

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "question_classifier"

DOMAIN_MODEL_PATH = MODEL_DIR / "domain_model_v3.pkl"
DOMAIN_VECTORIZER_PATH = MODEL_DIR / "domain_vectorizer_v3.pkl"

print("========== DOMAIN MODEL SELECTOR INITIALIZING ==========")
print("MODEL POOL: OLD, CONTEXT_V1, V3")
print("========================================================")

def _log_domain_selector_audit(question: str, selector_res: dict):
    """Format and log the [DOMAIN SELECTOR] audit block."""
    model_preds = selector_res.get("model_predictions", {})
    score_details = {sc["model"]: sc for sc in selector_res.get("score_details", [])}

    print("\n[DOMAIN SELECTOR]")
    print(f'Question: "{question}"')
    for m in ["OLD", "CONTEXT_V1", "V3"]:
        m_info = model_preds.get(m, {})
        sc_info = score_details.get(m, {})
        m_dom = m_info.get("domain", "unknown")
        m_conf = m_info.get("confidence", 0.0)
        m_score = sc_info.get("score", 0.0)
        print(f"  {m}:")
        print(f"    domain={m_dom}")
        print(f"    confidence={m_conf:.4f}")
        print(f"    score={m_score:.4f}")

    print("  SELECTED:")
    print(f"    domain={selector_res.get('selected_domain')}")
    print(f"    model={selector_res.get('selected_model')}")
    print("  REASON:")
    print(f"    {selector_res.get('selection_method')}\n")

def predict_domain_with_confidence(question: str) -> dict:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    selector_res = evaluate_and_select_domain(question.strip())
    _log_domain_selector_audit(question.strip(), selector_res)

    label = selector_res["selected_domain"]
    confidence = selector_res["confidence"]

    model_preds = selector_res.get("model_predictions", {})
    chosen_model = selector_res.get("selected_model", "V3")

    probabilities = {}
    if chosen_model in model_preds:
        probabilities = model_preds[chosen_model].get("probabilities", {})
    elif "V3" in model_preds:
        probabilities = model_preds["V3"].get("probabilities", {})

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "selected_model": chosen_model,
        "selection_method": selector_res.get("selection_method", "evaluation_weighted_selector"),
        "model_predictions": model_preds
    }

def predict_domain(question: str) -> str:
    res = predict_domain_with_confidence(question)
    return res["label"]

def get_domain_model():
    pool = load_domain_model_pool()
    if "V3" in pool:
        return pool["V3"][0], pool["V3"][1]
    elif "OLD" in pool:
        return pool["OLD"][0], pool["OLD"][1]
    return None, None

classify_domain = predict_domain
classify_domain_with_confidence = predict_domain_with_confidence