from pathlib import Path
import joblib

from .predictor import predict_with_confidence


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "question_classifier"

DOMAIN_MODEL_PATH = MODEL_DIR / "domain_model.pkl"
DOMAIN_VECTORIZER_PATH = MODEL_DIR / "domain_vectorizer.pkl"

_domain_model = None
_domain_vectorizer = None


def get_domain_model():
    global _domain_model, _domain_vectorizer
    if _domain_model is None:
        if DOMAIN_MODEL_PATH.exists() and DOMAIN_VECTORIZER_PATH.exists():
            _domain_model = joblib.load(DOMAIN_MODEL_PATH)
            _domain_vectorizer = joblib.load(DOMAIN_VECTORIZER_PATH)
    return _domain_model, _domain_vectorizer


# Direct attributes for tests expecting module-level variables
domain_model, domain_vectorizer = get_domain_model()


def predict_domain(question: str) -> str:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    model, vectorizer = get_domain_model()
    if model is None or vectorizer is None:
        return "general"

    result = predict_with_confidence(
        model,
        vectorizer,
        question
    )

    return result["label"]


def predict_domain_with_confidence(question: str) -> dict:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    model, vectorizer = get_domain_model()
    if model is None or vectorizer is None:
        return {
            "label": "general",
            "confidence": 0.0,
            "probabilities": {},
            "status": "awaiting_model_files"
        }

    return predict_with_confidence(
        model,
        vectorizer,
        question
    )


classify_domain = predict_domain
classify_domain_with_confidence = predict_domain_with_confidence