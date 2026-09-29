from pathlib import Path
import joblib


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


def predict_intent(question: str) -> str:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    model, vectorizer = get_intent_model()
    if model is None or vectorizer is None:
        return "general"

    X = vectorizer.transform([question.strip()])
    prediction = model.predict(X)

    return prediction[0]


classify_intent = predict_intent