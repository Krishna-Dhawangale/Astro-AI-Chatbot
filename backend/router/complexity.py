from pathlib import Path
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "question_classifier"

COMPLEXITY_MODEL_PATH = MODEL_DIR / "complexity_model.pkl"
COMPLEXITY_VECTORIZER_PATH = MODEL_DIR / "complexity_vectorizer.pkl"

_complexity_model = None
_complexity_vectorizer = None


def get_complexity_model():
    global _complexity_model, _complexity_vectorizer
    if _complexity_model is None:
        if COMPLEXITY_MODEL_PATH.exists() and COMPLEXITY_VECTORIZER_PATH.exists():
            _complexity_model = joblib.load(COMPLEXITY_MODEL_PATH)
            _complexity_vectorizer = joblib.load(COMPLEXITY_VECTORIZER_PATH)
    return _complexity_model, _complexity_vectorizer


def predict_complexity(question: str) -> str:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    model, vectorizer = get_complexity_model()
    if model is None or vectorizer is None:
        return "needs_chart"

    X = vectorizer.transform([question.strip()])
    prediction = model.predict(X)

    return prediction[0]


classify_complexity = predict_complexity