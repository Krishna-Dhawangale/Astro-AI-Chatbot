import numpy as np


def predict_with_confidence(model, vectorizer, question):
    """
    Run a classifier and return prediction + confidence.
    """

    question = str(question).strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    X = vectorizer.transform([question])

    probabilities = model.predict_proba(X)[0]

    predicted_index = np.argmax(probabilities)

    label = model.classes_[predicted_index]

    confidence = float(probabilities[predicted_index])

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            class_name: float(probability)
            for class_name, probability
            in zip(model.classes_, probabilities)
        }
    }