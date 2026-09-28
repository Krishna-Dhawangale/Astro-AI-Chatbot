import os
import joblib
import numpy as np

DOMAIN_KEYWORDS = {
    "career": ["career", "job", "profession", "work", "business", "promotion", "appraisal", "salary", "boss"],
    "marriage": ["marriage", "married", "wedding", "spouse", "partner", "relationship", "love", "divorce"],
    "wealth": ["money", "wealth", "finance", "financial", "property", "investment", "rich", "income"],
    "health": ["health", "disease", "illness", "body", "medical", "surgery", "fitness"],
    "travel": ["travel", "abroad", "foreign", "trip", "journey", "visa", "settle"],
    "children": ["child", "children", "baby", "pregnancy", "son", "daughter", "progeny"],
    "education": ["education", "study", "exam", "college", "degree", "learning"]
}

class IntentRouter:
    def __init__(self, model_path: str = "astrology_predict_model.pkl"):
        self.model_path = model_path
        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print("[OK] ML Model loaded successfully.")
            except Exception as e:
                print(f"[WARNING] Error loading ML model: {e}")

    def classify_and_predict(self, query: str, astro_features: list) -> dict:
        q_lower = query.lower()
        predicted_domain = "general"
        for dom, keywords in DOMAIN_KEYWORDS.items():
            if any(k in q_lower for k in keywords):
                predicted_domain = dom
                break

        confidence = 0.8
        if self.model and astro_features:
            try:
                feats = np.array(astro_features[:7]).reshape(1, -1)
                ml_pred = self.model.predict(feats)[0]
                if hasattr(self.model, "predict_proba"):
                    confidence = float(np.max(self.model.predict_proba(feats)[0]))
                if predicted_domain == "general":
                    predicted_domain = f"chart_archetype_{ml_pred}"
            except Exception as e:
                print(f"[WARNING] ML Prediction error: {e}")

        return {"predicted_domain": predicted_domain, "confidence_score": confidence}