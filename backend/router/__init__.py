"""Question routing components."""

from backend.router.pipeline import route_question
from backend.router.domain import predict_domain_with_confidence, predict_domain
from backend.router.complexity import predict_complexity
from backend.router.intent import predict_intent


class IntentRouter:
    """
    Adapter class for application callers.
    Delegates domain, complexity, and intent predictions directly to
    User's authoritative ML Router Engine.
    """
    def __init__(self, model_path: str = None):
        pass

    def classify_and_predict(self, query: str, astro_features: list = None) -> dict:
        routing = route_question(query)
        conf_result = predict_domain_with_confidence(query)
        
        return {
            "predicted_domain": routing.get("domain", "general"),
            "confidence_score": conf_result.get("confidence", 0.8),
            "complexity": routing.get("complexity", "needs_chart"),
            "intent": routing.get("intent", "general"),
            "route": routing.get("route", "astrology"),
            "probabilities": conf_result.get("probabilities", {})
        }


DOMAIN_KEYWORDS = {
    "career": ["career", "job", "profession", "work", "business", "promotion", "appraisal", "salary", "boss"],
    "marriage": ["marriage", "married", "wedding", "spouse", "partner", "relationship", "love", "divorce"],
    "wealth": ["money", "wealth", "finance", "financial", "property", "investment", "rich", "income"],
    "health": ["health", "disease", "illness", "body", "medical", "surgery", "fitness"],
    "travel": ["travel", "abroad", "foreign", "trip", "journey", "visa", "settle"],
    "children": ["child", "children", "baby", "pregnancy", "son", "daughter", "progeny"],
    "education": ["education", "study", "exam", "college", "degree", "learning"]
}

__all__ = [
    "route_question",
    "predict_domain",
    "predict_domain_with_confidence",
    "predict_complexity",
    "predict_intent",
    "IntentRouter",
    "DOMAIN_KEYWORDS",
]