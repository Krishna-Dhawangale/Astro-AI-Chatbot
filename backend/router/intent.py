from pathlib import Path
import re
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


def extract_context_features(question: str) -> Dict[str, Any]:
    """
    Non-destructively extracts structured contextual features, disambiguates polysemous
    terms (partner, invest, switch, property), identifies explicit user-stated time
    horizons, and detects multi-part query scopes WITHOUT modifying the raw query.
    """
    import re
    q_raw = question or ""
    q_low = q_raw.lower().strip()

    # 1. Detect Negated Scopes (e.g., "not interested in marriage", "don't tell me about jobs")
    negated_domains = set()
    neg_matches = re.finditer(
        r"\b(?:not|no|don't|dont|never|nahi|nahin|nhi)\s+(?:interested\s+in|looking\s+for|want|need|planning|chahiye|karni|karna|tell\s+me\s+about|talk\s+about|ask\s+about)\s+([^,?.!]+)",
        q_low
    )
    for m in neg_matches:
        scope = m.group(1)
        if any(w in scope for w in ["marriage", "wedding", "shaadi", "shadi", "relationship", "spouse"]):
            negated_domains.add("marriage")
        if any(w in scope for w in ["job", "jobs", "career", "naukri", "business", "work", "promotion"]):
            negated_domains.add("career")
        if any(w in scope for w in ["invest", "money", "finance", "paisa", "stock", "shares"]):
            negated_domains.add("finance")
        if any(w in scope for w in ["travel", "abroad", "foreign", "videsh"]):
            negated_domains.add("travel")
        if any(w in scope for w in ["property", "house", "car", "gadi", "flat"]):
            negated_domains.add("property")

    # 2. Context-Sensitive Polysemous Term Disambiguation
    disambiguated: Dict[str, str] = {}

    # (A) "partner": Business partner vs Marriage life partner
    has_partner = bool(re.search(r"\bpartners?\b", q_low))
    if has_partner:
        is_biz_partner = bool(re.search(
            r"\b(?:business|startup|firm|venture|company|commercial|trading)\s+partners?\b|"
            r"\bpartners?\s+(?:in|for)\s+(?:business|startup|firm|venture|company)\b|"
            r"\b(?:take|hire|find|have)\s+a\s+partner\s+for\s+(?:my\s+)?(?:startup|business|venture)\b",
            q_low
        ))
        is_life_partner = bool(re.search(
            r"\b(?:life|marriage|wedding|future|romantic|spouse|kundali)\s+partners?\b|"
            r"\bpartners?\s+(?:for\s+life|in\s+marriage|se\s+shadi)\b|"
            r"\bmeet\s+(?:my\s+)?(?:life\s+)?partner\b",
            q_low
        ))
        if is_biz_partner and not is_life_partner:
            disambiguated["partner"] = "business"
        elif is_life_partner and not is_biz_partner:
            disambiguated["partner"] = "marriage"
        elif any(w in q_low for w in ["business", "startup", "venture", "firm", "company"]):
            disambiguated["partner"] = "business"
        elif any(w in q_low for w in ["marry", "marriage", "wedding", "shaadi", "shadi", "love", "couple", "relationship"]):
            disambiguated["partner"] = "marriage"
        else:
            disambiguated["partner"] = "ambiguous"

    # (B) "invest" / "investment": Real estate vs Startup/Business vs Financial savings
    has_invest = bool(re.search(r"\b(?:invest|investing|investment|investments)\b", q_low))
    if has_invest:
        if re.search(r"\binvest\w*\s+(?:in\s+)?(?:a\s+)?(?:property|house|flat|land|real\s+estate|plot|ghar|makan)\b", q_low):
            disambiguated["investment"] = "property"
        elif re.search(r"\binvest\w*\s+(?:in\s+)?(?:a\s+)?(?:business|startup|firm|venture|company)\b", q_low):
            disambiguated["investment"] = "career"
        else:
            disambiguated["investment"] = "finance"

    # (C) "switch": Job change vs Internal team switch vs Relocation
    has_switch = bool(re.search(r"\bswitch\b", q_low))
    if has_switch:
        if re.search(r"\bswitch\s+(?:jobs?|career|profession|roles?|compan(?:y|ies)|naukri)\b|\b(?:job|career)\s+switch\b", q_low):
            disambiguated["switch"] = "career_job_change"
        elif re.search(r"\bswitch\s+(?:teams?|projects?|departments?)\b", q_low):
            disambiguated["switch"] = "career_team_change"
        elif re.search(r"\bswitch\s+(?:cities|locations?|countr(?:y|ies))\b", q_low):
            disambiguated["switch"] = "relocation"
        else:
            disambiguated["switch"] = "general"

    # (D) "property": Physical real estate vs Astrological/Planetary attributes
    has_property = bool(re.search(r"\bpropert(?:y|ies)\b", q_low))
    if has_property:
        if re.search(r"\b(?:buy|purchase|buying|invest\s+in|own|acquire|deal\s+in)\s+propert\w*\b|\bpropert\w*\s+(?:purchase|investment|market|rates?|deal)\b", q_low):
            disambiguated["property"] = "real_estate"
        elif re.search(r"\b(?:properties\s+of|astrological\s+propert|planetary\s+propert)\b", q_low):
            disambiguated["property"] = "astrological_attribute"
        else:
            disambiguated["property"] = "real_estate"

    # 3. Explicit User-Stated Temporal Horizons (Extracted user intent ONLY, not astrological claims)
    explicit_user_horizon = None
    horizon_match = re.search(
        r"\b(?:in|within|for)\s+(?:the\s+)?(?:next|coming)\s+(\d+\s+(?:months?|weeks?|years?)|few\s+(?:months?|weeks?|years?))\b|"
        r"\b(this\s+year|next\s+year|in\s+20\d\d)\b|"
        r"\b((?:aane\s+wale|agle)\s+\d+\s+(?:mahine|mahino|saal))\b",
        q_low
    )
    if horizon_match:
        explicit_user_horizon = next(g for g in horizon_match.groups() if g is not None).strip()

    # 4. Domain Signals with Polysemy & Negation Filtering
    dasha_signal = any(k in q_low for k in ["dasha", "mahadasha", "antardasha", "dasas", "bhukti", "pratyantardasha"])

    timing_signal = bool(
        explicit_user_horizon or
        any(k in q_low for k in ["right now", "now", "currently", "this year", "presently", "current period", "near future", "timeline", "which month", "which year"]) or
        re.search(r"\b(?:when|kab)\b", q_low)
    )

    career_signal = False
    if "career" not in negated_domains:
        career_keywords = ["career", "job", "work", "profession", "business", "promotion", "naukri", "office", "tarakki", "appraisal"]
        if any(re.search(rf"\b{re.escape(k)}\b", q_low) for k in career_keywords):
            career_signal = True
        elif disambiguated.get("partner") == "business":
            career_signal = True
        elif disambiguated.get("switch") in ("career_job_change", "career_team_change"):
            career_signal = True
        elif disambiguated.get("investment") == "career":
            career_signal = True
        elif re.search(r"\b(?:sarkari|corporate|startup|venture|employment)\b", q_low):
            career_signal = True

    marriage_signal = False
    if "marriage" not in negated_domains:
        marriage_keywords = ["marri", "spouse", "shaadi", "shadi", "vivah", "wife", "husband", "crush", "couple", "rishta", "jeevansathi"]
        if any(k in q_low for k in marriage_keywords):
            marriage_signal = True
        elif disambiguated.get("partner") == "marriage":
            marriage_signal = True
        elif has_partner and disambiguated.get("partner") != "business":
            marriage_signal = True

    finance_signal = False
    if "finance" not in negated_domains:
        finance_keywords = ["money", "finance", "finances", "wealth", "income", "paisa", "dhan", "arthik", "karza", "debt", "loan", "savings"]
        if any(re.search(rf"\b{re.escape(k)}\b", q_low) for k in finance_keywords):
            finance_signal = True
        elif disambiguated.get("investment") == "finance":
            finance_signal = True
        elif re.search(r"\b(?:invest\s+(?:my\s+)?savings?|financial\s+stability|loss|nuksan|stocks?|shares?|trading|fixed\s+deposits?|fd)\b", q_low):
            finance_signal = True

    health_signal = False
    if "health" not in negated_domains:
        health_keywords = ["health", "disease", "stress", "sehat", "energy", "sleep", "neend", "tired", "fatigue", "weakness", "exhausted", "bimar", "illness", "insomnia"]
        if any(re.search(rf"\b{re.escape(k)}\b", q_low) for k in health_keywords):
            health_signal = True

    property_signal = False
    if "property" not in negated_domains:
        if disambiguated.get("property") == "real_estate" or disambiguated.get("investment") == "property":
            property_signal = True
        else:
            has_vehicle = bool(re.search(r"\b(?:cars?|gadi|vehicles?|bike|automobile|vahana)\b", q_low))
            has_real_estate = bool(re.search(r"\b(?:buy|purchase|own|construct|renovate|invest\s+in|new|sell|deal\s+in)\s+(?:a\s+)?(?:house|home|flat|plot|land|property|makan|ghar)\b|\b(?:real\s+estate|flat|makan|plot|ghar|home)\b", q_low))
            is_residential_house = bool(re.search(r"\bhouse\b", q_low)) and not bool(re.search(r"\b(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+house\b|\bhouse\s+(?:lord|sign|rashi|ruler)\b", q_low))
            if has_vehicle or has_real_estate or is_residential_house:
                property_signal = True

    travel_signal = False
    if "travel" not in negated_domains:
        travel_keywords = ["foreign", "abroad", "travel", "visa", "pr", "videsh", "overseas", "yatra", "immigrate"]
        if any(re.search(rf"\b{re.escape(k)}\b", q_low) for k in travel_keywords):
            travel_signal = True
        elif disambiguated.get("switch") == "relocation":
            travel_signal = True

    domain_signals: List[str] = []
    if career_signal: domain_signals.append("career_signal")
    if marriage_signal: domain_signals.append("marriage_signal")
    if finance_signal: domain_signals.append("finance_signal")
    if health_signal: domain_signals.append("health_signal")
    if property_signal: domain_signals.append("property_signal")
    if travel_signal: domain_signals.append("travel_signal")

    is_comparison = bool(re.search(r"\b(?:vs|versus|or|ya|better|preferable|compare)\b", q_low)) and len(domain_signals) >= 1
    is_feasibility = bool(re.search(r"\b(?:should\s+i|can\s+i|is\s+it\s+(?:safe|good|advisable|favorable)\s+to|kya\s+mujhe|karna\s+chahiye|karu)\b", q_low))
    is_multi_part = len(domain_signals) >= 2 or (dasha_signal and len(domain_signals) >= 1)

    return {
        "raw_query": q_raw,
        "q_clean": q_low,
        "negated_domains": list(negated_domains),
        "disambiguated_terms": disambiguated,
        "explicit_user_horizon": explicit_user_horizon,
        "has_dasha_signal": dasha_signal,
        "has_timing_signal": timing_signal,
        "active_domain_signals": domain_signals,
        "is_comparison": is_comparison,
        "is_feasibility": is_feasibility,
        "is_multi_part": is_multi_part,
    }


def resolve_intent_overlay(question: str, raw_intent: str, raw_confidence: float, probabilities: Dict[str, float]) -> Tuple[str, str]:
    """
    Applies non-destructive multi-domain signal detection, context feature extraction,
    and structural intent resolution without overwriting raw_intent.
    Returns (resolved_intent, resolution_reason).
    """
    features = extract_context_features(question)
    domain_signals = features["active_domain_signals"]
    dasha_signal = features["has_dasha_signal"]
    timing_signal = features["has_timing_signal"]
    q_lower = features["q_clean"]

    # 1. Multi-domain & Temporal signal detection
    if dasha_signal and domain_signals:
        return "multi_domain", f"dasha_signal + {domain_signals[0]}"

    if timing_signal and "career_signal" in domain_signals and len(domain_signals) == 1:
        matched_kw = [k for k in ["right now", "now", "currently", "this year", "presently"] if k in q_lower]
        return "career_timing", f"temporal_timing_signal ({matched_kw[0] if matched_kw else 'timing'}) + career_signal"

    if timing_signal and "marriage_signal" in domain_signals and len(domain_signals) == 1:
        return "marriage_timing", "temporal_timing_signal + marriage_signal"

    if len(domain_signals) >= 2:
        return "multi_domain", f"multi_domain_signals ({'+'.join(domain_signals)})"

    # 2. Structural explicit pattern resolution for misclassified cases
    disambiguated = features.get("disambiguated_terms", {})
    negated_domains = features.get("negated_domains", [])

    if disambiguated.get("partner") == "business" and "marriage_signal" not in domain_signals:
        if raw_intent in ["marriage_timing", "love_marriage", "married_life", "relationship", "compatibility"]:
            return "career_business", "disambiguated_business_partner"

    if "marriage" in negated_domains and "career_signal" in domain_signals:
        if raw_intent in ["marriage_timing", "love_marriage", "married_life", "relationship", "compatibility"]:
            return "career_job", "negated_marriage_resolved_to_career"

    if "career" in negated_domains and "property_signal" in domain_signals:
        if raw_intent in ["career_job", "career_business", "career_change", "career_general", "career_promotion"]:
            return "wealth", "negated_career_resolved_to_property"

    if dasha_signal and not domain_signals:
        if raw_intent != "dasha":
            return "dasha", "explicit_dasha_keyword_match"

    if "promot" in q_lower and raw_intent != "career_promotion":
        return "career_promotion", "explicit_promotion_keyword_match"

    if any(k in q_lower for k in ["when will i get married", "wedding date", "marriage timing"]) and raw_intent != "marriage_timing":
        return "marriage_timing", "explicit_marriage_timing_keyword_match"

    if "income increase" in q_lower or "salary increase" in q_lower:
        return "income", "explicit_income_keyword_match"

    if "health_signal" in domain_signals and any(k in q_lower for k in ["tired", "fatigue", "exhausted", "weakness", "low energy", "draining", "lethargic"]):
        return "health_period", "explicit_health_fatigue_keyword_match"

    if any(k in q_lower for k in ["crush", "like them", "confess", "propose", "good couple"]) and raw_intent not in ["relationship", "love_marriage"]:
        return "relationship", "explicit_relationship_crush_keyword_match"

    if re.search(r"\b(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s*(?:house|bhava|ghar)\b|\bplanets?\s+in\s+.*house\b", q_lower):
        return "house_analysis", "explicit_house_query_match"

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