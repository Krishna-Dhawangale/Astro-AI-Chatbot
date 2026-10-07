"""
Rule Coverage Evaluator (backend/reasoning/rule_coverage.py)
------------------------------------------------------------
Pillar 2: Introduces explicit rule coverage evaluation.

Core Architectural Rules:
1. Domain match alone NEVER authorizes a Stage 8 rule.
2. A local rule executes ONLY when the rule is actually relevant to the detected intent/question
   and the required evidence is complete.
3. Inherently unsupported/deterministic claims (exact death date, lottery numbers, exact spouse name,
   exact stock price, guaranteed millionaire, exact daily schedule) return answerable_locally = False
   and trigger Tier 1 UNSUPPORTED boundary protection.
4. Valid domain questions without an exact local rule return answerable_locally = False,
   triggering Tier 3 EVIDENCE_GROUNDED_LLM.
"""

from typing import Dict, Any, List, Optional

# Tier 1 Explicit Unsupported Boundary Patterns
UNSUPPORTED_PATTERNS = [
    "past life", "previous life", "previous birth", "last birth", "pichle janam", "purana janam",
    "lottery", "winning numbers", "jackpot", "gambling", "lottery ke",
    "date i will die", "date of death", "exact death date", "when will i die", "maut ki date", "kab marunga",
    "exact stock price", "stock price of", "exact share price",
    "guarantee that i will", "guarantee i will become", "millionaire guarantee",
    "exact name of my future spouse", "exact name of spouse", "spouse name", "spouse ka exact naam", "exact naam kya",
    "exact name of my husband", "exact name of my wife",
    "exact events that will happen", "happen to me tomorrow at", "tomorrow at 3 pm",
    "every day for the next year", "exact number of children", "how many children"
]

# Explicit Intents that HAVE exact local rule coverage in our Stage 8 engine
EXACT_LOCAL_RULE_INTENTS = {
    "current_dasha",
    "dasha_career_interaction",
    "dasha_marriage_interaction",
    "career_suitability",
    "10th_lord_placement",
    "career_indicators",
    "7th_lord_placement",
    "marriage_timing",
    "relationship_stability",
    "financial_potential",
    "2nd_lord_placement",
    "11th_lord_placement",
    "direct_fact",
    "lagna_sign",
    "moon_sign",
    "sun_sign"
}


def is_unsupported_boundary_query(question: str) -> bool:
    """Checks if the query asks for an inherently unsupported/deterministic claim."""
    if not question:
        return False
    q_low = question.lower().strip()
    return any(p in q_low for p in UNSUPPORTED_PATTERNS)


def evaluate_rule_coverage(
    question: str,
    domain: str,
    intent: str,
    structured_evidence: Optional[Any] = None,
    matched_rules: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Evaluates rule relevance and coverage for a question.
    Returns a structured object detailing rule coverage and answerability.
    """
    q_low = (question or "").lower().strip()
    dom = (domain or "general").lower()
    ent = (intent or "general").lower()

    supported_domains = {"career", "marriage", "finance", "education", "property", "health", "multi_domain"}
    domain_match = dom in supported_domains or dom == "other"

    # 1. Boundary Check (Tier 1)
    if is_unsupported_boundary_query(question):
        return {
            "domain_match": domain_match,
            "intent_match": False,
            "exact_rule_match": False,
            "matching_rules": [],
            "required_evidence_complete": False,
            "coverage_score": 0.0,
            "answerable_locally": False,
            "reason": "Query asks for an unsupported deterministic boundary prediction (exact name/date/numbers)."
        }

    # 2. Check Evidence Completeness
    evidence_complete = False
    if structured_evidence is not None:
        if hasattr(structured_evidence, "is_sufficient"):
            evidence_complete = structured_evidence.is_sufficient()
        elif isinstance(structured_evidence, dict):
            evidence_complete = structured_evidence.get("is_sufficient", True)
    elif matched_rules:
        active = [r for r in matched_rules if r.get("matched", True)]
        evidence_complete = len(active) >= 1

    # Active matched rule IDs
    active_rule_ids = []
    if matched_rules:
        active_rule_ids = [r.get("rule_id", "") for r in matched_rules if r.get("matched", True)]

    # Intent match check
    intent_match = ent in EXACT_LOCAL_RULE_INTENTS or any(
        k in q_low for k in ["dasha", "mahadasha", "lagna", "10th lord", "7th lord", "moon sign", "which career"]
    )

    # 3. Exact Rule Match Evaluation
    # Local rule match requires intent match + active matched rules + complete evidence
    # AND must NOT be a complex custom/transition query beyond local rule scope
    complex_uncovered_keywords = [
        "transition", "move from", "switch from", "should i change", "explain my overall",
        "detail about my future", "tell me everything about", "career story", "guidance for my",
        "product management", "software engineering", "move to", "should i move",
        "business environment", "approach a career transition", "influence each other",
        "how should i approach"
    ]
    is_complex_uncovered = any(k in q_low for k in complex_uncovered_keywords)

    exact_rule_match = False
    if evidence_complete and intent_match and not is_complex_uncovered:
        exact_rule_match = True
    elif evidence_complete and not is_complex_uncovered and ("which career" in q_low or "what is my lagna" in q_low or q_low.strip("?.!") == "my current dasha"):
        exact_rule_match = True

    if exact_rule_match:
        return {
            "domain_match": domain_match,
            "intent_match": True,
            "exact_rule_match": True,
            "matching_rules": active_rule_ids or ["EXACT_LOCAL_RULE"],
            "required_evidence_complete": True,
            "coverage_score": 1.0,
            "answerable_locally": True,
            "reason": "Exact deterministic rule coverage available for intent"
        }

    # 4. Valid Domain Question but No Specific Local Rule (Triggers Tier 3: EVIDENCE_GROUNDED_LLM)
    return {
        "domain_match": domain_match,
        "intent_match": domain_match,
        "exact_rule_match": False,
        "matching_rules": [],
        "required_evidence_complete": evidence_complete,
        "coverage_score": 0.0,
        "answerable_locally": False,
        "reason": "Valid domain question but no sufficiently specific local rule coverage"
    }


def get_unsupported_reason(question: str) -> str:
    """Returns the specific unsupported reason code for a query."""
    if not question:
        return "GENERAL_UNSUPPORTED"
    q_low = question.lower().strip()
    if any(k in q_low for k in ["past life", "previous life", "previous birth", "last birth"]):
        return "PAST_LIFE_IDENTITY"
    if any(k in q_low for k in ["lottery", "winning numbers", "jackpot", "gambling"]):
        return "LOTTERY_PREDICTION"
    if any(k in q_low for k in ["date i will die", "date of death", "exact death date", "when will i die"]):
        return "DEATH_DATE"
    if any(k in q_low for k in ["name of my future spouse", "name of spouse", "spouse name", "name of my husband", "name of my wife"]):
        return "FUTURE_SPOUSE_NAME"
    if any(k in q_low for k in ["tomorrow at", "tomorrow at 3 pm", "events that will happen"]):
        return "EXACT_DAILY_EVENT"
    if any(k in q_low for k in ["stock price", "share price"]):
        return "EXACT_MARKET_PRICE"
    if any(k in q_low for k in ["guarantee that i will", "guarantee i will become", "millionaire guarantee"]):
        return "GUARANTEED_WEALTH"
    return "GENERAL_UNSUPPORTED"


def get_unsupported_limitation_response(question: str) -> str:
    """Generates a natural, reason-specific limitation response for unsupported queries."""
    reason = get_unsupported_reason(question)
    if reason == "PAST_LIFE_IDENTITY":
        return "No chart can reliably reveal your exact past-life identity or biography, so I wouldn't want to invent one. In Vedic Astrology, Rahu, Ketu, and 12th/9th house placements are interpreted symbolically for spiritual impressions and karmic patterns, but exact past identities cannot be derived from a chart."
    elif reason == "LOTTERY_PREDICTION":
        return "Astrology cannot predict exact winning lottery numbers or random gambling outcomes. Financial indicators in your chart show earning style, resource management, and prosperity windows, but cannot generate specific winning numbers."
    elif reason == "DEATH_DATE":
        return "Astrology cannot and should not be used to predict an exact date of death. While birth charts show general vitality indicators and health tendencies, claiming precise life span dates is ethically prohibited and astrologically invalid."
    elif reason == "FUTURE_SPOUSE_NAME":
        return "A birth chart cannot reveal the exact legal name or identity of your future spouse. Planetary placements in the 7th house describe relationship dynamics, qualities, and compatibility traits, but personal names are outside chart scope."
    elif reason == "EXACT_DAILY_EVENT":
        return "Astrology provides general timing windows and planetary transit influences rather than minute-by-minute event predictions for a specific hour."
    elif reason == "EXACT_MARKET_PRICE":
        return "Astrology cannot predict exact stock market prices or stock quotes. Financial markets are driven by external economic factors, whereas chart placements reflect personal financial tendencies and risk tolerance."
    elif reason == "GUARANTEED_WEALTH":
        return "No astrology system can guarantee a specific net worth or promise exact wealth sums. Your chart indicates financial potential and resource-building factors, but eventual wealth depends on career choices, investments, decisions, and circumstances."
    return "I don't currently have a supported chart-based analysis for that question. Please ask a career, health, marriage, finance, education, or property question based on your birth chart."

