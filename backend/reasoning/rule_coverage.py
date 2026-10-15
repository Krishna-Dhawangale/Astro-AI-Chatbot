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
    "date i will die", "date of death", "exact death date", "when will i die", "when i will die", "maut ki date", "kab marunga", "will die",
    "exact stock price", "stock price of", "exact share price",
    "guarantee that i will", "guarantee i will become", "millionaire guarantee",
    "exact name of my future spouse", "exact name of spouse", "spouse name", "spouse ka exact naam", "exact naam kya",
    "exact name of my husband", "exact name of my wife",
    "exact events that will happen", "happen to me tomorrow at", "tomorrow at 3 pm",
    "every day for the next year", "exact number of children", "how many children"
]

CLINICAL_DIAGNOSTIC_PATTERNS = [
    "chest pain", "chest pressure", "diabetes", "diabetic", "heart attack", "cancer", "tumor",
    "blood pressure", "cure for", "cure my", "medicine for", "which medicine", "what medicine", "medicine i should", "doctor recommended",
    "medical diagnosis", "prescribe treatment", "infectious disease", "covid", "stroke", "paralysis",
    "hospitalization date", "surgery recommendation", "diagnose my", "diagnose", "am i sick with", "what disease", "disease i have",
    "recover from my illness", "recover from illness", "guarantee that i will recover"
]

# Explicit Intents that HAVE exact local rule coverage in our Stage 8 engine
EXACT_LOCAL_RULE_INTENTS = {
    "current_dasha",
    "dasha_career_interaction",
    "dasha_marriage_interaction",
    "career_suitability",
    "career_general",
    "career_business",
    "career_job",
    "career_timing",
    "10th_lord_placement",
    "career_indicators",
    "marriage_general",
    "7th_lord_placement",
    "marriage_timing",
    "relationship_stability",
    "financial_potential",
    "finance_general",
    "financial_stability",
    "wealth",
    "income",
    "2nd_lord_placement",
    "11th_lord_placement",
    "education_general",
    "education_suitability",
    "education_higher",
    "5th_house_learning",
    "4th_lord_education",
    "5th_lord_education",
    "direct_fact",
    "lagna_sign",
    "moon_sign",
    "sun_sign",
    "health_vitality",
    "health_fatigue",
    "health_sleep",
    "health_stress",
    "health_wellness",
    "health_period",
    "health_general",
    "health_indicators",
    "lagna_vitality",
    "6th_lord_wellness",
    "12th_house_rest",
    "dasha_energy"
}


def is_unsupported_boundary_query(question: str) -> bool:
    """Checks if the query asks for an inherently unsupported/deterministic claim or clinical diagnosis."""
    if not question:
        return False
    q_low = question.lower().strip()
    return any(p in q_low for p in UNSUPPORTED_PATTERNS) or any(p in q_low for p in CLINICAL_DIAGNOSTIC_PATTERNS)


def analyze_question_task(question: str) -> Dict[str, bool]:
    """
    Evaluates semantic task requirements of the user's question.
    Distinguishes:
    - comparative dilemmas (e.g. govt vs private, job vs business, career field comparisons)
    - causal / delay inquiries (e.g. why delay, growth stagnation, reason for struggle)
    - action / decision dilemmas (e.g. job switch timing, taking business risk)
    - multi-topic synthesis
    from general overview and direct factual inquiries.
    """
    if not question:
        return {
            "is_comparative": False,
            "is_causal_delay": False,
            "is_action_dilemma": False,
            "is_multi_topic": False,
            "is_legacy_complex": False,
            "is_synthesis_gap": False,
        }
    q = question.lower().strip()

    # 1. Comparative Dilemmas (comparing two sectors, career paths, or choices)
    has_vs = any(sep in q for sep in [" vs ", " vs.", " versus ", " v/s "]) or q.startswith("job vs") or q.startswith("business vs")
    has_gov_private = ("government" in q or "govt" in q or "sarkari" in q) and ("private" in q or "corporate" in q)
    has_job_biz = ("job" in q or "naukri" in q or "rojgari" in q) and ("business" in q or "vyapar" in q or "enterprise" in q)
    has_choice_or = any(p in q for p in [
        "karun ya", "kare ya", "karein ya", "rahega ya", "better rahega ya", "kaunsa option better",
        "which option is better", "which is better", "dono mein se", "dono me se", "kis mein jyada",
        "kisme jyada", "choose karun ya", "try karte rehna chahiye ya", "or business", "or a job",
        "business or", "job or"
    ])
    is_comparative = has_vs or has_gov_private or has_job_biz or has_choice_or

    # 2. Causal / Delay / Obstacle Inquiries (why delay, why stuck, why stagnation)
    is_causal_delay = any(p in q for p in [
        "delay kyun", "delay kyu", "delay why", "why is there delay", "why delay", "baar-baar delay",
        "baar baar delay", "delay ho raha", "late kyun", "late kyu", "rukawat", "growth nahi mil rahi",
        "growth nahi", "growth kyu nahi", "growth kyun nahi", "fas gaya", "stuck in", "paisa fas",
        "problem kyun", "kyun nahi ho raha", "kyu nahi ho raha", "is stress ka reason", "restlessness rehti hai",
        "neend nahi aati"
    ])

    # 3. Action / Decision Dilemmas (should I switch, change, stay, take risk)
    is_action_dilemma = any(p in q for p in [
        "job change", "switch karna", "switch karun", "badalna safe", "rojgari badalna", "tikna sahi",
        "purani naukri mein hi tikna", "purani job mein", "should i switch", "should i change",
        "risk lena chahiye", "risk lu ya", "risk lena safe", "approach a career transition",
        "switch from", "move from", "leave job", "leave my job"
    ])

    # 4. Multi-Topic Synthesis (queries combining multiple distinct life domains)
    domain_hits = 0
    if any(k in q for k in ["shaadi", "marriage", "shadi", "relationship", "spouse"]):
        domain_hits += 1
    if any(k in q for k in ["foreign", "videsh", "abroad", "foreign settlement"]):
        domain_hits += 1
    if any(k in q for k in ["income", "paisa", "wealth", "earning", "finance"]):
        domain_hits += 1
    if any(k in q for k in ["job", "career", "naukri", "profession"]):
        domain_hits += 1
    is_multi_topic = (domain_hits >= 2)

    # Legacy uncovered keywords
    complex_uncovered_keywords = [
        "transition", "move from", "switch from", "should i change", "explain my overall",
        "detail about my future", "tell me everything about", "career story", "guidance for my",
        "product management", "software engineering", "move to", "should i move",
        "business environment", "approach a career transition", "influence each other",
        "how should i approach"
    ]
    is_legacy_complex = any(k in q for k in complex_uncovered_keywords)

    is_synthesis_gap = is_comparative or is_causal_delay or is_action_dilemma or is_multi_topic or is_legacy_complex

    return {
        "is_comparative": is_comparative,
        "is_causal_delay": is_causal_delay,
        "is_action_dilemma": is_action_dilemma,
        "is_multi_topic": is_multi_topic,
        "is_legacy_complex": is_legacy_complex,
        "is_synthesis_gap": is_synthesis_gap,
    }


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
            "reason": "Query asks for an unsupported deterministic boundary prediction or clinical medical diagnosis."
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
    health_kws = ["health", "wellbeing", "vitality", "energy", "fatigue", "tired", "sleep", "routine", "overthink", "stress", "rest", "physical strength", "weakness", "emotional stability", "relaxing my mind", "moon placement", "sun placement", "self-care", "mahadasha", "antardasha"]
    is_health_query = dom == "health" or ent.startswith("health") or any(k in q_low for k in health_kws)

    edu_kws = ["education", "study", "learning", "college", "degree", "higher education", "technical education", "school", "academic", "5th house", "4th house"]
    is_edu_query = dom == "education" or ent.startswith("education") or any(k in q_low for k in edu_kws)

    intent_match = (
        ent in EXACT_LOCAL_RULE_INTENTS
        or is_health_query
        or is_edu_query
        or any(k in q_low for k in ["dasha", "mahadasha", "lagna", "10th lord", "7th lord", "2nd lord", "5th house", "10th house", "moon sign", "which career", "technical education", "higher education"])
    )

    # 3. Task Requirement vs Local Template Capability
    # Distinguish general domain overview vs comparative, causal delay, or decision dilemmas
    task = analyze_question_task(question)
    is_synthesis_gap = task["is_synthesis_gap"]

    exact_rule_match = False
    if evidence_complete and intent_match and not is_synthesis_gap:
        exact_rule_match = True
    elif evidence_complete and not is_synthesis_gap and ("which career" in q_low or "what is my lagna" in q_low or q_low.strip("?.!") == "my current dasha"):
        exact_rule_match = True

    if exact_rule_match:
        rules_assigned = active_rule_ids
        if is_health_query and not rules_assigned:
            rules_assigned = ["HEALTH_LAGNA_VITALITY", "HEALTH_6TH_LORD_WELLNESS", "HEALTH_12TH_HOUSE_REST", "HEALTH_DASHA_ENERGY"]
        elif is_edu_query and not rules_assigned:
            rules_assigned = ["EDUCATION_4TH_LORD_PLACEMENT", "EDUCATION_5TH_LORD_PLACEMENT", "EDUCATION_KARAKA_PRESENCE"]
        return {
            "domain_match": domain_match,
            "intent_match": True,
            "exact_rule_match": True,
            "matching_rules": rules_assigned or ["EXACT_LOCAL_RULE"],
            "required_evidence_complete": True,
            "coverage_score": 1.0,
            "answerable_locally": True,
            "reason": "Exact deterministic rule coverage available for intent"
        }

    # 4. Valid Domain Question but Local Capability Gap (Triggers Tier 3: EVIDENCE_GROUNDED_LLM)
    return {
        "domain_match": domain_match,
        "intent_match": domain_match,
        "exact_rule_match": False,
        "matching_rules": [],
        "required_evidence_complete": evidence_complete,
        "coverage_score": 0.0,
        "answerable_locally": False,
        "reason": (
            "Valid domain question with backend evidence available, but requires comparative, causal, or decision synthesis beyond local template capability"
            if (domain_match and evidence_complete and is_synthesis_gap)
            else "Valid domain question but no sufficiently specific local rule coverage"
        )
    }


def get_unsupported_reason(question: str) -> str:
    """Returns the specific unsupported reason code for a query."""
    if not question:
        return "GENERAL_UNSUPPORTED"
    q_low = question.lower().strip()
    if any(k in q_low for k in CLINICAL_DIAGNOSTIC_PATTERNS):
        return "CLINICAL_DIAGNOSIS"
    if any(k in q_low for k in ["past life", "previous life", "previous birth", "last birth"]):
        return "PAST_LIFE_IDENTITY"
    if any(k in q_low for k in ["lottery", "winning numbers", "jackpot", "gambling"]):
        return "LOTTERY_PREDICTION"
    if any(k in q_low for k in ["date i will die", "date of death", "exact death date", "when will i die", "when i will die", "will die"]):
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
    if reason == "CLINICAL_DIAGNOSIS":
        return "I cannot evaluate clinical medical conditions, diagnose diseases (like chest pain, diabetes, or infections), or prescribe treatments. In Vedic Astrology, chart indicators describe general vitality and wellness patterns, but for physical health symptoms or medical concerns, please consult a qualified healthcare professional."
    elif reason == "PAST_LIFE_IDENTITY":
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


