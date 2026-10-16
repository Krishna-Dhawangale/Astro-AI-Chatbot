"""
backend/qa_router.py
Multi-Domain & Deep Micro-Intent Classifier with Hybrid Local Retrieval + LLM Synthesis
========================================================================================
Architecture:
1. Multi-Domain & Micro-Intent Parsing (returns List[str] of detected micro-intents).
2. Compressed Chart Summary & Aggregated Local Database Micro-Conclusions.
3. LLM Prose Refiner & Synthesizer Prompt Structure with strict token optimization.
4. Standardized Calculation Provenance Footer.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("AstroChatbot.QARouter")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_context_features(query: str) -> Dict[str, Any]:
    try:
        from backend.router.intent import extract_context_features
        return extract_context_features(query)
    except Exception:
        try:
            from router.intent import extract_context_features
            return extract_context_features(query)
        except Exception:
            return {
                "raw_query": query,
                "q_clean": query.lower().strip() if query else "",
                "negated_domains": [],
                "disambiguated_terms": {},
                "explicit_user_horizon": None,
                "has_dasha_signal": False,
                "has_timing_signal": False,
                "active_domain_signals": [],
                "is_comparison": False,
                "is_feasibility": False,
                "is_multi_part": False,
            }


# ---------------------------------------------------------------------------
# Micro-Intent Definition Model
# ---------------------------------------------------------------------------
@dataclass
class MicroIntentDefinition:
    domain: str
    intent_id: str
    qa_keys: List[str]
    description: str
    keywords: List[str]
    patterns: List[str]


# ---------------------------------------------------------------------------
# Canonical Multi-Domain Micro-Intent Registry
# ---------------------------------------------------------------------------
MICRO_INTENT_REGISTRY: Dict[str, MicroIntentDefinition] = {
    # ------------------ CAREER DOMAIN ------------------
    "career_timing": MicroIntentDefinition(
        domain="career",
        intent_id="career_timing",
        qa_keys=["job change", "career prediction", "promotion"],
        description="Timing of job changes, promotions, career stability, or transitions",
        keywords=["job", "career", "naukri", "promotion", "switch", "stable", "stability", "tarakki", "stabilize", "settle", "change"],
        patterns=[
            r"\b(career|job|naukri)\s+(?:kab|when)\s+(?:stable|theek|acchi)\s+(?:hoga|hogi|hogi\?)\b",
            r"\b(?:career|job)\s+stability\b",
            r"\b(?:will|can)\s+(?:my\s+)?(?:career|job)\s+(?:stabilize|settle|improve)\b",
            r"\b(?:career|job)\s+(?:kab\s+)?(?:stabilize|settle)\b",
            r"\b(kab\s+milegi\s+(?:naukri|job))\b",
            r"\b(job|naukri)\s+kab\s+(?:milegi|lagegi|hogi|change|badlu)\b",
            r"\b(promotion|appraisal|tarakki)\s+kab\s+(?:hoga|milega)\b",
            r"\bwhen\s+will\s+(?:my\s+)?(?:career|job)\s+(?:stabilize|settle|change|switch|improve)\b",
            r"\b(timing|time)\s+(?:for|of)\s+(?:jobs?|career|promotion|switch)\b",
            r"\b(right\s+time\s+(?:for|to)\s+(?:job\s+change|switch))\b",
            r"\b(?:switch|change)\s+(?:jobs?|career|profession)\b",
            r"\bjob\s+change\b",
        ]
    ),
    "career_domain_fit": MicroIntentDefinition(
        domain="career",
        intent_id="career_domain_fit",
        qa_keys=["business vs job", "career house", "government job yoga"],
        description="Suitability for business vs salaried job, industry choice, or venture",
        keywords=["business", "job", "industry", "field", "corporate", "startup", "invest in business", "partner", "career"],
        patterns=[
            r"\bbusiness\s+(?:karu|karein|karun|shuru\s+karu)\s+ya\s+(?:job|naukri)\b",
            r"\b(?:business\s+mein\s+invest|start\s+a\s+business|independent\s+venture)\b",
            r"\b(?:which|what)\s+(?:career\s+)?(?:field|career|business)\s+(?:is\s+best|suits)\b",
            r"\bwhat\s+(?:career|field)\s+is\s+best\b",
            r"\b(?:business\s+partner|partner\s+for\s+(?:my\s+)?(?:startup|business|firm|company))\b",
            r"\b(?:take|choose|have)\s+a\s+business\s+partner\b",
            r"\b(?:business\s+vs\s+job|job\s+vs\s+business)\b",
            r"\bnaukri\s+badlu\s+ya\s+business\b",
        ]
    ),
    "career_probability": MicroIntentDefinition(
        domain="career",
        intent_id="career_probability",
        qa_keys=["career prediction", "government job"],
        description="Chances of landing a job or succeeding in career",
        keywords=["job", "success", "chance", "probability", "sarkari", "selection"],
        patterns=[
            r"\b(job|naukri)\s+milegi\s+ya\s+nahi\b",
            r"\bchances\s+of\s+(?:getting\s+a\s+job|career\s+success)\b",
            r"\bkya\s+sarkari\s+naukri\s+milegi\b",
        ]
    ),
    "career_growth_prospects": MicroIntentDefinition(
        domain="career",
        intent_id="career_growth_prospects",
        qa_keys=["career prediction", "10th house", "job change", "promotion"],
        description="Career growth, promotions, professional elevation, and future prospects",
        keywords=["career", "growth", "prospects", "advancement", "tarakki", "progression", "job", "career growth"],
        patterns=[
            r"\b(?:career|job)\s+(?:growth|prospects|advancement|progression|trajectory|outlook)\b",
            r"\bgrowth\s+(?:in|for)\s+(?:career|job)\b",
            r"\bcareer\s+(?:grow|badhega|tarakki)\b",
            r"\b(tarakki|career\s+growth)\b",
            r"\b(?:how\s+will|will)\s+(?:my\s+)?career\s+(?:grow|advance|progress)\b",
            r"\b(?:can\s+i|should\s+i)\s+focus\s+on\s+(?:a\s+)?(?:job\s+change|career)\b",
        ]
    ),

    # ------------------ FINANCE DOMAIN ------------------
    "finance_stability": MicroIntentDefinition(
        domain="finance",
        intent_id="finance_stability",
        qa_keys=["financial stability", "2nd house", "11th house", "dhan yoga"],
        description="Financial stability, long-term wealth preservation, income security, and savings",
        keywords=["financial", "stability", "finance", "finances", "wealth", "income", "money", "arthik", "financial stability"],
        patterns=[
            r"\bfinancial\s+stability\b",
            r"\b(?:stability\s+in\s+finances?|financial\s+condition|financial\s+security|financial\s+prospects)\b",
            r"\b(?:financial|finances?)\s+(?:stable|stabilize|security|growth)\b",
            r"\barthik\s+(?:sthiti|stability)\b",
            r"\b(?:how\s+will|will)\s+(?:my\s+)?financial\s+stability\b",
        ]
    ),
    "finance_risk": MicroIntentDefinition(
        domain="finance",
        intent_id="finance_risk",
        qa_keys=["share market", "remedies for debt", "financial stability"],
        description="Assessing financial risks, business investments, loss prevention, debt, stock market",
        keywords=["loss", "nuksan", "invest", "investment", "risk", "debt", "karza", "share", "stock", "trading", "savings"],
        patterns=[
            r"\b(business|market|shares?)\s+(?:mein|me)\s+invest\s+(?:karna|karein|karna\s+chahiye|karu)\b",
            r"\b(financial\s+loss|loss|nuksan)\s+(?:rukega|se\s+bachav|hoga)\b",
            r"\b(risk\s+of\s+(?:financial\s+)?loss|investment\s+risk)\b",
            r"\b(?:safe\s+to\s+invest|should\s+i\s+invest|is\s+it\s+safe\s+to\s+invest)\b",
            r"\b(?:invest\s+(?:my\s+)?savings?|investing\s+(?:my\s+)?savings?)\b",
            r"\b(stock|share)\s+market\s+(?:mein|me)\s+(?:loss|invest)\b",
            r"\b(karza|loan|debt)\s+(?:chuke ga|lena\s+chahiye)\b",
        ]
    ),
    "finance_timing": MicroIntentDefinition(
        domain="finance",
        intent_id="finance_timing",
        qa_keys=["financial stability", "dhan yoga", "lakshmi yoga"],
        description="Timing of financial prosperity and wealth inflow",
        keywords=["paisa", "money", "wealth", "financial", "income", "kab", "when", "dhan"],
        patterns=[
            r"\bpaisa\s+kab\s+(?:aayega|milega|badhega)\b",
            r"\bfinancial\s+stability\s+kab\s+(?:hogi|aayegi)\b",
            r"\bwhen\s+will\s+(?:my\s+)?finances?\s+improve\b",
            r"\bdhan\s+laabh\s+kab\s+hoga\b",
            r"\bpaisa\s+(?:fas|phasa)\s+gaya\b",
            r"\bnikalne\s+ke\s+yog\b",
        ]
    ),
    "wealth_accumulation": MicroIntentDefinition(
        domain="finance",
        intent_id="wealth_accumulation",
        qa_keys=["dhan yoga", "gajakesari yoga", "2nd house", "11th house"],
        description="Long-term wealth prospects, savings accumulation, and assets",
        keywords=["savings", "wealth", "property", "dhan yog", "assets"],
        patterns=[
            r"\bwealth\s+prospects\b",
            r"\bsavings\s+kaise\s+(?:badhaye|karein)\b",
            r"\blong[-\s]term\s+wealth\b",
            r"\bproperty\s+(?:kab\s+banegi|investments)\b",
        ]
    ),

    # ------------------ REMEDIES & GEMSTONES ------------------
    "astrological_remedy_gemstone": MicroIntentDefinition(
        domain="remedies",
        intent_id="astrological_remedy_gemstone",
        qa_keys=["gemstones for career", "gemstone wearing rules"],
        description="Prescribing gemstones for career, wealth, and removing obstacles",
        keywords=["gemstone", "gemstones", "ratna", "pehnne", "ring", "panna", "pukhraj", "neelam", "manikya", "moonga", "moti"],
        patterns=[
            r"\b(?:kaunsa|konsa|which)\s+gemstone\s+(?:pehnne|pehnu|wear|suit)\b",
            r"\bgemstone\s+(?:pehnne|se|for)\b",
            r"\bwhich\s+gemstone\s+(?:suits\s+me|should\s+i\s+wear)\b",
            r"\bratna\s+(?:dharan|kaunsa\s+pehnu)\b",
            r"\b(panna|pukhraj|neelam|ruby|emerald|sapphire|coral)\s+(?:pehnna|suit)\b",
        ]
    ),
    "astrological_remedy_mantra": MicroIntentDefinition(
        domain="remedies",
        intent_id="astrological_remedy_mantra",
        qa_keys=["gayatri mantra", "maha mrityunjaya mantra", "shani shanti"],
        description="Mantras, chanting, and spiritual remedies for mitigation",
        keywords=["mantra", "jaap", "stotra", "chalisa", "upay", "remedy"],
        patterns=[
            r"\bkonsa\s+mantra\s+(?:jaap\s+kare|padhe)\b",
            r"\b(mantra|chalisa|stotra)\s+(?:recommendation|upay)\b",
            r"\bwhich\s+mantra\s+(?:should\s+i\s+chant|helps)\b",
        ]
    ),

    # ------------------ MARRIAGE DOMAIN ------------------
    "marriage_timing": MicroIntentDefinition(
        domain="marriage",
        intent_id="marriage_timing",
        qa_keys=["marriage timing", "7th house"],
        description="Marriage timing, wedding period, and 7th house activation",
        keywords=["marriage", "wedding", "shaadi", "shadi", "vivah", "kab", "when", "planning", "plan"],
        patterns=[
            r"\b(?:shaadi|shadi|vivah)\s*(?:kab|when|ke\s+chances)\s*(?:hogi|hoga|tak|ban\s+rahe)?\b",
            r"\bwhen\s+will\s+i\s+(?:get\s+)?marr(?:y|ied)\b",
            r"\b(?:planning|plan)\s+(?:my\s+)?(?:marriage|shaadi|wedding)\b",
            r"\b(?:marriage|wedding|shaadi)\s+(?:plans?|planning)\b",
            r"\bmarriage\s+(?:timing|timeline|date|window)\b",
        ]
    ),
    "marriage_compatibility": MicroIntentDefinition(
        domain="marriage",
        intent_id="marriage_compatibility",
        qa_keys=["marriage house", "kundali milan", "love marriage", "arranged marriage"],
        description="Spouse nature, relationship harmony, kundali matching",
        keywords=["spouse", "partner", "nature", "compatibility", "milan", "love", "arranged"],
        patterns=[
            r"\bpartner(?:'s)?\s+(?:nature|personality|kaisa\s+hoga)\b",
            r"\b(love|arranged)\s+marriage\b",
            r"\bkundali\s+milan\b",
        ]
    ),

    # ------------------ HEALTH DOMAIN ------------------
    "health_timing": MicroIntentDefinition(
        domain="health",
        intent_id="health_timing",
        qa_keys=["health house", "health remedies"],
        description="Health recovery, wellness vitality, and recovery timing",
        keywords=["health", "illness", "bimar", "recovery", "sehat"],
        patterns=[
            r"\bhealth\s+kab\s+(?:theek|thik|acchi)\s+hogi\b",
            r"\bwhen\s+will\s+(?:my\s+)?health\s+improve\b",
            r"\bsehat\s+kab\s+sudhre gi\b",
        ]
    ),
    "health_sleep_rest": MicroIntentDefinition(
        domain="health",
        intent_id="health_sleep_rest",
        qa_keys=["health_rest_general", "sleep and rest", "sleep", "rest and health", "remedies for mental peace"],
        description="Sleep patterns, nighttime restlessness, insomnia, relaxation, fatigue, and physical rejuvenation",
        keywords=["sleep", "neend", "rest", "insomnia", "tired", "fatigue", "exhaustion", "shayan", "dreams", "relax", "health"],
        patterns=[
            r"\b(?:sleep|neend|rest|insomnia|shayan|fatigue|tired)\b",
            r"\b(?:not\s+able\s+to\s+sleep|sleep\s+issue|trouble\s+sleeping|can't\s+sleep|lack\s+of\s+sleep)\b",
            r"\bneend\s+(?:nahi\s+aati|kharab|nahi\s+aa\s+rahi|kam\s+aati)\b",
            r"\b(?:rest\s+and\s+health|health\s+and\s+rest|sleep\s+and\s+health|sleep\s+and\s+rest)\b",
        ]
    ),
    "health_mitigation": MicroIntentDefinition(
        domain="health",
        intent_id="health_mitigation",
        qa_keys=["remedies for mental peace", "mental stress", "chronic illness"],
        description="Stress relief, mental anxiety mitigation, wellness habits",
        keywords=["stress", "mental", "peace", "anxiety", "depression", "tension"],
        patterns=[
            r"\bmental\s+(?:peace|stress|anxiety)\b",
            r"\btension\s+(?:se\s+bachav|dur\s+kare)\b",
            r"\bhow\s+to\s+reduce\s+mental\s+stress\b",
        ]
    ),

    # ------------------ EDUCATION DOMAIN ------------------
    "education_field": MicroIntentDefinition(
        domain="education",
        intent_id="education_field",
        qa_keys=["education house", "higher education", "mercury for education"],
        description="Field of study, degree selection, stream alignment",
        keywords=["education", "study", "stream", "college", "degree", "field"],
        patterns=[
            r"\bwhich\s+(?:field|stream|subject)\s+(?:for\s+studies|suits\s+me)\b",
            r"\bkaun\s+si\s+stream\s+(?:acchi|suit)\s+karegi\b",
            r"\bhigher\s+education\b",
        ]
    ),
    "education_exams": MicroIntentDefinition(
        domain="education",
        intent_id="education_exams",
        qa_keys=["competitive exams", "government job yoga"],
        description="Competitive exams, entrance tests, government exams",
        keywords=["exam", "exams", "competitive", "entrance", "upsc", "test"],
        patterns=[
            r"\bcompetitive\s+exams?\b",
            r"\bexam\s+clear\s+(?:hoga|hoga\s+kya)\b",
            r"\bwill\s+i\s+clear\s+(?:my\s+)?exams?\b",
        ]
    ),

    # ------------------ TRAVEL DOMAIN ------------------
    "travel_foreign": MicroIntentDefinition(
        domain="travel",
        intent_id="travel_foreign",
        qa_keys=["foreign settlement", "foreign travel", "will i travel abroad"],
        description="Foreign travel, overseas settlement, PR, visas",
        keywords=["foreign", "abroad", "travel", "settlement", "visa", "pr", "overseas"],
        patterns=[
            r"\b(?:foreign\s+travel|abroad|foreign\s+settlement)\b",
            r"\bkya\s+foreign\s+travel\s+ke\s+yog\s+hai\b",
            r"\bwhen\s+will\s+i\s+travel\s+abroad\b",
        ]
    ),
    "travel_foreign_timing": MicroIntentDefinition(
        domain="travel",
        intent_id="travel_foreign_timing",
        qa_keys=["foreign travel timing", "foreign travel", "foreign settlement", "travel abroad"],
        description="Foreign travel timing, overseas journeys, PR, visa approval, relocation",
        keywords=["foreign", "abroad", "travel", "visa", "pr", "overseas", "videsh", "settlement", "relocation", "yatra", "go abroad"],
        patterns=[
            r"\b(?:foreign\s+travel|travel\s+abroad|go\s+abroad|videsh)\s+(?:kab|when|yog|timing)\b",
            r"\bwhen\s+will\s+i\s+(?:travel|go|move|settle)\s+(?:abroad|overseas|foreign)\b",
            r"\bvisa\s+(?:kab\s+milega|approval|timing)\b",
            r"\bforeign\s+settlement\s+kab\b",
        ]
    ),

    # ------------------ PROPERTY & VEHICLES DOMAIN ------------------
    "property_vehicle_timing": MicroIntentDefinition(
        domain="property",
        intent_id="property_vehicle_timing",
        qa_keys=["vehicle purchase", "property and vehicles", "property purchase"],
        description="Timing of vehicle purchase, buying a car, property acquisition, land, home",
        keywords=["car", "vehicle", "gadi", "property", "house", "flat", "makan", "land", "plot", "buy car", "buy house", "vahana", "assets"],
        patterns=[
            r"\b(?:car|gadi|vehicle|bike|automobile)\s+(?:kab|when|buy|purchase|khareedu|khareedne|lena|lenge)\b",
            r"\b(?:buy|purchase|buying|invest\s+in)\s+(?:a\s+)?(?:cars?|vehicles?|house|property|flat|home)\b",
            r"\b(?:property|house|flat|makan|plot|ghar)\s+(?:kab|when|buy|purchase|banega)\b",
            r"\bwhen\s+will\s+i\s+(?:buy|purchase)\s+(?:a\s+)?(?:cars?|vehicles?|house|property|home)\b",
            r"\bvahana\s+(?:sukha|yoga)\b",
        ]
    ),
}

VALID_PLANETS: Dict[str, str] = {
    "rahu": "rahu", "ketu": "ketu", "saturn": "saturn", "shani": "saturn",
    "jupiter": "jupiter", "guru": "jupiter", "brihaspati": "jupiter",
    "mercury": "mercury", "budh": "mercury", "budha": "mercury",
    "mars": "mars", "mangal": "mars", "venus": "venus", "shukra": "venus",
    "sun": "sun", "surya": "sun", "moon": "moon", "chandra": "moon"
}


# ---------------------------------------------------------------------------
# 1. Multi-Domain & Micro-Intent Classifier
# ---------------------------------------------------------------------------
def classify_multi_domain_micro_intents(query: str) -> List[str]:
    """
    Parses a user query and returns ALL detected domain micro-intents.
    Decomposes multi-domain queries into separate micro-intents.
    """
    if not query:
        return ["career_timing"]

    q_clean = query.lower().strip()
    detected_intents: List[str] = []

    features = _get_context_features(query)
    disambiguated = features.get("disambiguated_terms", {})
    negated_domains = set(features.get("negated_domains", []))
    active_domain_signals = features.get("active_domain_signals", [])

    # 1. Explicit Planetary Dasha Decomposition
    planets_regex = r"(rahu|ketu|saturn|shani|jupiter|guru|brihaspati|mercury|budh|budha|mars|mangal|venus|shukra|sun|surya|moon|chandra)"
    dasha_pair_match = re.search(
        rf"\b{planets_regex}\s*(?:mahadasha|maha\s*dasha|dasha)?\s*(?:and|aur|with|-|&|\/|,)?\s*{planets_regex}\s*(?:antardasha|antar\s*dasha|sub-period|bhukti|dasha)\b",
        q_clean
    )
    if not dasha_pair_match:
        dasha_pair_match = re.search(
            rf"\b(?:under|during|in)\s+{planets_regex}(?:'s)?\s*(?:mahadasha|dasha)?\s*(?:and|aur|with|-)\s*{planets_regex}\s*(?:antardasha|sub-period|bhukti|dasha)?\b",
            q_clean
        )

    if dasha_pair_match:
        p1 = VALID_PLANETS.get(dasha_pair_match.group(1).lower(), "rahu")
        p2 = VALID_PLANETS.get(dasha_pair_match.group(2).lower(), "saturn")
        dasha_intent_id = f"dasha_impact_{p1}_{p2}"
        if dasha_intent_id not in detected_intents:
            detected_intents.append(dasha_intent_id)
    else:
        single_dasha_match = re.search(
            rf"\b{planets_regex}\s+(?:mahadasha|maha\s*dasha|antardasha|dasha)\b",
            q_clean
        )
        if single_dasha_match:
            p1 = VALID_PLANETS.get(single_dasha_match.group(1).lower(), "rahu")
            dasha_intent_id = f"dasha_impact_{p1}"
            if dasha_intent_id not in detected_intents:
                detected_intents.append(dasha_intent_id)

    # 2. General Dasha mention without planet names
    if not any(it.startswith("dasha_impact") for it in detected_intents):
        if re.search(r"\b(?:under|during|in)\s+.*(?:dasha|mahadasha)\b", q_clean) or re.search(r"\b(?:mahadasha|antardasha)\s+(?:effects?|impact|results?)\b", q_clean):
            detected_intents.append("dasha_dynamics")

    # 3. Canonical Registry Matching
    for intent_id, intent_def in MICRO_INTENT_REGISTRY.items():
        matched = False
        for pattern in intent_def.patterns:
            if re.search(pattern, q_clean, re.IGNORECASE):
                matched = True
                break

        if not matched:
            kw_hits = sum(1 for kw in intent_def.keywords if re.search(rf"\b{re.escape(kw)}\b", q_clean))
            if kw_hits >= 2:
                matched = True

        if matched and intent_id not in detected_intents:
            detected_intents.append(intent_id)

    # 4. Synchronize Multi-Domain Context Signals
    for signal in active_domain_signals:
        if signal == "career_signal" and "career" not in negated_domains:
            if not any(MICRO_INTENT_REGISTRY.get(it) and MICRO_INTENT_REGISTRY[it].domain == "career" for it in detected_intents):
                if any(w in q_clean for w in ["timing", "when", "kab", "change", "switch", "stabilize", "settle", "badlu"]):
                    detected_intents.append("career_timing")
                elif any(w in q_clean for w in ["business", "partner", "startup", "venture", "field", "best"]):
                    detected_intents.append("career_domain_fit")
                else:
                    detected_intents.append("career_growth_prospects")

        elif signal == "marriage_signal" and "marriage" not in negated_domains:
            if not any(MICRO_INTENT_REGISTRY.get(it) and MICRO_INTENT_REGISTRY[it].domain == "marriage" for it in detected_intents):
                if any(w in q_clean for w in ["timing", "when", "kab", "date", "window", "planning", "plan", "chances"]):
                    detected_intents.append("marriage_timing")
                else:
                    detected_intents.append("marriage_compatibility")

        elif signal == "finance_signal" and "finance" not in negated_domains:
            if not any(MICRO_INTENT_REGISTRY.get(it) and MICRO_INTENT_REGISTRY[it].domain == "finance" for it in detected_intents):
                if any(w in q_clean for w in ["risk", "loss", "nuksan", "invest", "safe", "debt", "karza"]):
                    detected_intents.append("finance_risk")
                elif any(w in q_clean for w in ["when", "kab", "aayega", "inflow", "timing"]):
                    detected_intents.append("finance_timing")
                else:
                    detected_intents.append("finance_stability")

        elif signal == "property_signal" and "property" not in negated_domains:
            if disambiguated.get("property") != "astrological_attribute":
                if not any(MICRO_INTENT_REGISTRY.get(it) and MICRO_INTENT_REGISTRY[it].domain == "property" for it in detected_intents):
                    detected_intents.append("property_vehicle_timing")

        elif signal == "travel_signal" and "travel" not in negated_domains:
            if not any(MICRO_INTENT_REGISTRY.get(it) and MICRO_INTENT_REGISTRY[it].domain == "travel" for it in detected_intents):
                detected_intents.append("travel_foreign_timing")

        elif signal == "health_signal" and "health" not in negated_domains:
            if not any(MICRO_INTENT_REGISTRY.get(it) and MICRO_INTENT_REGISTRY[it].domain == "health" for it in detected_intents):
                if any(w in q_clean for w in ["sleep", "neend", "rest", "insomnia", "tired", "fatigue"]):
                    detected_intents.append("health_sleep_rest")
                else:
                    detected_intents.append("health_timing")

    # 5. Disambiguation Filtering
    if disambiguated.get("partner") == "business":
        if "marriage_compatibility" in detected_intents:
            if not any(re.search(rf"\b{w}\b", q_clean) for w in ["marriage", "shaadi", "vivah", "spouse", "wife", "husband"]):
                detected_intents.remove("marriage_compatibility")

    if disambiguated.get("property") == "astrological_attribute":
        if "property_vehicle_timing" in detected_intents:
            detected_intents.remove("property_vehicle_timing")

    # 6. Negation Filtering
    if negated_domains:
        detected_intents = [
            it for it in detected_intents
            if not (MICRO_INTENT_REGISTRY.get(it) and MICRO_INTENT_REGISTRY[it].domain in negated_domains)
        ]

    if "career_growth_prospects" in detected_intents and "career_timing" in detected_intents:
        timing_words = ["kab", "when", "switch", "change", "right time", "timeline", "date", "stabilize", "settle"]
        if not any(tw in q_clean for tw in timing_words):
            detected_intents.remove("career_timing")

    if "finance_stability" in detected_intents and "finance_timing" in detected_intents:
        timing_words = ["kab", "when", "timeline", "date", "aayega"]
        if not any(tw in q_clean for tw in timing_words):
            detected_intents.remove("finance_timing")

    # Fallback heuristics
    if not detected_intents:
        is_astro_house = bool(re.search(r"\b(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+house\b|\bhouse\s+(?:lord|sign|rashi|ruler)\b", q_clean))
        if any(re.search(rf"\b{re.escape(w)}\b", q_clean) for w in ["car", "cars", "vehicle", "vehicles", "gadi", "flat", "makan", "plot", "land", "vahana"]):
            if disambiguated.get("property") != "astrological_attribute" and "property" not in negated_domains:
                detected_intents.append("property_vehicle_timing")
        elif re.search(r"\bhouse\b", q_clean) and not is_astro_house:
            if disambiguated.get("property") != "astrological_attribute" and "property" not in negated_domains:
                detected_intents.append("property_vehicle_timing")
        elif "property" in q_clean and disambiguated.get("property") != "astrological_attribute" and "property" not in negated_domains:
            detected_intents.append("property_vehicle_timing")
        elif any(re.search(rf"\b{re.escape(w)}\b", q_clean) for w in ["foreign", "abroad", "travel", "visa", "overseas", "videsh"]):
            if "travel" not in negated_domains:
                detected_intents.append("travel_foreign_timing")
        elif any(re.search(rf"\b{re.escape(w)}\b", q_clean) for w in ["career", "job", "work", "naukri"]):
            if "career" not in negated_domains:
                detected_intents.append("career_growth_prospects")
        elif any(re.search(rf"\b{re.escape(w)}\b", q_clean) for w in ["paisa", "money", "finance", "dhan"]):
            if "finance" not in negated_domains:
                detected_intents.append("finance_stability")
        elif any(re.search(rf"\b{re.escape(w)}\b", q_clean) for w in ["shadi", "shaadi", "marriage", "vivah"]):
            if "marriage" not in negated_domains:
                detected_intents.append("marriage_timing")
        elif any(re.search(rf"\b{re.escape(w)}\b", q_clean) for w in ["sleep", "neend", "rest", "insomnia", "tired", "fatigue", "health"]):
            if "health" not in negated_domains:
                detected_intents.append("health_sleep_rest")

    return detected_intents


# ---------------------------------------------------------------------------
# 2. Aggregated Local Database Retrieval & Chart Personalization
# ---------------------------------------------------------------------------
ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

LAGNA_GEMSTONES = {
    "Aries": "Red Coral (Moonga)",
    "Taurus": "Diamond or White Sapphire",
    "Gemini": "Emerald (Panna)",
    "Cancer": "Pearl (Moti)",
    "Leo": "Ruby (Manikya)",
    "Virgo": "Emerald (Panna)",
    "Libra": "Diamond or White Sapphire",
    "Scorpio": "Red Coral (Moonga)",
    "Sagittarius": "Yellow Sapphire (Pukhraj)",
    "Capricorn": "Blue Sapphire (Neelam)",
    "Aquarius": "Blue Sapphire (Neelam)",
    "Pisces": "Yellow Sapphire (Pukhraj)",
}

SIGN_RULERS = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
    "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
    "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter",
}


def build_chart_context_summary(
    user_profile: Dict[str, Any],
    active_houses: Optional[str] = None
) -> str:
    """
    Compresses user profile metadata into a single-line compact notation (~35 tokens):
    '{lagna} Lagna, {moon_sign} Moon | Dasha: {mahadasha}-{antardasha} | Transits: {active_houses} active.'
    """
    lagna = (
        user_profile.get("ascendant_sign")
        or user_profile.get("ascendant")
        or user_profile.get("lagna")
        or "Virgo"
    )
    moon_sign = user_profile.get("moon_sign") or "Taurus"
    mahadasha = user_profile.get("mahadasha") or "Rahu"
    antardasha = user_profile.get("antardasha") or "Saturn"

    if not active_houses:
        active_houses = user_profile.get("active_houses") or "10th/11th"

    return f"{lagna} Lagna, {moon_sign} Moon | Dasha: {mahadasha}-{antardasha} | Transits: {active_houses} active."


def retrieve_aggregated_local_context(
    detected_intents: List[str],
    user_profile: Dict[str, Any],
    qa_db: Dict[str, Any]
) -> str:
    """
    Extracts lean micro-conclusions (1 line per domain, ~110 tokens total)
    plus a single-line chart context summary rather than verbose paragraph dumps.
    """
    try:
        from backend.dasha_calculator import resolve_safe_user_dasha_profile
    except ImportError:
        from dasha_calculator import resolve_safe_user_dasha_profile

    safe_profile = resolve_safe_user_dasha_profile(user_profile)
    asc_sign = safe_profile.get("ascendant_sign", "Virgo")
    moon_sign = safe_profile.get("moon_sign", "Taurus")
    sun_sign = safe_profile.get("sun_sign", "Gemini")
    mahadasha = safe_profile.get("mahadasha", "Rahu")
    antardasha = safe_profile.get("antardasha", "Saturn")
    pratyantardasha = safe_profile.get("pratyantardasha", "Mercury")

    # Determine primary inquiry domains
    domain_map = {
        "property": "4th",
        "vehicle": "4th",
        "travel": "9th/12th",
        "foreign": "9th/12th",
        "finance": "2nd/11th",
        "marriage": "7th",
        "health": "6th/12th",
        "education": "5th/9th",
        "career": "10th/6th"
    }

    active_house_parts = []
    primary_domain = "career"
    for it in detected_intents:
        for dom_key, house_val in domain_map.items():
            if dom_key in it:
                if house_val not in active_house_parts:
                    active_house_parts.append(house_val)
                primary_domain = dom_key
                break

    active_houses_str = ", ".join(active_house_parts) if active_house_parts else "10th/11th"

    # Compute dynamic Gochar timing window via timing_engine if available
    timing_window = "Favorable Transit Window"
    try:
        from backend.timing_engine import calculate_astrological_timing_window
        resolved_birth = (
            user_profile.get("session_birth_data")
            or safe_profile.get("session_birth_data")
        )
        if not resolved_birth and any(k in user_profile for k in ("birth_year", "birth_month", "birth_day")):
            resolved_birth = user_profile

        resolved_chart = user_profile.get("chart_data") or safe_profile.get("chart_data")
        d_start = safe_profile.get("dasha_start_date") or user_profile.get("dasha_start_date")
        d_end = safe_profile.get("dasha_end_date") or user_profile.get("dasha_end_date")

        calced_timing = calculate_astrological_timing_window(
            domain=primary_domain,
            ascendant_sign=asc_sign,
            moon_sign=moon_sign,
            mahadasha=mahadasha,
            antardasha=antardasha,
            session_birth_data=resolved_birth,
            chart_data=resolved_chart,
            dasha_start_date=d_start,
            dasha_end_date=d_end
        )
        timing_window = calced_timing.get("timing_window") or timing_window
    except Exception:
        pass

    # 1. Single-line compact chart context summary (~35 tokens)
    chart_summary_line = build_chart_context_summary(safe_profile, active_houses=active_houses_str)

    # 2. Extract lean micro-conclusions (1 line per domain, ~110 tokens total)
    seen_domains = set()
    micro_conclusions: List[str] = []

    for intent_id in detected_intents:
        if intent_id.startswith("dasha_impact_"):
            if "dasha" not in seen_domains:
                seen_domains.add("dasha")
                parts = intent_id.replace("dasha_impact_", "").split("_")
                p1 = parts[0] if len(parts) > 0 else "rahu"
                p2 = parts[1] if len(parts) > 1 else "saturn"
                micro_conclusions.append(
                    f"[DASHA]: {p1.capitalize()}-{p2.capitalize()} Dasha combines {p1.capitalize()}'s ambition "
                    f"with {p2.capitalize()}'s discipline, requiring methodical execution."
                )
            continue

        intent_def = MICRO_INTENT_REGISTRY.get(intent_id)
        if not intent_def or intent_def.domain in seen_domains:
            continue

        seen_domains.add(intent_def.domain)
        retrieved_fact = ""
        for qk in intent_def.qa_keys:
            val = qa_db.get(qk)
            if val and isinstance(val, str):
                # Personalize placeholders
                clean_val = val.replace("[Mahadasha]", mahadasha)
                clean_val = clean_val.replace("[Antardasha]", antardasha)
                clean_val = clean_val.replace("[Pratyantardasha]", pratyantardasha)
                clean_val = clean_val.replace("[Job Change Window]", timing_window)
                clean_val = clean_val.replace("{moon_sign}", moon_sign)
                clean_val = clean_val.replace("{ascendant}", asc_sign)
                clean_val = clean_val.replace("{sun_sign}", sun_sign)
                # Take exactly 1 concise sentence
                sentences = [s.strip() for s in re.split(r"[.\n]+", clean_val) if len(s.strip()) > 15]
                if sentences:
                    retrieved_fact = sentences[0]
                    break

        if retrieved_fact:
            micro_conclusions.append(f"[{intent_def.domain.upper()}]: {retrieved_fact}.")
        else:
            # Fallback lean micro-conclusion if template lookup is empty
            dom = intent_def.domain
            if dom == "career":
                micro_conclusions.append(f"[CAREER]: 10th house indicates steady advancement through perseverance; favorable window: {timing_window}.")
            elif dom == "finance":
                micro_conclusions.append("[FINANCE]: 2nd and 11th houses govern accumulated wealth; conservative financial planning is advised.")
            elif dom == "property":
                micro_conclusions.append(f"[PROPERTY]: 4th house alignments support vehicle and fixed asset acquisition during {timing_window}.")
            elif dom == "travel":
                micro_conclusions.append(f"[TRAVEL]: 9th and 12th houses activate foreign journeys and documentation clearance ({timing_window}).")
            elif dom == "marriage":
                micro_conclusions.append(f"[MARRIAGE]: 7th house indicates auspicious partnership prospects within {timing_window}.")
            elif dom == "health":
                micro_conclusions.append("[HEALTH]: 6th and 12th houses emphasize restorative rest and balanced evening routines.")

    evidence_body = "\n".join(micro_conclusions) if micro_conclusions else f"[{primary_domain.upper()}]: Supportive planetary cycle ({timing_window})."

    return f"{chart_summary_line}\n\n{evidence_body}"


# ---------------------------------------------------------------------------
# 3. LLM Prose Refiner & Synthesizer Prompt Structure
# ---------------------------------------------------------------------------
def build_llm_refiner_prompts(
    query: str,
    raw_context: str,
    is_hinglish: bool = False,
    domains: Optional[List[str]] = None
) -> Tuple[str, str]:
    """
    Constructs the system prompt and user prompt for the LLM Prose Refiner.
    Strictly instructs the LLM to synthesize all detected domains without hallucination.
    """
    try:
        from backend.answer_synthesizer import build_multidomain_system_prompt
    except ImportError:
        from answer_synthesizer import build_multidomain_system_prompt

    system_prompt = build_multidomain_system_prompt(domains=domains, is_hinglish=is_hinglish)
    user_prompt = (
        f"Q: {query}\n\n"
        f"Evidence:\n"
        f"{raw_context}"
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# 4. Standardized Multi-Domain Attribution Footer Helper
# ---------------------------------------------------------------------------
def append_multidomain_attribution_footer(answer: str, domains: Optional[List[str]] = None) -> str:
    """
    Standardizes the final response by appending the required Multi-Domain Hybrid Synthesis footer.
    """
    try:
        from backend.answer_synthesizer import append_multidomain_footer
        return append_multidomain_footer(answer, domains or ["career", "finance"])
    except ImportError:
        from answer_synthesizer import append_multidomain_footer
        return append_multidomain_footer(answer, domains or ["career", "finance"])


# ---------------------------------------------------------------------------
# Backwards Compatibility Helpers for Career Timing
# ---------------------------------------------------------------------------
def is_career_timing_query(query: str) -> bool:
    """Retains backward compatibility for single-intent career queries."""
    q_low = (query or "").lower().strip()
    has_career = any(w in q_low for w in ["job", "career", "naukri", "promotion", "switch", "work"])
    intents = classify_multi_domain_micro_intents(query)
    return any(k in intents for k in ["career_timing", "career_growth_prospects"]) and has_career


async def execute_multidomain_hybrid_synthesis(
    query: str,
    detected_intents: List[str],
    user_profile: Dict[str, Any],
    qa_db: Dict[str, Any],
    session_id: str = "default_user",
    custom_gemini_client: Any = None,
    is_hinglish: bool = False
) -> Dict[str, Any]:
    """
    Executes the end-to-end Multi-Domain Local Retrieval + LLM Refiner synthesis.
    """
    try:
        from backend.answer_synthesizer import (
            build_context_payload,
            synthesizer_multidomain_response,
            format_multidomain_footer,
        )
    except ImportError:
        from answer_synthesizer import (
            build_context_payload,
            synthesizer_multidomain_response,
            format_multidomain_footer,
        )

    domains = []
    for it in detected_intents:
        if it.startswith("dasha_impact") or "dasha" in it:
            if "dasha" not in domains:
                domains.append("dasha")
        elif it in MICRO_INTENT_REGISTRY:
            d = MICRO_INTENT_REGISTRY[it].domain
            if d not in domains:
                domains.append(d)

    try:
        from backend.dasha_calculator import resolve_safe_user_dasha_profile
    except ImportError:
        from dasha_calculator import resolve_safe_user_dasha_profile

    safe_profile = resolve_safe_user_dasha_profile(user_profile)

    raw_context = retrieve_aggregated_local_context(detected_intents, safe_profile, qa_db)
    context_payload = build_context_payload(raw_context, detected_intents, safe_profile)

    return await synthesizer_multidomain_response(
        query=query,
        context_payload=context_payload,
        detected_intents=detected_intents,
        domains=domains,
        user_profile=safe_profile,
        session_id=session_id,
        custom_gemini_client=custom_gemini_client,
        is_hinglish=is_hinglish
    )


def handle_career_timing_query(
    query: str,
    raw_template: str,
    session_birth_data: Dict[str, Any],
    ascendant_sign: str = "Virgo",
    moon_longitude: Optional[float] = None,
    is_hinglish: bool = False
) -> Dict[str, Any]:
    from backend.dasha_calculator import format_career_timing_response
    return format_career_timing_response(
        query=query,
        raw_template=raw_template,
        session_birth_data=session_birth_data,
        ascendant_sign=ascendant_sign,
        moon_longitude=moon_longitude,
        is_hinglish=is_hinglish
    )


def is_timing_query(query: str) -> bool:
    """Detects whether query seeks astrological timing, date ranges, or favorable windows."""
    try:
        from backend.timing_engine import is_timing_query as _itq
        return _itq(query)
    except ImportError:
        from timing_engine import is_timing_query as _itq
        return _itq(query)


async def handle_timing_query(
    query: str,
    user_profile: Dict[str, Any],
    session_birth_data: Optional[Dict[str, Any]] = None,
    chart_data: Optional[Dict[str, Any]] = None,
    session_id: str = "default_user",
    custom_gemini_client: Any = None,
    is_hinglish: bool = False
) -> Dict[str, Any]:
    """
    Executes the specialized Timing Engine pipeline.
    """
    try:
        from backend.timing_engine import execute_timing_pipeline
    except ImportError:
        from timing_engine import execute_timing_pipeline

    return await execute_timing_pipeline(
        query=query,
        user_profile=user_profile,
        session_birth_data=session_birth_data,
        chart_data=chart_data,
        session_id=session_id,
        custom_gemini_client=custom_gemini_client,
        is_hinglish=is_hinglish
    )
