"""
Multi-Domain Micro-Intent NLP Router & Local QA Database Resolver
=================================================================
Module: backend/qa_router.py

Implements fine-grained micro-intent categorization across Vedic astrology
domains (Career, Finance, Health, Marriage, Education, Travel, Spirituality, Gemstones).
Replaces broad, generic keyword matching with specialized syntactic & semantic
pattern recognition distinguishing timing, probability, domain fit, risk, and mitigation.

Supports:
1. Multi-Domain Micro-Intent Classification with Hinglish + English parsing.
2. Dedicated mapping to hyper-personalized qa_database.json templates.
3. Fallback Safety: Detects ambiguity when queries span multiple competing micro-intents
   and triggers the Hybrid Local DB + LLM synthesis pipeline.
"""

import re
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("AstroChatbot.QARouter")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class MicroIntentDefinition:
    domain: str
    intent_id: str
    qa_key: str
    description: str
    keywords: List[str]
    patterns: List[str]
    sample_queries: List[str]


# ---------------------------------------------------------------------------
# Canonical Multi-Domain Micro-Intent Registry
# ---------------------------------------------------------------------------
MICRO_INTENT_REGISTRY: Dict[str, MicroIntentDefinition] = {
    # ------------------ CAREER DOMAIN ------------------
    "career_timing": MicroIntentDefinition(
        domain="career",
        intent_id="career_timing",
        qa_key="career_timing",
        description="Timing of job changes, promotions, interviews, or hiring milestones",
        keywords=["job", "jobs", "career", "naukri", "promotion", "appraisal", "switch", "transfer", "joining"],
        patterns=[
            r"\b(kab\s+milegi\s+(?:naukri|job))\b",
            r"\b(job|naukri)\s+kab\s+(milegi|lagegi|hogi)\b",
            r"\b(promotion|appraisal|tarakki)\s+kab\s+(hoga|milega)\b",
            r"\bwhen\s+will\s+i\s+(?:get|change|switch|find)\s+(?:a\s+)?(?:jobs?|career|work|promotion)\b",
            r"\b(timing|time)\s+(?:for|of)\s+(?:jobs?|career|promotion|switch)\b",
            r"\b(job\s+switch|job\s+change|promotion)\s+(kab|when|timing)\b",
            r"\bkab\s+(?:hoga|milega)\s+(?:promotion|transfer|increment)\b",
        ],
        sample_queries=["kab milegi job", "promotion kab hoga", "when will I change jobs", "job switch kab karun"]
    ),
    "career_probability": MicroIntentDefinition(
        domain="career",
        intent_id="career_probability",
        qa_key="career_probability",
        description="Likelihood, chances, and probability of landing a job or succeeding in career",
        keywords=["job", "career", "naukri", "success", "chance", "probability", "asar", "kismat"],
        patterns=[
            r"\b(job|naukri)\s+milegi\s+ya\s+nahi\b",
            r"\bkya\s+(?:mujhe\s+)?(job|naukri)\s+milegi\b",
            r"\bcareer\s+(?:mein|me)\s+(?:success|safalta)\s+ke\s+(?:asar|chances)\b",
            r"(?<!when\s)\bwill\s+i\s+(?:get|clear|succeed\s+in)\s+(?:a\s+)?(?:job|interview|career)\b",
            r"\bchances\s+of\s+(?:getting\s+a\s+job|career\s+success|promotion)\b",
            r"\bkya\s+career\s+accha\s+rahega\b",
            r"\bsuccess\s+milegi\s+ya\s+nahi\b",
        ],
        sample_queries=["job milegi ya nahi", "career mein success ke asar", "will I get a good job", "kya mujhe naukri milegi"]
    ),
    "career_domain_fit": MicroIntentDefinition(
        domain="career",
        intent_id="career_domain_fit",
        qa_key="career_domain_fit",
        description="Suitability for business vs salaried job, industry choice, corporate vs startup",
        keywords=["business", "job", "industry", "field", "corporate", "startup", "profession", "suit"],
        patterns=[
            r"\bbusiness\s+(?:karu|karein|karun)\s+ya\s+(?:job|naukri)\b",
            r"\bjob\s+acchi\s+hai\s+ya\s+business\b",
            r"\bwhich\s+(?:industry|field|career|profession)\s+(?:suits|is\s+best\s+for)\s+(?:me|my\s+lagna|my\s+chart)\b",
            r"\bkaunsa\s+(?:field|industry|business)\s+(?:mere\s+liye|suit)\s+(?:karega|accha\s+hai)\b",
            r"\b(?:business\s+vs\s+job|job\s+vs\s+business)\b",
            r"\bcorporate\s+vs\s+startup\b",
            r"\bwhat\s+(?:career|job)\s+suits\s+me\b",
        ],
        sample_queries=["business karu ya job", "which industry suits me", "business vs job", "kaunsa field mere liye accha hai"]
    ),

    # ------------------ FINANCE DOMAIN ------------------
    "finance_timing": MicroIntentDefinition(
        domain="finance",
        intent_id="finance_timing",
        qa_key="finance_timing",
        description="Timing of financial prosperity, wealth inflow, and liquidity relief",
        keywords=["paisa", "money", "wealth", "financial", "stability", "income", "kab", "when"],
        patterns=[
            r"\bpaisa\s+kab\s+(?:aayega|milega|badhega)\b",
            r"\bfinancial\s+stability\s+kab\s+(?:hogi|aayegi)\b",
            r"\bwhen\s+will\s+(?:my\s+)?(?:finances|wealth|financial\s+situation)\s+improve\b",
            r"\bwhen\s+will\s+i\s+(?:become|get)\s+(?:financially\s+stable|wealthy|rich)\b",
            r"\btiming\s+of\s+(?:financial\s+gain|wealth|money)\b",
            r"\bdhan\s+laabh\s+kab\s+hoga\b",
        ],
        sample_queries=["paisa kab aayega", "financial stability kab hogi", "when will my finances improve", "dhan kab aayega"]
    ),
    "finance_risk": MicroIntentDefinition(
        domain="finance",
        intent_id="finance_risk",
        qa_key="finance_risk",
        description="Assessing financial risks, debt burden, loans, and stock market volatility",
        keywords=["loan", "debt", "karza", "loss", "nuksan", "share", "stock", "trading", "risk"],
        patterns=[
            r"\bloan\s+lena\s+chahiye\s+kya\b",
            r"\bkarza\s+(?:hoga|chuke ga|lena\s+sahi\s+hai)\b",
            r"\bstock\s+market\s+(?:mein|me)\s+(?:loss|nuksan|invest)\b",
            r"\bshare\s+market\s+(?:mein|me)\s+(?:invest|paisa|trade|loss|nuksan)\b",
            r"\bshares?\s+(?:mein|me)\s+(?:invest\s+karu|paisa\s+lagau)\b",
            r"\bis\s+it\s+safe\s+to\s+(?:invest|take\s+a\s+loan|trade)\b",
            r"\brisk\s+of\s+(?:financial\s+loss|debt|bankruptcy)\b",
            r"\bkya\s+share\s+market\s+mein\s+loss\s+hoga\b",
        ],
        sample_queries=["loan lena chahiye kya", "stock market mein loss hoga kya", "debt risk in my chart", "is investing in shares safe"]
    ),
    "wealth_accumulation": MicroIntentDefinition(
        domain="finance",
        intent_id="wealth_accumulation",
        qa_key="wealth_accumulation",
        description="Long-term wealth prospects, asset creation, compounding, and savings accumulation",
        keywords=["wealth", "savings", "accumulation", "assets", "property", "dhan", "bachat"],
        patterns=[
            r"\bwealth\s+prospects\b",
            r"\bsavings\s+kaise\s+(?:badhaye|karein|increase\s+kare)\b",
            r"\blong[-\s]term\s+wealth\s+(?:accumulation|prospects)\b",
            r"\b2nd\s+and\s+11th\s+house\b",
            r"\bproperty\s+(?:kab\s+banegi|investments|assets)\b",
            r"\bhow\s+to\s+accumulate\s+wealth\b",
            r"\bmere\s+chart\s+mein\s+dhan\s+yog\s+kaisa\s+hai\b",
        ],
        sample_queries=["wealth prospects", "savings kaise badhaye", "long term wealth accumulation", "dhan yog prospects"]
    ),

    # ------------------ HEALTH DOMAIN ------------------
    "health_timing": MicroIntentDefinition(
        domain="health",
        intent_id="health_timing",
        qa_key="health_timing",
        description="Timing of recovery, health improvement, and vitality resurgence",
        keywords=["health", "recovery", "bimar", "illness", "kab", "theek", "vitality", "when"],
        patterns=[
            r"\bhealth\s+kab\s+(?:theek|thik|acchi)\s+(?:hogi|hoga)\b",
            r"\brecovery\s+(?:period|kab\s+hogi)\b",
            r"\bwhen\s+will\s+(?:my\s+)?(?:health|illness|vitality)\s+(?:improve|recover|heal)\b",
            r"\bbimari\s+se\s+chhutkara\s+kab\s+milega\b",
            r"\bsehat\s+kab\s+sudhre gi\b",
        ],
        sample_queries=["health kab theek hogi", "recovery period", "when will my health improve", "sehat kab theek hogi"]
    ),
    "health_mitigation": MicroIntentDefinition(
        domain="health",
        intent_id="health_mitigation",
        qa_key="health_mitigation",
        description="Identifying chronic risks, stress management, mental peace, and health remedies",
        keywords=["roag", "disease", "stress", "mental", "peace", "remedies", "upay", "chronic", "anxiety"],
        patterns=[
            r"\bkonsa\s+(?:roag|rog|disease|illness|stress)\s+(?:aane\s+wala|hone\s+ke\s+chances)\b",
            r"\bremedies\s+for\s+mental\s+(?:peace|calm|stress)\b",
            r"\bmental\s+stress\s+(?:kaise\s+dur\s+kare|se\s+bachav)\b",
            r"\bhealth\s+remedies\b",
            r"\bprevent\s+(?:illness|chronic\s+disease|fatigue)\b",
            r"\bsehat\s+ke\s+upay\b",
        ],
        sample_queries=["konsa roag ya stress aane wala hai", "remedies for mental peace", "health remedies", "chronic stress prevention"]
    ),

    # ------------------ MARRIAGE DOMAIN ------------------
    "marriage_timing": MicroIntentDefinition(
        domain="marriage",
        intent_id="marriage_timing",
        qa_key="marriage_timing",
        description="Marriage timing, wedding period, and 7th house activation cycle",
        keywords=["marriage", "wedding", "shaadi", "shadi", "vivah", "kab", "when", "timing", "hogi", "hoga", "date"],
        patterns=[
            r"\b(?:meri\s+|mera\s+)?(?:shaadi|shadi|vivah)\s*(?:kab|when|timing)\s*(?:hogi|hoga|tak)?\b",
            r"\bkab\s+(?:hogi|hoga)\s*(?:meri\s+)?(?:shaadi|shadi|vivah)\b",
            r"\bwhen\s+will\s+i\s+(?:get\s+)?marr(?:y|ied)\b",
            r"\b(?:marriage|wedding)\s+(?:timing|timeline|date|year)\b",
            r"\b7th\s+house\s+(?:marriage\s+)?activation\b",
            r"\bvivah\s+kab\s+hoga\b",
        ],
        sample_queries=["meri shadi kab hogi", "shaadi kab hogi", "when will I get married", "marriage timing", "7th house marriage activation"]
    ),
    "marriage_probability": MicroIntentDefinition(
        domain="marriage",
        intent_id="marriage_probability",
        qa_key="marriage_probability",
        description="Likelihood, chances, and probability of getting married vs denial/delays",
        keywords=["marriage", "wedding", "shaadi", "shadi", "vivah", "asar", "chance", "chances", "probability", "yog", "hogi", "hoga", "nahi", "nhi"],
        patterns=[
            r"\b(?:shaadi|shadi|vivah)\s*(?:hone\s+ke\s+)?(?:asar|chances?|sambhavna|yog)\s*(?:hai|hain)?\s*(?:ya\s*(?:nahi|nhi|na))?\b",
            r"\b(?:shaadi|shadi|vivah)\s*(?:hogi|hoga)\s*ya\s*(?:nahi|nhi|na)\b",
            r"(?<!when\s)\bwill\s+i\s+(?:ever\s+)?get\s+married\b",
            r"\bchances\s+of\s+(?:getting\s+)?marr(?:y|ied)\b",
            r"\bmarriage\s+(?:chances|probability)\b",
        ],
        sample_queries=["mere shadi hone ke asar hai ya nhi", "shaadi hogi ya nahi", "will i get married", "marriage chances"]
    ),
    "marriage_decision": MicroIntentDefinition(
        domain="marriage",
        intent_id="marriage_decision",
        qa_key="marriage_decision",
        description="Should I marry or not, marriage suitability vs singlehood, commitment decision",
        keywords=["marriage", "wedding", "shaadi", "shadi", "vivah", "karun", "karu", "karein", "chahiye", "single", "decision", "marry", "nahi", "nhi"],
        patterns=[
            r"\b(?:mai|mujhe)?\s*(?:shaadi|shadi|vivah)\s*(?:karun|karu|karna|karni)\s*(?:chahiye\s*)?(?:ya|or)\s*(?:nahi|nhi|na)\b",
            r"\bshould\s+i\s+(?:get\s+)?marr(?:y|ied)\s*(?:or\s+not|or\s+stay\s+single)?\b",
            r"\b(?:shaadi|shadi|vivah)\s*(?:karun|karu|karein|karna\s+chahiye)\b",
            r"\bmarriage\s+(?:decision|suitability)\b",
        ],
        sample_queries=["mai shadi karu ya nhi", "should i get married or not", "shaadi karun ya nahi"]
    ),
    "partner_profile": MicroIntentDefinition(
        domain="marriage",
        intent_id="partner_profile",
        qa_key="partner_profile",
        description="Future life partner personality traits, appearance, nature, and background",
        keywords=["partner", "spouse", "husband", "wife", "traits", "nature", "kaisa", "personality"],
        patterns=[
            r"\b(?:life\s+)?partner\s+(?:kaisa|kaisi)\s+(?:hoga|hogi)\b",
            r"\bpersonality\s+traits\s+of\s+(?:my\s+)?(?:future\s+)?(?:spouse|partner|husband|wife)\b",
            r"\bspouse\s+(?:nature|personality|appearance|look)\b",
            r"\bpatni\s+ya\s+pati\s+kaisa\s+milega\b",
        ],
        sample_queries=["life partner kaisa hoga", "future life partner personality traits", "spouse nature and traits"]
    ),
    "relationship_compatibility": MicroIntentDefinition(
        domain="marriage",
        intent_id="relationship_compatibility",
        qa_key="relationship_compatibility",
        description="Compatibility analysis, love vs arranged marriage, and Kundali Milan prospects",
        keywords=["compatibility", "love", "arranged", "milan", "guna", "relationship"],
        patterns=[
            r"\b(?:love\s+marriage\s+or\s+arranged|arranged\s+vs\s+love)\b",
            r"\bcompatibility\s+kaisa\s+hai\b",
            r"\bkundali\s+milan\b",
            r"\bwill\s+i\s+have\s+a\s+love\s+marriage\b",
            r"\bmarital\s+harmony\b",
        ],
        sample_queries=["love or arranged marriage", "compatibility kaisa hai", "kundali milan evaluation", "love marriage chances"]
    ),
    "relationship_mitigation": MicroIntentDefinition(
        domain="marriage",
        intent_id="relationship_mitigation",
        qa_key="relationship_mitigation",
        description="Remedies for marriage delays, resolving marital friction, and Mangal Dosha mitigation",
        keywords=["delay", "upay", "remedies", "mangal", "dosha", "conflict", "friction"],
        patterns=[
            r"\bshaadi\s+mein\s+delay\s+ke\s+upay\b",
            r"\bremedies\s+for\s+(?:marriage|relationship)\s+delays?\b",
            r"\bmangal\s+dosha\s+remedies\b",
            r"\bresolve\s+relationship\s+(?:delays|conflict|obstacles)\b",
        ],
        sample_queries=["shaadi mein delay ke upay", "remedies for marriage delay", "which astrological remedies resolve relationship delays"]
    ),

    # ------------------ EDUCATION DOMAIN ------------------
    "education_field_fit": MicroIntentDefinition(
        domain="education",
        intent_id="education_field_fit",
        qa_key="education_field_fit",
        description="Educational stream selection, higher education field alignment with 5th and 9th houses",
        keywords=["education", "study", "stream", "field", "degree", "college", "higher"],
        patterns=[
            r"\bwhich\s+field\s+of\s+higher\s+education\b",
            r"\bstream\s+selection\b",
            r"\bkaunsa\s+subject\s+ya\s+field\s+padhu\b",
            r"\bscience\s+arts\s+commerce\b",
            r"\b5th\s+and\s+9th\s+house\s+education\b",
        ],
        sample_queries=["which field of higher education suits me", "stream selection in study", "kaunsa field padhu"]
    ),
    "exam_timing_success": MicroIntentDefinition(
        domain="education",
        intent_id="exam_timing_success",
        qa_key="exam_timing_success",
        description="Competitive exam success probability, timing of academic achievements",
        keywords=["exam", "competitive", "clear", "pass", "upsc", "sarkari", "test"],
        patterns=[
            r"\bcompetitive\s+exam\s+(?:clear\s+hoga|success|timing)\b",
            r"\bwill\s+i\s+pass\s+(?:my\s+)?exam\b",
            r"\bexam\s+clear\s+hoga\s+kya\b",
            r"\bacademic\s+success\s+period\b",
        ],
        sample_queries=["competitive exam clear hoga kya", "will I clear competitive exams", "exam timing and success"]
    ),
    "study_abroad": MicroIntentDefinition(
        domain="education",
        intent_id="study_abroad",
        qa_key="study_abroad",
        description="Favorable yogas and dasha timing for studying in foreign universities",
        keywords=["abroad", "foreign", "study", "overseas", "university", "masters"],
        patterns=[
            r"\bstudying\s+abroad\b",
            r"\bforeign\s+education\s+yoga\b",
            r"\bbahar\s+padhai\s+karne\s+ke\s+chances\b",
            r"\bcan\s+i\s+study\s+abroad\b",
        ],
        sample_queries=["are there favorable yogas for studying abroad", "can I study abroad", "foreign study chances"]
    ),

    # ------------------ TRAVEL / RELOCATION DOMAIN ------------------
    "foreign_travel_timing": MicroIntentDefinition(
        domain="travel",
        intent_id="foreign_travel_timing",
        qa_key="foreign_travel_timing",
        description="Timing of international travel, vacations, and short-term overseas journeys",
        keywords=["travel", "foreign", "trip", "journey", "abroad", "kab", "when"],
        patterns=[
            r"\bforeign\s+travel\s+kab\s+hoga\b",
            r"\bwhen\s+will\s+i\s+travel\s+abroad\b",
            r"\btravel\s+abroad\s+timing\b",
            r"\bforeign\s+trip\s+kab\b",
        ],
        sample_queries=["when is the most favorable time for foreign travel", "foreign travel kab hoga", "when will I travel abroad"]
    ),
    "foreign_settlement_fit": MicroIntentDefinition(
        domain="travel",
        intent_id="foreign_settlement_fit",
        qa_key="foreign_settlement_fit",
        description="Permanent foreign residence, immigration, PR opportunities via 12th house",
        keywords=["settlement", "permanent", "pr", "citizenship", "relocate", "relocation", "12th"],
        patterns=[
            r"\bpermanent\s+foreign\s+settlement\b",
            r"\bbahar\s+settle\s+ho\s+sakta\s+hu\s+kya\b",
            r"\bwill\s+i\s+settle\s+abroad\b",
            r"\bforeign\s+pr\s+yoga\b",
            r"\bsettlement\s+via\s+12th\s+house\b",
        ],
        sample_queries=["does my chart support permanent foreign settlement", "foreign PR chances", "bahar settle hone ke yog"]
    ),
    "travel_obstacle_mitigation": MicroIntentDefinition(
        domain="travel",
        intent_id="travel_obstacle_mitigation",
        qa_key="travel_obstacle_mitigation",
        description="Remedies to eliminate visa rejections, relocation hurdles, and paperwork roadblocks",
        keywords=["visa", "hurdles", "obstacles", "remedies", "rejection", "upay"],
        patterns=[
            r"\bvisa\s+(?:problem|rejection|obstacles|delay)\s+remedies\b",
            r"\bvisa\s+clear\s+karne\s+ke\s+upay\b",
            r"\bremedies\s+for\s+visa\s+or\s+relocation\s+obstacles\b",
            r"\bvisa\s+kab\s+aayega\b",
        ],
        sample_queries=["which remedies remove visa or relocation obstacles", "visa problem upay", "visa rejection remedies"]
    ),

    # ------------------ SPIRITUALITY DOMAIN ------------------
    "ishta_devata_guidance": MicroIntentDefinition(
        domain="spirituality",
        intent_id="ishta_devata_guidance",
        qa_key="ishta_devata_guidance",
        description="Discovering personal Ishta Devata, resonant mantras, and deity alignment via 5th house",
        keywords=["ishta", "devata", "deity", "mantra", "puja", "worship", "bhagwan"],
        patterns=[
            r"\bwho\s+is\s+my\s+ishta\s+devata\b",
            r"\bwhich\s+deity\s+or\s+mantra\s+aligns\s+with\s+my\s+ishta\s+devata\b",
            r"\bmere\s+ishta\s+dev\s+kaun\s+hai\b",
            r"\bkaunse\s+bhagwan\s+ki\s+puja\s+karu\b",
        ],
        sample_queries=["which deity or mantra aligns with my ishta devata", "who is my ishta devata", "ishta devata guidance"]
    ),
    "karmic_dosha_mitigation": MicroIntentDefinition(
        domain="spirituality",
        intent_id="karmic_dosha_mitigation",
        qa_key="karmic_dosha_mitigation",
        description="Remedies for active karmic doshas like Pitra Dosha, Kaal Sarp, and Sade Sati",
        keywords=["pitra", "kaal sarp", "sade sati", "dosha", "remedies", "upay", "shanti"],
        patterns=[
            r"\bremedies\s+(?:for\s+)?(?:karmic\s+)?doshas?\b",
            r"\bpitra\s+dosha\s+remedies\b",
            r"\bkaal\s+sarp\s+(?:dosha\s+)?remedies\b",
            r"\bsade\s+sati\s+(?:se\s+bachav|remedies|upay)\b",
            r"\bmitigate\s+active\s+karmic\s+doshas\b",
        ],
        sample_queries=["what remedies mitigate my active karmic doshas", "pitra dosha remedies", "kaal sarp upay"]
    ),
    "spiritual_growth_dharma": MicroIntentDefinition(
        domain="spirituality",
        intent_id="spiritual_growth_dharma",
        qa_key="spiritual_growth_dharma",
        description="9th house spiritual evolution, dharma path, and enlightenment indicators",
        keywords=["dharma", "spiritual", "growth", "evolution", "moksha", "9th"],
        patterns=[
            r"\bspiritual\s+(?:growth|evolution|path)\b",
            r"\b9th\s+house\s+(?:spiritual\s+)?dharma\b",
            r"\bmoksha\s+(?:prospects|path)\b",
            r"\badhyatmik\s+unnati\b",
        ],
        sample_queries=["how does my 9th house shape my spiritual dharma and growth", "spiritual growth path", "dharma guidance"]
    ),

    # ------------------ GEMSTONES DOMAIN ------------------
    "gemstone_recommendation": MicroIntentDefinition(
        domain="gemstones",
        intent_id="gemstone_recommendation",
        qa_key="gemstone_recommendation",
        description="Identifying most auspicious gemstone for Lagna lord, career lord, or wealth lord",
        keywords=["gemstone", "ratna", "pukhraj", "neelam", "manik", "panna", "recommend", "wear"],
        patterns=[
            r"\bwhich\s+gemstone\s+is\s+most\s+auspicious\b",
            r"\bkaunsa\s+ratna\s+(?:pahanu|pehnna\s+chahiye)\b",
            r"\bgemstone\s+for\s+(?:career|lagna|fortune)\b",
            r"\bwhich\s+gemstone\s+suits\s+me\b",
        ],
        sample_queries=["which gemstone is most auspicious for my lagna lord", "kaunsa ratna pahanu", "gemstone recommendation"]
    ),
    "gemstone_wearing_rules": MicroIntentDefinition(
        domain="gemstones",
        intent_id="gemstone_wearing_rules",
        qa_key="gemstone_wearing_rules",
        description="Vedic energization rituals, Shukla Paksha timings, fingers, and metals for wearing gemstones",
        keywords=["rules", "timing", "energize", "shukla paksha", "pran pratishtha", "finger", "niyam"],
        patterns=[
            r"\bvedic\s+rules\s+and\s+timings\s+for\s+energizing\s+my\s+gemstone\b",
            r"\bgemstone\s+(?:pehnne|pahanne)\s+ke\s+niyam\b",
            r"\bhow\s+to\s+energize\s+gemstones?\b",
            r"\bgemstone\s+wearing\s+rules\b",
        ],
        sample_queries=["what are the vedic rules and timings for energizing my gemstone", "gemstone wearing rules", "gemstone energization"]
    ),
}


# ---------------------------------------------------------------------------
# NLP Normalization & Keyword Matching Helpers
# ---------------------------------------------------------------------------
def normalize_text_nlp(text: str) -> str:
    """Lowercases, removes extra punctuation, and standardizes common Hinglish phonetic variants."""
    if not text:
        return ""
    cleaned = text.lower().strip()
    cleaned = re.sub(r"[\?\!\.\,\;\:\/\-\(\)]+", " ", cleaned)
    tokens = cleaned.split()

    # Canonical mapping for common Hinglish transliterations
    token_map = {
        "shadi": "shaadi",
        "shaddi": "shaadi",
        "sadi": "shaadi",
        "shadii": "shaadi",
        "shadiyan": "shaadi",
        "nhi": "nahi",
        "nahin": "nahi",
        "nhn": "nahi",
        "karu": "karun",
        "karoo": "karun",
        "karo": "karun",
        "kru": "karun",
        "kare": "karein",
        "paise": "paisa",
        "pesa": "paisa",
        "rupya": "paisa",
        "rupaye": "paisa",
        "naukri": "naukri",
        "nokri": "naukri",
        "bussiness": "business",
        "bizness": "business",
        "upay": "upay",
        "upaay": "upay",
    }
    normalized_tokens = [token_map.get(t, t) for t in tokens]
    return " ".join(normalized_tokens)


def compute_intent_match_score(query_normalized: str, intent_def: MicroIntentDefinition) -> float:
    """
    Computes a normalized matching confidence score (0.0 to 1.0) for a micro-intent.
    Combines:
    1. Exact regex pattern match (+0.60 to +0.80)
    2. Domain keyword match (+0.20)
    3. Syntactic intent modal verbs / particles (+0.20)
    """
    score = 0.0

    # 1. Regex Pattern Matches
    for pat in intent_def.patterns:
        if re.search(pat, query_normalized, re.IGNORECASE):
            score += 0.65
            break

    # 2. Sample Query Substring / Inversion
    for sample in intent_def.sample_queries:
        s_norm = normalize_text_nlp(sample)
        if s_norm in query_normalized or query_normalized in s_norm:
            score += 0.25
            break

    # 3. Domain Keywords Coverage
    kw_hits = sum(1 for kw in intent_def.keywords if re.search(r"\b" + re.escape(kw) + r"\b", query_normalized))
    if kw_hits >= 1:
        score += min(0.20, kw_hits * 0.10)

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Multi-Domain Micro-Intent Classifier
# ---------------------------------------------------------------------------
def classify_micro_intent(query: str, threshold: float = 0.35) -> Dict[str, Any]:
    """
    Classifies an incoming query into a fine-grained micro-intent.

    Features:
    - Resolves timing vs probability vs domain-fit vs risk vs mitigation.
    - Ambiguity Detection: If top 2 micro-intents have high scores within 0.15 delta,
      marks `is_ambiguous=True`, triggering the Hybrid Local DB + LLM synthesis pipeline.
    """
    norm_q = normalize_text_nlp(query)
    if not norm_q:
        return {
            "micro_intent": "general",
            "domain": "general",
            "confidence": 0.0,
            "is_ambiguous": False,
            "candidate_intents": [],
            "qa_key": None,
            "routing_action": "STANDARD_REASONING"
        }

    scored_intents: List[Tuple[str, float, MicroIntentDefinition]] = []
    for intent_id, intent_def in MICRO_INTENT_REGISTRY.items():
        score = compute_intent_match_score(norm_q, intent_def)
        if score >= threshold:
            scored_intents.append((intent_id, score, intent_def))

    # Sort descending by score
    scored_intents.sort(key=lambda x: x[1], reverse=True)

    if not scored_intents:
        # Fallback to ML / Router default
        return {
            "micro_intent": "general",
            "domain": "general",
            "confidence": 0.0,
            "is_ambiguous": False,
            "candidate_intents": [],
            "qa_key": None,
            "routing_action": "STANDARD_REASONING"
        }

    top_intent_id, top_score, top_def = scored_intents[0]

    # Check for Ambiguity / Multi-Intent Balance
    is_ambiguous = False
    candidate_intents = [top_intent_id]

    if len(scored_intents) >= 2:
        second_intent_id, second_score, second_def = scored_intents[1]
        score_diff = abs(top_score - second_score)

        has_compound_marker = any(conj in f" {norm_q} " for conj in [" aur ", " and ", " as well as ", " tatha ", " evam ", " saath hi "])

        # Ambiguous condition: close scores or compound multi-aspect query (e.g. timing + risk)
        if (top_score >= 0.50 and second_score >= 0.50 and score_diff <= 0.15) or \
           (has_compound_marker and top_score >= 0.60 and second_score >= 0.60):
            is_ambiguous = True
            candidate_intents.append(second_intent_id)

    routing_action = "HYBRID_LLM" if is_ambiguous else "LOCAL_DB"

    return {
        "micro_intent": top_intent_id if not is_ambiguous else f"ambiguous_{top_intent_id}_{candidate_intents[1]}",
        "domain": top_def.domain,
        "confidence": round(top_score, 4),
        "is_ambiguous": is_ambiguous,
        "candidate_intents": candidate_intents,
        "qa_key": top_def.qa_key if not is_ambiguous else None,
        "routing_action": routing_action,
        "scored_candidates": [(item[0], round(item[1], 4)) for item in scored_intents[:4]]
    }


# ---------------------------------------------------------------------------
# Local Database Resolution with User Chart Dynamic Injection
# ---------------------------------------------------------------------------
_CACHED_QA_DB: Optional[Dict[str, str]] = None


def load_qa_database_dict() -> Dict[str, str]:
    """Loads and caches qa_database.json from canonical locations."""
    global _CACHED_QA_DB
    if _CACHED_QA_DB is not None:
        return _CACHED_QA_DB

    candidate_paths = [
        os.path.join(BASE_DIR, "qa_database.json"),
        os.path.join(BASE_DIR, "..", "qa_database.json"),
        os.path.join(os.getcwd(), "qa_database.json"),
    ]
    for p in candidate_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    _CACHED_QA_DB = json.load(f)
                    return _CACHED_QA_DB
            except Exception as err:
                logger.warning("Error loading QA DB from %s: %s", p, err)

    _CACHED_QA_DB = {}
    return _CACHED_QA_DB


def resolve_micro_intent_local_template(
    query: str,
    chart_context: Dict[str, Any],
    qa_db: Optional[Dict[str, str]] = None,
    is_hinglish: Optional[bool] = None,
    is_first_turn: bool = True,
) -> Optional[Tuple[str, List[str], str]]:
    """
    Attempts to resolve an unambiguous micro-intent directly via qa_database.json.
    Injects user chart parameters: {user_name}, {lagna}, {house_10_sign}, {house_7_lord}, etc.
    Mirrors language: I/P Hinglish -> O/P Hinglish, I/P English -> O/P English.
    Enforces greeting policy: Greets only on the starting turn of chat.
    Returns (personalized_answer, domain_locked_followups, domain_name) or None if ambiguous/unmatched.
    """
    res = classify_micro_intent(query)
    if res.get("is_ambiguous") or not res.get("qa_key"):
        return None

    db = qa_db or load_qa_database_dict()
    target_key = res["qa_key"]

    # Detect language if not provided
    if is_hinglish is None:
        try:
            from backend.llm_router import is_hinglish_query
            is_hinglish = is_hinglish_query(query)
        except Exception:
            try:
                from llm_router import is_hinglish_query
                is_hinglish = is_hinglish_query(query)
            except Exception:
                is_hinglish = False

    template = None
    if is_hinglish:
        template = db.get(f"{target_key}_hinglish") or db.get(f"{target_key}_hi")

    if not template:
        template = db.get(target_key)

    if not template:
        return None

    # Inject dynamic user birth chart parameters
    personalized_text = template
    for token, val in chart_context.items():
        placeholder = f"{{{token}}}"
        personalized_text = personalized_text.replace(placeholder, str(val))

    if not is_first_turn:
        try:
            from .response_builder import strip_leading_greeting
        except ImportError:
            try:
                from response_builder import strip_leading_greeting
            except ImportError:
                strip_leading_greeting = lambda t: t
        personalized_text = strip_leading_greeting(personalized_text)

    domain = res["domain"]

    # Import follow-ups generator from response_builder
    try:
        from .response_builder import get_domain_locked_followups
    except ImportError:
        try:
            from response_builder import get_domain_locked_followups
        except ImportError:
            get_domain_locked_followups = lambda dom, current_query=None, is_hinglish=False: []

    follow_ups = get_domain_locked_followups(domain, current_query=query, is_hinglish=is_hinglish)
    return personalized_text.strip(), follow_ups, domain
