# -*- coding: utf-8 -*-
"""
backend/faq_engine.py - Strict FAQ Eligibility & Educational Matching Engine (Proposal 3)

Distinguishes between:
1. ELIGIBLE_EDUCATIONAL: Pure definitional/terminological/educational queries.
   Returns verified classical definitions (0 LLM tokens, 0 ephemeris calls).
2. ELIGIBLE_GREETING: Conversational greetings and system introductions.
3. INELIGIBLE_CHART_DEPENDENT: Personal horoscope queries requiring chart grounding
   (planets, houses, dashas, transits, timing, predictions). Routes to Stage 8.
4. INELIGIBLE_AMBIGUOUS_REASONING: Queries touching on placements, aspects, or life
   indications without personal pronouns that require the reasoning pipeline to evaluate.

Guarantees:
- Never intercepts personal chart inquiries.
- Never serves generic canned predictions (9 predictive keys strictly excluded).
- Indexes only the ~76 audited educational concepts from qa_database.json.
"""

from enum import Enum
import json
import logging
import os
import re
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class FAQEligibilityOutcome(str, Enum):
    ELIGIBLE_GREETING = "ELIGIBLE_GREETING"
    ELIGIBLE_EDUCATIONAL = "ELIGIBLE_EDUCATIONAL"
    INELIGIBLE_CHART_DEPENDENT = "INELIGIBLE_CHART_DEPENDENT"
    INELIGIBLE_AMBIGUOUS_REASONING = "INELIGIBLE_AMBIGUOUS_REASONING"


# The 9 predictive/collision keys strictly excluded from the static FAQ matcher
# to prevent intercepting real user chart inquiries or serving canned predictions.
EXCLUDED_PREDICTIVE_KEYS: Set[str] = {
    "career prediction",
    "job change",
    "promotion",
    "marriage timing",
    "divorce or separation",
    "financial stability",
    "will i travel abroad",
    "mental stress",
    "chronic illness",
}

# Standalone greetings and introductory phrases
GREETING_WORDS: Set[str] = {
    "hi", "hello", "hey", "namaste", "namaskar", "greetings",
    "good morning", "good evening", "good afternoon", "pranam",
    "jai shree ram", "hare krishna", "who are you", "what can you do"
}

# Synonyms and aliases mapping conversational concepts to clean keys in qa_database.json
CONCEPT_ALIASES: Dict[str, str] = {
    # Dasha / Timing concepts
    "antardasha": "dasha",
    "mahadasha": "dasha",
    "vimshottari dasha": "dasha",
    "vimshottari": "dasha",
    "pratyantardasha": "dasha",
    # Charts & Divisonals
    "navamsha chart": "navamsha",
    "navamsha": "navamsha",
    "navamsa chart": "navamsha",
    "navamsa": "navamsha",
    "d9 chart": "navamsha",
    "d9": "navamsha",
    # Remedies & Gemstones
    "rules for wearing gemstones": "gemstone wearing rules",
    "gemstone wearing rules": "gemstone wearing rules",
    "gemstones rules": "gemstone wearing rules",
    "wearing gemstones": "gemstone wearing rules",
    "gemstones for career": "gemstones for career",
    "wearing gemstone": "gemstone wearing rules",
    # Mantras
    "gayatri mantra": "gayatri mantra",
    "maha mrityunjaya mantra": "maha mrityunjaya mantra",
    "mahamrityunjaya mantra": "maha mrityunjaya mantra",
    "maha mrityunjaya": "maha mrityunjaya mantra",
    "mahamrityunjaya": "maha mrityunjaya mantra",
    # Classical Concepts & Yogas
    "lagna": "lagna",
    "ascendant": "ascendant",
    "moon sign": "moon sign",
    "rashi": "rashi",
    "sun sign": "sun sign",
    "nakshatra": "nakshatra",
    "kundali": "kundali",
    "kundli": "kundali",
    "sade sati": "sade sati",
    "sadesati": "sade sati",
    "shani sade sati": "sade sati",
    "dhayya": "dhayya",
    "rahu ketu": "rahu ketu",
    "retrograde": "retrograde",
    "mangal dosha": "mangal dosha",
    "manglik dosha": "mangal dosha",
    "kuja dosha": "mangal dosha",
    "kundali milan": "kundali milan",
    "horoscope matching": "kundali milan",
    "gun milan": "kundali milan",
    "gajakesari yoga": "gajakesari yoga",
    "gaja kesari yoga": "gajakesari yoga",
    "dhan yoga": "dhan yoga",
    "dhana yoga": "dhan yoga",
    "lakshmi yoga": "lakshmi yoga",
    "laxmi yoga": "lakshmi yoga",
    "kaal sarp dosha": "kaal sarp dosha",
    "kaalsarp dosha": "kaal sarp dosha",
    "kaal sarp": "kaal sarp dosha",
    "pitra dosha": "pitra dosha",
    "pitru dosha": "pitra dosha",
    "pitra dosha remedies": "pitra dosha remedies",
    "shani shanti": "shani shanti",
    "surya arghya": "surya arghya",
    # Planetary significations
    "saturn in astrology": "saturn in astrology",
    "saturn": "saturn in astrology",
    "shani in astrology": "saturn in astrology",
    "shani": "saturn in astrology",
    "jupiter in astrology": "jupiter in astrology",
    "jupiter": "jupiter in astrology",
    "guru in astrology": "jupiter in astrology",
    "guru": "jupiter in astrology",
    "brihaspati": "jupiter in astrology",
    "mars in astrology": "mars in astrology",
    "mars": "mars in astrology",
    "mangal in astrology": "mars in astrology",
    "venus in astrology": "venus in astrology",
    "venus": "venus in astrology",
    "shukra in astrology": "venus in astrology",
    "shukra": "venus in astrology",
    "mercury in astrology": "mercury in astrology",
    "mercury": "mercury in astrology",
    "budh in astrology": "mercury in astrology",
    "budh": "mercury in astrology",
    "sun in astrology": "sun in astrology",
    "sun": "sun in astrology",
    "surya in astrology": "sun in astrology",
    "surya": "sun in astrology",
    "moon in astrology": "moon in astrology",
    "moon": "moon in astrology",
    "chandra in astrology": "moon in astrology",
    "chandra": "moon in astrology",
    "rahu in astrology": "rahu in astrology",
    "rahu": "rahu in astrology",
    "ketu in astrology": "ketu in astrology",
    "ketu": "ketu in astrology",
}

HOUSE_NUMBER_MAP: Dict[str, str] = {
    "1st": "1st house", "first": "1st house", "1": "1st house",
    "2nd": "2nd house", "second": "2nd house", "2": "2nd house",
    "3rd": "3rd house", "third": "3rd house", "3": "3rd house",
    "4th": "4th house", "fourth": "4th house", "4": "4th house",
    "5th": "5th house", "fifth": "5th house", "5": "5th house",
    "6th": "6th house", "sixth": "6th house", "6": "6th house",
    "7th": "7th house", "seventh": "7th house", "7": "7th house",
    "8th": "8th house", "eighth": "8th house", "8": "8th house",
    "9th": "9th house", "ninth": "9th house", "9": "9th house",
    "10th": "10th house", "tenth": "10th house", "10": "10th house",
    "11th": "11th house", "eleventh": "11th house", "11": "11th house",
    "12th": "12th house", "twelfth": "12th house", "12": "12th house",
}

_CLEAN_FAQ_CACHE: Optional[Dict[str, str]] = None


def normalize_query_for_eligibility(query: str) -> str:
    """Normalize whitespace and punctuation for strict intent evaluation."""
    clean = re.sub(r"[^\w\s'-]", " ", query.casefold()).strip()
    return re.sub(r"\s+", " ", clean)


def classify_faq_eligibility(query: str) -> Tuple[FAQEligibilityOutcome, Optional[str]]:
    """
    Classify query eligibility into one of four distinct outcomes:
    1. ELIGIBLE_GREETING: Conversational greetings / system inquiries.
    2. ELIGIBLE_EDUCATIONAL: Pure definitional inquiries without personal chart requirements.
    3. INELIGIBLE_CHART_DEPENDENT: Requires user chart grounding or predictive analysis.
    4. INELIGIBLE_AMBIGUOUS_REASONING: Placement/aspect or indication query requiring Stage 8.
    """
    clean = normalize_query_for_eligibility(query)
    words = clean.split()

    # ---------------------------------------------------------------------------
    # Tier 1: Greetings & System Questions
    # ---------------------------------------------------------------------------
    if clean in GREETING_WORDS:
        return FAQEligibilityOutcome.ELIGIBLE_GREETING, "exact_greeting"
    if len(words) <= 2 and any(w in GREETING_WORDS for w in words):
        # Exclude questions that happen to contain short words like 'hi' inside other words
        astro_check_words = ["house", "chart", "planet", "saturn", "sun", "moon", "lagna", "rashi"]
        if not any(astro_w in clean for astro_w in astro_check_words):
            return FAQEligibilityOutcome.ELIGIBLE_GREETING, "short_greeting"

    # ---------------------------------------------------------------------------
    # Tier 2: Chart Grounding & Personal Reference Screening (STRICTLY INELIGIBLE)
    # ---------------------------------------------------------------------------
    chart_patterns = [
        # Explicit personal chart references: "my chart", "in my chart", "my lagna", "my horoscope"
        r"\b(?:my|mera|meri|mere)\s+(?:chart|horoscope|kundali|kundli|janmapatri|lagna|rashi|ascendant|dasha|mahadasha|antardasha|bhava|house|lord|graha|planet|life|career|marriage|job|wealth|finances?|health|education|future|placement|placements|transit|transits|star|nakshatra|zodiac)\b",
        r"\b(?:in|for|from|according\s+to|regarding)\s+(?:my|mera|meri|mere)\s+(?:chart|horoscope|kundali|kundli|lagna|rashi|birth\s*chart)\b",
        r"\b(?:in|for|from|according\s+to)\s+my\b",
        r"\b(?:tell\s+me\s+about\s+my|predict\s+my|check\s+my|read\s+my|analyze\s+my|explain\s+my|describe\s+my)\b",
        r"\b(?:my\s+(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s*(?:house|bhava|lord))\b",
        r"\b(?:according\s+to\s+my\s+(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s*(?:house|bhava|lord|dasha))\b",
        r"\b(?:which\s+career\s+suits\s+me|career\s+for\s+me|remedies\s+for\s+me|sade\s+sati\s+for\s+me|good\s+for\s+me|bad\s+for\s+me)\b",
        r"\b(?:condition\s+of\s+\w+\s+in\s+my)\b",
        r"\b(?:saturn\s+in\s+my|jupiter\s+in\s+my|rahu\s+in\s+my|ketu\s+in\s+my|mars\s+in\s+my|venus\s+in\s+my|mercury\s+in\s+my|sun\s+in\s+my|moon\s+in\s+my)\b",
        r"\b(?:during\s+my\s+\w+\s*(?:period|dasha|mahadasha|antardasha))\b",
        r"\b(?:what\s+is\s+my|batao\s+mera|batao\s+meri)\b",
    ]
    for pat in chart_patterns:
        if re.search(pat, clean):
            return FAQEligibilityOutcome.INELIGIBLE_CHART_DEPENDENT, "chart_grounded"

    # ---------------------------------------------------------------------------
    # Tier 3: Predictive Timing, Life Event Outcomes & Decisions (STRICTLY INELIGIBLE)
    # ---------------------------------------------------------------------------
    predictive_patterns = [
        # Timing inquiries
        r"\b(?:when\s+will|kab\s+(?:hoga|hogi|hoge|milega|milegi|aayega|aayegi|hona)|timing\s+of|next\s+year|in\s+202[0-9]|in\s+the\s+future)\b",
        # Predictive outcomes / chances
        r"\b(?:will\s+i|shall\s+i|can\s+i\s+get|chances\s+of|probability\s+of|kya\s+main|kya\s+mujhe)\b",
        r"\b(?:should\s+i\s+(?:switch|change|marry|move|wear|buy|invest|settle|start))\b",
        # Life events
        r"\b(?:promotion|job\s+switch|job\s+change|marriage\s+timing|foreign\s+settlement|foreign\s+travel|travel\s+abroad|clear\s+(?:my\s+)?competitive\s+exam|pass\s+(?:my\s+)?exam|getting\s+married|getting\s+divorced|divorce|separation)\b",
        r"\b(?:during\s+(?:my\s+)?(?:rahu|jupiter|saturn|mars|mercury|venus|sun|moon|ketu)\s*(?:period|dasha|mahadasha|antardasha))\b",
    ]
    for pat in predictive_patterns:
        if re.search(pat, clean):
            return FAQEligibilityOutcome.INELIGIBLE_CHART_DEPENDENT, "predictive_timing"

    # ---------------------------------------------------------------------------
    # Tier 4: Astrological Placement Configurations & Ambiguous Indications
    # (Bypass FAQ -> Route to Stage 8 Reasoning)
    # ---------------------------------------------------------------------------
    ambiguous_patterns = [
        # House lords in houses: "10th lord in 1st house", "lord of 7th in 10th"
        r"\b\d+(?:st|nd|rd|th)?\s*lord\s+in\s+\d+(?:st|nd|rd|th)?\s*house\b",
        r"\blord\s+of\s+\d+(?:st|nd|rd|th)?\s*(?:house)?\s+in\s+\d+(?:st|nd|rd|th)?\b",
        # Planet in house: "Saturn placed in 7th house", "Mars in 8th house"
        r"\b(?:sun|moon|mars|mercury|jupiter|venus|saturn|rahu|ketu)\s+(?:in|placed\s+in|sitting\s+in|posited\s+in)\s+(?:the\s+)?\d+(?:st|nd|rd|th)?\s*house\b",
        r"\b\d+(?:st|nd|rd|th)?\s*house\s+(?:sun|moon|mars|mercury|jupiter|venus|saturn|rahu|ketu)\b",
        # Specific astrological aspects & states: "Saturn aspects 7th house", "Mars combust"
        r"\b(?:aspecting|aspects|aspected\s+by|conjunction|conjoined|combust|retrograde\s+in|debilitated\s+in|exalted\s+in)\b",
        # Indication / impact / effect / result questions:
        # e.g. "What does the 10th house indicate about career?", "Does Saturn delay marriage?"
        r"\b(?:indicate|indicates|indication|affect|affects|affecting|impact|impacts|influence|influences|cause|causes)\s+(?:about|my|our|career|marriage|wealth|health|job|growth|delay|problems?)\b",
        r"\b(?:does|do|can|is)\s+.*\b(?:delay|support|favor|cause|give|bring)\s+(?:marriage|career|wealth|education|divorce|success)\b",
        r"\b(?:support|supports)\s+(?:higher\s+)?education\b",
        r"\bfinancial\s+prospects\b",
        r"\binfluence\s+relationships\b",
    ]
    for pat in ambiguous_patterns:
        if re.search(pat, clean):
            return FAQEligibilityOutcome.INELIGIBLE_AMBIGUOUS_REASONING, "placement_or_indication"

    # ---------------------------------------------------------------------------
    # Tier 5: Pure Educational Phrasing Patterns (ELIGIBLE FOR FAQ MATCHING)
    # ---------------------------------------------------------------------------
    educational_patterns = [
        # Definitional: "What is a Lagna?", "What is Sade Sati?", "Kya hota hai Mangal Dosha"
        r"^(?:what\s+is|what\s+are|kya\s+hai|kya\s+hota\s+hai)\s+(?:a|an|the\s+)?(?:[\w-]+\s*){1,6}$",
        # Conversational definitional: "Tell me what a Lagna is", "Help me understand what 8th house means"
        r"^(?:can\s+you\s+)?(?:tell\s+me|help\s+me\s+understand|explain\s+to\s+me|explain|batao|samjhao)\s+(?:what|meaning\s+of|about)?\s*(?:a|an|the\s+)?(?:[\w-]+\s*){1,6}$",
        r"^(?:can\s+you\s+)?explain\s+(?:what\s+is\s+)?(?:a|an|the\s+)?(?:[\w-]+\s*){1,6}$",
        r"^(?:tell\s+me\s+)?(?:the\s+)?meaning\s+of\s+(?:[\w-]+\s*){1,6}$",
        # Karakatva / General significations of a planet or house:
        # e.g. "What does the 10th house represent in Vedic astrology?", "What does Saturn signify in Vedic astrology?"
        r"^(?:what\s+does|what\s+do)\s+(?:the\s+)?(?:[\w-]+\s+){1,3}(?:represent|signify|govern|mean)\s*(?:in\s+(?:vedic\s+)?astrology)?$",
        # Prescriptive rules / text: "What are the rules for wearing gemstones?", "Gayatri Mantra text"
        r"^(?:what\s+are\s+the\s+)?rules\s+for\s+(?:[\w-]+\s*){1,6}$",
        r"^(?:what\s+is\s+the\s+)?(?:gayatri|maha\s*mrityunjaya)\s+mantra$",
    ]
    for pat in educational_patterns:
        if re.search(pat, clean):
            return FAQEligibilityOutcome.ELIGIBLE_EDUCATIONAL, "educational_definition"

    # Default fallback: If not explicitly proven educational, route to reasoning
    return FAQEligibilityOutcome.INELIGIBLE_AMBIGUOUS_REASONING, "default_reasoning"


def is_faq_eligible_query(query: str) -> bool:
    """Return True if the query is safe for fast zero-token FAQ interception."""
    outcome, _ = classify_faq_eligibility(query)
    return outcome in (
        FAQEligibilityOutcome.ELIGIBLE_EDUCATIONAL,
        FAQEligibilityOutcome.ELIGIBLE_GREETING,
    )


def load_clean_educational_faq_db(db_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load qa_database.json and index ONLY the ~76 clean educational entries,
    strictly omitting the 9 dangerous predictive keys.
    """
    global _CLEAN_FAQ_CACHE
    if _CLEAN_FAQ_CACHE is not None:
        return _CLEAN_FAQ_CACHE

    possible_paths = [
        db_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qa_database.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa_database.json"),
        "qa_database.json",
    ]
    resolved_path = None
    for p in possible_paths:
        if p and os.path.exists(p):
            resolved_path = p
            break

    if not resolved_path:
        logger.warning("Could not find qa_database.json; returning empty educational index")
        _CLEAN_FAQ_CACHE = {}
        return _CLEAN_FAQ_CACHE

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw_db = json.load(f)
            clean_db: Dict[str, str] = {}
            for key, val in raw_db.items():
                clean_k = key.lower().strip()
                if clean_k in EXCLUDED_PREDICTIVE_KEYS:
                    continue
                if isinstance(val, str):
                    clean_db[clean_k] = val
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and item.get("answer"):
                            for kw in item.get("keywords", []):
                                kw_str = str(kw).lower().strip()
                                if kw_str and kw_str not in EXCLUDED_PREDICTIVE_KEYS:
                                    clean_db[kw_str] = item["answer"]
            _CLEAN_FAQ_CACHE = clean_db
            logger.info("Loaded clean educational FAQ database (%d entries) from %s", len(_CLEAN_FAQ_CACHE), resolved_path)
            return _CLEAN_FAQ_CACHE
    except Exception as e:
        logger.error("Error reading %s: %s", resolved_path, e)
        _CLEAN_FAQ_CACHE = {}
        return _CLEAN_FAQ_CACHE


def match_educational_faq(query: str, db_path: Optional[str] = None) -> Optional[str]:
    """
    Strict FAQ Resolver:
    1. Evaluates eligibility: Returns None if query is chart-dependent or ambiguous.
    2. Matches educational concepts against the 76 clean educational entries.
    3. Returns the static educational answer template or None.
    """
    outcome, reason = classify_faq_eligibility(query)
    if outcome not in (FAQEligibilityOutcome.ELIGIBLE_EDUCATIONAL, FAQEligibilityOutcome.ELIGIBLE_GREETING):
        return None

    clean_db = load_clean_educational_faq_db(db_path)
    clean = normalize_query_for_eligibility(query)

    # 1. Handle Greetings
    if outcome == FAQEligibilityOutcome.ELIGIBLE_GREETING:
        for greeting_k in ["hello", "hi", "namaste"]:
            if greeting_k in clean_db:
                return clean_db[greeting_k]
        return "Namaste! I am your Vedic Astrology Assistant. How can I assist you with your astrological journey today?"

    # 2. Check House Definitions: e.g. "10th house", "tenth house", "7th house"
    house_match = re.search(
        r"\b(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|11th|12th|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+house\b",
        clean,
    )
    if house_match:
        target = HOUSE_NUMBER_MAP.get(house_match.group(1))
        if target and target in clean_db:
            return clean_db[target]

    # 3. Check Aliases (longest phrases first)
    for alias in sorted(CONCEPT_ALIASES.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", clean):
            target_key = CONCEPT_ALIASES[alias]
            if target_key in clean_db:
                return clean_db[target_key]

    # 4. Check Direct Clean Database Keys (longest phrases first)
    for key in sorted(clean_db.keys(), key=len, reverse=True):
        if len(key) < 3 or key in GREETING_WORDS:
            continue
        if re.search(r"\b" + re.escape(key) + r"\b", clean):
            return clean_db[key]

    return None
