"""
backend/intent_router.py
========================
Intent-First Execution Pipeline & Pre-Execution Guardrails
---------------------------------------------------------
Enforces the mandatory 5-step classification architecture:
1. Garbage / Special Character Filtering (bypasses LLM/astrology calculations).
2. Strict Profanity & Vulgarity Guardrail (Hindi/Hinglish/English keyword filter).
3. Safety Boundaries (medical diagnosis & death predictions).
4. Domain Prefilter (out-of-scope rejection).
5. Astrological Intent Classification (house_analysis, career_timing, multi_domain,
   educational_faq, direct_fact, general_horoscope).
6. Robust Conversational Hinglish Detection.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Requirement 2: Refusal Prompt Constants
# ---------------------------------------------------------------------------
GARBAGE_REFUSAL_MESSAGE = "Please share an appropriate question so I can assist you better with your astrology query."
PROFANITY_REFUSAL_MESSAGE = "⚠️ Warning (1/3): Inappropriate or vulgar language is not permitted."
PROFANITY_WARNING_TEMPLATE = "⚠️ Warning ({count}/3): Inappropriate or vulgar language is not permitted."
PROFANITY_TERMINATED_MESSAGE = "🚫 Session Terminated: You have exceeded the limit for inappropriate language."

# ---------------------------------------------------------------------------
# Requirement 1: Enhanced Hinglish Vocabulary & Detection Markers
# ---------------------------------------------------------------------------
CONVERSATIONAL_HINGLISH_MARKERS = {
    # Pronouns & Address
    "aap", "aapka", "aapki", "aapke", "aapko", "hum", "hamara", "hamari", "hamare",
    "mera", "meri", "mere", "mujhe", "mujhko", "apna", "apni", "apne",
    "tum", "tumhara", "tumhari", "tumhare", "tera", "teri", "tere", "tujhe",
    # Question words
    "kya", "kab", "kaise", "kyun", "kyu", "kahan", "kidhar", "kitna", "kitni",
    "kitne", "kaisa", "kaisi", "kaun", "kaunsa", "kaunsi", "kaunse", "kis",
    # Verbs & Auxiliary
    "hai", "hain", "ho", "hoon", "hun", "hoga", "hogi", "honge", "tha", "thi",
    "raha", "rahi", "rahe", "rahega", "rahegi", "karo", "karein", "karna",
    "karni", "kar", "kiya", "kiye", "chahiye", "chahta", "chahti", "sakta",
    "sakti", "sakte", "batao", "bataiye", "bataye", "bata", "batao na", "dekh",
    "dekho", "dekhiye", "bata do", "bata de", "milega", "milegi", "milenge",
    "badhega", "badhegi", "hona", "hoti", "hota", "lagta", "lagti", "likha",
    # Prepositions & Conjunctions
    "mein", "se", "ko", "par", "pe", "ka", "ki", "ke", "liye", "aur",
    "ya", "toh", "bhi", "nahi", "nahin", "nhi", "na", "ab", "kabtak",
    "kabh", "bohot", "bahut", "thoda", "thodi", "accha", "achha", "sahi",
    "galat", "theek", "thik", "bilkul", "kripya", "namaste", "dhanyawad",
    # Life Domain Hinglish Terms
    "naukri", "nokri", "kaam", "vyapar", "dhandha", "rojgar", "paisa", "paise",
    "dhan", "sampatti", "karz", "karja", "shaadi", "shadi", "vivah", "rishta",
    "pyaar", "prem", "dulhan", "dulha", "patni", "pati", "saathi", "swasthya",
    "sehat", "bimari", "dard", "ilaj", "dawai", "tanav", "videsh", "yatra",
    "safar", "pardesh", "padhai", "shiksha", "pariksha", "imtihan", "bachha",
    "baccha", "bacche", "santaan", "beta", "beti", "parivar", "upay", "totka"
}

# Technical Sanskrit terms that appear in both English & Hindi astrology queries
ASTROLOGICAL_SANSKRIT_TERMS = {
    "kundali", "kundli", "janam", "janampatri", "patrika", "graha", "rashi",
    "lagna", "dasha", "mahadasha", "antardasha", "pratyantardasha", "shani",
    "mangal", "budh", "guru", "shukra", "surya", "chandra", "rahu", "ketu",
    "sade sati", "dosh", "dosha", "bhava", "bhav", "ghar", "shubh",
    "ashubh", "yog", "yoga", "ratna", "puja"
}

# Alias for backwards compatibility
HINGLISH_DETECTION_MARKERS = CONVERSATIONAL_HINGLISH_MARKERS.union(ASTROLOGICAL_SANSKRIT_TERMS)


def is_hinglish_query(text: str) -> bool:
    """
    Detects if a user question is formulated in conversational Hinglish
    (Hindi expressed in the Roman/Latin script).
    Sanskrit technical terms (e.g. 'dasha', 'kundali', 'lagna') alone in an
    otherwise English sentence do not trigger Hinglish.
    """
    if not text or not text.strip():
        return False
    clean = text.lower().strip()
    words = set(re.findall(r"\b[a-zA-Z']+\b", clean))
    if not words:
        return False

    # 1. Distinct conversational Hindi markers
    matched_conv = words.intersection(CONVERSATIONAL_HINGLISH_MARKERS)
    if matched_conv:
        return True

    # 2. Multi-word Hinglish phrase patterns
    hinglish_phrases = [
        r"\bkab\s+(?:hogi|hoga|milega|milegi|lagegi|aayega|aayegi)\b",
        r"\bkya\s+(?:hoga|hogi|karein|karna|chahiye|hai|hain|likha)\b",
        r"\bkaisa\s+(?:rahega|hoga|hai)\b",
        r"\bkaisi\s+(?:rahegi|hogi|hai)\b",
        r"\b(?:meri|mera|mere)\s+(?:kundali|kundli|career|shadi|shaadi|job|future|health|graha|dasha)\b",
        r"\b(?:kundali|kundli)\s+me\b",
        r"\bme\s+(?:kya|hai|hoga|hogi|likha)\b",
        r"\b(?:batao|bataiye|bataye|bata\s+do)\b",
        r"\b(?:hogi\s+ya\s+nahi|hoga\s+ya\s+nahi)\b",
        r"\b(?:7th|10th|1st|4th|5th|9th|2nd|11th)\s*(?:ghar|bhav|bhava)\b",
    ]
    for pattern in hinglish_phrases:
        if re.search(pattern, clean):
            return True

    return False


KEYBOARD_MASH_ROWS = [
    "qwertyuiop", "asdfghjkl", "zxcvbnm",
    "poiuytrewq", "lkjhgfdsa", "mnbvcxz",
    "1234567890", "0987654321"
]


# ---------------------------------------------------------------------------
# Requirement 2: Garbage & Special Character Input Filtering
# ---------------------------------------------------------------------------
def is_garbage_or_special_char_input(text: str) -> Tuple[bool, str]:
    """
    Evaluates whether input consists of single letters, single punctuation,
    strings of only symbols (e.g. '???', '@@@'), numbers only, or character repetition.
    Returns (is_garbage: bool, reason: str).
    """
    if not text:
        return True, "empty_input"

    stripped = text.strip()
    if not stripped:
        return True, "whitespace_only"

    # Case 0: Exclude profanities/slang from being falsely flagged as mere garbage
    is_profane, _ = check_profanity_guardrail(stripped)
    if is_profane:
        return False, "profane_term"

    # Case 1: Single alphabetical character (e.g., 'a', 'x', 'k')
    if len(stripped) == 1 and stripped.isalpha():
        return True, "single_alphabetical_character"

    # Case 2: Single punctuation mark (e.g., '?', '!', '.')
    if len(stripped) == 1 and not stripped.isalnum():
        return True, "single_punctuation"

    # Case 3: Only punctuation / special characters (e.g., '???', '@@@', '!@#$%', '...', '---', '&&&')
    if re.fullmatch(r"[\W_]+", stripped):
        return True, "special_characters_only"

    # Case 4: Digits only or numbers with whitespace/punctuation (e.g. '123', '99999', '12.5', '00')
    if re.fullmatch(r"[\d\s.,+\-*/=]+", stripped):
        return True, "numbers_only"

    # Case 5: Single character repeated 4+ times (e.g. 'aaaaa', 'zzzz', '?????')
    if re.fullmatch(r"(.)\1{3,}", stripped.lower()):
        return True, "character_repetition"

    # Case 6: Less than 2 alphanumeric characters total (e.g. '?a?', '#1', '..x..')
    alnum_chars = re.findall(r"[a-zA-Z0-9]", stripped)
    if len(alnum_chars) <= 1:
        return True, "insufficient_alphanumeric_content"

    VALID_CONSONANT_EXEMPTIONS = {
        "strengths", "strength", "lengths", "length", "twelfths", "eighths",
        "sixth", "eighth", "twelfth", "rhythm", "rhythms", "symphysy", "psych",
        "sh", "shh", "psst", "hmm", "hmmm", "bhava", "bhav", "rashi",
        "d1", "d9", "d10", "mc", "ic", "dr", "mr", "mrs", "ms", "vs", "st", "nd", "rd", "th"
    }

    # Evaluate keyboard mashing, consonant clusters, and vowel presence on individual tokens
    # to avoid false refusals when legitimate words cross boundaries (e.g. "birth chart" -> "rthch")
    for raw_token in stripped.split():
        clean_word = re.sub(r"[^a-zA-Z]", "", raw_token.lower())
        if not clean_word:
            continue

        # Explicit exemption for legitimate English and astrological vocabulary
        if clean_word in VALID_CONSONANT_EXEMPTIONS:
            continue

        # Case 7: Keyboard row mashing (e.g. 'asdfghjk', 'qwerty', 'lkjhgfdsa')
        if len(clean_word) >= 4 and any(clean_word in row for row in KEYBOARD_MASH_ROWS):
            return True, "keyboard_row_mash"

        # Case 8: Long consonant cluster of 5+ letters within a single token (e.g. 'sdfghjk')
        if re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", clean_word):
            return True, "consonant_cluster_gibberish"

        # Case 9: Consonant-only gibberish of 3+ letters without vowels within a single token (e.g. 'ghjk', 'qwrt', 'xyz')
        if len(clean_word) >= 3 and not any(v in clean_word for v in ["a", "e", "i", "o", "u", "y"]):
            return True, "consonant_gibberish"

    return False, ""


# ---------------------------------------------------------------------------
# Requirement 3: Robust Profanity & Vulgarity Guardrail (Hindi / Hinglish / English)
# ---------------------------------------------------------------------------
RAW_PROFANITY_PATTERNS = [
    # Hindi / Hinglish Profanity (Single Words & Slang)
    r"\bb+c+\b",  # bc
    r"\bm+c+\b",  # mc
    r"\bb+s+d+k+\b",  # bsdk, b.s.d.k
    r"\bb+k+l+\b",  # bkl
    r"\bb+h+o+s+[a-z]*d+i+k+e+\b",  # bhosdike, bhosadike
    r"\bb+h+o+s+[a-z]*d+i+(?:wala|wali|wale)?\b",  # bhosdi, bhosdiwali
    r"\bb+h+o+s+d+a+\b",  # bhosda
    r"\b(?:behen|bhen|bhn)\s*c+h+o+d+\b",  # behenchod
    r"\b(?:madar|maadar|mader)\s*c+h+o+d+\b",  # madarchod
    r"\bmadarja+t+\b",  # madarjaat
    r"\bc+h+u+t+i+y+a+(?:s|n|pan)?\b",  # chutiya, chutiyapan
    r"\bc+h+u+t+i+y+e+\b",  # chutiye
    r"\bc+h+u+t+y+a+\b",  # chutya
    r"\bc+h+u+t+i+y+a+p+a*\b",  # chutiyapa
    r"\bc+h+o+o+t+i+y+a+\b",  # chootiya
    r"\bc+h+o+o+t+\b",  # choot
    r"\bc+h+u+t+\b",  # chut
    r"\bg+a+n+d+u+(?:s)?\b",  # gandu
    r"\bg+a+a+n+d+u+(?:s)?\b",  # gaandu
    r"\bg+a+n+d+\b",  # gand
    r"\bg+a+a+n+d+\b",  # gaand
    r"\bg+a+a+n+d+\s*(?:marna|mara|mari|faad|fat|masti|chhed)?\b",
    r"\bh+a+r+a+m+i+(?:yo|pan)?\b",  # harami
    r"\bh+a+r+a+m+k+h+o+r+\b",  # haramkhor
    r"\bh+a+r+a+m+z+a+d+[ae]\b",  # haramzada
    r"\bk+a+m+i+n+e+\b",  # kamine
    r"\bk+a+m+i+n+a+\b",  # kamina
    r"\bk+a+m+i+n+i+\b",  # kamini
    r"\bl+a+u+d+[aeiu]+\b",  # lauda, laude
    r"\bl+o+d+[aeiu]+\b",  # loda, lode
    r"\bl+a+v+d+[aeiu]+\b",  # lavda, lavde
    r"\bl+u+n+d+(?:s)?\b",  # lund, lunds
    r"\br+a+n+d+i+(?:ya|baaz)?\b",  # randi
    r"\br+a+n+d+w+a+\b",  # randwa
    r"\br+a+a+n+d+\b",  # raand
    r"\bc+h+i+n+a+l+\b",  # chinal
    r"\bc+h+i+n+a+r+\b",  # chinar
    r"\bt+a+t+t+e+\b",  # tatte
    r"\bt+a+t+t+a+\b",  # tatta
    r"\bj+h+a+a*t+\b",  # jhaat, jhat
    r"\bj+h+a+a*t+u+\b",  # jhaatu, jhatu
    r"\bm+u+t+h+a+l+\b",  # muthal
    r"\bm+u+t+t+h+a+l+\b",  # mutthal
    r"\b(?:chudai|chodna|chodo|chode|chuda|chudwa)\b",
    r"\bteri\s+(?:maa|ma|behen|bhen)\s+ki\b",
    r"\bmaa\s+ki\s+choot\b",
    r"\bmaa\s+ka\s+bhosda\b",
    r"\bteri\s+maa\s+ka\b",
    r"\bsaale\s+kutte\b",

    # English Profanity
    r"\bf+u*c+k+(?:er|ing|ed|s|off)?\b",
    r"\bf+c+k+\b",
    r"\bmother\s*f+u*c+k+(?:er)?\b",
    r"\bb+i+t+c+h+(?:es|ing)?\b",
    r"\bs+h+i+t+(?:ty|s|ting)?\b",
    r"\bbullshit\b",
    r"\ba+s+s+h+o+l+e+(?:s)?\b",
    r"\ba+r+s+e+h+o+l+e+(?:s)?\b",
    r"\b(?:dumb|jack)ass\b",
    r"\bb+a+s+t+a+r+d+(?:s)?\b",
    r"\bd+i+c+k+(?:head|s)?\b",
    r"\bp+u+s+s+y+(?:es)?\b",
    r"\bc+u+n+t+(?:s)?\b",
    r"\bs+l+u+t+(?:s)?\b",
    r"\bw+h+o+r+e+(?:s)?\b",
    r"\bn+i+g+g+e+r+(?:s)?\b",
    r"\bf+a+g+g+o+t+(?:s)?\b",
    r"\br+e+t+a+r+d+(?:s|ed)?\b",
    r"\bp+o+r+n+(?:o|ography)?\b",
    r"\bblowjob\b",
    r"\bhandjob\b",
    r"\bmasturbat\w*\b",
    r"\b(?:nude|nudes|boobs|tits|penis|vagina|incest|xxx|xvideos|porno)\b",

    # Devanagari Script Hindi Profanity
    r"मादरचोद", r"बहनचोद", r"भोसड़ीके", r"भोसड़ी", r"भोसड़ा",
    r"चूतिया", r"चूतिये", r"गांडू", r"गांड", r"हरामी",
    r"कमीने", r"लौड़े", r"लौड़ा", r"लंड", r"रंडी",
    r"साले\s*कुत्ते", r"चोदना", r"चुदाई", r"मुठ"
]

COMPILED_PROFANITY_PATTERNS = [re.compile(pat, re.IGNORECASE) for pat in RAW_PROFANITY_PATTERNS]

LEET_REPLACEMENTS = {
    "@": "a", "4": "a", "$": "s", "5": "s", "1": "i", "!": "i", "0": "o", "3": "e", "*": "u"
}


def normalize_leetspeak(text: str) -> str:
    """Normalize common leetspeak substitutions to standard characters."""
    t = text.lower()
    for symbol, char in LEET_REPLACEMENTS.items():
        t = t.replace(symbol, char)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)  # Collapse elongated letters (e.g. fuuuuck -> fuuck)
    return t


def check_profanity_guardrail(query: str) -> Tuple[bool, Optional[str]]:
    """
    Evaluates query against calibrated profanity patterns.
    Enforces negative contextual exemptions for historical dates (e.g. '1500 BC')
    and astrological midheaven ('MC').
    """
    if not query or not query.strip():
        return False, None

    normalized = normalize_leetspeak(query)
    q_low = query.lower()

    # Contextual check for 'bc' (historical Before Christ exemption)
    is_bc_date = bool(re.search(r"\b\d+\s*bc\b", q_low)) or any(w in q_low for w in ["century bc", "era", "ancient", "history", "bce"])
    is_mc_astro = bool(re.search(r"\b(?:my|natal|the|sign\s+of)\s+mc\b", q_low)) or any(w in q_low for w in ["midheaven", "medium coeli", "zenith", "meridian"])

    # 1. Direct Regex Pattern Matching
    for pattern in COMPILED_PROFANITY_PATTERNS:
        match = pattern.search(query) or pattern.search(normalized)
        if match:
            matched_word = match.group(0).lower()
            if matched_word in ["bc", "b.c"] and is_bc_date:
                continue
            if matched_word in ["mc", "m.c"] and is_mc_astro:
                continue
            return True, match.group(0)

    # 2. Dot-separated swear acronyms (b.s.d.k, b.c, m.c)
    dot_clean = re.sub(r"\.", "", q_low)
    for bad_acronym in ["bsdk", "bkl"]:
        if re.search(r"\b" + bad_acronym + r"\b", dot_clean):
            return True, bad_acronym

    return False, None


# ---------------------------------------------------------------------------
# Requirement 5: Intent Classification Result Model
# ---------------------------------------------------------------------------
@dataclass
class IntentClassificationResult:
    primary_intent: str
    detected_intents: List[str]
    is_hinglish: bool
    is_refusal: bool = False
    refusal_message: Optional[str] = None
    refusal_source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def classify_query_intent(
    query: str,
    user_profile: Optional[Dict[str, Any]] = None
) -> IntentClassificationResult:
    """
    Executes Intent-First Classification Pipeline:
    Analyzes *every* incoming question before executing LLMs or astrological calculations.
    
    Order of precedence:
    1. Garbage / Special Character Filtering -> 'garbage_input'
    2. Profanity & Vulgarity Guardrail       -> 'profanity'
    3. Safety Boundaries (Medical / Death)   -> 'safety_boundary'
    4. Out-of-Scope Pre-Filter               -> 'out_of_scope'
    5. Astrological House Analysis           -> 'house_analysis'
    6. Timing Windows & Predictions          -> 'career_timing' / 'timing_window'
    7. Direct Fact & Dasha Lookups           -> 'dasha' / 'direct_fact'
    8. Multi-Domain Composite Inquiries       -> 'multi_domain'
    9. Educational Definition FAQ            -> 'educational_faq'
    10. General Horoscope / In-depth Chart   -> 'general_horoscope'
    """
    is_hinglish = is_hinglish_query(query)

    # Step 1: Profanity & Vulgarity Guardrail (Hindi / Hinglish / English)
    is_profane, flagged_term = check_profanity_guardrail(query)
    if is_profane:
        return IntentClassificationResult(
            primary_intent="profanity",
            detected_intents=["profanity"],
            is_hinglish=is_hinglish,
            is_refusal=True,
            refusal_message=PROFANITY_REFUSAL_MESSAGE,
            refusal_source="PROFANITY_GUARDRAIL",
            metadata={"flagged_term": flagged_term}
        )

    # Step 2: Garbage & Special Character Input Filtering
    is_garbage, garbage_reason = is_garbage_or_special_char_input(query)
    if is_garbage:
        return IntentClassificationResult(
            primary_intent="garbage_input",
            detected_intents=["garbage_input"],
            is_hinglish=is_hinglish,
            is_refusal=True,
            refusal_message=GARBAGE_REFUSAL_MESSAGE,
            refusal_source="INPUT_SANITIZER",
            metadata={"reason": garbage_reason}
        )

    # Step 3: Clinical Medical & Death Prediction Boundaries
    try:
        from backend.reasoning.rule_coverage import is_unsupported_boundary_query, get_unsupported_limitation_response
        if is_unsupported_boundary_query(query):
            limit_msg = get_unsupported_limitation_response(query)
            return IntentClassificationResult(
                primary_intent="safety_boundary",
                detected_intents=["safety_boundary"],
                is_hinglish=is_hinglish,
                is_refusal=True,
                refusal_message=limit_msg,
                refusal_source="SAFETY_BOUNDARY",
                metadata={"boundary": "medical_or_death"}
            )
    except Exception:
        pass

    # Step 4: Out-of-Scope Pre-Filter
    try:
        from backend.domain_filter import check_domain_prefilter, FilterAction, OUT_OF_SCOPE_RESPONSE
        filter_dec = check_domain_prefilter(query)
        if filter_dec.action == FilterAction.REJECT_OUT_OF_SCOPE:
            return IntentClassificationResult(
                primary_intent="out_of_scope",
                detected_intents=["out_of_scope"],
                is_hinglish=is_hinglish,
                is_refusal=True,
                refusal_message=filter_dec.message or OUT_OF_SCOPE_RESPONSE,
                refusal_source="OUT_OF_SCOPE_PREFILTER",
                metadata={"reason": filter_dec.reason}
            )
    except Exception:
        pass

    q_lower = query.lower().strip()

    # Step 5: Astrological House-Related Query Handling (Pure Factual Lookup Only)
    try:
        from backend.astrology.house_engine import is_pure_factual_house_query, extract_house_number
        if is_pure_factual_house_query(query):
            target_house = extract_house_number(query) or 7
            return IntentClassificationResult(
                primary_intent="house_analysis",
                detected_intents=["house_analysis"],
                is_hinglish=is_hinglish,
                is_refusal=False,
                metadata={"house_number": target_house}
            )
    except Exception:
        pass

    # Step 6: Timing Queries (Career / Marriage / Finance Timing)
    try:
        from backend.timing_engine import is_timing_query
        if is_timing_query(query):
            if any(k in q_lower for k in ["job", "career", "naukri", "promotion", "switch", "work"]):
                timing_intent = "career_timing"
            elif any(k in q_lower for k in ["marriage", "married", "marry", "shaadi", "shadi", "vivah", "wedding", "spouse"]):
                timing_intent = "marriage_timing"
            else:
                timing_intent = "timing_window"
            return IntentClassificationResult(
                primary_intent=timing_intent,
                detected_intents=[timing_intent],
                is_hinglish=is_hinglish,
                is_refusal=False,
                metadata={"timing_type": timing_intent}
            )
    except Exception:
        pass

    # Step 7: Direct Deterministic Fact / Dasha Queries
    is_direct_dasha = any(
        k in q_lower for k in [
            "what is my dasha", "what is my current dasha", "my current dasha",
            "current dasha", "my mahadasha", "what is my mahadasha", "active dasha",
            "current antardasha", "meri dasha kya hai", "kaunsi dasha chal rahi hai"
        ]
    ) and not any(ck in q_lower for ck in ["career", "job", "work", "marriage", "when"])
    if is_direct_dasha:
        return IntentClassificationResult(
            primary_intent="dasha",
            detected_intents=["dasha"],
            is_hinglish=is_hinglish,
            is_refusal=False
        )

    # Step 8: Multi-Domain Composite Inquiries
    try:
        from backend.qa_router import classify_multi_domain_micro_intents
        micro_intents = classify_multi_domain_micro_intents(query)
        if len(micro_intents) > 1:
            all_intents = ["multi_domain"] + [m for m in micro_intents if m != "multi_domain"]
            return IntentClassificationResult(
                primary_intent="multi_domain",
                detected_intents=all_intents,
                is_hinglish=is_hinglish,
                is_refusal=False,
                metadata={"micro_intents": micro_intents}
            )
    except Exception:
        pass

    # Step 9: Educational FAQ Concepts
    try:
        from backend.faq_engine import is_faq_eligible_query, match_educational_faq
        if is_faq_eligible_query(query):
            faq_match = match_educational_faq(query)
            if faq_match:
                return IntentClassificationResult(
                    primary_intent="educational_faq",
                    detected_intents=["educational_faq"],
                    is_hinglish=is_hinglish,
                    is_refusal=False,
                    metadata={"has_faq_template": True}
                )
    except Exception:
        pass

    # Step 10: General Horoscope / Specific Astrological Inquiries
    # Determine domain if evident
    if any(k in q_lower for k in ["career", "job", "naukri", "profession", "business", "promotion", "salary"]):
        domain_intent = "career"
    elif any(k in q_lower for k in ["marriage", "wedding", "shaadi", "shadi", "relationship", "spouse", "love"]):
        domain_intent = "marriage"
    elif any(k in q_lower for k in ["money", "wealth", "finance", "paisa", "paise", "income", "property"]):
        domain_intent = "finance"
    elif any(k in q_lower for k in ["health", "swasthya", "sehat", "illness", "disease", "stress", "mental"]):
        domain_intent = "health"
    elif any(k in q_lower for k in ["travel", "abroad", "foreign", "videsh", "visa"]):
        domain_intent = "travel"
    elif any(k in q_lower for k in ["study", "education", "exam", "college", "degree", "padhai"]):
        domain_intent = "education"
    else:
        domain_intent = "general_horoscope"

    return IntentClassificationResult(
        primary_intent=domain_intent,
        detected_intents=[domain_intent],
        is_hinglish=is_hinglish,
        is_refusal=False
    )
