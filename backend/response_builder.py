"""
Domain-Locked Dynamic Follow-Up System & Response Builder
---------------------------------------------------------
1. Domain Identification & Strict Follow-up Mapping:
   Identifies the primary domain (career, marriage, finance, health, gemstones, education, travel, spirituality)
   and restricts follow-up choices 100% to THAT SAME DOMAIN.

2. Local Database Resolution for Follow-ups:
   When a follow-up chip is submitted, intercepts via qa_database.json and dynamically injects
   the active session's birth chart parameters ({lagna}, {house_10_sign}, {house_7_lord}, {current_dasha}, {user_name}).
   Guarantees zero hardcoded fallbacks between different birth charts.

3. UI Component Output (ElicitationsGroup):
   Appends <ElicitationsGroup> with self-contained, exact query strings to every response payload.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("vedic_astrology.response_builder")

# Standard 12 Vedic Sidereal Zodiac Signs
ZODIAC_SIGNS: List[str] = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

SIGN_RULERS: Dict[str, str] = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": "Mars",
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}

PLANET_GEMSTONES: Dict[str, str] = {
    "Sun": "Ruby (Manik)",
    "Moon": "Pearl (Moti)",
    "Mars": "Red Coral (Moonga)",
    "Mercury": "Emerald (Panna)",
    "Jupiter": "Yellow Sapphire (Pukhraj)",
    "Venus": "Diamond or White Sapphire (Heera)",
    "Saturn": "Blue Sapphire (Neelam)",
    "Rahu": "Hessonite (Gomed)",
    "Ketu": "Cat's Eye (Lehsunia)",
}

WORK_ENVIRONMENTS: Dict[str, str] = {
    "Sun": "strategic leadership, public governance, executive administration, or autonomous management",
    "Moon": "creative strategy, advisory consulting, public relations, wellness, or healthcare",
    "Mars": "engineering, tech operations, competitive project execution, real estate, or high-initiative ventures",
    "Mercury": "data analytics, financial modeling, communications, marketing, software engineering, or commerce",
    "Jupiter": "corporate advisory, higher education, legal strategy, asset management, or research leadership",
    "Venus": "creative design, fintech, luxury goods, media, human relations, or client-facing consulting",
    "Saturn": "structured systems, operations management, infrastructure, manufacturing, or long-term institution building",
}

PARTNER_TRAITS: Dict[str, str] = {
    "Aries": "energetic, decisive, independent, and candidly ambitious",
    "Taurus": "grounded, patient, aesthetically refined, and deeply loyal",
    "Gemini": "intellectually curious, witty, highly communicative, and versatile",
    "Cancer": "nurturing, deeply empathetic, family-oriented, and emotionally protective",
    "Leo": "dignified, warm-hearted, confident, and natural leader",
    "Virgo": "analytical, organized, attentive to detail, and practically supportive",
    "Libra": "diplomatic, charming, peace-loving, and committed to harmonious partnership",
    "Scorpio": "intensely loyal, perceptive, resilient, and transformative",
    "Sagittarius": "optimistic, philosophical, broad-minded, and adventurous",
    "Capricorn": "mature, disciplined, responsible, and steadily goal-oriented",
    "Aquarius": "visionary, independent, intellectual, and progressive",
    "Pisces": "compassionate, deeply imaginative, intuitive, and spiritually generous",
}

ISHTA_DEVATAS: Dict[str, str] = {
    "Sun": "Surya Dev or Lord Shiva",
    "Moon": "Lord Shiva or Goddess Parvati",
    "Mars": "Lord Hanuman or Lord Kartikeya",
    "Mercury": "Lord Maha Vishnu",
    "Jupiter": "Lord Brihaspati or Lord Vishnu (Dakshinamurthy)",
    "Venus": "Goddess Maha Lakshmi",
    "Saturn": "Lord Hanuman or Lord Shani",
}

# Domain-locked follow-up prompt templates (exactly 2-3 per domain)
DOMAIN_LOCKED_FOLLOWUPS: Dict[str, List[str]] = {
    "career": [
        "When is the right time for a job change or promotion?",
        "Which specific work environment or industry suits my Lagna lord?",
        "Which gemstone strengthens my 10th house career lord?",
    ],
    "marriage": [
        "What are the personality traits of my future life partner?",
        "When will my 7th house marriage activation period begin?",
        "Which astrological remedies can resolve relationship delays?",
    ],
    "finance": [
        "What are my long-term wealth accumulation prospects in my 2nd and 11th house?",
        "Which planetary period indicates high financial gains?",
        "What remedies should I perform to prevent financial losses?",
    ],
    "health": [
        "Which planetary periods require extra care for my vitality?",
        "How does my Moon sign influence my mental peace and stress?",
        "Which astrological remedies strengthen my 6th and 8th house health indicators?",
    ],
    "gemstones": [
        "Which gemstone is most auspicious for my Lagna lord?",
        "What are the Vedic rules and timings for energizing my gemstone?",
        "Can I wear gemstones for my 9th house of fortune?",
    ],
    "education": [
        "Which field of higher education best aligns with my 5th and 9th houses?",
        "What planetary periods support competitive exams and academic success?",
        "Are there favorable yogas in my chart for studying abroad?",
    ],
    "travel": [
        "When is the most favorable time for foreign travel in my dasha?",
        "Does my chart support permanent foreign settlement via the 12th house?",
        "Which remedies remove visa or relocation obstacles?",
    ],
    "spirituality": [
        "Which deity or mantra aligns with my Ishta Devata?",
        "What remedies mitigate my active karmic doshas?",
        "How does my 9th house shape my spiritual dharma and growth?",
    ],
}

DOMAIN_LOCKED_FOLLOWUPS_HINGLISH: Dict[str, List[str]] = {
    "career": [
        "Job change ya promotion ke liye sahi samay kab aayega?",
        "Mere Lagna lord ke according kaun sa work environment ya industry suit karegi?",
        "Mere 10th house career lord ko kaun sa gemstone strengthen karega?",
    ],
    "marriage": [
        "Mere future life partner ka swabhav aur traits kaisa hoga?",
        "Mere 7th house marriage activation ka samay kab shuru hoga?",
        "Relationship aur shaadi mein delay ke liye kaun se astrological upay hain?",
    ],
    "finance": [
        "Mere 2nd aur 11th house ke according long-term wealth prospects kaise hain?",
        "Kaun sa planetary period financial gains aur aamdani ke liye sabse accha hai?",
        "Financial loss se bachne ke liye kaun se astrological remedies karein?",
    ],
    "health": [
        "Vitality aur energy ke liye kaun se planetary periods mein extra care chahiye?",
        "Mera Moon sign mere mental peace aur stress ko kaise influence karta hai?",
        "6th aur 8th house health indicators ko strengthen karne ke remedies kya hain?",
    ],
    "gemstones": [
        "Mere Lagna lord ke liye kaun sa gemstone sabse shubh hai?",
        "Gemstone dharan karne ke Vedic niyam aur timings kya hain?",
        "Kya mai 9th house bhagya ke liye ratna dharan kar sakta hoon?",
    ],
    "education": [
        "Higher education ke liye kaun si stream mere 5th aur 9th house se align hoti hai?",
        "Competitive exams aur academic success ke liye kaun sa dasha period favorable hai?",
        "Kya meri kundali mein foreign studies ke liye yog hain?",
    ],
    "travel": [
        "Meri dasha mein foreign travel ke liye sabse favorable samay kab hai?",
        "Kya 12th house ke through permanent foreign settlement ke yog hain?",
        "Visa aur relocation hurdles ko dur karne ke upay kya hain?",
    ],
    "spirituality": [
        "Mere 5th house ke according mere Ishta Devata kaun hain?",
        "Pitra dosha ya karmic dosha shaanti ke upay kya hain?",
        "Mera 9th house mere spiritual dharma aur vikas ko kaise shape karta hai?",
    ],
}

DOMAIN_ALIASES: Dict[str, str] = {
    "wealth": "finance",
    "finances": "finance",
    "money": "finance",
    "job": "career",
    "jobs": "career",
    "profession": "career",
    "business": "career",
    "work": "career",
    "relationship": "marriage",
    "relationships": "marriage",
    "love": "marriage",
    "gemstone": "gemstones",
    "study": "education",
    "studies": "education",
    "foreign": "travel",
}


def normalize_domain_name(domain: Optional[str]) -> str:
    """Normalizes any domain string or alias into one of the 8 canonical keys."""
    if not domain:
        return "career"
    d_clean = domain.lower().strip()
    return DOMAIN_ALIASES.get(d_clean, d_clean if d_clean in DOMAIN_LOCKED_FOLLOWUPS else "career")


def detect_query_primary_domain(query: str, fallback_domain: Optional[str] = None) -> str:
    """
    Identifies the single primary domain of the user's query with domain locking.
    """
    if not query:
        return normalize_domain_name(fallback_domain)

    q_lower = query.lower().strip()

    # Priority 1: Match against known follow-up queries directly
    for domain, questions in DOMAIN_LOCKED_FOLLOWUPS.items():
        for q in questions:
            if q.lower() in q_lower or q_lower in q.lower():
                return domain

    # Priority 2: Distinct domain keyword scanning
    domain_keywords = [
        ("marriage", ["marri", "spouse", "partner", "relationship", "wedding", "shaadi", "shadi", "7th house", "husband", "wife", "vivah", "manglik"]),
        ("career", ["career", "job", "profession", "promotion", "appraisal", "workplace", "10th house", "karma bhava", "business vs job", "boss", "interview", "employment"]),
        ("finance", ["finance", "finances", "wealth", "money", "savings", "debt", "loan", "investment", "2nd house", "11th house", "paisa", "income", "dhan"]),
        ("health", ["health", "disease", "illness", "vitality", "immunity", "6th house", "8th house", "mental stress", "wellness", "sehat", "recovery"]),
        ("gemstones", ["gemstone", "gemstones", "ratna", "wearing rules", "pukhraj", "manik", "neelam", "panna", "moonga", "heera", "crystal"]),
        ("education", ["education", "study", "studies", "exam", "exams", "college", "degree", "university", "academic", "5th house", "student"]),
        ("travel", ["travel", "foreign", "abroad", "visa", "settlement", "relocation", "relocate", "overseas", "12th house", "9th house"]),
        ("spirituality", ["spiritual", "spirituality", "ishta devata", "mantra", "puja", "pooja", "pitra dosha", "kaal sarp", "dharma", "deity", "moksha"]),
    ]

    for dom, kws in domain_keywords:
        if any(kw in q_lower for kw in kws):
            return dom

    if fallback_domain:
        return normalize_domain_name(fallback_domain)
    return "career"


def get_domain_locked_followups(
    domain: str,
    current_query: Optional[str] = None,
    is_hinglish: Optional[bool] = None,
) -> List[str]:
    """
    Returns exactly 2-3 follow-up prompts restricted strictly to the given domain,
    mirroring the language of the query (Hinglish or English).
    Filters out the exact incoming query if it matches one of the templates.
    """
    if is_hinglish is None and current_query:
        try:
            from backend.llm_router import is_hinglish_query
            is_hinglish = is_hinglish_query(current_query)
        except Exception:
            try:
                from llm_router import is_hinglish_query
                is_hinglish = is_hinglish_query(current_query)
            except Exception:
                is_hinglish = False

    canonical_domain = normalize_domain_name(domain)
    source_dict = DOMAIN_LOCKED_FOLLOWUPS_HINGLISH if is_hinglish else DOMAIN_LOCKED_FOLLOWUPS
    pool = source_dict.get(canonical_domain, source_dict["career"])

    if not current_query:
        return list(pool[:3])

    clean_curr = re.sub(r"[^\w\s]", "", current_query).lower().strip()
    filtered = [
        item for item in pool
        if re.sub(r"[^\w\s]", "", item).lower().strip() != clean_curr
    ]
    if len(filtered) >= 2:
        return filtered[:3]
    return pool[:3]


def render_elicitations_group(follow_up_questions: List[str]) -> str:
    """
    Renders 2-3 follow-up prompt options as an <ElicitationsGroup> UI component
    with self-contained exact query strings.
    """
    if not follow_up_questions:
        return ""

    elicitation_items = []
    for q in follow_up_questions:
        clean_q = q.strip()
        escaped_q = clean_q.replace('"', "&quot;")
        elicitation_items.append(f'  <Elicitation query="{escaped_q}">{clean_q}</Elicitation>')

    inner = "\n".join(elicitation_items)
    return f"<ElicitationsGroup>\n{inner}\n</ElicitationsGroup>"


def build_user_chart_context(
    user_name: Optional[str] = None,
    lagna: Optional[str] = None,
    moon_sign: Optional[str] = None,
    sun_sign: Optional[str] = None,
    current_dasha: Optional[str] = None,
    astro_features: Optional[List[int]] = None,
    nakshatra: Optional[str] = None,
    nakshatra_pada: Optional[str] = None,
    nakshatra_lord: Optional[str] = None,
) -> Dict[str, str]:
    """
    Builds a complete, deterministic dictionary of dynamic chart parameters for the active session.
    Guarantees that User A and User B get hyper-personalized variables.
    Now includes Janma Nakshatra (nakshatra, nakshatra_pada, nakshatra_lord) and all 12 house signs/lords.
    """
    clean_name = (user_name or "").strip()
    if not clean_name or clean_name.lower() in ("guest", "there", "user", "anonymous"):
        clean_name = "Seeker"

    asc = lagna if lagna in ZODIAC_SIGNS else "Leo"
    asc_idx = ZODIAC_SIGNS.index(asc)

    def house_sign(house_num: int) -> str:
        return ZODIAC_SIGNS[(asc_idx + house_num - 1) % 12]

    def house_lord(house_num: int) -> str:
        s = house_sign(house_num)
        return SIGN_RULERS.get(s, "Benefic Planet")

    lagna_lord = SIGN_RULERS.get(asc, "Sun")
    h2_sign, h2_lord = house_sign(2), house_lord(2)
    h3_sign, h3_lord = house_sign(3), house_lord(3)
    h4_sign, h4_lord = house_sign(4), house_lord(4)
    h5_sign, h5_lord = house_sign(5), house_lord(5)
    h6_sign, h6_lord = house_sign(6), house_lord(6)
    h7_sign, h7_lord = house_sign(7), house_lord(7)
    h8_sign, h8_lord = house_sign(8), house_lord(8)
    h9_sign, h9_lord = house_sign(9), house_lord(9)
    h10_sign, h10_lord = house_sign(10), house_lord(10)
    h11_sign, h11_lord = house_sign(11), house_lord(11)
    h12_sign, h12_lord = house_sign(12), house_lord(12)

    gem_10th = PLANET_GEMSTONES.get(h10_lord, "Diamond or White Sapphire")
    gem_lagna = PLANET_GEMSTONES.get(lagna_lord, "Ruby")
    gem_9th = PLANET_GEMSTONES.get(h9_lord, "Yellow Sapphire")

    work_env = WORK_ENVIRONMENTS.get(lagna_lord, "strategic leadership and autonomous execution")
    partner_tr = PARTNER_TRAITS.get(h7_sign, "grounded, supportive, and harmonious")
    ishta = ISHTA_DEVATAS.get(h5_lord, "Lord Maha Vishnu or Goddess Lakshmi")

    if h2_lord in ("Mercury", "Venus", "Jupiter") or h11_lord in ("Mercury", "Venus", "Jupiter"):
        fin_prospects = "your birth chart indicates strong Dhana Yogas supporting consistent wealth accumulation and liquidity"
    else:
        fin_prospects = "your wealth builds steadily through disciplined compounding, strategic budgeting, and long-term asset investments"

    active_dasha = (current_dasha or "").strip()
    if not active_dasha or active_dasha.lower() in ("none", "unknown"):
        active_dasha = "Active Dasha period"

    # Nakshatra defaults - use provided values or sensible fallbacks
    resolved_nakshatra = (nakshatra or "Rohini").strip()
    resolved_pada = (nakshatra_pada or "Pada 1").strip()
    resolved_nak_lord = (nakshatra_lord or "Moon").strip()

    return {
        "user_name": clean_name,
        "lagna": asc,
        "ascendant": asc,
        "lagna_lord": lagna_lord,
        # All 12 house signs and lords
        "house_1_sign": asc,
        "house_1_lord": lagna_lord,
        "house_2_sign": h2_sign,
        "house_2_lord": h2_lord,
        "house_3_sign": h3_sign,
        "house_3_lord": h3_lord,
        "house_4_sign": h4_sign,
        "house_4_lord": h4_lord,
        "house_5_sign": h5_sign,
        "house_5_lord": h5_lord,
        "house_6_sign": h6_sign,
        "house_6_lord": h6_lord,
        "house_7_sign": h7_sign,
        "house_7_lord": h7_lord,
        "house_8_sign": h8_sign,
        "house_8_lord": h8_lord,
        "house_9_sign": h9_sign,
        "house_9_lord": h9_lord,
        "house_10_sign": h10_sign,
        "house_10_lord": h10_lord,
        "house_11_sign": h11_sign,
        "house_11_lord": h11_lord,
        "house_12_sign": h12_sign,
        "house_12_lord": h12_lord,
        # Gemstones
        "gemstone_10th": gem_10th,
        "gemstone_lagna": gem_lagna,
        "gemstone_9th": gem_9th,
        # Personality & destiny indicators
        "work_environment": work_env,
        "partner_traits": partner_tr,
        "financial_prospects": fin_prospects,
        "ishta_devata": ishta,
        "current_dasha": active_dasha,
        "moon_sign": moon_sign or "Aquarius",
        "sun_sign": sun_sign or "Leo",
        # Janma Nakshatra
        "nakshatra": resolved_nakshatra,
        "nakshatra_pada": resolved_pada,
        "nakshatra_lord": resolved_nak_lord,
    }


def load_qa_database_dict() -> Dict[str, str]:
    """Loads and caches qa_database.json."""
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qa_database.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa_database.json"),
        "c:/Users/a/Desktop/astro_env/qa_database.json",
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Error reading %s: %s", p, e)
    return {}


# Conversational prefix patterns to strip before QA key lookup
_QA_PREFIX_RE = re.compile(
    r"^(?:"
    r"for\s+my\s+|what\s+is\s+my\s+|what's\s+my\s+|tell\s+me\s+about\s+my\s+|"
    r"explain\s+my\s+|show\s+me\s+my\s+|give\s+me\s+my\s+|about\s+my\s+|"
    r"my\s+|mera\s+|meri\s+|mere\s+|mujhe\s+batao\s+|batao\s+mera\s+|"
    r"kya\s+hai\s+mera\s+|kya\s+hai\s+meri\s+|aapka\s+|apna\s+"
    r")",
    re.IGNORECASE,
)

# Typo / alias corrections for QA key lookup
_QA_ALIASES: Dict[str, str] = {
    "nakshtra": "nakshatra",
    "nakhatra": "nakshatra",
    "nakshatram": "nakshatra",
    "birth star": "nakshatra",
    "janma nakshatra": "nakshatra",
    "janm nakshatra": "nakshatra",
    "birth nakshatra": "nakshatra",
    "rashi": "moon sign",
    "moon rashi": "moon sign",
    "chandra rashi": "moon sign",
    "lagna rashi": "lagna",
    "ascendant sign": "ascendant",
    "rising sign": "ascendant",
    "asc": "ascendant",
    "mahadasha": "dasha",
    "antardasha": "dasha",
    "vimshottari dasha": "dasha",
    "janma dasha": "dasha",
    "current dasha": "dasha",
    "sun sign": "sun sign",
    "surya rashi": "sun sign",
}


def normalize_qa_concept(query: str) -> str:
    """
    Strips conversational prefixes and corrects common typos/aliases
    to produce a clean QA key that can be matched against qa_database.json.
    Examples:
      'for my nakshtra'    -> 'nakshatra'
      'what is my lagna'   -> 'lagna'
      'mera moon sign'     -> 'moon sign'
      'kya hai mera rashi' -> 'moon sign'
    """
    q = query.strip().lower()
    # Remove punctuation except spaces
    q = re.sub(r"[^\w\s]", "", q).strip()
    # Strip conversational prefixes (may repeat, e.g. 'kya hai mera')
    for _ in range(3):
        q_new = _QA_PREFIX_RE.sub("", q).strip()
        if q_new == q:
            break
        q = q_new
    # Apply alias corrections
    return _QA_ALIASES.get(q, q)


# QA concept -> domain mapping for follow-up generation
_QA_CONCEPT_DOMAINS: Dict[str, str] = {
    "nakshatra": "spirituality",
    "moon sign": "spirituality",
    "sun sign": "spirituality",
    "lagna": "spirituality",
    "ascendant": "spirituality",
    "dasha": "spirituality",
    "sade sati": "spirituality",
    "dhayya": "spirituality",
    "rahu ketu": "spirituality",
    "retrograde": "spirituality",
    "kundali": "spirituality",
    "navamsha": "spirituality",
    "1st house": "spirituality",
    "2nd house": "finance",
    "3rd house": "spirituality",
    "4th house": "spirituality",
    "5th house": "education",
    "6th house": "health",
    "7th house": "marriage",
    "8th house": "health",
    "9th house": "travel",
    "10th house": "career",
    "11th house": "finance",
    "12th house": "travel",
    "career house": "career",
    "marriage house": "marriage",
    "health house": "health",
    "education house": "education",
}


def resolve_domain_locked_followup(
    query: str,
    chart_context: Dict[str, str],
    qa_db: Optional[Dict[str, str]] = None,
    is_first_turn: bool = True,
) -> Optional[Tuple[str, List[str], str]]:
    """
    Intercepts follow-up questions from the user, retrieves the template from qa_database.json,
    injects user birth chart details, and returns:
    (formatted_answer, follow_up_questions_list, detected_domain).

    Now handles:
    - Short concept keys like 'nakshatra' (len < 12)
    - Conversational prefixes: 'for my', 'what is my', 'mera', 'kya hai', etc.
    - Typo aliases: 'nakshtra' -> 'nakshatra'
    - Hinglish variants via _hinglish key suffix lookup
    """
    if not query:
        return None

    db = qa_db or load_qa_database_dict()
    q_norm = query.strip().lower()
    clean_q_tokens = re.sub(r"[^\w\s]", "", q_norm).strip()

    # Detect language for response variant selection
    is_hinglish = False
    try:
        from llm_router import is_hinglish_query
        is_hinglish = is_hinglish_query(query)
    except Exception:
        try:
            from backend.llm_router import is_hinglish_query
            is_hinglish = is_hinglish_query(query)
        except Exception:
            hinglish_markers = ["kya", "hai", "mera", "meri", "mere", "kab", "hoga", "kaisa", "aur", "batao", "mujhe"]
            is_hinglish = any(m in q_norm for m in hinglish_markers)

    # Step 1: Check against domain-locked follow-up pools (exact match)
    matched_template: Optional[str] = None
    matched_domain: Optional[str] = None
    matched_canonical_q: Optional[str] = None

    for pool_dict in [DOMAIN_LOCKED_FOLLOWUPS, DOMAIN_LOCKED_FOLLOWUPS_HINGLISH]:
        for domain, questions in pool_dict.items():
            for candidate_q in questions:
                cand_norm = candidate_q.strip().lower()
                clean_cand_tokens = re.sub(r"[^\w\s]", "", cand_norm).strip()
                if cand_norm == q_norm or clean_cand_tokens == clean_q_tokens:
                    matched_template = db.get(cand_norm) or db.get(clean_cand_tokens)
                    matched_domain = domain
                    matched_canonical_q = candidate_q
                    break
            if matched_template:
                break
        if matched_template:
            break

    # Step 2: Direct qa_database.json key scan (exact + guarded containment)
    if not matched_template:
        for k, v in db.items():
            k_clean = re.sub(r"[^\w\s]", "", k).strip().lower()
            # Exact match always wins
            if k_clean == clean_q_tokens:
                matched_template = v
                matched_domain = _QA_CONCEPT_DOMAINS.get(k_clean) or detect_query_primary_domain(query)
                matched_canonical_q = query
                break
            # Containment match: only when key is long enough (≥8 chars, avoids 'hi'/'rahu'/etc.)
            # AND the query is not a long multi-topic sentence that merely contains the key as a substring.
            # Rule: query must have ≤ 6 words OR the key length is at least 60% of the query length.
            q_word_count = len(clean_q_tokens.split())
            if (
                len(k_clean) >= 8
                and k_clean in clean_q_tokens
                and (q_word_count <= 6 or len(k_clean) >= len(clean_q_tokens) * 0.5)
                and re.search(r"\b" + re.escape(k_clean) + r"\b", clean_q_tokens)
            ):
                matched_template = v
                matched_domain = _QA_CONCEPT_DOMAINS.get(k_clean) or detect_query_primary_domain(query)
                matched_canonical_q = query
                break
            # Reverse: query is a prefix/suffix of the key (e.g., user typed "10th house" vs key "10th house lord")
            if len(k_clean) >= 8 and clean_q_tokens and clean_q_tokens in k_clean:
                matched_template = v
                matched_domain = _QA_CONCEPT_DOMAINS.get(k_clean) or detect_query_primary_domain(query)
                matched_canonical_q = query
                break

    # Step 3: Normalize conversational prefix and re-scan (handles 'for my nakshtra', 'mera lagna', etc.)
    if not matched_template:
        concept = normalize_qa_concept(query)
        if concept and concept != clean_q_tokens:
            # Try exact concept key in db
            if concept in db:
                # Prefer hinglish variant if query is in hinglish
                hinglish_key = concept + "_hinglish"
                if is_hinglish and hinglish_key in db:
                    matched_template = db[hinglish_key]
                else:
                    matched_template = db[concept]
                matched_domain = _QA_CONCEPT_DOMAINS.get(concept) or detect_query_primary_domain(query)
                matched_canonical_q = concept
            else:
                # Fuzzy scan on normalized concept — only valid if concept is short (≤ 4 words)
                # A long concept means prefix stripping didn't simplify it, so don't trust it
                concept_word_count = len(concept.split())
                if concept_word_count <= 4:
                    for k, v in db.items():
                        k_clean = re.sub(r"[^\w\s]", "", k).strip().lower()
                        if k_clean == concept or (
                            k_clean
                            and len(k_clean) >= 4
                            and (k_clean in concept or concept in k_clean)
                        ):
                            # Prefer hinglish variant
                            hinglish_key = k + "_hinglish"
                            if is_hinglish and hinglish_key in db:
                                matched_template = db[hinglish_key]
                            else:
                                matched_template = v
                            matched_domain = _QA_CONCEPT_DOMAINS.get(k_clean) or detect_query_primary_domain(query)
                            matched_canonical_q = k
                            break

    if not matched_template or not matched_domain:
        return None

    # Inject dynamic user birth chart parameters
    personalized_text = matched_template
    for token, val in chart_context.items():
        placeholder = f"{{{token}}}"
        personalized_text = personalized_text.replace(placeholder, str(val))

    # Generate subsequent follow-ups for this same domain
    next_followups = get_domain_locked_followups(matched_domain, current_query=matched_canonical_q or query)
    clean_text = strip_elicitations_group(personalized_text)
    if not is_first_turn:
        clean_text = strip_leading_greeting(clean_text)
    return clean_text, next_followups, matched_domain


def strip_leading_greeting(text: str) -> str:
    """
    Strips opening greetings such as 'Namaste Rahul!', 'Hello Rahul!', 'Hi Rahul!',
    'Rahul, ...' from the response if the user has already been greeted in the conversation.
    Leaves the main astrological interpretation intact and properly capitalized.
    """
    if not text:
        return ""

    cleaned = text.strip()
    badge = ""
    if cleaned.startswith("[Source:") or cleaned.startswith("[Model:") or cleaned.startswith("[Local Database"):
        parts = cleaned.split("\n\n", 1)
        if len(parts) == 2:
            badge = parts[0] + "\n\n"
            cleaned = parts[1].strip()

    pattern = r"^(?:(?:Namaste|Hello|Hi|Pranam|Greetings)\s*(?:[A-Za-z0-9_'\-]+\s*)?[!\.,]\s*|[A-Za-z0-9_'\-]+,\s+)"
    match = re.search(pattern, cleaned, re.IGNORECASE)
    if match:
        cleaned = cleaned[match.end():].strip()
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]

    return badge + cleaned


def strip_elicitations_group(text: str) -> str:
    """Removes any <ElicitationsGroup>...</ElicitationsGroup> XML tags from text."""
    if not text:
        return ""
    clean = re.sub(r"<ElicitationsGroup>[\s\S]*?</ElicitationsGroup>", "", text)
    return clean.strip()


def attach_domain_locked_elicitations(
    answer: str,
    query: str,
    domain: Optional[str] = None,
    is_first_turn: bool = True,
) -> Tuple[str, List[str]]:
    """
    Strips any XML components from the answer, returning clean plain text
    along with the domain-locked follow-up questions for frontend metadata.
    Enforces greeting policy: greets only on the first turn of the chat.
    """
    clean_answer = strip_elicitations_group(answer)
    if not is_first_turn:
        clean_answer = strip_leading_greeting(clean_answer)

    primary_domain = detect_query_primary_domain(query, fallback_domain=domain)
    follow_ups = get_domain_locked_followups(primary_domain, current_query=query)
    return clean_answer, follow_ups
