"""
Domain Filter Module: Conservative, Deterministic Auxiliary Pre-Filter.
------------------------------------------------------------------------
Implements a 3-tier auxiliary screening architecture to reject clearly
unrelated requests (code generation, math solving, recipes, trivia, weather)
before the ML classifier, while strictly preserving:
1. Legitimate astrology inquiries, chart interpretations, and life guidance.
2. Colloquial wellness and physical state inquiries (tiredness, fatigue, etc.).
3. Technical and modern career inquiries (Python developer, data science, AWS).
4. Mixed-intent queries (e.g. "Explain my career prospects and write a Python script").
5. Ambiguous or indirect distress signals (which continue downstream to ML / Stage 8).
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Set, List


class FilterAction(str, Enum):
    CONTINUE = "CONTINUE"
    REJECT_OUT_OF_SCOPE = "REJECT_OUT_OF_SCOPE"


@dataclass
class DomainFilterDecision:
    action: FilterAction
    reason: str
    confidence: float
    message: Optional[str] = None


# Standard Out-of-Scope rejection response
OUT_OF_SCOPE_RESPONSE = (
    "I am an Astro AI assistant specialized strictly in Vedic astrology, birth charts, "
    "and life guidance. Please ask a question related to your chart, career, marriage, health, or life path!"
)

# ---------------------------------------------------------------------------
# 1. In-Scope Domain & Astrological Vocabulary (Override / Whitelist Signals)
# ---------------------------------------------------------------------------
ASTROLOGY_TERMS: Set[str] = {
    # Astrological entities & concepts
    "astrology", "vedic", "horoscope", "kundali", "kundli", "chart", "charts", "birth",
    "natal", "lagna", "ascendant", "rashi", "moon", "sun", "planet", "planets", "graha",
    "grahas", "transit", "transits", "gochar", "dasha", "dashas", "mahadasha", "antardasha",
    "vimshottari", "house", "houses", "bhava", "bhavas", "yoga", "yogas", "dosha", "doshas",
    "manglik", "mangal", "sade", "sati", "dhaya", "kaal", "sarp", "rahu", "ketu", "shani",
    "saturn", "jupiter", "guru", "brihaspati", "venus", "shukra", "mercury", "budh", "mars",
    "surya", "chandra", "gemstone", "gemstones", "remedy", "remedies", "mantra", "mantras",
    "puja", "upay", "upaye", "totka", "totke", "nakshatra", "navamsha", "panna", "ruby",
    "pearl", "sapphire", "emerald", "coral", "diamond", "heera", "moti", "manik", "pukhraj",
    "neelam", "rudraksha", "ephemeris", "sidereal", "zodiac", "astrological"
}

LIFE_DOMAIN_TERMS: Set[str] = {
    # Career & Work
    "career", "job", "profession", "professional", "work", "business", "promotion", "appraisal",
    "salary", "boss", "naukri", "nokri", "kaam", "vyapar", "dhandha", "employment", "office",
    "interview", "startup", "vocation", "transfer", "resignation", "growth", "tarakki", "safalta",
    "bakery", "shop", "firm", "company", "vocation",

    # Marriage & Relationships
    "marriage", "married", "wedding", "spouse", "partner", "relationship", "relationships",
    "love", "divorce", "shaadi", "shadi", "vivah", "rishta", "pyaar", "prem", "pati", "patni",
    "compatibility", "milan", "matching", "breakup", "soulmate", "dulha", "dulhan", "remarriage",

    # Wealth & Finance
    "money", "wealth", "finance", "financial", "finances", "property", "investment", "rich",
    "income", "paisa", "paise", "dhan", "sampatti", "karz", "loan", "debt", "savings", "bachat",
    "loss", "profit", "assets", "ameer", "daulat", "kamao", "kamayi", "nivesh",

    # Health, Wellness & Physical States (Broadened for colloquial & Hinglish)
    "health", "disease", "illness", "body", "medical", "surgery", "fitness", "sehat", "swasthya",
    "bimari", "pain", "mental", "depression", "anxiety", "stress", "tanav", "chinta", "recovery",
    "longevity", "ayush", "swasth", "dard", "ilaj", "dawai", "tired", "tiredness", "fatigue",
    "lethargy", "exhaustion", "exhausted", "weakness", "weak", "low energy", "energy", "headache",
    "headaches", "chakkar", "susti", "tabiyat", "neend", "sleep", "insomnia", "vitality",

    # Education & Learning
    "education", "study", "studies", "exam", "exams", "college", "degree", "learning", "padhai",
    "shiksha", "pariksha", "imtihan", "university", "school", "academic", "pass", "fail", "score",
    "certification", "test",

    # Travel & Foreign Settlement
    "travel", "abroad", "foreign", "trip", "journey", "visa", "settle", "settlement", "yatra",
    "videsh", "pardesh", "relocation", "immigration", "pr", "relocate",

    # Life Guidance & Distress Signals
    "life", "future", "destiny", "fate", "kismat", "bhagya", "karma", "moksha", "prediction",
    "predictions", "guidance", "guide", "muhurat", "shubh", "ashubh", "favorable", "unfavorable",
    "timing", "luck", "lucky", "prospects", "outlook", "stuck", "pareshan", "pareshani", "hope",
    "hopeless", "confused", "help"
}

CONVERSATIONAL_GREETINGS: Set[str] = {
    "hi", "hello", "hey", "namaste", "pranam", "greetings", "good morning",
    "good afternoon", "good evening", "who are you", "what can you do", "help me",
    "who are u", "what is your name", "what do you do", "introduce yourself",
    "kya kar sakte ho", "kaise ho", "help", "guide me", "how are you"
}

# ---------------------------------------------------------------------------
# 2. Explicit Out-of-Scope Blacklist Patterns
# ---------------------------------------------------------------------------
EXPLICIT_OUT_OF_SCOPE_PATTERNS = [
    # 1. Code Generation, Script Writing, Programming Problem Solving
    (
        "coding_task",
        re.compile(
            r"\b(?:write|code|debug|compile|fix|implement|create|generate)\s+.*?\b(?:python|javascript|typescript|c\+\+|java|html|css|sql|react|vue|angular|node|script|regex|function|class|algorithm|query|loop|program)\b",
            re.IGNORECASE
        )
    ),
    (
        "code_snippet_request",
        re.compile(
            r"\b(?:python|javascript|typescript|c\+\+|java|golang|rust|php)\s+(?:code|script|syntax|error|tutorial|function|library|framework)\b",
            re.IGNORECASE
        )
    ),
    (
        "developer_cli_or_repo",
        re.compile(
            r"\b(?:leetcode|stackoverflow|git\s+push|git\s+commit|pip\s+install|npm\s+install)\b",
            re.IGNORECASE
        )
    ),
    (
        "how_to_code",
        re.compile(
            r"\bhow\s+to\s+(?:code|program|build\s+an?\s+app|make\s+a\s+website|write\s+a\s+script)\b",
            re.IGNORECASE
        )
    ),
    (
        "pure_coding_slang",
        re.compile(
            r"^\s*python\s+.*?\b(?:reverse|string|sort|list|array|function|script)\b",
            re.IGNORECASE
        )
    ),

    # 2. Math, Equations & Quantitative Calculations
    (
        "math_equation_solve",
        re.compile(
            r"\b(?:solve|calculate|evaluate|integrate|differentiate|compute)\s+.*?(?:equation|integral|derivative|algebra|matrix|quadratic|pythagorean|\d+\s*[\+\-\*\/])\b",
            re.IGNORECASE
        )
    ),
    (
        "pure_arithmetic",
        re.compile(
            r"^\s*(?:what\s+is\s+)?\d+\s*[\+\-\*\/\^]\s*\d+\s*\??$",
            re.IGNORECASE
        )
    ),
    (
        "math_formulas",
        re.compile(
            r"\b(?:square\s+root\s+of|derivative\s+of|integral\s+of|value\s+of\s+pi|quadratic\s+formula|fibonacci\s+series)\b",
            re.IGNORECASE
        )
    ),

    # 3. Cooking Recipes & Food Preparation
    (
        "culinary_recipe",
        re.compile(
            r"\b(?:recipe\s+for|ingredients\s+for|how\s+(?:do\s+i|to|can\s+i|2)\s+(?:bake|cook|make)\s+.*?\b(?:cake|cookies?|bread|pasta|biryani|tea|coffee|curry|pizza|pie|dish|snack))\b",
            re.IGNORECASE
        )
    ),

    # 4. General Trivia, Geography, History, Pop Culture
    (
        "geography_trivia",
        re.compile(
            r"\b(?:what\s+is\s+the\s+capital\s+of|who\s+was\s+napoleon|history\s+of\s+(?:rome|france|world\s+war)|who\s+invented)\b",
            re.IGNORECASE
        )
    ),
    (
        "political_officeholder_trivia",
        re.compile(
            r"\bwho\s+is\s+(?:the\s+)?(?:prime\s+minister|president|chief\s+minister|governor|king|queen|chancellor|ceo)\s+of\b",
            re.IGNORECASE
        )
    ),
    (
        "vehicle_and_speed_trivia",
        re.compile(
            r"\b(?:which|what)\s+is\s+(?:the\s+)?(?:fastest|fasest|quickest|speediest)\s+(?:car|vehicle|plane|train|bike|automobile)\b",
            re.IGNORECASE
        )
    ),
    (
        "sports_pop_trivia",
        re.compile(
            r"\b(?:who\s+won\s+the\s+(?:world\s+cup|ipl|super\s+bowl|oscar)|who\s+directed|lyrics\s+of)\b",
            re.IGNORECASE
        )
    ),
    (
        "school_science_trivia",
        re.compile(
            r"\b(?:explain\s+photosynthesis|how\s+do\s+airplanes\s+fly|speed\s+of\s+light|quantum\s+physics|theory\s+of\s+relativity)\b",
            re.IGNORECASE
        )
    ),

    # 5. Real-Time Weather & Logistics
    (
        "weather_inquiry",
        re.compile(
            r"\b(?:weather\s+in|weather\s+forecast|temperature\s+in|flight\s+status|traffic\s+in)\b",
            re.IGNORECASE
        )
    ),

    # 6. Politics & Partisan Elections
    (
        "political_debate",
        re.compile(
            r"\b(?:presidential\s+election|prime\s+minister\s+election|democrat\s+vs\s+republican|bjp\s+vs\s+congress|israel\s+palestine|russia\s+ukraine)\b",
            re.IGNORECASE
        )
    )
]


def _has_in_scope_signal(query_lower: str) -> bool:
    """
    Checks if the query contains any Vedic astrology entities, life domains,
    or wellness indicators that warrant continued evaluation in the astrology pipeline.
    """
    # Tokenize words ignoring punctuation
    words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query_lower))

    # Direct token intersection
    if any(t in words for t in ASTROLOGY_TERMS):
        return True
    if any(t in words for t in LIFE_DOMAIN_TERMS):
        return True

    # Multi-word phrase checks
    multiword_phrases = [
        "moon sign", "sun sign", "birth chart", "natal chart", "sade sati",
        "kaal sarp", "mangal dosha", "10th house", "7th house", "2nd house",
        "1st house", "8th house", "11th house", "tired all the time",
        "low energy", "feel tired", "guide me", "life path", "what should i do",
        "things get better", "feels stuck", "pareshan hoon", "tabiyat theek"
    ]
    if any(phrase in query_lower for phrase in multiword_phrases):
        return True

    return False


def _is_conversational_greeting(query_lower: str) -> bool:
    """Checks if the query is a simple greeting or identity question."""
    cleaned = re.sub(r"[^\w\s]", "", query_lower).strip()
    if cleaned in CONVERSATIONAL_GREETINGS:
        return True
    if any(cleaned.startswith(g) for g in ["hi ", "hello ", "hey ", "namaste ", "pranam "]):
        return True
    if bool(re.search(r"\b(?:who\s+are\s+you|what\s+can\s+you\s+do|introduce\s+yourself|help\s+me)\b", query_lower)):
        return True
    return False


def check_domain_prefilter(query: str) -> DomainFilterDecision:
    """
    Evaluates an incoming query through the conservative 3-tier pre-filter.

    Rules:
    1. Conversational greetings -> CONTINUE (in-scope).
    2. Explicit Out-of-Scope Blacklist match:
       - If mixed with legitimate astrology/career/wellness context -> CONTINUE (preserve astrology reasoning).
       - If purely out-of-scope with NO legitimate context -> REJECT_OUT_OF_SCOPE.
    3. All other queries (in-scope or ambiguous) -> CONTINUE (never reject on keyword absence alone).

    Returns:
        DomainFilterDecision: action (CONTINUE or REJECT_OUT_OF_SCOPE), reason, confidence, and message.
    """
    if not query or not query.strip():
        return DomainFilterDecision(
            action=FilterAction.CONTINUE,
            reason="empty_query",
            confidence=1.0
        )

    q_clean = query.strip()
    q_low = q_clean.lower()

    # 1. Greetings fast-track
    if _is_conversational_greeting(q_low):
        return DomainFilterDecision(
            action=FilterAction.CONTINUE,
            reason="conversational_greeting",
            confidence=1.0
        )

    # 2. Check explicit blacklist patterns
    matched_rule: Optional[str] = None
    for rule_name, pat in EXPLICIT_OUT_OF_SCOPE_PATTERNS:
        if pat.search(q_clean):
            matched_rule = rule_name
            break

    if matched_rule:
        # Refinement 1: Conservative exception check
        # Does the query contain legitimate chart, career, wellness, or life context?
        has_legit_context = _has_in_scope_signal(q_low)

        if has_legit_context:
            # e.g., "Will I get a job as a Python developer according to my chart?"
            # or "Explain my career prospects and write a Python script"
            return DomainFilterDecision(
                action=FilterAction.CONTINUE,
                reason=f"mixed_intent_with_astrology_context:{matched_rule}",
                confidence=0.85
            )

        # Purely out-of-scope with NO astrological or life-path context
        return DomainFilterDecision(
            action=FilterAction.REJECT_OUT_OF_SCOPE,
            reason=f"explicit_out_of_scope:{matched_rule}",
            confidence=0.99,
            message=OUT_OF_SCOPE_RESPONSE
        )

    # 3. Refinement 2: If query does NOT match any explicit out-of-scope blacklist,
    # it MUST ALWAYS CONTINUE downstream to the ML router, indirect intent resolver,
    # and Stage-8 reasoning. Never reject purely because an in-scope keyword was absent!
    return DomainFilterDecision(
        action=FilterAction.CONTINUE,
        reason="in_scope_or_ambiguous",
        confidence=1.0
    )
