import os
import re
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np

# Standard Out-of-Scope rejection response mandated by Rule Book
OUT_OF_SCOPE_RESPONSE = (
    "I am an Astro AI assistant specialized strictly in Vedic astrology, birth charts, "
    "and life guidance. Please ask a question related to your chart, career, marriage, or health!"
)

# Core Domain Keywords for life-path domains and astrology
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "career": [
        "career", "job", "profession", "work", "business", "promotion", "appraisal",
        "salary", "boss", "naukri", "kaam", "vyapar", "dhandha", "employment", "office",
        "interview", "startup", "vocation", "transfer", "resignation"
    ],
    "marriage": [
        "marriage", "married", "wedding", "spouse", "partner", "relationship", "love",
        "divorce", "shaadi", "shadi", "vivah", "rishta", "pyaar", "prem", "pati", "patni",
        "compatibility", "gun milan", "kundali matching", "breakup", "soulmate"
    ],
    "wealth": [
        "money", "wealth", "finance", "financial", "property", "investment", "rich",
        "income", "paisa", "paise", "dhan", "sampatti", "karz", "loan", "debt",
        "savings", "bachat", "loss", "profit", "assets"
    ],
    "health": [
        "health", "disease", "illness", "body", "medical", "surgery", "fitness",
        "sehat", "swasthya", "bimari", "pain", "mental health", "depression", "anxiety",
        "stress", "tanav", "chinta", "recovery", "longevity", "ayush"
    ],
    "travel": [
        "travel", "abroad", "foreign", "trip", "journey", "visa", "settle",
        "yatra", "videsh", "pardesh", "relocation", "immigration", "pr"
    ],
    "children": [
        "child", "children", "baby", "pregnancy", "son", "daughter", "progeny",
        "bachcha", "baccha", "bacche", "santaan", "santan", "fertility", "family"
    ],
    "education": [
        "education", "study", "exam", "college", "degree", "learning", "padhai",
        "shiksha", "pariksha", "imtihan", "university", "school", "academic"
    ],
    "astrology": [
        "kundali", "kundli", "horoscope", "birth chart", "natal chart", "lagna",
        "ascendant", "rashi", "moon sign", "sun sign", "planet", "graha", "transit",
        "gochar", "dasha", "mahadasha", "antardasha", "vimshottari", "house", "bhava",
        "yoga", "dosha", "manglik", "mangal dosha", "sade sati", "dhaya", "kaal sarp",
        "rahu", "ketu", "shani", "saturn", "jupiter", "guru", "venus", "shukra",
        "mercury", "budh", "mars", "sun", "surya", "moon", "chandra", "gemstone",
        "remedy", "mantra", "puja", "upay", "nakshatra", "navamsha"
    ]
}

# Conversational starters and greetings allowed without rejection
CONVERSATIONAL_GREETINGS = {
    "hi", "hello", "hey", "namaste", "pranam", "greetings", "good morning",
    "good afternoon", "good evening", "who are you", "what can you do", "help me",
    "who are u", "what is your name", "what do you do", "introduce yourself",
    "kya kar sakte ho", "kaise ho", "help", "guide me", "how are you"
}

# Explicit Out-of-Scope Patterns (Coding, Math, Politics, General Trivia)
EXPLICIT_OUT_OF_SCOPE_REGEX = [
    # 1. Coding, Programming & Tech Support
    r"\b(?:write|code|debug|compile|fix|implement|create|generate)\s+.*?\b(?:python|javascript|typescript|c\+\+|java|html|css|sql|react|vue|angular|node|docker|kubernetes|api|script|regex|function|class|algorithm|query|loop)\b",
    r"\b(?:python|javascript|typescript|c\+\+|java|golang|rust|php|swift|kotlin|ruby)\s+(?:code|script|syntax|error|tutorial|function|library|framework)\b",
    r"\b(?:leetcode|stackoverflow|git\s+push|git\s+commit|pip\s+install|npm\s+install)\b",
    r"\bhow\s+to\s+(?:code|program|build\s+an?\s+app|make\s+a\s+website|write\s+a\s+script)\b",
    r"\bwhat\s+is\s+(?:docker|kubernetes|recursion|object\s+oriented|polymorphism|async\s+await)\b",

    # 2. Math, Equations & Arithmetic Calculations
    r"\b(?:solve|calculate|evaluate|integrate|differentiate|compute)\s+.*?(?:equation|integral|derivative|algebra|matrix|pythagorean|\d+\s*[\+\-\*\/])\b",
    r"^\s*(?:what\s+is\s+)?\d+\s*[\+\-\*\/\^]\s*\d+\s*\??$",
    r"\b(?:square\s+root\s+of|derivative\s+of|integral\s+of|value\s+of\s+pi|quadratic\s+formula|fibonacci\s+series)\b",

    # 3. Politics, Government & Geopolitics
    r"\b(?:election|elections|presidential|president|prime\s+minister|parliament|senate|politics|political|democrat|republican|bjp|congress|putin|zelensky|biden|modi|trump)\b",
    r"\b(?:israel\s+palestine|russia\s+ukraine|united\s+nations|nato\s+summit|geopolitics)\b",

    # 4. General Trivia, Cooking Recipes, Pop Culture & School Science
    r"\b(?:recipe\s+for|how\s+to\s+cook|how\s+to\s+bake|ingredients\s+for|how\s+to\s+make\s+(?:cake|pasta|biryani|tea|coffee))\b",
    r"\b(?:who\s+won\s+the\s+(?:world\s+cup|ipl|super\s+bowl|oscar)|who\s+directed|lyrics\s+of)\b",
    r"\b(?:what\s+is\s+the\s+capital\s+of|who\s+was\s+napoleon|history\s+of\s+(?:rome|world\s+war))\b",
    r"\b(?:explain\s+photosynthesis|how\s+do\s+airplanes\s+fly|speed\s+of\s+light|quantum\s+physics|theory\s+of\s+relativity)\b",
    r"\b(?:write\s+an\s+essay|write\s+a\s+poem|tell\s+me\s+a\s+joke\s+about\s+(?:animals|dogs|cats))\b",
]

COMPILED_OUT_OF_SCOPE_REGEX = [re.compile(pat, re.IGNORECASE) for pat in EXPLICIT_OUT_OF_SCOPE_REGEX]

# Comprehensive In-Scope Life-Path and Astrology Tokens
IN_SCOPE_TOKENS = {
    # Astrology Concepts & Entities
    "astrology", "vedic", "horoscope", "kundali", "kundli", "chart", "charts", "birth",
    "natal", "lagna", "ascendant", "rashi", "moon", "sun", "planet", "planets", "graha",
    "grahas", "transit", "transits", "gochar", "dasha", "dashas", "mahadasha", "antardasha",
    "vimshottari", "house", "houses", "bhava", "bhavas", "yoga", "yogas", "dosha", "doshas",
    "manglik", "mangal", "sade", "sati", "dhaya", "kaal", "sarp", "rahu", "ketu", "shani",
    "saturn", "jupiter", "guru", "brihaspati", "venus", "shukra", "mercury", "budh", "mars",
    "surya", "chandra", "gemstone", "gemstones", "remedy", "remedies", "mantra", "mantras",
    "puja", "upay", "upaye", "totka", "totke", "nakshatra", "navamsha", "panna", "ruby",
    "pearl", "sapphire", "emerald", "coral", "diamond", "heera", "moti", "manik", "pukhraj",
    "neelam", "rudraksha",

    # Career & Work
    "career", "job", "profession", "professional", "work", "business", "promotion", "appraisal",
    "salary", "boss", "naukri", "nokri", "kaam", "vyapar", "dhandha", "employment", "office",
    "interview", "startup", "vocation", "transfer", "resignation", "growth", "tarakki", "safalta",

    # Marriage & Love
    "marriage", "married", "wedding", "spouse", "partner", "relationship", "relationships",
    "love", "divorce", "shaadi", "shadi", "vivah", "rishta", "pyaar", "prem", "pati", "patni",
    "compatibility", "milan", "matching", "breakup", "soulmate", "dulha", "dulhan", "remarriage",

    # Wealth & Finance
    "money", "wealth", "finance", "financial", "finances", "property", "investment", "rich",
    "income", "paisa", "paise", "dhan", "sampatti", "karz", "loan", "debt", "savings", "bachat",
    "loss", "profit", "assets", "ameer", "daulat", "kamao", "kamayi", "nivesh",

    # Health & Well-being
    "health", "disease", "illness", "body", "medical", "surgery", "fitness", "sehat", "swasthya",
    "bimari", "pain", "mental", "depression", "anxiety", "stress", "tanav", "chinta", "recovery",
    "longevity", "ayush", "swasth", "dard", "ilaj", "dawai",

    # Family & Children
    "child", "children", "baby", "pregnancy", "son", "daughter", "progeny", "bachcha", "baccha",
    "bacche", "santaan", "santan", "fertility", "family", "parivar", "beta", "beti", "parents",

    # Education & Exam
    "education", "study", "studies", "exam", "exams", "college", "degree", "learning", "padhai",
    "shiksha", "pariksha", "imtihan", "university", "school", "academic", "pass", "fail", "score",

    # Travel & Foreign Settlement
    "travel", "abroad", "foreign", "trip", "journey", "visa", "settle", "settlement", "yatra",
    "videsh", "pardesh", "relocation", "immigration", "pr", "relocate",

    # General Life Path & Destiny
    "life", "future", "destiny", "fate", "kismat", "bhagya", "karma", "moksha", "prediction",
    "predictions", "guidance", "muhurat", "shubh", "ashubh", "favorable", "unfavorable",
    "timing", "luck", "lucky"
}


def is_astrology_query(query: str) -> Tuple[bool, Optional[str]]:
    """
    Fast Intent Classifier & Guardrail Check (<1ms execution).
    Rule 1 (Domain Restriction):
      - If query is outside Vedic astrology, horoscope, transits, dashas, or life-path domains
        (coding, general trivia, politics, math, etc.), reject it immediately.
      - Returns (is_valid: bool, reason: Optional[str]).
    """
    if not query:
        return False, "empty_query"

    q_clean = query.strip()
    q_lower = q_clean.lower()

    # 1. Allow conversational greetings or assistance queries
    cleaned_greeting = re.sub(r"[^\w\s]", "", q_lower).strip()
    if (
        cleaned_greeting in CONVERSATIONAL_GREETINGS
        or any(cleaned_greeting.startswith(g) for g in ["hi ", "hello ", "hey ", "namaste ", "pranam "])
        or bool(re.search(r"\b(?:who\s+are\s+you|what\s+can\s+you\s+do|introduce\s+yourself|help\s+me)\b", q_lower))
    ):
        return True, None

    # 2. Check explicit out-of-scope patterns (Coding, Math, Politics, Trivia)
    # Exception: If user mentions chart/kundali/astrology/career/exam in context (e.g. "Will I pass python exam in my career?"),
    # we verify whether it's truly an astrology question about life path or a request to code.
    for pat in COMPILED_OUT_OF_SCOPE_REGEX:
        if pat.search(q_clean):
            # Check if this is a meta astrological inquiry about a career/exam field
            has_chart_context = any(
                k in q_lower for k in ("chart", "kundali", "kundli", "horoscope", "astrology", "rashi", "lagna", "dasha", "graha")
            )
            is_coding_request = bool(
                re.search(r"\b(write|generate|debug|fix|create|compile)\s+.*?\b(code|script|function|program|class|sql)\b", q_lower)
            )
            if is_coding_request or not has_chart_context:
                return False, "explicit_out_of_scope"

    # 3. Check for presence of in-scope astrology or life-path vocabulary
    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower)
    has_in_scope_token = any(w in IN_SCOPE_TOKENS for w in words)

    if has_in_scope_token:
        return True, None

    # Check multi-word phrase matches from DOMAIN_KEYWORDS
    for dom_keywords in DOMAIN_KEYWORDS.values():
        if any(kw in q_lower for kw in dom_keywords):
            return True, None

    # If query has no overlap with Vedic astrology or life-path domains, reject it
    return False, "unrelated_domain"


def validate_astrology_domain(query: str) -> Tuple[bool, str]:
    """
    Convenience validator returning (is_valid: bool, response_text: str).
    If is_valid is False, response_text is the standard OUT_OF_SCOPE_RESPONSE.
    """
    is_valid, _ = is_astrology_query(query)
    if is_valid:
        return True, ""
    return False, OUT_OF_SCOPE_RESPONSE


class IntentRouter:
    """
    Intent Classifier and Domain Router.
    Enforces domain validation guardrails before downstream chart and LLM processing.
    """

    def __init__(self, model_path: str = "astrology_predict_model.pkl"):
        self.model_path = model_path
        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print("[OK] ML Model loaded successfully.")
            except Exception as e:
                print(f"[WARNING] Error loading ML model: {e}")

    def validate_domain(self, query: str) -> Tuple[bool, str]:
        """Validates query against Vedic astrology and life-path domain guardrails."""
        return validate_astrology_domain(query)

    def classify_and_predict(self, query: str, astro_features: list) -> dict:
        q_lower = query.lower()
        predicted_domain = "general"
        for dom, keywords in DOMAIN_KEYWORDS.items():
            if any(k in q_lower for k in keywords):
                predicted_domain = dom
                break

        confidence = 0.8
        if self.model and astro_features:
            try:
                feats = np.array(astro_features[:7]).reshape(1, -1)
                ml_pred = self.model.predict(feats)[0]
                if hasattr(self.model, "predict_proba"):
                    confidence = float(np.max(self.model.predict_proba(feats)[0]))
                if predicted_domain == "general":
                    predicted_domain = f"chart_archetype_{ml_pred}"
            except Exception as e:
                print(f"[WARNING] ML Prediction error: {e}")

        return {"predicted_domain": predicted_domain, "confidence_score": confidence}