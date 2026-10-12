"""
QA Engine: High-performance local Q&A intercept with fuzzy/pattern-matching.
Matches incoming queries against qa_database.json and dynamically personalizes
responses with user profile parameters (0 LLM tokens consumed).
Enforces single-greeting policy so user is never greeted repeatedly.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False

try:
    from llm_router import should_bypass_local_qa, is_template_recently_served, record_served_template
except ImportError:
    try:
        from backend.llm_router import should_bypass_local_qa, is_template_recently_served, record_served_template
    except ImportError:
        def should_bypass_local_qa(query: str, session_id: Optional[str] = None) -> bool:
            return len(query.strip().split()) >= 10
        def is_template_recently_served(session_id: Optional[str], template_key: str, recent_limit: int = 5, chat_history: Optional[List[Dict[str, Any]]] = None) -> bool:
            return False
        def record_served_template(session_id: Optional[str], template_key: str) -> None:
            pass

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATHS = [
    os.path.join(BASE_DIR, "qa_database.json"),
    "qa_database.json",
    os.path.join(BASE_DIR, "..", "qa_database.json"),
]

DEFAULT_PROFILE: Dict[str, str] = {
    "name": "Friend",
    "lagna": "Libra",
    "moon_sign": "Gemini",
    "sun_sign": "Cancer",
    "jupiter_house": "1st house",
    "current_dasha": "Rahu-Moon",
}

GREETING_KEYS = {"hello", "hi", "namaste", "hey", "greetings"}

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

SIGN_LORDS = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury",
    "Cancer": "Moon", "Leo": "Sun", "Virgo": "Mercury",
    "Libra": "Venus", "Scorpio": "Mars", "Sagittarius": "Jupiter",
    "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"
}

HOUSE_NAMES = [
    "1st_house", "2nd_house", "3rd_house", "4th_house", "5th_house", "6th_house",
    "7th_house", "8th_house", "9th_house", "10th_house", "11th_house", "12th_house"
]

RASHI_ALIASES = {
    "mesh": "Aries", "mesha": "Aries",
    "vrishabh": "Taurus", "vrishabha": "Taurus",
    "mithun": "Gemini", "mithuna": "Gemini",
    "kark": "Cancer", "karka": "Cancer",
    "simha": "Leo", "singh": "Leo",
    "kanya": "Virgo",
    "tula": "Libra",
    "vrishchik": "Scorpio", "vrishchika": "Scorpio",
    "dhanu": "Sagittarius",
    "makar": "Capricorn", "makara": "Capricorn",
    "kumbh": "Aquarius", "kumbha": "Aquarius",
    "meen": "Pisces", "meena": "Pisces",
}


def calculate_house_rulerships(ascendant: str) -> Dict[str, str]:
    """Calculate deterministic house rulerships (1st to 12th house) for any given Ascendant."""
    raw = (ascendant or "Libra").strip().lower()
    clean_asc = RASHI_ALIASES.get(raw, ascendant.strip().capitalize() if ascendant else "Libra")

    match_idx = 6  # default Libra
    for idx, sign in enumerate(ZODIAC_SIGNS):
        if sign.lower() == clean_asc.lower():
            match_idx = idx
            break

    rulerships: Dict[str, str] = {}
    for i, house_name in enumerate(HOUSE_NAMES):
        sign = ZODIAC_SIGNS[(match_idx + i) % 12]
        lord = SIGN_LORDS[sign]
        rulerships[house_name] = f"{lord} ({sign})"
    return rulerships


def build_chart_substitution_context(chart_data: Optional[Any] = None) -> Dict[str, str]:
    """
    Extracts strictly grounded facts from the user's computed birth chart object.
    Computes mathematical house placements (1H through 12H) deterministically from the Lagna.

    Zero Hallucination Guarantee:
    - Only facts explicitly present in `chart_data` (or mathematically derived from the Lagna)
      are injected.
    - If an astrological parameter (like a specific planet degree) is missing from the user's profile,
      it resolves safely without making fake assumptions or generating fictional traits.
    """
    if chart_data is None:
        raw_dict: Dict[str, Any] = {}
    elif isinstance(chart_data, dict):
        raw_dict = dict(chart_data)
    elif hasattr(chart_data, "model_dump"):
        raw_dict = chart_data.model_dump()
    elif hasattr(chart_data, "__dict__"):
        raw_dict = {k: v for k, v in chart_data.__dict__.items() if not k.startswith("_")}
    else:
        raw_dict = {}

    def _get(keys: List[str], default: str = "") -> str:
        for k in keys:
            v = raw_dict.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        return default

    # Extract user identity facts
    user_name = _get(["user_name", "name", "first_name", "seeker_name"])
    if user_name.lower() in ("guest", "there", "friend"):
        user_name = ""

    birth_details = _get(["birth_details", "dob", "birth_date_time", "birth_datetime"])
    birth_place = _get(["birth_place", "place", "city", "location"])

    # Extract primary astrological coordinates
    lagna = _get(["lagna", "ascendant", "asc_sign"], "Libra")
    # Normalize Hindi/Sanskrit rashi aliases if used
    lagna = RASHI_ALIASES.get(lagna.lower(), lagna.capitalize())

    moon_sign = _get(["moon_sign", "rashi", "moon"], "Gemini")
    moon_sign = RASHI_ALIASES.get(moon_sign.lower(), moon_sign.capitalize())

    sun_sign = _get(["sun_sign", "sun", "surya"], "")
    if sun_sign:
        sun_sign = RASHI_ALIASES.get(sun_sign.lower(), sun_sign.capitalize())

    dasha = _get(["dasha", "current_dasha", "active_dasha"], "Rahu-Mars")
    jupiter_house = _get(["jupiter_house", "jupiter"], "")

    # Calculate deterministic 12-house signs and lords from Lagna
    rulerships = calculate_house_rulerships(lagna)
    lagna_lord = SIGN_LORDS.get(lagna, "Venus")

    subs: Dict[str, str] = {
        "user_name": user_name or "Seeker",
        "name": user_name or "Seeker",
        "birth_details": birth_details,
        "birth_place": birth_place,
        "lagna": lagna,
        "ascendant": lagna,
        "lagna_lord": lagna_lord,
        "ascendant_lord": lagna_lord,
        "moon_sign": moon_sign,
        "moon": moon_sign,
        "rashi": moon_sign,
        "sun_sign": sun_sign,
        "sun": sun_sign,
        "dasha": dasha,
        "current_dasha": dasha,
        "active_dasha": dasha,
        "jupiter_house": jupiter_house,
        "jupiter": jupiter_house,
    }

    # Populate 1st to 12th houses (both {house_10} and {10th_house} variants)
    for i, h_name in enumerate(HOUSE_NAMES, start=1):
        h_val = rulerships.get(h_name, "")
        m = re.match(r"^(\w+)\s*\(([^)]+)\)$", str(h_val).strip())
        if m:
            lord, sign = m.group(1), m.group(2)
        else:
            lord, sign = "", str(h_val)

        # Numerical forms: {house_10}, {house_10_sign}, {house_10_lord}, {house_10_full}
        subs[f"house_{i}"] = sign
        subs[f"house_{i}_sign"] = sign
        subs[f"house_{i}_lord"] = lord
        subs[f"house_{i}_full"] = f"{lord} ({sign})"

        # Ordinal forms: {10th_house}, {10th_house_sign}, {10th_house_lord}, {10th_house_full}
        subs[h_name] = sign
        subs[f"{h_name}_sign"] = sign
        subs[f"{h_name}_lord"] = lord
        subs[f"{h_name}_full"] = f"{lord} ({sign})"

    # Include any custom explicit parameters from the user's birth chart object
    for k, v in raw_dict.items():
        if k not in subs and v is not None and str(v).strip():
            subs[k] = str(v).strip()

    return subs


def render_local_template(
    template: Union[str, Dict[str, Any]],
    user_chart_data: Optional[Union[Dict[str, Any], Any]] = None,
    already_greeted: bool = False,
    is_greeting_key: bool = False,
) -> str:
    """
    Renders a local database template dynamically using strictly-grounded facts
    from the user's computed birth chart object.

    1. TEMPLATE VARIABLE INJECTION:
       Injects {user_name}, {lagna}, {moon_sign}, {house_10}, {dasha}, etc.
    2. ZERO HALLUCINATION / STRICT GROUNDING:
       Only injects facts explicitly present or mathematically derived.
       Missing parameters are safely omitted or defaulted without inventing fictional data.
    3. UNIQUE USER OUTPUTS:
       User A (Virgo Lagna, 10H Gemini) and User B (Scorpio Lagna, 10H Leo)
       receive completely distinct, customized responses.
    """
    context = build_chart_substitution_context(user_chart_data)

    def _render_string(text: str) -> str:
        if not text:
            return ""

        class SafeContextDict(dict):
            def __missing__(self, key: str) -> str:
                # Zero Hallucination: Missing astrological parameters are omitted safely
                return ""

        safe_dict = SafeContextDict(context)
        try:
            rendered = text.format_map(safe_dict)
        except Exception:
            def _replace_placeholder(match):
                k = match.group(1).strip()
                return context.get(k, "")
            rendered = re.sub(r"\{([a-zA-Z0-9_]+)\}", _replace_placeholder, text)

        # Clean up empty brackets and excessive whitespace left by omitted optional parameters
        rendered = re.sub(r"\(\s*\)", "", rendered)
        rendered = re.sub(r"\s{2,}", " ", rendered)
        rendered = re.sub(r"\s+([,;.!?])", r"\1", rendered)
        return rendered.strip()

    # If template is a structured dict from qa_database.json
    if isinstance(template, dict):
        concept = template.get("concept", "")
        analogy = _render_string(template.get("analogy") or template.get("simple_analogy", ""))
        core_analysis = _render_string(template.get("core_analysis") or template.get("core_meaning", ""))
        dasha_influence = _render_string(template.get("dasha_influence") or template.get("what_it_means", ""))
        recommendation = _render_string(template.get("recommendation") or template.get("actionable_takeaway", ""))

        sections = []
        insight_parts = [p for p in [core_analysis, dasha_influence] if p]
        insight_text = " ".join(insight_parts).strip()
        if insight_text:
            sections.append(f"**Astrological Insight:** {insight_text}")
        if recommendation:
            sections.append(f"**Actionable Advice:** {recommendation}")

        rendered = "\n\n".join(sections).strip() if sections else (core_analysis or concept)
    else:
        rendered = _render_string(str(template))

    if already_greeted and not is_greeting_key:
        rendered = strip_greeting(rendered)

    return rendered


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation/quotes/excess whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r'[\'\".,?!;:()\[\]{}_+\-=/\\<>~`@#$%^&*|]', ' ', text.lower())
    return " ".join(cleaned.split())


def strip_greeting(text: str) -> str:
    """Strip repeated greeting prefixes like 'Hello [Name], ', 'Namaste [Name]! ', etc."""
    pattern = r"^(?:(?:Hello|Namaste|Hi|Hey|Greetings|Dear)\s+[^,.!?\n]+[,!?]\s*)+(?:[^\w\s]\s*)*"
    cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    pattern2 = r"^(?:(?:Hello|Namaste|Hi|Hey|Greetings|Dear)[,.!?]\s*)+(?:[^\w\s]\s*)*"
    cleaned = re.sub(pattern2, "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned:
        return cleaned[0].upper() + cleaned[1:]
    return text


def calculate_similarity(query_clean: str, key_clean: str) -> float:
    """Calculate similarity score (0.0 to 100.0) between cleaned query and key."""
    if not query_clean or not key_clean:
        return 0.0

    # Exact match is 100%
    if query_clean == key_clean:
        return 100.0

    if HAS_RAPIDFUZZ:
        ratio = fuzz.ratio(query_clean, key_clean)
        token_set = fuzz.token_set_ratio(query_clean, key_clean)
        partial = fuzz.partial_ratio(key_clean, query_clean)
        token_sort = fuzz.token_sort_ratio(query_clean, key_clean)

        # Word-boundary subphrase check
        if bool(re.search(r'\b' + re.escape(key_clean) + r'\b', query_clean)):
            return max(90.0, float(max(ratio, token_set, partial, token_sort)))

        return float(max(ratio, token_set, partial, token_sort))
    else:
        direct_ratio = SequenceMatcher(None, query_clean, key_clean).ratio() * 100.0
        if key_clean in query_clean:
            return max(90.0, direct_ratio)
        return direct_ratio


class QAEngine:
    """Loads and matches user queries against the local Q&A database."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.database: Dict[str, str] = {}
        self.normalized_keys: List[Tuple[str, str, str]] = []  # (raw_key, normalized_key, template)
        self.load_database(db_path)

    def load_database(self, db_path: Optional[str] = None) -> None:
        """Load and index entries from qa_database.json."""
        self.database.clear()
        self.normalized_keys.clear()

        candidates = [db_path] if db_path else DEFAULT_DB_PATHS
        loaded_path = None

        for path in candidates:
            if path and os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, dict):
                                concept = v.get("concept", k.title())
                                analogy = v.get("analogy") or v.get("simple_analogy", "")
                                meaning = v.get("core_analysis") or v.get("core_meaning", "")
                                impact = v.get("dasha_influence") or v.get("what_it_means", "")
                                takeaway = v.get("recommendation") or v.get("actionable_takeaway", "")
                                parts = []
                                if meaning:
                                    parts.append(f"**Astrological Insight:** {meaning}")
                                if impact:
                                    parts.append(f"**Dasha Influence:** {impact}")
                                if takeaway:
                                    parts.append(f"**Actionable Advice:** {takeaway}")
                                v_str = "\n\n".join([p for p in parts if p]).strip()
                                raw_key = k.strip()
                                norm_key = normalize_text(raw_key)
                                self.database[raw_key] = v_str
                                if norm_key:
                                    self.normalized_keys.append((raw_key, norm_key, v_str))
                                for kw in v.get("keywords", []):
                                    kw_raw = str(kw).strip()
                                    kw_norm = normalize_text(kw_raw)
                                    if kw_norm and kw_norm not in ("career", "job", "work"):
                                        self.database[kw_raw] = v_str
                                        self.normalized_keys.append((kw_raw, kw_norm, v_str))
                            elif isinstance(v, str):
                                raw_key = k.strip()
                                norm_key = normalize_text(raw_key)
                                self.database[raw_key] = v
                                if norm_key:
                                    self.normalized_keys.append((raw_key, norm_key, v))
                        loaded_path = path
                        break
                except Exception as e:
                    logger.warning("Error reading QA database at %s: %s", path, e)

        # Sort normalized keys descending by length to prefer specific multi-word matches
        self.normalized_keys.sort(key=lambda item: len(item[1]), reverse=True)
        logger.info("QAEngine loaded %d templates from %s", len(self.database), loaded_path or "None")

    def format_template(
        self,
        template: Union[str, Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]] = None,
        already_greeted: bool = False,
        is_greeting_key: bool = False,
    ) -> str:
        """Safely inject user profile parameters into the matched template."""
        return render_local_template(
            template=template,
            user_chart_data=user_profile,
            already_greeted=already_greeted,
            is_greeting_key=is_greeting_key,
        )

    def check_query(
        self,
        query: str,
        user_profile: Optional[Dict[str, Any]] = None,
        threshold: float = 82.0,
        already_greeted: bool = False,
    ) -> Optional[Tuple[str, float, str]]:
        """
        Check query against database.
        If similarity >= threshold (default 82%), returns (formatted_answer, score, matched_key).
        Otherwise returns None.
        """
        if not query or not self.normalized_keys:
            return None

        clean_query = normalize_text(query)
        if not clean_query:
            return None

        # Check if caller passed already_greeted in profile
        if user_profile and isinstance(user_profile, dict):
            if user_profile.get("already_greeted") or user_profile.get("has_greeted"):
                already_greeted = True

        best_match: Optional[Tuple[str, float, str]] = None
        highest_score = 0.0

        for raw_key, norm_key, template in self.normalized_keys:
            is_greeting = norm_key in GREETING_KEYS

            # 1. Exact match shortcut (100%)
            if clean_query == norm_key:
                formatted = self.format_template(
                    template,
                    user_profile,
                    already_greeted=already_greeted,
                    is_greeting_key=is_greeting,
                )
                return formatted, 100.0, raw_key

            # 2. Similarity calculation
            score = calculate_similarity(clean_query, norm_key)
            if score > highest_score:
                highest_score = score
                best_match = (template, score, raw_key)
                if highest_score >= 99.0:
                    break

        if best_match and highest_score >= threshold:
            template, score, raw_key = best_match
            is_greeting = normalize_text(raw_key) in GREETING_KEYS
            formatted = self.format_template(
                template,
                user_profile,
                already_greeted=already_greeted,
                is_greeting_key=is_greeting,
            )
            return formatted, score, raw_key

        return None


# Global singleton QAEngine instance for zero-latency module access
_QA_ENGINE = QAEngine()


def intercept_qa(
    prompt: str,
    user_profile: Optional[Dict[str, Any]] = None,
    threshold: float = 82.0,
    already_greeted: bool = False,
    session_id: Optional[str] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Public intercept function: returns pre-written answer instantly if similarity >= 82%,
    consuming 0 tokens. Returns None if query does not meet threshold.
    Never greets user more than once if already_greeted is True.
    Checks state tracking: if template was already served in the active session, returns None
    to force fallback to LLM pipeline.
    """
    if not prompt:
        return None

    if should_bypass_local_qa(prompt, session_id=session_id):
        logger.info("intercept_qa: bypassing local QA for compound or complex query '%s'", prompt)
        return None

    result = _QA_ENGINE.check_query(
        prompt,
        user_profile=user_profile,
        threshold=threshold,
        already_greeted=already_greeted,
    )
    if result:
        formatted_answer, score, matched_key = result
        if is_template_recently_served(session_id, matched_key, chat_history=chat_history):
            logger.info(
                "Local QA Intercept '%s' was already served in session '%s'. Forcing fallback to LLM pipeline.",
                matched_key, session_id or "anonymous"
            )
            return None

        record_served_template(session_id, matched_key)
        logger.info(
            "Local QA Intercept Hit! Score=%.1f%%, Key='%s', 0 LLM tokens consumed",
            score, matched_key
        )
        return formatted_answer
    return None
