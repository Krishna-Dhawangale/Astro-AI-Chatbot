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
from typing import Any, Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False

try:
    from llm_router import should_bypass_local_qa
except ImportError:
    try:
        from backend.llm_router import should_bypass_local_qa
    except ImportError:
        def should_bypass_local_qa(query: str) -> bool:
            return len(query.strip().split()) >= 10

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
                                analogy = v.get("simple_analogy", "")
                                meaning = v.get("core_meaning", "")
                                impact = v.get("what_it_means", "")
                                takeaway = v.get("actionable_takeaway", "")
                                parts = [f"{concept}:", analogy, meaning]
                                if impact:
                                    parts.append(f"What it means: {impact}")
                                if takeaway:
                                    parts.append(f"Actionable takeaway: {takeaway}")
                                v_str = " ".join([p for p in parts if p]).strip()
                                raw_key = k.strip()
                                norm_key = normalize_text(raw_key)
                                self.database[raw_key] = v_str
                                if norm_key:
                                    self.normalized_keys.append((raw_key, norm_key, v_str))
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
        template: str,
        user_profile: Optional[Dict[str, Any]] = None,
        already_greeted: bool = False,
        is_greeting_key: bool = False,
    ) -> str:
        """Safely inject user profile parameters into the matched template."""
        merged: Dict[str, Any] = {**DEFAULT_PROFILE}
        if user_profile and isinstance(user_profile, dict):
            for k, v in user_profile.items():
                if v is not None and str(v).strip():
                    merged[k] = str(v).strip()

        # Backward compatibility aliases
        if "ascendant" not in merged and "lagna" in merged:
            merged["ascendant"] = merged["lagna"]
        if "lagna" not in merged and "ascendant" in merged:
            merged["lagna"] = merged["ascendant"]
        if "moon_sign" not in merged and "moon" in merged:
            merged["moon_sign"] = merged["moon"]
        if "sun_sign" not in merged and "sun" in merged:
            merged["sun_sign"] = merged["sun"]
        if "current_dasha" not in merged and "dasha" in merged:
            merged["current_dasha"] = merged["dasha"]
        if "jupiter_house" not in merged and "jupiter" in merged:
            merged["jupiter_house"] = merged["jupiter"]

        try:
            formatted = template.format(**merged)
        except KeyError as exc:
            logger.warning("Missing template key %s in profile; applying safe fallback", exc)
            class SafeDict(dict):
                def __missing__(self, k: str) -> str:
                    return f"{{{k}}}"
            formatted = template.format_map(SafeDict(**merged))
        except Exception as exc:
            logger.warning("Error formatting template: %s", exc)
            formatted = template

        # Strip repeated greeting if user was already greeted and this is not a greeting query
        if already_greeted and not is_greeting_key:
            formatted = strip_greeting(formatted)

        return formatted

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

        # Strict Length & Intent Guard for Local QA (< 10 words, no personal intents)
        if should_bypass_local_qa(query):
            return None

        words = query.strip().split()
        if len(words) >= 10:
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
) -> Optional[str]:
    """
    Public intercept function: returns pre-written answer instantly if similarity >= 82%,
    consuming 0 tokens. Returns None if query does not meet threshold.
    Never greets user more than once if already_greeted is True.
    """
    if not prompt:
        return None

    # Strict Length & Intent Guard for Local QA (< 10 words, no personal intents)
    if should_bypass_local_qa(prompt):
        return None

    words = prompt.strip().split()
    if len(words) >= 10:
        return None

    result = _QA_ENGINE.check_query(
        prompt,
        user_profile=user_profile,
        threshold=threshold,
        already_greeted=already_greeted,
    )
    if result:
        formatted_answer, score, matched_key = result
        logger.info(
            "Local QA Intercept Hit! Score=%.1f%%, Key='%s', 0 LLM tokens consumed",
            score, matched_key
        )
        return formatted_answer
    return None
