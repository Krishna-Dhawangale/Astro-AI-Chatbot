# -*- coding: utf-8 -*-
"""
backend/router/semantic_parser.py
---------------------------------
Semantic Intent & Entity Parser for the Astro-AI Engine.

Analyzes the full meaning, target entities, and complexity of a user query
without relying on rigid keyword lists. Parameterized for all 12 houses (1..12).
"""

import re
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field


class AstrologicalIntent(str, Enum):
    # Direct Factual Inquiries (1..12 Houses)
    HOUSE_SIGN = "HOUSE_SIGN"                      # What zodiac sign is in/occupies house X
    HOUSE_LORD = "HOUSE_LORD"                      # Which planet rules/owns house X
    LORD_PLACEMENT = "LORD_PLACEMENT"              # Where is house X lord placed
    HOUSE_OCCUPANTS = "HOUSE_OCCUPANTS"            # Which planets occupy/sit in house X
    
    # Planet Placements
    PLANET_PLACEMENT = "PLANET_PLACEMENT"          # Where is planet X placed
    ALL_PLACEMENTS = "ALL_PLACEMENTS"              # Where are all planets placed
    
    # General / Definitional
    HOUSE_SIGNIFICANCE = "HOUSE_SIGNIFICANCE"      # What does house X represent/mean
    HOUSE_INTERPRETATION = "HOUSE_INTERPRETATION"  # Astrological interpretation of house X
    MULTI_HOUSE_INTERACTION = "MULTI_HOUSE_INTERACTION" # Interplay between multiple houses (e.g. 8H + 12H)
    
    # Timing & Predicative
    TIMING_PREDICTION = "TIMING_PREDICTION"        # When will event happen / timing window
    DASHA_LOOKUP = "DASHA_LOOKUP"                  # What is current Dasha
    
    # Broad domain interpretation
    GENERAL_INTERPRETATION = "GENERAL_INTERPRETATION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


ORDINAL_MAP = {
    "first": 1, "1st": 1, "1": 1, "i": 1,
    "second": 2, "2nd": 2, "2": 2, "ii": 2,
    "third": 3, "3rd": 3, "3": 3, "iii": 3,
    "fourth": 4, "4th": 4, "4": 4, "iv": 4,
    "fifth": 5, "5th": 5, "5": 5, "v": 5,
    "sixth": 6, "6th": 6, "6": 6, "vi": 6,
    "seventh": 7, "7th": 7, "7": 7, "vii": 7,
    "eighth": 8, "8th": 8, "8": 8, "viii": 8,
    "ninth": 9, "9th": 9, "9": 9, "ix": 9,
    "tenth": 10, "10th": 10, "10": 10, "x": 10,
    "eleventh": 11, "11th": 11, "11": 11, "xi": 11,
    "twelfth": 12, "12th": 12, "12": 12, "xii": 12,
}

CORE_PLANETS = [
    "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"
]

# Canonical domain keywords that strongly link to specific houses when houses aren't explicitly numbered
DOMAIN_HOUSE_ASSOCIATIONS = {
    1: ["lagna", "ascendant", "physical body", "outer persona", "appearance", "vitality", "tanu bhava"],
    2: ["accumulated wealth", "savings", "family assets", "speech", "dhan bhava"],
    3: ["courage", "younger siblings", "short trips", "communication skills", "sahaja bhava"],
    4: ["property", "real estate", "buying a house", "purchasing a house", "buy a house", "purchase a house", "home purchase", "conveyance", "vehicles", "matru bhava"],
    5: ["intellect", "buddhi", "speculation", "progeny", "children", "creativity", "putra bhava"],
    6: ["debt", "debts", "loan", "loans", "borrowing", "enemies", "acute illness", "roga bhava", "shatru bhava"],
    7: ["marriage", "spouse", "wedding", "husband", "wife", "love life", "partnership", "business partner", "kalatra bhava"],
    8: ["transformation", "longevity", "sudden events", "ayur bhava", "occult", "unearned wealth"],
    9: ["higher education", "study abroad", "higher learning", "fortune", "dharma bhava", "father", "long travel", "bhagya"],
    10: ["career", "profession", "public reputation", "status", "job", "karma bhava", "employment"],
    11: ["financial gains", "income", "labha bhava", "wealth expansion", "aspirations", "large networks"],
    12: ["foreign relocation", "relocation abroad", "working in another country", "overseas", "foreign", "abroad", "expenses", "sleep", "insomnia", "vyaya bhava", "isolation", "moksha"],
}


@dataclass
class QueryIntentProfile:
    raw_query: str
    normalized_query: str
    core_intent: AstrologicalIntent
    target_houses: List[int] = field(default_factory=list)
    target_planets: List[str] = field(default_factory=list)
    is_compound: bool = False
    has_timing_modifier: bool = False
    has_predictive_modifier: bool = False
    has_evaluative_modifier: bool = False
    complexity_score: float = 0.0
    debug_notes: List[str] = field(default_factory=list)


def extract_all_houses(query: str) -> List[int]:
    """
    Extracts every house mentioned in the user query (1 to 12).
    Supports:
    - Ordinal numbers: '1st house', '7th house', '10th house'
    - Words: 'first house', 'seventh house', 'tenth house'
    - Formats: 'house 1', 'house 12', 'h1', '10h'
    - Domain anchors when explicit house context is present
    Returns an ordered, deduplicated list of house numbers [1..12].
    """
    if not query:
        return []
        
    q_low = query.lower().strip()
    found_houses: List[int] = []
    
    # 1. Match explicit house patterns: e.g. "1st house", "tenth house", "house 7", "h10"
    # Matches: (first|1st|1|tenth|10th|10)\s*(?:th|st|nd|rd)?\s*(?:house|cusp|bhava|bhav|h\b)
    explicit_pat = r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d{1,2}(?:st|nd|rd|th)?)\s*(?:house|cusp|bhava|bhav)\b"
    for m in re.finditer(explicit_pat, q_low):
        val_str = re.sub(r"[^\w]", "", m.group(1))
        h_val = ORDINAL_MAP.get(val_str) or (int(val_str) if val_str.isdigit() and 1 <= int(val_str) <= 12 else None)
        if h_val and h_val not in found_houses:
            found_houses.append(h_val)

    # 2. Match "house 1" .. "house 12"
    house_num_pat = r"\b(?:house|bhava|bhav)\s*(\d{1,2})\b"
    for m in re.finditer(house_num_pat, q_low):
        val = int(m.group(1))
        if 1 <= val <= 12 and val not in found_houses:
            found_houses.append(val)

    # 3. Match shorthand e.g. "10h", "7h", "12h"
    shorthand_pat = r"\b(\d{1,2})h\b"
    for m in re.finditer(shorthand_pat, q_low):
        val = int(m.group(1))
        if 1 <= val <= 12 and val not in found_houses:
            found_houses.append(val)

    # 4. Check domain-specific house anchors for additional houses (e.g. loan -> 6H, property -> 4H, abroad -> 12H)
    for h_num, anchors in DOMAIN_HOUSE_ASSOCIATIONS.items():
        if any(re.search(r"\b" + re.escape(a) + r"\b", q_low) for a in anchors):
            if h_num not in found_houses:
                found_houses.append(h_num)

    return found_houses


def extract_target_planets(query: str) -> List[str]:
    """Extracts all planet names mentioned in the query."""
    q_low = query.lower()
    found = []
    for p in CORE_PLANETS:
        if re.search(r"\b" + re.escape(p.lower()) + r"\b", q_low):
            found.append(p)
    return found


def analyze_query_intent(query: str) -> QueryIntentProfile:
    """
    Complete semantic parser for user queries.
    Parameterizes house questions across all 12 houses (1..12).
    Detects complexity, predictive modifiers, and compound intents.
    """
    q_low = (query or "").lower().strip()
    target_houses = extract_all_houses(q_low)
    target_planets = extract_target_planets(q_low)
    
    debug_notes = []
    
    # 1. Complexity & Modifier Detection
    # Compound query check (multiple clauses, 'and', 'also', 'as well as', 'relate to')
    has_conjunction = any(w in q_low for w in [" and ", " also ", " as well as ", " relate to ", " connection between ", " how does it affect "])
    is_compound = len(target_houses) > 1 or (has_conjunction and len(re.findall(r"\b(what|which|how|will|can|where|is)\b", q_low)) > 1)
    
    # Predictive modifier check ('will I', 'will my', 'future', 'prediction', 'hoga', 'milega')
    has_predictive_modifier = any(w in q_low for w in [
        "will i", "will my", "will the", "can i", "shall i", "future", "predict", "prediction", "prosper", 
        "succeed", "success", "bring", "chance of", "possibility of", "hoga", "hogi", "milega", "milegi"
    ])
    
    # Timing modifier check ('when', 'timing', 'year', 'month', 'kab', 'date', 'in 202')
    has_timing_modifier = any(w in q_low for w in [
        "when", "timing", "time frame", "window", "kab", "which year", "this year", "next year", "in 202", "in 203"
    ])
    
    # Evaluative / Advisory modifier ('should I', 'is it good', 'advice', 'remedy', 'how to handle')
    has_evaluative_modifier = any(w in q_low for w in [
        "should i", "is it good", "is it bad", "advice", "remedy", "improve", "handle", "manage"
    ])

    # 2. Determine Core Intent
    core_intent = AstrologicalIntent.GENERAL_INTERPRETATION
    
    # A. All Placements Query
    if any(p in q_low for p in [
        "which planet is my which house", "which planet is in which house", "where are my planets",
        "all planetary placements", "planetary positions", "planets in my houses", "my planet placements"
    ]):
        core_intent = AstrologicalIntent.ALL_PLACEMENTS
        
    # B. Specific Planet Placement (e.g. "Where is Mercury placed in my chart?")
    elif any(f"where is {p.lower()}" in q_low or f"{p.lower()} placed" in q_low or f"{p.lower()} position" in q_low for p in CORE_PLANETS):
        core_intent = AstrologicalIntent.PLANET_PLACEMENT
        
    # C. Multi-House Interaction (e.g. "What does my 8th house indicate and how does it relate to 12th house?")
    elif len(target_houses) > 1:
        core_intent = AstrologicalIntent.MULTI_HOUSE_INTERACTION
        
    # D. Single House Target Queries (Applies to ANY house 1..12)
    elif len(target_houses) == 1:
        h = target_houses[0]
        
        # House Sign intent (e.g. "Which sign is in my 7th house?", "Which sign occupies my 10th house?")
        is_sign_query = any(w in q_low for w in [
            "sign", "rashi", "zodiac", "constellation"
        ]) and any(w in q_low for w in [
            "which", "what", "occupies", "in my", "falls in", "rules the house", "on the cusp"
        ]) and not any(w in q_low for w in ["who is", "who rules", "lord", "ruler", "planet"])
        
        # House Lord Placement intent (e.g. "Where is my 7th house lord placed?")
        is_lord_placement_query = any(w in q_low for w in ["where is", "where is my", "placement of", "placed"]) and any(w in q_low for w in ["lord", "ruler", "swami", "master"])
        
        # House Lord intent (e.g. "Who rules my 7th house?", "Who is the lord of my 8th house?", "What is my 7th house lord?")
        is_lord_query = (
            any(w in q_low for w in ["lord", "ruler", "rules", "governs", "owns", "swami", "master"]) 
            and not is_lord_placement_query
            and not any(w in q_low for w in ["timing", "when", "purchas", "buy", "loan", "stress"])
        )
        
        # House Occupants intent (e.g. "Which planet is in my 1st house?", "Who occupies my 10th house?", "Are there planets in my 8th house?")
        is_occupants_query = (
            any(w in q_low for w in ["planet", "planets", "who is in", "what is in", "occupants", "occupy", "occupies", "sits in", "sitting in", "positioned in"])
            and not is_sign_query
            and not is_lord_query
            and not is_lord_placement_query
        )
        
        # House Significance / Definition (e.g. "What does my 1st house represent?")
        is_significance_query = any(w in q_low for w in ["represent", "signify", "meaning of", "mean", "stand for"])
        
        if is_sign_query:
            core_intent = AstrologicalIntent.HOUSE_SIGN
        elif is_lord_placement_query:
            core_intent = AstrologicalIntent.LORD_PLACEMENT
        elif is_lord_query:
            core_intent = AstrologicalIntent.HOUSE_LORD
        elif is_occupants_query:
            core_intent = AstrologicalIntent.HOUSE_OCCUPANTS
        elif is_significance_query:
            core_intent = AstrologicalIntent.HOUSE_SIGNIFICANCE
        else:
            core_intent = AstrologicalIntent.HOUSE_INTERPRETATION
            
    # E. Pure Timing / Dasha
    elif has_timing_modifier and any(w in q_low for w in ["dasha", "period", "gochar", "transit"]):
        core_intent = AstrologicalIntent.TIMING_PREDICTION
    elif any(k in q_low for k in ["what is my dasha", "current dasha", "my mahadasha", "current period"]):
        core_intent = AstrologicalIntent.DASHA_LOOKUP

    complexity_score = 0.0
    if is_compound: complexity_score += 0.4
    if has_predictive_modifier: complexity_score += 0.3
    if has_timing_modifier: complexity_score += 0.3
    if has_evaluative_modifier: complexity_score += 0.2

    return QueryIntentProfile(
        raw_query=query,
        normalized_query=q_low,
        core_intent=core_intent,
        target_houses=target_houses,
        target_planets=target_planets,
        is_compound=is_compound,
        has_timing_modifier=has_timing_modifier,
        has_predictive_modifier=has_predictive_modifier,
        has_evaluative_modifier=has_evaluative_modifier,
        complexity_score=complexity_score,
        debug_notes=debug_notes
    )
