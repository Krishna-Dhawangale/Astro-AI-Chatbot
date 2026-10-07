"""
Controlled Vocabulary Engine (backend/reasoning/controlled_vocab.py)
---------------------------------------------------------------------
Pillar 3 & Safeguard 2: Controlled Vocabulary is Strictly Evidence-Gated.

Converts structured evidence keys into natural English and Hinglish semantic components.
Components CANNOT be generated unless their corresponding evidence key exists in the
StructuredEvidence object (0 unsupported claims).
"""

from typing import Dict, Any, List, Optional
from backend.reasoning.structured_evidence import StructuredEvidence


CONTROLLED_VOCABULARY: Dict[str, Dict[str, Dict[str, str]]] = {
    # ------------------------------------------------------------------
    # CAREER DOMAIN SEMANTIC COMPONENTS
    # ------------------------------------------------------------------
    "career": {
        "strong_10th_lord": {
            "en": "your 10th-house lord is in a strong placement, highlighting natural strengths in analytical decision-making, professional management, and structured execution",
            "hinglish": "aapka 10th-house lord strong placement mein hai, jo analytical leadership aur structured execution ko support karta hai"
        },
        "moderate_10th_lord": {
            "en": "your 10th-house lord shows a balanced placement, indicating steady career progress through analytical decision-making, consistent skill consolidation, and adaptability",
            "hinglish": "aapka 10th-house lord balanced placement mein hai, jo steady effort aur skill development se growth support karta hai"
        },
        "10th_lord_requires_patience": {
            "en": "your 10th-house lord indicates that career growth develops best through patient perseverance, adaptability, and structured discipline over sudden changes",
            "hinglish": "aapka 10th-house lord suggest karta hai ki career growth patient perseverance aur structured discipline se aayegi"
        },
        "karaka_favorable": {
            "en": "career karaka planets favor steady professional advancement, skill-based growth, and organizational leadership",
            "hinglish": "career-related planets steady professional growth, organizational leadership, aur skill-based advancement ko strength dete hain"
        },
        "dasha_career_active": {
            "en": "your active Dasha period connects constructively with your career indicators, making this a supportive phase for professional development and expanding responsibilities",
            "hinglish": "aapka active Dasha period career indicators ko trigger kar raha hai, jo naye leadership opportunities aur growth ko encourage karta hai"
        },
        "saturn_transit_patience": {
            "en": "current transit influences suggest progress develops best through consistent effort, disciplined organization, and patience rather than sudden opportunities",
            "hinglish": "current transit influences suggest karte hain ki progress steady effort, disciplined routine, aur patience se aayegi"
        }
    },

    # ------------------------------------------------------------------
    # MARRIAGE & RELATIONSHIP SEMANTIC COMPONENTS
    # ------------------------------------------------------------------
    "marriage": {
        "strong_7th_lord": {
            "en": "your 7th-house lord is in a supportive placement, emphasizing mutual trust, clear communication, and emotional harmony",
            "hinglish": "aapka 7th-house lord supportive placement mein hai, jo mutual trust, clear communication, aur emotional harmony ko emphasize karta hai"
        },
        "moderate_7th_lord": {
            "en": "your 7th-house lord shows a balanced placement, favoring gradual relationship harmony and mutual understanding",
            "hinglish": "aapka 7th-house lord balanced placement show karta hai, jo mutual understanding se harmony badhata hai"
        },
        "7th_lord_requires_patience": {
            "en": "your 7th-house lord indicates that relationship stability develops best through patient communication and clear expectations",
            "hinglish": "aapka 7th-house lord relationship stability ke liye patient communication ko highlight karta hai"
        },
        "marriage_karaka_favorable": {
            "en": "relationship karaka planets favor building a supportive, enduring partnership foundation built on shared values",
            "hinglish": "relationship indicators long-term emotional stability aur strong partnership bond ko support karte hain"
        },
        "dasha_marriage_active": {
            "en": "your active Dasha period connects constructively with your relationship house factors, highlighting a supportive timing window",
            "hinglish": "aapka active Dasha period relationship house factors ko highlight karta hai, jo supportive timing window banata hai"
        },
        "relationship_patience": {
            "en": "long-term emotional stability is strengthened through patient communication and mutual understanding",
            "hinglish": "long-term relationship stability patient communication aur mutual understanding se strengthen hoti hai"
        }
    },


    # ------------------------------------------------------------------
    # WEALTH & FINANCE SEMANTIC COMPONENTS
    # ------------------------------------------------------------------
    "wealth": {
        "strong_2nd_11th_lord": {
            "en": "your 2nd and 11th house indicators favor steady wealth accumulation, disciplined budgeting, and prudent investments",
            "hinglish": "2nd aur 11th house factors steady financial accumulation, disciplined budgeting, aur prudent investments ko support karte hain"
        },
        "gradual_wealth_accumulation": {
            "en": "active planetary alignments support multiple income streams and gradual financial consolidation over time",
            "hinglish": "active planetary alignments multiple income streams aur gradual financial growth ko support karte hain"
        }
    },

    # ------------------------------------------------------------------
    # HEALTH SEMANTIC COMPONENTS
    # ------------------------------------------------------------------
    "health": {
        "ascendant_vitality_support": {
            "en": "your Ascendant and health indicators support physical resilience, natural recovery, and daily vitality",
            "hinglish": "Ascendant strength physical vitality, natural recovery, aur daily energy levels ko align rakhti hai"
        }
    },

    # ------------------------------------------------------------------
    # EDUCATION SEMANTIC COMPONENTS
    # ------------------------------------------------------------------
    "education": {
        "strong_4th_5th_house_intellect": {
            "en": "your 4th and 5th house placements highlight strong analytical intellect, deep comprehension, and focused study skills",
            "hinglish": "4th aur 5th house factors high conceptual learning, analytical intellect, aur focused study skills ko favor karte hain"
        }
    },

    # ------------------------------------------------------------------
    # PROPERTY SEMANTIC COMPONENTS
    # ------------------------------------------------------------------
    "property": {
        "property_house_support": {
            "en": "your 4th house and property indicators show constructive alignment for home stability and real estate matters",
            "hinglish": "4th house aur property indicators property matters aur home stability mein favorable alignment show karte hain"
        }
    },

    # ------------------------------------------------------------------
    # DOMAIN THEME VOCABULARY
    # ------------------------------------------------------------------
    "themes": {
        "analysis": {"en": "analytical problem-solving", "hinglish": "analytical reasoning"},
        "communication": {"en": "strategic communication", "hinglish": "communication skills"},
        "technology": {"en": "technology and systems management", "hinglish": "technical management"},
        "strategic_planning": {"en": "strategic decision-making", "hinglish": "strategic planning"},
        "governance": {"en": "executive governance and administration", "hinglish": "administrative management"},
        "systems_management": {"en": "systems organization and operations", "hinglish": "systems operations"},
        "emotional_harmony": {"en": "emotional harmony and trust", "hinglish": "emotional understanding"},
        "mutual_trust": {"en": "mutual trust and partnership", "hinglish": "mutual respect"},
        "long_term_commitment": {"en": "enduring commitment", "hinglish": "long-term bond"},
        "budgeting": {"en": "disciplined financial planning", "hinglish": "disciplined budgeting"},
        "steady_accumulation": {"en": "steady wealth accumulation", "hinglish": "steady financial growth"},
        "vitality": {"en": "physical resilience and vitality", "hinglish": "physical energy aur vitality"},
        "intellect": {"en": "deep conceptual intellect", "hinglish": "high learning intellect"}
    }
}


def get_gated_semantic_phrases(evidence: StructuredEvidence, is_hinglish: bool = False) -> Dict[str, List[str]]:
    """
    Safeguard 2: Returns ONLY semantic phrases whose underlying rule keys exist
    in evidence.interpretation or evidence.facts.
    
    Prevents the synthesizer from generating any sentence that is not backed by
    the evidence object.
    """
    lang = "hinglish" if is_hinglish else "en"
    domain = evidence.domain
    interp = evidence.interpretation

    positive_keys = interp.get("positive_factors", [])
    challenging_keys = interp.get("challenging_factors", [])
    theme_keys = interp.get("themes", [])

    matched_positives: List[str] = []
    matched_challenges: List[str] = []
    matched_themes: List[str] = []

    # Domain vocabulary lookup
    domain_vocab = CONTROLLED_VOCABULARY.get(domain, {})
    for key in positive_keys:
        if key in domain_vocab:
            matched_positives.append(domain_vocab[key][lang])

    for key in challenging_keys:
        if key in domain_vocab:
            matched_challenges.append(domain_vocab[key][lang])

    theme_vocab = CONTROLLED_VOCABULARY.get("themes", {})
    for t_key in theme_keys:
        if t_key in theme_vocab:
            matched_themes.append(theme_vocab[t_key][lang])

    return {
        "positive_phrases": matched_positives,
        "challenging_phrases": matched_challenges,
        "theme_phrases": matched_themes,
    }
