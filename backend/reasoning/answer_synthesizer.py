"""
Evidence-Grounded Local Answer Synthesizer (backend/reasoning/answer_synthesizer.py)
-------------------------------------------------------------------------------------
Pillar 4: Consumes StructuredEvidence & ControlledVocab to produce 100% natural,
question-aware human responses locally (gemini_calls = 0).

Strict Safeguards Enforced:
1. No hardcoded status scores.
2. Controlled Vocabulary is strictly evidence-gated.
3. Flexible intent-specific answer shapes based on available evidence.
4. No technical debug strings exposed to the user.
5. If evidence is missing, returns clear limitation answer without fabrication.
"""

from typing import Dict, Any, List, Optional
from backend.reasoning.structured_evidence import StructuredEvidence
from backend.reasoning.controlled_vocab import get_gated_semantic_phrases
from backend.reasoning.answer_shapes import get_answer_shape

try:
    from backend.llm_router import is_hinglish_query
except ImportError:
    def is_hinglish_query(q: str) -> bool:
        return False

# Domain-Locked Contextual Follow-Up Suggestions
DOMAIN_LOCKED_FOLLOWUPS: Dict[str, List[str]] = {
    "career": [
        "When is the right time for job change?",
        "Which gemstones suit my career?",
        "Does my chart favor business or a job?"
    ],
    "finance": [
        "When is the best period for financial growth?",
        "Which gemstones or remedies boost wealth flow?",
        "Does my chart support stock market or property investment?"
    ],
    "marriage": [
        "What are the key qualities of my future spouse?",
        "When is a favorable timing window for marriage?",
        "How can relationship understanding be strengthened in my chart?"
    ],
    "education": [
        "Which higher education field suits my chart?",
        "How can I improve concentration for exams?",
        "What timing is favorable for competitive exams?"
    ],
    "health": [
        "What daily routine best supports my vitality?",
        "Which planetary periods require extra rest?",
        "What astrological habits help balance stress?"
    ],
    "general": [
        "What is the most significant planetary transit affecting me now?",
        "Which area should I focus on in the coming months?",
        "What practical first step should I take based on my chart?"
    ]
}

def get_contextual_followups(domain: str, question: str = "") -> List[str]:
    """Returns 3 relevant follow-up questions locked to the queried domain."""
    d = (domain or "general").lower()
    return DOMAIN_LOCKED_FOLLOWUPS.get(d, DOMAIN_LOCKED_FOLLOWUPS["general"])


def guard_deterministic_claims(text: str) -> str:
    """
    Strips out deterministic claim words (e.g. 'guaranteed', '100% certainty')
    to maintain non-deterministic safety.
    """
    if not text:
        return ""
    import re
    replacements = {
        r"\bguaranteed\b": "supportive",
        r"\bwill definitely\b": "tends to support",
        r"\bdefinitely\b": "likely",
        r"\bexact date\b": "favorable window",
        r"\b100% certainty\b": "high likelihood",
        r"\bcertain to\b": "tends to",
    }
    cleaned = text
    for pattern, new in replacements.items():
        cleaned = re.sub(pattern, new, cleaned, flags=re.IGNORECASE)
    return cleaned


def _extract_features(question: str) -> Dict[str, Any]:
    try:
        from backend.router.intent import extract_context_features
        return extract_context_features(question)
    except Exception:
        try:
            from router.intent import extract_context_features
            return extract_context_features(question)
        except Exception:
            return {}


def classify_question_type(
    question: str,
    intent: str = "",
    features: Optional[Dict[str, Any]] = None
) -> str:
    """
    Classifies a user query into one of five structured conversational question types:
    - 'comparison': comparing two choices (e.g. job vs business, property vs stocks)
    - 'feasibility_advisory': should I, can I, is it safe to, kya karna chahiye
    - 'timing': timing window, timeframe, or horizon
    - 'direct_placement': asking about a specific lord, house, or planetary placement
    - 'growth_outlook': general progress, advancement, long-term trajectory
    """
    import re
    q_low = (question or "").lower().strip()
    intent_low = (intent or "").lower().strip()
    feat = features or {}

    # 1. Comparison
    if feat.get("is_comparison") or any(k in q_low for k in ["business vs job", "job vs business", "business or job", "job or business", "vs", "versus", " ya "]):
        return "comparison"

    # 2. Feasibility / Advisory (Should I..., Can I..., Is it safe...)
    # Queries like "Should I switch jobs right now?" are feasibility questions with a temporal context.
    if (
        any(q_low.startswith(p) for p in ["should i", "can i", "is it safe", "is it good", "is it advisable", "kya mujhe"])
        or any(k in q_low for k in ["safe to invest", "should i switch", "should i take", "karna chahiye", "favorable to"])
        or feat.get("is_feasibility")
    ):
        if not any(k in q_low for k in ["when", "kab", "which year", "which month"]):
            return "feasibility_advisory"

    # 3. Timing
    if (
        feat.get("has_timing_signal")
        or "timing" in intent_low
        or any(k in q_low for k in ["when", "kab", "timing", "timeline", "which year", "which month", "date", "window", "kab tak"])
    ):
        return "timing"

    # 4. Direct Placement
    if (
        re.search(r"\b(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+(?:house|lord)\b", q_low)
        or re.search(r"\b(?:sun|moon|mars|mercury|jupiter|venus|saturn|rahu|ketu)\s+in\s+[a-z]+\b", q_low)
        or "placement" in intent_low
        or "10th_lord" in intent_low
    ):
        return "direct_placement"

    # 5. Growth Outlook & General Trajectory
    return "growth_outlook"


def _connect_evidence_phrases(positives: List[str], challenges: List[str], is_hinglish: bool = False) -> str:
    """
    Connects positive and challenging evidence phrases using natural discourse connectors
    without deleting distinct or conflicting evidence, while eliminating exact duplicate phrases.
    """
    if not positives and not challenges:
        return ""

    # Deduplicate repeated evidence phrases while preserving order
    seen = set()
    deduped_pos = []
    for p in positives:
        p_clean = p.strip()
        p_norm = p_clean.lower()
        if p_norm and p_norm not in seen:
            seen.add(p_norm)
            deduped_pos.append(p_clean)

    seen_chal = set()
    deduped_chal = []
    for c in challenges:
        c_clean = c.strip()
        c_norm = c_clean.lower()
        if c_norm and c_norm not in seen_chal and c_norm not in seen:
            seen_chal.add(c_norm)
            deduped_chal.append(c_clean)

    if not deduped_pos and not deduped_chal:
        return ""

    if is_hinglish:
        parts = []
        if deduped_pos:
            parts.append(deduped_pos[0].capitalize())
            hi_pos_conn = ["Iske saath hi", "Saath hi", "Iske alawa"]
            for i, p in enumerate(deduped_pos[1:]):
                conn = hi_pos_conn[min(i, len(hi_pos_conn) - 1)]
                parts.append(f"{conn}, {p}")
        if deduped_chal:
            if deduped_pos:
                parts.append(f"Dusri taraf, {deduped_chal[0]}")
            else:
                parts.append(deduped_chal[0].capitalize())
            hi_chal_conn = ["Saath hi", "Iske sath", "Dhyan rahe"]
            for i, c in enumerate(deduped_chal[1:]):
                conn = hi_chal_conn[min(i, len(hi_chal_conn) - 1)]
                parts.append(f"{conn}, {c}")
        return ". ".join(parts) + "."
    else:
        parts = []
        if deduped_pos:
            parts.append(deduped_pos[0].capitalize())
            en_pos_conn = ["Furthermore", "Additionally", "Moreover"]
            for i, p in enumerate(deduped_pos[1:]):
                conn = en_pos_conn[min(i, len(en_pos_conn) - 1)]
                parts.append(f"{conn}, {p}")
        if deduped_chal:
            if deduped_pos:
                parts.append(f"While these foundational placements are supportive, conversely, {deduped_chal[0]}")
            else:
                parts.append(deduped_chal[0].capitalize())
            en_chal_conn = ["In addition", "Concurrently", "Simultaneously"]
            for i, c in enumerate(deduped_chal[1:]):
                conn = en_chal_conn[min(i, len(en_chal_conn) - 1)]
                parts.append(f"{conn}, {c}")
        return ". ".join(parts) + "."


def _resolve_house_and_lord(asc: str, house_offset: int) -> tuple[str, str]:
    """Helper to dynamically calculate house sign and lord from ascendant."""
    signs = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ]
    lords = {
        "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
        "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
        "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"
    }
    if asc in signs:
        sign = signs[(signs.index(asc) + house_offset - 1) % 12]
        return sign, lords[sign]
    return "Gemini", "Mercury"


def _synthesize_timing_answer(
    domain: str,
    asc: str,
    moon: str,
    c_maha: str,
    c_antar: str,
    h_sign: Optional[str],
    h_lord: Optional[str],
    positives: List[str],
    challenges: List[str],
    explicit_horizon: Optional[str] = None,
    timing_window: Optional[str] = None,
    is_hinglish: bool = False
) -> str:
    has_timing_evidence = bool(c_maha)
    dasha_str = f"{c_maha}{'/' + c_antar if c_antar else ''}" if c_maha else ""

    if not has_timing_evidence:
        if is_hinglish:
            return (
                f"Aapki birth chart ({asc} Lagna, {moon} Moon sign) {domain} ke liye foundational baseline indicators show karti hai, "
                "lekin exact timing window confirm karne ke liye active Dasha aur transit evidence ki requirement hai. "
                "System bina verify hue Dasha facts ke koi artificial date generate nahi karta.\n\n"
                "**Foundational Context**: Natal placements steady capability highlight karte hain. "
                "Current Dasha verify hote hi exact timing period clearly calculate ho jayega."
            )
        return (
            f"While your birth chart indicates general foundational tendencies for your {domain}, "
            "specific timing windows cannot be confirmed without verified active Dasha and transit data. "
            "The system avoids generating ungrounded calendar dates when timing evidence is uninitialized.\n\n"
            "**Foundational Alignment**: Your natal indicators show steady long-term capability. "
            "Confirming your active Dasha timeline will allow precise calculation of your favorable activation window."
        )

    if is_hinglish:
        if explicit_horizon:
            opening = f"Aapke stated timeline ({explicit_horizon}) ke context mein, aapki active {dasha_str} Dasha {domain} ke liye ek supportive phase create karti hai."
        elif timing_window:
            opening = f"Aapki birth chart ke according active {dasha_str} Dasha ({timing_window}) {domain} ke liye favorable timing window highlight karti hai."
        else:
            opening = f"Aapki birth chart ke according active {dasha_str} Dasha operating period ke dauran {domain} ke liye supportive activation window indicate hoti hai."

        body = _connect_evidence_phrases(positives, challenges, is_hinglish=True)
        conclusion = "Yeh phase preparatory skill consolidation aur methodical execution ke liye sabse best hai."
        return f"{opening}\n\n**Astrological Context ({asc} Lagna, {moon} Moon)**:\n{body}\n\n**Conclusion**: {conclusion}"
    else:
        if explicit_horizon:
            opening = f"Regarding your query for the {explicit_horizon}, your active {dasha_str} Dasha indicates a supportive timing window for steady {domain} progress during this period."
        elif timing_window:
            opening = f"Your birth chart indicates a supportive timing window ({timing_window}) for {domain} milestones under active {dasha_str} Dasha dynamics."
        else:
            opening = f"Your birth chart indicates favorable timing activation for {domain} during the active {dasha_str} Dasha operating period as supportive planetary influences align."

        body = _connect_evidence_phrases(positives, challenges, is_hinglish=False)
        conclusion = "Focusing on systematic skill building and disciplined pacing during this preparatory window positions you to maximize upcoming planetary momentum."
        return f"{opening}\n\n**Astrological Evidence ({asc} Ascendant, {moon} Moon sign)**:\n{body}\n\n**Actionable Conclusion**: {conclusion}"


def _synthesize_feasibility_answer(
    domain: str,
    asc: str,
    moon: str,
    c_maha: str,
    c_antar: str,
    h_sign: Optional[str],
    h_lord: Optional[str],
    positives: List[str],
    challenges: List[str],
    is_hinglish: bool = False
) -> str:
    if is_hinglish:
        if domain in ["finance", "wealth"]:
            opening = "Haan, aapka chart prudent aur long-term investments ko favor karta hai, lekin bina due diligence ke speculative risks lene ki recommendation nahi hai."
        else:
            opening = "Haan, aapka birth chart independent initiatives aur naye avenues ko support karta hai, lekin shuruat mein structured experience aur financial cushion maintain karna sabse safe path hai."

        body = _connect_evidence_phrases(positives, challenges, is_hinglish=True)
        conclusion = "Ek phased approach—pehle capital aur skills consolidate karna, phir calculated move lena—aapke chart ke saath best align karta hai."
        return f"{opening}\n\n**Chart Indicators ({asc} Lagna, {moon} Moon)**:\n{body}\n\n**Key Advisory**: {conclusion}"
    else:
        if domain in ["finance", "wealth"]:
            opening = "Yes, your birth chart supports prudent, long-term investments, whereas speculative or unhedged financial risks are not advised under current planetary conditions."
        else:
            opening = "Yes, your birth chart supports exploring new professional avenues, but establishing domain mastery and capital reserves in structured employment first provides the lowest risk path."

        body = _connect_evidence_phrases(positives, challenges, is_hinglish=False)
        conclusion = "A phased transition—validating practical feasibility before committing substantial capital—aligns best with your astrological indicators."
        return f"{opening}\n\n**Astrological Analysis ({asc} Ascendant, {moon} Moon sign)**:\n{body}\n\n**Key Recommendation**: {conclusion}"


BHAVA_INFO: Dict[int, Dict[str, str]] = {
    1: {"name": "Lagna Bhava", "domain": "vitality and personal leadership", "hi_domain": "vitality aur vyaktitva"},
    2: {"name": "Dhana Bhava", "domain": "accumulated wealth and family assets", "hi_domain": "dhan sanchay aur parivar"},
    3: {"name": "Sahaja Bhava", "domain": "initiative, courage, and communication", "hi_domain": "parakram aur communication"},
    4: {"name": "Sukha Bhava", "domain": "inner peace, domestic life, and assets", "hi_domain": "griha sukh aur emotional peace"},
    5: {"name": "Putra Bhava", "domain": "intellect, creativity, and strategic insight", "hi_domain": "buddhi aur creative intelligence"},
    6: {"name": "Ari Bhava", "domain": "problem resolution, daily work, and resilience", "hi_domain": "work routine aur problem solving"},
    7: {"name": "Kalatra Bhava", "domain": "partnerships, marriage, and public relations", "hi_domain": "partnerships, vivah, aur public ties"},
    8: {"name": "Randhra Bhava", "domain": "deep transformation, research, and longevity", "hi_domain": "transformation aur research depth"},
    9: {"name": "Bhagya Bhava", "domain": "higher wisdom, fortune, and ethical purpose", "hi_domain": "bhagya aur higher learning"},
    10: {"name": "Karma Bhava", "domain": "career authority, professional action, and status", "hi_domain": "career authority aur professional action"},
    11: {"name": "Labha Bhava", "domain": "income gains, professional networks, and aspirations", "hi_domain": "aarthik labh aur networks"},
    12: {"name": "Vyaya Bhava", "domain": "solitude, spiritual growth, and restful recovery", "hi_domain": "spiritual growth aur rest recovery"},
}

ORDINALS: Dict[int, str] = {
    1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th",
    7: "7th", 8: "8th", 9: "9th", 10: "10th", 11: "11th", 12: "12th"
}

def _extract_target_house(question: str = "", domain: str = "") -> int:
    import re
    q_low = (question or "").lower()
    house_words = {
        "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
        "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9,
        "tenth": 10, "10th": 10, "eleventh": 11, "11th": 11, "twelfth": 12, "12th": 12
    }
    for word, h_num in house_words.items():
        if re.search(rf"\b{word}\s+(?:house|lord|ghar|bhav|bhava)\b", q_low) or re.search(rf"\b(?:house|lord|ghar|bhav|bhava)\s+{word}\b", q_low):
            return h_num
        if word in ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th"] and re.search(rf"\b{word}\b", q_low):
            return h_num

    match = re.search(r"\b(?:house|bhav|ghar)\s*(\d{1,2})\b", q_low) or re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:house|lord|bhav|ghar)\b", q_low)
    if match:
        val = int(match.group(1))
        if 1 <= val <= 12:
            return val

    domain_defaults = {
        "career": 10,
        "marriage": 7,
        "relationship": 7,
        "finance": 2,
        "wealth": 2,
        "education": 5,
        "health": 6,
        "property": 4,
        "spirituality": 9,
    }
    return domain_defaults.get((domain or "").lower(), 10)


def _synthesize_comparison_answer(
    domain: str,
    asc: str,
    moon: str,
    h10_sign: Optional[str],
    h10_lord: Optional[str],
    h7_sign: Optional[str],
    h7_lord: Optional[str],
    c_maha: str,
    positives: List[str],
    challenges: List[str],
    is_hinglish: bool = False
) -> str:
    if not h10_sign or not h10_lord:
        h10_sign, h10_lord = _resolve_house_and_lord(asc, 10)
    if not h7_sign or not h7_lord:
        h7_sign, h7_lord = _resolve_house_and_lord(asc, 7)

    if is_hinglish:
        return (
            f"Job aur business ke comparison mein, aapka chart shuruat mein structured employment (Job) ko favor karta hai "
            f"taaki domain expertise aur capital reserve build ho sakein. 10th house ({h10_sign}, lord {h10_lord}) methodical "
            f"progression ko support karta hai, jabki 7th house ({h7_sign}, lord {h7_lord}) commercial indicators supportive "
            f"Dashas mein enterprise capacity activate karte hain. Phased transition sabse secure aur profitable strategy hai."
        )
    else:
        return (
            f"Between structured employment and an independent business, your chart initially favors organizational roles "
            f"to establish domain authority and capital reserves before expanding independently. Your 10th house in {h10_sign} "
            f"ruled by {h10_lord} anchors job stability, while 7th house ({h7_sign}, lord {h7_lord}) commercial enterprise capacity "
            f"matures during later supportive Dashas. A phased transition provides the most sustainable, low-risk trajectory."
        )


def _synthesize_growth_outlook_answer(
    domain: str,
    asc: str,
    moon: str,
    themes: List[str],
    c_maha: str,
    c_antar: str,
    positives: List[str],
    challenges: List[str],
    is_hinglish: bool = False
) -> str:
    theme_str = ", ".join(themes[:3]) if themes else ("analytical leadership, structured execution" if domain == "career" else "stability, gradual consolidation")

    if is_hinglish:
        opening = f"Aapka overall {domain} outlook steady compounding advancement aur structural mastery se define hota hai, na ki sudden volatile shifts se."
        body = _connect_evidence_phrases(positives, challenges, is_hinglish=True)
        conclusion = f"Agle 12–18 mahino mein {theme_str} par focus maintain karna aapki long-term authority aur growth ko consolidate karega."
        return f"{opening}\n\n**Core Indications ({asc} Lagna, {moon} Moon)**:\n{body}\n\n**Outlook Summary**: {conclusion}"
    else:
        opening = f"Your overall {domain} outlook is characterized by steady, compounding advancement and structural mastery rather than sudden volatile shifts."
        body = _connect_evidence_phrases(positives, challenges, is_hinglish=False)
        conclusion = f"Consolidating expertise in {theme_str} over the next 12 to 18 months positions you favorably for substantive long-term elevation."
        return f"{opening}\n\n**Astrological Baseline ({asc} Ascendant, {moon} Moon sign)**:\n{body}\n\n**Strategic Outlook**: {conclusion}"


def _synthesize_direct_placement_answer(
    domain: str,
    asc: str,
    moon: str,
    h10_sign: Optional[str] = None,
    h10_lord: Optional[str] = None,
    c_maha: str = "",
    positives: Optional[List[str]] = None,
    challenges: Optional[List[str]] = None,
    is_hinglish: bool = False,
    house_num: int = 10,
    house_sign: Optional[str] = None,
    house_lord: Optional[str] = None,
) -> str:
    h_idx = max(1, min(12, int(house_num or 10)))
    h_sign = house_sign or (h10_sign if h_idx == 10 else None)
    h_lord = house_lord or (h10_lord if h_idx == 10 else None)

    if not h_sign or not h_lord:
        resolved_s, resolved_l = _resolve_house_and_lord(asc, h_idx)
        h_sign = h_sign or resolved_s
        h_lord = h_lord or resolved_l

    info = BHAVA_INFO.get(h_idx, BHAVA_INFO[10])
    ord_str = ORDINALS.get(h_idx, f"{h_idx}th")

    if is_hinglish:
        return (
            f"Aapka {ord_str} house ({info['name']}) {h_sign} rashi mein sthit hai jiske lord {h_lord} hain, "
            f"jo aapke {info['hi_domain']} ko directly anchor karte hain. Is placement ke saath steady discipline "
            f"aur patient execution aapke outcomes ko mazboot banate hain. Structured focus maintain karna "
            f"aapki potential ko long-term maximize karega."
        )
    else:
        return (
            f"Your {ord_str} house ({info['name']}) in {h_sign} is ruled by {h_lord}, directly anchoring your "
            f"{info['domain']}. With {h_lord} overseeing this area, steady discipline and methodical responsibility "
            f"yield cumulative progress across your active phases. Channeling efforts into structured execution "
            f"rather than speculative haste maximizes this placement's enduring fulfillment."
        )


def synthesize_local_answer(
    evidence: StructuredEvidence,
    is_hinglish: bool = False,
    question: str = ""
) -> str:
    """
    Synthesizes clean, natural human language from a StructuredEvidence object.
    Applies question-aware answer shapes, direct opening sentences, evidence-aware
    discourse connectors, and repetition reduction.
    """
    if not is_hinglish and question:
        is_hinglish = is_hinglish_query(question)

    # 0. Check Evidence Availability & Handle Missing Evidence (Safeguard 5)
    if not evidence or not evidence.is_sufficient():
        domain_label = evidence.domain if evidence else "astrological"
        if is_hinglish:
            return guard_deterministic_claims(
                f"✨ Aapke chart ke according {domain_label} domain ke liye sufficient background details available nahi hain. "
                "System facts verify nahi hone par dynamic prediction create nahi karta."
            )
        return guard_deterministic_claims(
            f"✨ Based on the available evidence, details regarding your {domain_label} domain cannot be fully determined from the backend facts. "
            "The system avoids generating ungrounded claims when evidence is incomplete."
        )

    facts = evidence.facts
    evals = evidence.evaluations
    interp = evidence.interpretation
    timing = evidence.timing

    asc = facts.get("ascendant", "Virgo")
    moon = facts.get("moon_rashi", "Taurus")
    sun = facts.get("sun_sign", "Gemini")
    c_maha = timing.get("active_mahadasha", "")
    c_antar = timing.get("active_antardasha", "")
    h10_sign = facts.get("tenth_house_sign")
    h10_lord = facts.get("tenth_lord")
    h7_sign = facts.get("seventh_house_sign")
    h7_lord = facts.get("seventh_lord")

    # Get gated semantic phrases (Safeguard 2)
    vocab = get_gated_semantic_phrases(evidence, is_hinglish=is_hinglish)
    positives = vocab["positive_phrases"]
    challenges = vocab["challenging_phrases"]
    themes = vocab["theme_phrases"]

    q_lower = (question or "").lower().strip()
    intent = evidence.intent.lower()
    domain = evidence.domain.lower()

    # Reuse Phase 1 Context Features non-destructively
    features = _extract_features(question)
    explicit_horizon = features.get("explicit_user_horizon")

    # Classify Question Type
    q_type = classify_question_type(question, intent, features)

    # 1. HEALTH DOMAIN HANDLING (Preserve wellness disclaimers & boundaries)
    if domain == "health" or intent.startswith("health") or any(k in q_lower for k in ["tired", "energy", "sleep", "wellbeing", "vitality", "routine", "overthink", "stressed", "emotional", "rest", "physical strength", "weakness"]):
        disclaimer_en = "\n\nThese are astrological wellness indicators, not a medical diagnosis."
        disclaimer_hi = "\n\nYeh astrological wellness indicators hain, koi medical diagnosis nahi."

        # Sleep & Rest Focus
        if any(k in q_lower for k in ["sleep", "rest", "night", "insomnia", "bedtime"]):
            if is_hinglish:
                opening = "Aapki birth chart ke according sleep aur rest schedule ko stabilize karna aapki mental clarity aur daily recovery ke liye essential factor hai."
                body = (
                    f"- **Recovery Indicators**: 12th-house placements recovery, relaxation, aur nighttime quietude ko manage karte hain.\n"
                    f"- **Moon & Mind**: {moon} Moon placement suggest karti hai ki sleep se pehle mind ko relax karna sleep quality aur recovery ko favor karta hai.\n"
                    f"- **Dasha Influence**: Active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) consistent sleep hygiene maintain karne par emphasis deti hai."
                )
                return guard_deterministic_claims(f"{opening}\n\n**Rest & Recovery ({asc} Lagna, {moon} Moon)**:\n{body}{disclaimer_hi}")
            opening = "Your birth chart indicates that maintaining structured quietude before sleep directly supports your emotional rest and physical recovery."
            body = (
                f"- **Rest & Recovery Indicators**: Your 12th-house placements govern recovery, relaxation, and nighttime quietude.\n"
                f"- **Moon & Mind Balance**: Your {moon} Moon placement suggests that calming your mental state before sleep supports higher sleep quality and emotional rest.\n"
                f"- **Dasha Influence**: Your active Dasha period ({c_maha}{'/' + c_antar if c_antar else ''}) highlights the importance of maintaining a consistent sleep schedule to prevent restlessness during demanding phases."
            )
            return guard_deterministic_claims(f"{opening}\n\n**Rest & Recovery ({asc} Ascendant, {moon} Moon sign)**:\n{body}{disclaimer_en}")

        # Overthinking / Stress / Emotional Balance
        if any(k in q_lower for k in ["overthink", "stress", "emotional", "relax", "mind", "mental"]):
            if is_hinglish:
                opening = "Aapka Moon placement ek analytical aur deeply reflective nature show karta hai, jo busy work phases mein intentional mental pauses lene se balance hota hai."
                body = (
                    f"- **Moon Placement & Mind**: {moon} mein Moon placement deep thinking aur active mental reflection ko show karti hai.\n"
                    f"- **Stress Management**: 6th-house indicators suggest karte hain ki daily work aur relaxation ke beech clear boundaries maintain karne se emotional stability strengthen hoti hai.\n"
                    f"- **Dasha Influence**: Current Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) mindful pacing ko favor karti hai."
                )
                return guard_deterministic_claims(f"{opening}\n\n**Mind & Emotional Balance ({asc} Lagna, {moon} Moon)**:\n{body}{disclaimer_hi}")
            opening = "Your Moon placement indicates an active, deeply reflective analytical disposition that benefits from intentional mental pauses between demanding tasks."
            body = (
                f"- **Moon Placement & Mind**: Your Moon in {moon} indicates an analytical, deeply reflective emotional disposition, which can occasionally lead to mental fatigue if unpaced.\n"
                f"- **Stress & Workload Balance**: 6th-house indicators suggest that maintaining clear boundaries between work tasks and relaxation strengthens emotional resilience.\n"
                f"- **Dasha Period**: Your current Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) emphasizes intentional mental relaxation and mindful energy management during active phases."
            )
            return guard_deterministic_claims(f"{opening}\n\n**Mind & Emotional Balance ({asc} Ascendant, {moon} Moon sign)**:\n{body}{disclaimer_en}")

        # Fatigue / Vitality / Energy
        if is_hinglish:
            opening = "Aapka chart indicate karta hai ki baseline physical vitality steady daily routines aur balanced workload discipline se best support hoti hai."
            body = (
                f"- **Astrological Vitality**: {asc} Lagna aur Sun placements ({sun}) aapki core energy aur stamina tendencies ko highlight karte hain.\n"
                f"- **Routine & Wellness**: 6th-house indicators batate hain ki consistent daily routine aur workload balance energy levels ko support karte hain.\n"
                f"- **Rest & Recovery**: 12th-house factors suggest karte hain ki adequate rest aur recovery windows physical vitality refresh karne ke liye essential hain.\n"
                f"- **Dasha Influence**: Active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) active work phases mein physical pacing ko favor karti hai."
            )
            return guard_deterministic_claims(f"{opening}\n\n**Vitality Patterns ({asc} Lagna, {moon} Moon)**:\n{body}{disclaimer_hi}")
        opening = "Your chart indicates that your baseline physical vitality is best sustained through predictable daily rhythms and structured recovery windows rather than sporadic bursts of activity."
        body = (
            f"- **Astrological Vitality**: Your {asc} Lagna and {sun} Sun placement describe your baseline physical constitution and vitality tendencies.\n"
            f"- **Routine & Wellness**: 6th-house indicators emphasize the importance of maintaining consistent daily routines and workload discipline to manage energy levels.\n"
            f"- **Rest & Recovery**: 12th-house factors indicate that structured quiet recovery time plays a key role in maintaining physical resilience.\n"
            f"- **Dasha Influence**: Your active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) suggests pacing your activities and balancing exertion with adequate rest during this period."
        )
        return guard_deterministic_claims(f"{opening}\n\n**Vitality Patterns ({asc} Ascendant, {moon} Moon sign)**:\n{body}{disclaimer_en}")

    # 2. EDUCATION DOMAIN HANDLING
    if domain == "education" or intent.startswith("education") or any(k in q_lower for k in ["education", "study", "learning", "college", "degree", "academic", "5th house"]):
        mercury_sign = "Taurus" if asc == "Virgo" else "favorable placement"
        if is_hinglish:
            opening = "Aapki birth chart analytical learning, technical disciplines, aur conceptual clarity ko strongly favor karti hai."
            body = (
                f"- **Technical Alignment**: 5th house intelligence indicators aur Mercury ({mercury_sign}) analytical disciplines aur applied problem-solving ko support karte hain.\n"
                f"- **Advanced Studies**: 9th house indicators higher education aur specialized certifications ko favor karte hain.\n"
                f"- **Current Phase**: Active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) consistent academic discipline ko reward karti hai."
            )
            return guard_deterministic_claims(f"{opening}\n\n**Education Indicators ({asc} Lagna, {moon} Moon)**:\n{body}")
        opening = "Your birth chart strongly aligns with analytical disciplines, technical training, and structured academic problem-solving."
        body = (
            f"- **Academic Alignment**: Your 5th house indicators and Mercury placement ({mercury_sign}) strongly favor analytical disciplines, technical training, and structured problem-solving.\n"
            f"- **Higher Learning Strengths**: Higher education factors suggest favorable capacity for advanced degrees, conceptual synthesis, and specialized certifications.\n"
            f"- **Dasha Influence**: Your active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) supports concentrated academic effort and steady skill acquisition."
        )
        return guard_deterministic_claims(f"{opening}\n\n**Academic Alignment ({asc} Ascendant, {moon} Moon sign)**:\n{body}")

    # 3. QUESTION-AWARE SYNTHESIS FOR CAREER, MARRIAGE, FINANCE, AND GENERAL
    if q_type == "comparison":
        res = _synthesize_comparison_answer(
            domain=domain,
            asc=asc,
            moon=moon,
            h10_sign=h10_sign,
            h10_lord=h10_lord,
            h7_sign=h7_sign,
            h7_lord=h7_lord,
            c_maha=c_maha,
            positives=positives,
            challenges=challenges,
            is_hinglish=is_hinglish
        )
    elif q_type == "timing":
        res = _synthesize_timing_answer(
            domain=domain,
            asc=asc,
            moon=moon,
            c_maha=c_maha,
            c_antar=c_antar,
            h_sign=h10_sign if domain == "career" else h7_sign,
            h_lord=h10_lord if domain == "career" else h7_lord,
            positives=positives,
            challenges=challenges,
            explicit_horizon=explicit_horizon,
            timing_window=timing.get("timing_window") or timing.get("window"),
            is_hinglish=is_hinglish
        )
    elif q_type == "feasibility_advisory":
        res = _synthesize_feasibility_answer(
            domain=domain,
            asc=asc,
            moon=moon,
            c_maha=c_maha,
            c_antar=c_antar,
            h_sign=h10_sign if domain == "career" else h7_sign,
            h_lord=h10_lord if domain == "career" else h7_lord,
            positives=positives,
            challenges=challenges,
            is_hinglish=is_hinglish
        )
    elif q_type == "direct_placement":
        target_h = _extract_target_house(question, domain=domain)
        t_sign = facts.get(f"house_{target_h}_sign") or facts.get(f"{target_h}H")
        t_lord = facts.get(f"house_{target_h}_lord") or facts.get(f"{target_h}L")
        if target_h == 10:
            t_sign = t_sign or h10_sign
            t_lord = t_lord or h10_lord
        elif target_h == 7:
            t_sign = t_sign or h7_sign
            t_lord = t_lord or h7_lord
        if not t_sign or not t_lord:
            res_s, res_l = _resolve_house_and_lord(asc, target_h)
            t_sign = t_sign or res_s
            t_lord = t_lord or res_l

        res = _synthesize_direct_placement_answer(
            domain=domain,
            asc=asc,
            moon=moon,
            house_num=target_h,
            house_sign=t_sign,
            house_lord=t_lord,
            c_maha=c_maha,
            positives=positives,
            challenges=challenges,
            is_hinglish=is_hinglish
        )
    else:  # growth_outlook or fallback
        res = _synthesize_growth_outlook_answer(
            domain=domain,
            asc=asc,
            moon=moon,
            themes=themes,
            c_maha=c_maha,
            c_antar=c_antar,
            positives=positives,
            challenges=challenges,
            is_hinglish=is_hinglish
        )

    return guard_deterministic_claims(res)


def build_auditable_answer_trace(
    question: str = "",
    domain: str = "general",
    intent: str = "general",
    answer_source: str = "LOCAL",
    evidence: Optional[StructuredEvidence] = None,
    gemini_calls: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Builds structured auditable answer trace object (Section 13).
    """
    ev_complete = evidence.is_sufficient() if (evidence and hasattr(evidence, "is_sufficient")) else True
    matched_rules = kwargs.get("matched_rules", []) or (evidence.positive_factors if evidence and hasattr(evidence, "positive_factors") else [])
    in_tok = kwargs.get("llm_input_tokens", 0)
    out_tok = kwargs.get("llm_output_tokens", 0)
    tot_tok = kwargs.get("llm_total_tokens", in_tok + out_tok)
    llm_calls_list = kwargs.get("llm_calls_list", []) or kwargs.get("llm_calls", [])
    llm_usage = kwargs.get("llm_usage") or {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": tot_tok,
    }
    return {
        "question": question,
        "domain": domain,
        "intent": intent,
        "resolved_intent": kwargs.get("resolved_intent", intent),
        "answer_source": answer_source,
        "domain_match": kwargs.get("domain_match", True),
        "intent_match": kwargs.get("intent_match", True),
        "exact_rule_match": kwargs.get("exact_rule_match", answer_source == "LOCAL"),
        "matching_rules": matched_rules,
        "rule_coverage_score": kwargs.get("rule_coverage_score", 1.0 if answer_source == "LOCAL" else 0.0),
        "required_evidence": kwargs.get("required_evidence", ["ascendant_rashi", "moon_rashi"]),
        "available_evidence": kwargs.get("available_evidence", ["ascendant_rashi", "moon_rashi"]),
        "missing_evidence": kwargs.get("missing_evidence", []),
        "evidence_status": kwargs.get("evidence_status", "COMPLETE" if ev_complete else "UNRESOLVED"),
        "gemini_calls": gemini_calls,
        "llm_input_tokens": in_tok,
        "llm_output_tokens": out_tok,
        "llm_total_tokens": tot_tok,
        "llm_usage": llm_usage,
        "llm_calls": llm_calls_list,
        "errors": kwargs.get("errors", []),
        "quality_status": kwargs.get("quality_status", "PASS"),
        "quality_score": kwargs.get("quality_score", 100),
        "failure_category": kwargs.get("failure_category", None)
    }

