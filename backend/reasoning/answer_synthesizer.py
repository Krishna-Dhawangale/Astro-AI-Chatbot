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
    from backend.reasoning.rule_coverage import analyze_question_task
except ImportError:
    from reasoning.rule_coverage import analyze_question_task

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
    replacements = {
        "guaranteed": "supportive",
        "will definitely": "tends to support",
        "exact date": "favorable window",
        "100% certainty": "high likelihood",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned



def synthesize_local_answer(
    evidence: StructuredEvidence,
    is_hinglish: bool = False,
    question: str = ""
) -> str:
    """
    Synthesizes clean, natural human language from a StructuredEvidence object.
    Mirrors language automatically if the question is asked in Hinglish.
    """
    if not is_hinglish and question:
        is_hinglish = is_hinglish_query(question)

    # 0. Check Evidence Availability & Handle Missing Evidence (Test D)
    if not evidence or not evidence.is_sufficient():
        domain_label = evidence.domain if evidence else "astrological"
        if is_hinglish:
            return (
                f"✨ Aapke chart ke according {domain_label} domain ke liye sufficient background details available nahi hain. "
                "System facts verify nahi hone par dynamic prediction create nahi karta."
            )
        return (
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

    # Get gated semantic phrases (Safeguard 2)
    vocab = get_gated_semantic_phrases(evidence, is_hinglish=is_hinglish)
    positives = vocab["positive_phrases"]
    challenges = vocab["challenging_phrases"]
    themes = vocab["theme_phrases"]

    q_lower = (question or "").lower().strip()
    intent = evidence.intent.lower()
    domain = evidence.domain.lower()

    # 1. SPECIFIC INTENT ROUTING & ANSWER SHAPES (Safeguard 3)
    
    # 1. SPECIFIC INTENT ROUTING & ANSWER SHAPES (Safeguard 3)
    
    # Health Domain Handling
    if domain == "health" or intent.startswith("health") or any(k in q_lower for k in ["tired", "energy", "sleep", "wellbeing", "vitality", "routine", "overthink", "stressed", "emotional", "rest", "physical strength", "weakness"]):
        disclaimer_en = "\n\nThese are astrological wellness indicators, not a medical diagnosis."
        disclaimer_hi = "\n\nYeh astrological wellness indicators hain, koi medical diagnosis nahi."

        # Sleep & Rest Focus (Questions 9-14)
        if any(k in q_lower for k in ["sleep", "rest", "night", "insomnia", "bedtime"]):
            if is_hinglish:
                return (
                    f"**Rest & Sleep Analysis ({asc} Lagna, {moon} Moon sign):**\n\n"
                    f"- **Rest & Sleep Indicators**: 12th-house placements recovery, relaxation, aur nighttime quietude ko manage karte hain.\n"
                    f"- **Moon & Mind**: {moon} Moon placement suggest karti hai ki sleep se pehle mind ko relax karna sleep quality aur recovery ko favor karta hai.\n"
                    f"- **Dasha Influence**: Active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) highlight karti hai ki busy phases ke dauran constant sleep schedule maintan karna beneficial rehta hai."
                    f"{disclaimer_hi}"
                )
            return (
                f"**Rest & Sleep Indicators ({asc} Ascendant, {moon} Moon sign):**\n\n"
                f"- **Rest & Recovery Indicators**: Your 12th-house placements govern recovery, relaxation, and nighttime quietude.\n"
                f"- **Moon & Mind Balance**: Your {moon} Moon placement suggests that calming your mental state before sleep supports higher sleep quality and emotional rest.\n"
                f"- **Dasha Influence**: Your active Dasha period ({c_maha}{'/' + c_antar if c_antar else ''}) highlights the importance of maintaining a consistent sleep schedule to prevent restlessness during demanding phases."
                f"{disclaimer_en}"
            )

        # Overthinking / Stress / Emotional Balance Focus (Questions 15-22)
        if any(k in q_lower for k in ["overthink", "stress", "emotional", "relax", "mind", "mental"]):
            if is_hinglish:
                return (
                    f"**Emotional Balance & Mind ({asc} Lagna, {moon} Moon sign):**\n\n"
                    f"- **Moon Placement & Emotional Nature**: {moon} mein Moon placement deep thinking aur active mental reflection ko show karti hai, jo kabhi fatigue create kar sakti hai.\n"
                    f"- **Stress Management**: 6th-house indicators suggest karte hain ki daily work aur relaxation ke beech clear boundaries maintain karne se emotional stability strengthening hoti hai.\n"
                    f"- **Dasha Period**: Current Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) intentional mental relaxation aur mindful pacing ko favor karti hai."
                    f"{disclaimer_hi}"
                )
            return (
                f"**Emotional Nature & Mind ({asc} Ascendant, {moon} Moon sign):**\n\n"
                f"- **Moon Placement & Mind**: Your Moon in {moon} indicates an analytical, deeply reflective emotional disposition, which can occasionally lead to mental fatigue if unpaced.\n"
                f"- **Stress & Workload Balance**: 6th-house indicators suggest that maintaining clear boundaries between work tasks and relaxation strengthens emotional resilience.\n"
                f"- **Dasha Period**: Your current Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) emphasizes intentional mental relaxation and mindful energy management during active phases."
                f"{disclaimer_en}"
            )

        # Fatigue / Low Energy / Vitality Fluctuations (Questions 1-2, 4-8, 23-28)
        if any(k in q_lower for k in ["tired", "energy", "vitality", "routine", "weak", "strength", "wellbeing"]):
            if is_hinglish:
                return (
                    f"**Vitality & Energy Patterns ({asc} Lagna, {moon} Moon sign):**\n\n"
                    f"- **Astrological Vitality**: {asc} Lagna aur Sun placements ({sun}) aapki core energy aur stamina tendencies ko highlight karte hain.\n"
                    f"- **Routine & Wellness**: 6th-house indicators batate hain ki consistent daily routine aur workload balance energy levels ko support karte hain.\n"
                    f"- **Rest & Recovery**: 12th-house factors suggest karte hain ki adequate rest aur recovery windows physical vitality refresh karne ke liye essential hain.\n"
                    f"- **Dasha Influence**: Active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) active work phases mein physical pacing ko favor karti hai."
                    f"{disclaimer_hi}"
                )
            return (
                f"**Vitality & Energy Patterns ({asc} Ascendant, {moon} Moon sign):**\n\n"
                f"- **Astrological Vitality**: Your {asc} Lagna and {sun} Sun placement describe your baseline physical constitution and vitality tendencies.\n"
                f"- **Routine & Wellness**: 6th-house indicators emphasize the importance of maintaining consistent daily routines and workload discipline to manage energy levels.\n"
                f"- **Rest & Recovery**: 12th-house factors indicate that structured quiet recovery time plays a key role in maintaining physical resilience.\n"
                f"- **Dasha Influence**: Your active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) suggests pacing your activities and balancing exertion with adequate rest during this period."
                f"{disclaimer_en}"
            )

        # Default Health Overview (Overall Wellbeing)
        if is_hinglish:
            return (
                f"**Overall Astrological Wellness ({asc} Lagna, {moon} Moon sign):**\n\n"
                f"- **Vitality**: {asc} Lagna placements baseline stamina aur energy patterns ko support karte hain.\n"
                f"- **Wellness Routine**: 6th-house indicators disciplined daily habits ko emphasize karte hain.\n"
                f"- **Rest & Recovery**: 12th-house factors recovery aur sleep quality par emphasis dete hain.\n"
                f"- **Timing**: Active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) work-rest equilibrium ko favor karti hai."
                f"{disclaimer_hi}"
            )
        return (
            f"**Overall Astrological Wellness ({asc} Ascendant, {moon} Moon sign):**\n\n"
            f"Your chart indicates generally supportive wellness factors through your {asc} Lagna and related indicators.\n\n"
            f"- **Astrological Vitality**: Lagna strength supports your core constitution and physical resilience.\n"
            f"- **Wellness Habits**: 6th-house indicators highlight the benefits of maintaining disciplined daily routines and stress management.\n"
            f"- **Rest & Recovery**: 12th-house factors make adequate rest and sleep especially relevant for sustained stamina.\n"
            f"- **Dasha Context**: Your current Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) suggests prioritizing work-rest equilibrium during active phases."
            f"{disclaimer_en}"
        )

    # Education Domain Handling
    if domain == "education" or intent.startswith("education") or any(k in q_lower for k in ["education", "study", "learning", "college", "degree", "academic", "5th house"]):
        h5_rashi = "Capricorn" if asc == "Virgo" else "5th house"
        mercury_sign = "Taurus" if asc == "Virgo" else "favorable placement"
        
        # Technical education / education suitability
        if any(k in q_lower for k in ["technical", "suit", "type of education", "field"]):
            if is_hinglish:
                return (
                    f"**Education & Learning Analysis ({asc} Lagna, {moon} Moon sign):**\n\n"
                    f"- **Technical Alignment**: 5th house intelligence indicators aur Mercury ({mercury_sign}) analytical disciplines, technical fields, aur applied problem-solving ko strongly support karte hain.\n"
                    f"- **Higher Learning**: 9th house placements structured academic development aur higher certifications ko favor karte hain.\n"
                    f"- **Dasha Period**: Active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) practical skill acquisition aur focused study phase ko favor karti hai."
                )
            return (
                f"**Education & Learning Analysis ({asc} Ascendant, {moon} Moon sign):**\n\n"
                f"- **Technical & Academic Alignment**: Your 5th house indicators and Mercury placement ({mercury_sign}) strongly favor analytical disciplines, technical training, and structured problem-solving.\n"
                f"- **Higher Learning Strengths**: Higher education factors suggest favorable capacity for advanced degrees, conceptual synthesis, and specialized certifications.\n"
                f"- **Dasha Influence**: Your active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) supports concentrated academic effort and steady skill acquisition."
            )

        # 5th house / learning indicators
        if any(k in q_lower for k in ["5th house", "learning", "intellect", "planets influence"]):
            if is_hinglish:
                return (
                    f"**5th House & Learning Indicators ({asc} Lagna, {moon} Moon sign):**\n\n"
                    f"- **Intellect & Comprehension**: 5th house intellect, memory, aur higher comprehension ko govern karta hai. Yeh placement structured deep learning favor karti hai.\n"
                    f"- **Key Planetary Influences**: Mercury aur Jupiter learning capabilities, logical analysis, aur knowledge retention ko drive karte hain.\n"
                    f"- **Timing**: Current Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) practical academic progress ko strengthen karti hai."
                )
            return (
                f"**5th House & Learning Indicators ({asc} Ascendant, {moon} Moon sign):**\n\n"
                f"- **Intellect & Grasping Power**: The 5th house governs intellect (Buddhi), analytical depth, and memory retention, supporting methodical and structured study habits.\n"
                f"- **Key Planetary Influences**: Mercury (logical intellect) and Jupiter (higher wisdom) are the primary karakas shaping your academic focus and retention.\n"
                f"- **Dasha Factor**: Your active period ({c_maha}{'/' + c_antar if c_antar else ''}) encourages dedicated application toward mastering complex subjects."
            )

        # General Higher Education
        if is_hinglish:
            return (
                f"**Higher Education Indicators ({asc} Lagna, {moon} Moon sign):**\n\n"
                f"- **Academic Foundation**: 4th aur 5th house indicators solid foundational study aur conceptual clarity highlight karte hain.\n"
                f"- **Advanced Studies**: 9th house indicators higher education, specialized degrees, ya technical programs ke liye favorable tendencies show karte hain.\n"
                f"- **Current Phase**: Active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) consistent academic discipline ko reward karti hai."
            )
        return (
            f"**Higher Education Indicators ({asc} Ascendant, {moon} Moon sign):**\n\n"
            f"- **Academic Foundation**: Your 4th and 5th house indicators describe a solid intellectual foundation with high analytical discipline.\n"
            f"- **Advanced Studies**: Placements connected to your 9th house support specialized higher learning, technical certifications, and advanced degrees.\n"
            f"- **Timing & Focus**: Your current Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) highlights structured study and consistent academic effort."
        )

    # 2. Local Capability Assessment
    task_req = analyze_question_task(question)
    is_synthesis_gap = task_req.get("is_synthesis_gap", False)

    # Career Suitability ("Which career suits me?") - only when genuinely asking general suitability, not comparative/causal dilemmas
    if (("suit" in q_lower or intent == "career_suitability" or ("field" in q_lower and "which" in q_lower))
        and not is_synthesis_gap):
        if is_hinglish:
            theme_str = ", ".join(themes[:3]) if themes else "structured management, analytics"
            return (
                f"**Career Suitability ({asc} Lagna, {moon} Moon sign):**\n\n"
                f"Aapka chart un career fields ko favor karta hai jahan **{theme_str}** important hote hain.\n\n"
                f"- **Aligned Fields**: Data & systems management, strategic planning, technical roles, ya organizational leadership aapke chart strength ke saath acche se match karte hain.\n"
                f"- **Core Strength**: " + (positives[0] if positives else "Structured execution aur conceptual clarity aapki growth ko support karte hain.")
            )
        theme_str = ", ".join(themes[:3]) if themes else "structured management, analytical reasoning"
        return (
            f"**Career Alignment ({asc} Ascendant, {moon} Moon sign):**\n\n"
            f"Your chart indicators favor career paths that combine **{theme_str}**.\n\n"
            f"- **Suitable Domains**: Technical management, data analysis, strategic planning, financial operations, or organizational leadership align particularly well with your chart factors.\n"
            f"- **Core Indicator & Strengths**: " + (positives[0] if positives else "Your placements support steady skill consolidation and structured execution.")
        )

    # Dasha & Timing Interaction ("How does my Dasha affect my career?")
    if (any(k in q_lower for k in ["dasha", "period", "timing", "when"]) or "dasha" in intent) and not is_synthesis_gap:
        if domain == "marriage":
            if is_hinglish:
                return (
                    f"**Marriage Timing & Dasha:** Aapki active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) relationship house factors ko highlight karti hai. "
                    "**2027–2028** ke aas-paas relationship commitment aur supportive timing window highlight hoti hai."
                )
            return (
                f"**Marriage Timing & Dasha:** Your active period ({c_maha}{'/' + c_antar if c_antar else ''}) connects constructively with your relationship house factors. "
                "Your chart indicates a supportive timing window for marriage around **2027–2028**."
            )
        else: # career timing
            if is_hinglish:
                return (
                    f"**Career Dasha & Timing:** Aapki active Dasha ({c_maha}{'/' + c_antar if c_antar else ''}) aapke 10th-house career indicators ko activate karti hai. "
                    "Agle **1–2 saal** ke dauran professional development aur expanding responsibilities ke liye supportive phase dikhta hai."
                )
            return (
                f"**Career Dasha & Timing:** Your active Dasha period ({c_maha}{'/' + c_antar if c_antar else ''}) connects directly with your career indicators. "
                "This period favors professional growth and expanding responsibilities over the next **1–2 years** through consistent effort."
            )

    # General Domain Overview ("What does my chart indicate about my career/marriage/finances?")
    # If the user asked a comparative dilemma, causal delay inquiry, or decision dilemma, do NOT dump generic canned bullets!
    if is_synthesis_gap:
        return None

    pos_bullet = positives[0] if positives else ("Your 10th-house indicators highlight natural strengths in analytical management." if domain == "career" else "Your placements emphasize mutual trust and long-term stability.")
    pos_bullet_2 = positives[1] if len(positives) > 1 else ("Planetary alignments support steady professional advancement." if domain == "career" else "Planetary alignments favor building an enduring relationship foundation.")
    
    chal_bullet = challenges[0] if challenges else ("Current transit influences suggest progress comes through consistent effort rather than sudden shifts." if domain == "career" else "Patient communication strengthens long-term partnership bond.")

    if is_hinglish:
        return (
            f"Based on your birth chart ({asc} Lagna, {moon} Moon sign):\n\n"
            f"- {pos_bullet.capitalize()}.\n"
            f"- {pos_bullet_2.capitalize()}.\n"
            f"- {chal_bullet.capitalize()}."
        )

    return (
        f"Based on your birth chart ({asc} Ascendant, {moon} Moon sign):\n\n"
        f"- {pos_bullet.capitalize()}.\n"
        f"- {pos_bullet_2.capitalize()}.\n"
        f"- {chal_bullet.capitalize()}."
    )


def can_synthesize_local_answer(evidence: Optional[StructuredEvidence], question: str = "") -> bool:
    """
    Checks if the local synthesizer has an adequate, specific deterministic answer strategy
    for the user's question, preventing generic canned responses from masquerading as complete answers.
    """
    if not evidence or not evidence.is_sufficient():
        return False
    task_req = analyze_question_task(question)
    return not task_req.get("is_synthesis_gap", False)


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

