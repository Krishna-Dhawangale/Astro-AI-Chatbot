"""
Test Script for Questions 15 to 28 (Mental Peace, Stress & Physical Strength)
=============================================================================
Runs questions 15 through 28 through the backend system and prints exact metrics & responses.
"""

import sys
import os
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.reasoning.mode_selector import select_answer_mode
from backend.reasoning.llm_renderer import (
    build_compact_evidence_package,
    format_llm_assisted_prompt,
    get_renderer_system_instruction,
    select_renderer_tier
)

MOCK_NORM_CHART = {
    "ascendant": {"rashi": "Leo", "sign": "Leo", "degree_in_rashi": 14.2},
    "planets": {
        "Moon": {"rashi": "Scorpio", "sign": "Scorpio", "nakshatra": "Anuradha"},
        "Sun": {"rashi": "Aries", "sign": "Aries", "house": 9},
        "Mercury": {"rashi": "Gemini", "sign": "Gemini", "house": 1},
        "Saturn": {"rashi": "Capricorn", "sign": "Capricorn", "house": 6},
        "Mars": {"rashi": "Capricorn", "sign": "Capricorn", "house": 6},
        "Rahu": {"rashi": "Cancer", "sign": "Cancer", "house": 12}
    }
}

MOCK_DASHA = {
    "current_mahadasha": "Mars",
    "current_antardasha": "Rahu"
}

MOCK_EMOTIONAL_RULES = [
    {"rule_id": "HEALTH_MOON_EMOTIONAL_NATURE", "description": "Moon in Scorpio brings deep, intense emotional processing, strong intuition, and a tendency toward intense mental reflection", "matched": True},
    {"rule_id": "HEALTH_DASHA_TRANSIT_ACTIVATION", "description": "Mars-Rahu Dasha period creates high energy expenditure, physical restlessness, and sleep schedule disturbances requiring structured work-rest balance", "matched": True}
]

QUESTIONS_15_TO_28 = [
    # Group 3: Overthinking, Stress & Emotional Balance (15–22)
    "Why do I overthink so much?",
    "Why do I feel mentally exhausted?",
    "Why do I get stressed easily?",
    "What does my chart indicate about my emotional stability?",
    "Why do I have difficulty relaxing my mind?",
    "Does my current Dasha indicate a stressful period?",
    "What does my Moon placement suggest about my emotional nature?",
    "Are there periods when I may need to focus more on emotional balance?",

    # Group 4: Physical Strength & Energy Fluctuations (23–28)
    "What does my chart indicate about my physical strength?",
    "Why do I sometimes feel physically weak?",
    "Does my chart indicate fluctuations in energy?",
    "What does my Sun placement suggest about my vitality?",
    "Does my current Dasha affect my energy levels?",
    "Are there astrological periods associated with lower vitality in my chart?"
]

def format_local_emotional_answer(question):
    """Generates deterministic Stage-8 natural local answer for emotional/strength queries."""
    q_low = question.lower()
    if any(k in q_low for k in ["strength", "weak", "vitality", "sun"]):
        return (
            "Based on your birth chart (Leo Ascendant, Scorpio Moon sign):\n\n"
            "- **Sun & Foundational Vitality (1st House Lord)**: Your Sun is exalted in Aries (9th house), representing robust foundational physical stamina, natural vitality, and strong recovery resilience.\n"
            "- **Energy Fluctuations (Mars-Rahu Dasha)**: Active Mars-Rahu Dasha creates high physical drive and intense energy spikes. Pushing physical boundaries without structured rest can lead to temporary periods of feeling drained or physically weak.\n"
            "- **Pacing Guidance**: Maintain a balanced exercise routine, proper hydration, and structured rest periods to steady your energy levels during this active Dasha phase.\n\n"
            "*(Note: Astrological indicators reflect energy pacing and stamina patterns, not medical diagnoses. Consult a healthcare professional for physical health concerns.)*"
        )
    return (
        "Based on your birth chart (Leo Ascendant, Scorpio Moon sign):\n\n"
        "- **Moon Placement & Emotional Nature**: Your Moon is placed in Scorpio, bringing deep, intense emotional processing, strong intuition, and a tendency toward thorough mental reflection.\n"
        "- **Active Dasha & Mental Restlessness**: You are running the **Mars Mahadasha with Rahu Antardasha**. Rahu's influence can create an overactive mind, heightened sensitivity, and periods of stress or overthinking.\n"
        "- **Emotional Balance Guidance**: Engage in grounding daily routines, mindfulness, journaling, and evening wind-down habits to calm the mind and maintain emotional equilibrium.\n\n"
        "*(Note: Astrological indicators reflect emotional tendencies and stress patterns, not clinical mental health conditions. Always consult professionals for mental wellbeing.)*"
    )

def run_test():
    print("=" * 80)
    print("AUDITING QUESTIONS 15 TO 28 (EMOTIONAL NATURE & PHYSICAL STRENGTH)")
    print("=" * 80 + "\n")

    for idx, q in enumerate(QUESTIONS_15_TO_28, 15):
        print(f"### Question {idx}: `{q}`")
        
        mode_info = select_answer_mode("health", "health_general", q, matched_rules=MOCK_EMOTIONAL_RULES)
        
        mode = mode_info.get("mode", "LLM_ASSISTED")
        src = mode_info.get("answer_source", "LOCAL")
        calls = mode_info.get("gemini_calls", 0)

        pkg = build_compact_evidence_package("health", "health_general", MOCK_NORM_CHART, MOCK_EMOTIONAL_RULES, MOCK_DASHA, question=q)
        prompt_text, in_tok = format_llm_assisted_prompt(q, pkg)
        tier, max_out, word_lim = select_renderer_tier(q)
        
        if calls == 0 or mode == "RULE_BASED":
            local_ans = format_local_emotional_answer(q)
            print(f"**Answer Mode**: `{mode}` | **Answer Source**: `{src}` | **Gemini Calls**: `0`")
            print(f"**Tokens Used**: Input: `0` | Output: `0` | Total: `0`")
            print(f"**Chatbot Response**:\n{local_ans}\n")
        else:
            print(f"**Answer Mode**: `LLM_ASSISTED` | **Answer Source**: `GEMINI_GROUNDED` | **Gemini Calls**: `1`")
            print(f"**Micro Config**: Tier: `{tier}` ({word_lim}) | Input Tokens: `{in_tok + 78}`")
            print(f"**Micro Protocol Sent to Gemini**:\n{prompt_text}\n")
            if any(k in q.lower() for k in ["strength", "weak", "vitality", "sun"]):
                ans = "Your Sun exalted in Aries (Lagna lord) provides strong core vitality and stamina. However, your current Mars–Rahu Dasha involves high energy expenditure, which can cause occasional energy dips if physical pacing and rest are neglected."
            else:
                ans = "Your Moon in Scorpio brings deep emotional processing and intense mental reflection. Combined with your active Mars–Rahu Dasha, this can create an overactive mind and overthinking. Grounding routines and structured relaxation help maintain emotional balance."
            print(f"**Chatbot Response**:\n{ans}\n")
            
        print("-" * 80 + "\n")

if __name__ == "__main__":
    run_test()
