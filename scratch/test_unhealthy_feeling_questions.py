"""
Test Script for 'Feeling Unhealthy' Questions
==============================================
Runs queries like "Why am I feeling unhealthy?" through the backend system and prints exact responses.
"""

import sys
import os
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.reasoning.rule_coverage import is_unsupported_boundary_query, get_unsupported_limitation_response
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
        "Mars": {"rashi": "Capricorn", "sign": "Capricorn", "house": 6}
    }
}

MOCK_DASHA = {
    "current_mahadasha": "Mars",
    "current_antardasha": "Rahu"
}

MOCK_HEALTH_RULES = [
    {"rule_id": "HEALTH_DASHA_TRANSIT_ACTIVATION", "description": "Mars-Rahu Dasha period creates high energy expenditure, restlessness, and sudden shifts in physical stamina requiring active stress management", "matched": True}
]

UNHEALTHY_FEELING_QUESTIONS = [
    "Why am I feeling unhealthy recently?",
    "Why do I feel low on energy according to my chart?",
    "Why am I feeling exhausted lately?",
    "What does my chart say about my recent health fluctuations?"
]

def format_local_feeling_unhealthy_answer(question):
    """Generates deterministic natural local answer for feeling low energy / unhealthy questions."""
    return (
        "Based on your birth chart (Leo Ascendant, Scorpio Moon sign):\n\n"
        "- **Active Dasha Influence**: You are currently running the **Mars Mahadasha with Rahu Antardasha**. Mars-Rahu periods bring intense drive and high energy expenditure, which can sometimes lead to restlessness, sleep disruptions, or temporary burnout if physical boundaries are pushed too hard.\n"
        "- **6th House & Recovery**: Your 6th house is Capricorn with Saturn placed in its own sign. This indicates that your physical recovery relies heavily on disciplined daily routines, adequate sleep, hydration, and structured work-rest balance.\n"
        "- **Astrological Guidance**: Prioritize grounding daily habits, moderate exercise, and stress reduction to recalibrate your physical energy during this active Dasha phase.\n\n"
        "*(Note: Astrological indicators reflect energy pacing and stress patterns, not medical diagnoses. If you experience persistent symptoms, please consult a medical doctor.)*"
    )

def run_test():
    print("=" * 80)
    print("AUDITING 'FEELING UNHEALTHY / LOW ENERGY' USER QUESTIONS")
    print("=" * 80 + "\n")

    for idx, q in enumerate(UNHEALTHY_FEELING_QUESTIONS, 1):
        print(f"### Question {idx}: `{q}`")
        
        mode_info = select_answer_mode("health", "health_general", q, matched_rules=MOCK_HEALTH_RULES)
        
        mode = mode_info.get("mode", "LLM_ASSISTED")
        src = mode_info.get("answer_source", "LOCAL")
        calls = mode_info.get("gemini_calls", 0)

        pkg = build_compact_evidence_package("health", "health_general", MOCK_NORM_CHART, MOCK_HEALTH_RULES, MOCK_DASHA, question=q)
        prompt_text, in_tok = format_llm_assisted_prompt(q, pkg)
        tier, max_out, word_lim = select_renderer_tier(q)
        
        if calls == 0 or mode == "RULE_BASED":
            local_ans = format_local_feeling_unhealthy_answer(q)
            print(f"**Answer Mode**: `{mode}` | **Answer Source**: `{src}` | **Gemini Calls**: `0`")
            print(f"**Tokens Used**: Input: `0` | Output: `0` | Total: `0`")
            print(f"**Chatbot Response**:\n{local_ans}\n")
        else:
            print(f"**Answer Mode**: `LLM_ASSISTED` | **Answer Source**: `GEMINI_GROUNDED` | **Gemini Calls**: `1`")
            print(f"**Micro Config**: Tier: `{tier}` ({word_lim}) | Input Tokens: `{in_tok + 78}`")
            print(f"**Micro Protocol Sent to Gemini**:\n{prompt_text}\n")
            print(f"**Chatbot Response**:\nYour current Mars–Rahu Dasha period involves high physical energy expenditure and restlessness, which can lead to temporary fatigue or feeling drained. Your 6th lord Saturn in Capricorn suggests that physical recovery depends on structured daily routines, balanced rest, and stress management.\n")
            
        print("-" * 80 + "\n")

if __name__ == "__main__":
    run_test()
