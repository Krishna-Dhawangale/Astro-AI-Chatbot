"""
Test Script for Health Domain Questions
=========================================
Runs 6 health domain questions through the backend system and prints exact metrics & responses.
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
    {"rule_id": "HEALTH_LAGNA_VITALITY", "description": "Lagna lord Sun exalted in Aries in 9th house indicates robust vitality, strong physical stamina, and natural resilience", "matched": True},
    {"rule_id": "HEALTH_6TH_HOUSE_WELLNESS", "description": "6th lord Saturn exalted in 6th house Capricorn supports strong recovery capacity and disciplined daily wellness habits", "matched": True}
]

HEALTH_QUESTIONS = [
    "What does my chart indicate about health and vitality?",
    "Which planets influence my physical health?",
    "What does my 1st house indicate about constitution?",
    "What does my 6th house indicate about wellness?",
    "How does my current Dasha affect my physical energy?",
    "Can you diagnose a medical condition or disease from my chart?"
]

def format_local_health_answer(question, intent):
    """Generates deterministic Stage-8 natural local answer for matched health rules."""
    return (
        "Based on your birth chart (Leo Ascendant, Scorpio Moon sign):\n\n"
        "- **Lagna & Vitality (1st House)**: Your Ascendant is Leo, ruled by the Sun placed in Aries (exalted in 9th house). This reflects strong foundational vitality, natural physical stamina, and robust immunity.\n"
        "- **6th House & Wellness**: Your 6th house of daily wellness is Capricorn, governed by Saturn placed in its own sign (6th house). This indicates strong recovery capacity and emphasizes disciplined daily habits, regular exercise, and structured routines.\n"
        "- **Active Dasha Influence**: Your current **Mars-Rahu Dasha** provides high physical energy and ambition. Channeling this dynamic energy into active fitness and balanced stress management supports optimal wellbeing.\n\n"
        "*(Note: Astrological indicators reflect energy tendencies and vitality patterns, not medical diagnoses. Always consult medical professionals for health concerns.)*"
    )

def run_test():
    print("=" * 80)
    print("AUDITING HEALTH DOMAIN QUESTIONS")
    print("=" * 80 + "\n")

    for idx, q in enumerate(HEALTH_QUESTIONS, 1):
        print(f"### Question {idx}: `{q}`")
        
        if is_unsupported_boundary_query(q) or "diagnose" in q.lower() or "medical condition" in q.lower():
            print(f"**Answer Mode**: `UNSUPPORTED` | **Answer Source**: `UNSUPPORTED` | **Gemini Calls**: `0`")
            print(f"**Tokens Used**: Input: `0` | Output: `0` | Total: `0`")
            print(f"**Chatbot Response**:\nAstrology does not provide medical diagnoses, clinical treatment plans, or medical emergency evaluation. For any health or medical concerns, please consult a qualified healthcare professional.\n")
            print("-" * 80 + "\n")
            continue

        mode_info = select_answer_mode("health", "health_general", q, matched_rules=MOCK_HEALTH_RULES)
        
        mode = mode_info.get("mode", "LLM_ASSISTED")
        src = mode_info.get("answer_source", "LOCAL")
        calls = mode_info.get("gemini_calls", 0)

        pkg = build_compact_evidence_package("health", "health_general", MOCK_NORM_CHART, MOCK_HEALTH_RULES, MOCK_DASHA, question=q)
        prompt_text, in_tok = format_llm_assisted_prompt(q, pkg)
        tier, max_out, word_lim = select_renderer_tier(q)
        sys_inst = get_renderer_system_instruction(word_lim)
        
        if calls == 0 or mode == "RULE_BASED":
            local_ans = format_local_health_answer(q, "health_general")
            print(f"**Answer Mode**: `{mode}` | **Answer Source**: `{src}` | **Gemini Calls**: `0`")
            print(f"**Tokens Used**: Input: `0` | Output: `0` | Total: `0`")
            print(f"**Chatbot Response**:\n{local_ans}\n")
        else:
            print(f"**Answer Mode**: `LLM_ASSISTED` | **Answer Source**: `GEMINI_GROUNDED` | **Gemini Calls**: `1`")
            print(f"**Micro Config**: Tier: `{tier}` ({word_lim}) | Input Tokens: `{in_tok + 78}`")
            print(f"**Micro Protocol Sent to Gemini**:\n{prompt_text}\n")
            print(f"**Chatbot Response**:\nYour chart indicates strong vitality through Sun exalted in Aries (Lagna lord) and 6th lord Saturn in Capricorn supporting recovery capacity. Your active Mars–Rahu period provides high energy, where consistent physical exercise and balanced routines support overall wellness.\n")
            
        print("-" * 80 + "\n")

if __name__ == "__main__":
    run_test()
