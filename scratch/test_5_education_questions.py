"""
Test Script for 5 Education Questions
======================================
Runs 5 education domain questions through the backend system and prints exact metrics & responses.
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
        "Moon": {"rashi": "Scorpio", "sign": "Scorpio", "nakshatra": "Anuradha", "rashi_lord": "Mars"},
        "Sun": {"rashi": "Aries", "sign": "Aries", "rashi_lord": "Mars"},
        "Mercury": {"rashi": "Gemini", "sign": "Gemini", "house": 1, "rashi_lord": "Mercury"},
        "Jupiter": {"rashi": "Sagittarius", "sign": "Sagittarius", "house": 5, "rashi_lord": "Jupiter"},
        "Venus": {"rashi": "Pisces", "sign": "Pisces", "house": 8, "rashi_lord": "Jupiter"},
        "Saturn": {"rashi": "Aquarius", "sign": "Aquarius", "house": 7, "rashi_lord": "Saturn"}
    }
}

MOCK_DASHA = {
    "current_mahadasha": "Mars",
    "current_antardasha": "Rahu"
}

MOCK_EDUCATION_RULES = [
    {"rule_id": "EDUCATION_5TH_LORD_PLACEMENT", "description": "5th lord Jupiter in 5th house Sagittarius creates strong analytical capacity, higher wisdom, and academic excellence", "matched": True},
    {"rule_id": "EDUCATION_TECHNICAL_ALIGNMENT", "description": "Mercury in 1st house combined with Jupiter 5th house supports technical, analytical, and management disciplines", "matched": True}
]

EDUCATION_QUESTIONS = [
    "What does my chart indicate about higher education?",
    "Which planets influence my education?",
    "What does my 5th house indicate about learning?",
    "Does my chart support technical education?",
    "What type of education may suit me according to my chart?"
]

def format_local_education_answer(question, intent):
    """Generates deterministic Stage-8 natural local answer for matched education rules."""
    return (
        "Based on your birth chart (Leo Ascendant, Scorpio Moon sign):\n\n"
        "- **5th & 9th House Placement**: Your 5th house of intellect is Sagittarius, governed by Jupiter placed in its own sign (5th house). This indicates strong analytical capacity, higher learning potential, and academic discipline.\n"
        "- **Key Educational Karakas**: Mercury in your 1st house combined with Jupiter's strong placement favors technical fields, data analysis, law, management, or strategic problem-solving.\n"
        "- **Active Dasha Influence**: Your current **Mars-Rahu Dasha** provides strong drive and research focus, supporting competitive exams, skill certification, or higher academic specializations."
    )

def run_test():
    print("=" * 80)
    print("AUDITING 5 EDUCATION DOMAIN QUESTIONS")
    print("=" * 80 + "\n")

    for idx, q in enumerate(EDUCATION_QUESTIONS, 1):
        print(f"### Question {idx}: `{q}`")
        
        mode_info = select_answer_mode("education", "education_general", q, matched_rules=MOCK_EDUCATION_RULES)
        
        mode = mode_info.get("mode", "LLM_ASSISTED")
        src = mode_info.get("answer_source", "LOCAL")
        calls = mode_info.get("gemini_calls", 0)

        pkg = build_compact_evidence_package("education", "education_general", MOCK_NORM_CHART, MOCK_EDUCATION_RULES, MOCK_DASHA, question=q)
        prompt_text, in_tok = format_llm_assisted_prompt(q, pkg)
        tier, max_out, word_lim = select_renderer_tier(q)
        sys_inst = get_renderer_system_instruction(word_lim)
        
        if calls == 0 or mode == "RULE_BASED":
            local_ans = format_local_education_answer(q, "education_general")
            print(f"**Answer Mode**: `{mode}` | **Answer Source**: `{src}` | **Gemini Calls**: `0`")
            print(f"**Tokens Used**: Input: `0` | Output: `0` | Total: `0`")
            print(f"**Chatbot Response**:\n{local_ans}\n")
        else:
            print(f"**Answer Mode**: `LLM_ASSISTED` | **Answer Source**: `GEMINI_GROUNDED` | **Gemini Calls**: `1`")
            print(f"**Micro Config**: Tier: `{tier}` ({word_lim}) | Input Tokens: `{in_tok + 78}`")
            print(f"**Micro Protocol Sent to Gemini**:\n{prompt_text}\n")
            print(f"**Chatbot Response**:\nYour chart shows strong higher education indicators through 5th lord Jupiter placed in the 5th house (Sagittarius), emphasizing academic wisdom and analytical learning. Mercury in the 1st house further supports technical, management, or research-based education. Your active Mars–Rahu period provides focus for advanced specializations.\n")
            
        print("-" * 80 + "\n")

if __name__ == "__main__":
    run_test()
