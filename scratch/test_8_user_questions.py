"""
Test Script for 8 User Questions
=================================
Runs all 8 factual/comparison questions through the backend system and prints exact metrics & responses.
"""

import sys
import os
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.reasoning.direct_fact_engine import is_direct_fact_query, extract_direct_fact
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
        "Moon": {"rashi": "Scorpio", "sign": "Scorpio", "nakshatra": "Anuradha", "nakshatra_pada": 3, "rashi_lord": "Mars"},
        "Sun": {"rashi": "Aries", "sign": "Aries", "rashi_lord": "Mars"},
        "Mercury": {"rashi": "Gemini", "sign": "Gemini", "house": 10, "rashi_lord": "Mercury"},
        "Venus": {"rashi": "Pisces", "sign": "Pisces", "house": 8, "rashi_lord": "Jupiter"},
        "Jupiter": {"rashi": "Libra", "sign": "Libra", "house": 3, "rashi_lord": "Venus"}
    }
}

MOCK_DASHA = {
    "current_mahadasha": "Mars",
    "current_antardasha": "Rahu"
}

QUESTIONS = [
    "What is my Sun sign?",
    "What is my Moon sign?",
    "What is my Rashi?",
    "What is my Lagna?",
    "What is the difference between my Sun sign and Moon sign?",
    "Is my Rashi the same as my Moon sign?",
    "How is my Lagna different from my Rashi?",
    "What is the difference between my Nakshatra and Rashi?"
]

def run_test():
    print("=" * 80)
    print("AUDITING 8 USER FACTUAL & COMPARISON QUESTIONS")
    print("=" * 80 + "\n")

    for idx, q in enumerate(QUESTIONS, 1):
        print(f"### Question {idx}: `{q}`")
        
        is_direct = is_direct_fact_query(q)
        if is_direct:
            direct_res = extract_direct_fact(q, MOCK_NORM_CHART, MOCK_DASHA)
            if direct_res:
                print(f"**Answer Mode**: `{direct_res.get('answer_mode', 'DIRECT')}` | **Answer Source**: `DIRECT_FACT_ENGINE` | **Gemini Calls**: `0`")
                print(f"**Tokens**: Input: `0` | Output: `0` | Total: `0`")
                print(f"**Chatbot Response**:\n{direct_res['answer']}\n")
                print("-" * 80 + "\n")
                continue

        # If not direct fact lookup, select mode
        mode_info = select_answer_mode("other", "general", q)
        print(f"**Answer Mode**: `{mode_info['mode']}` | **Answer Source**: `{mode_info['answer_source']}` | **Gemini Calls**: `{mode_info['gemini_calls']}`")
        
        pkg = build_compact_evidence_package("other", "general", MOCK_NORM_CHART, [], MOCK_DASHA, question=q)
        prompt_text, in_tok = format_llm_assisted_prompt(q, pkg)
        tier, max_out, word_lim = select_renderer_tier(q)
        sys_inst = get_renderer_system_instruction(word_lim)
        
        print(f"**Prompt Config**: Tier: `{tier}` ({word_lim}) | Input Tokens: `{in_tok + 78}`")
        print(f"**Formatted Prompt Sent to LLM**:\n{prompt_text}\n")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    run_test()
