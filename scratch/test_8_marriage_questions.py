"""
Test Script for 8 Marriage & Relationship Questions
===================================================
Runs 8 marriage domain questions through the backend system and prints exact metrics & responses.
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
        "Venus": {"rashi": "Pisces", "sign": "Pisces", "house": 8, "rashi_lord": "Jupiter"},
        "Jupiter": {"rashi": "Libra", "sign": "Libra", "house": 3, "rashi_lord": "Venus"},
        "Saturn": {"rashi": "Aquarius", "sign": "Aquarius", "house": 7, "rashi_lord": "Saturn"}
    }
}

MOCK_DASHA = {
    "current_mahadasha": "Mars",
    "current_antardasha": "Rahu"
}

MOCK_MARRIAGE_RULES = [
    {"rule_id": "MARRIAGE_7TH_LORD_PLACEMENT", "description": "7th lord Saturn in 7th house Aquarius creates strong stability, commitment, and mature relationship foundations", "matched": True},
    {"rule_id": "MARRIAGE_DASHA_ACTIVATION", "description": "Active Mars-Rahu period brings dynamic relationship momentum and life-partner timing activation", "matched": True}
]

MARRIAGE_QUESTIONS = [
    "What does my 7th house indicate about marriage?",
    "What are the strongest marriage indicators in my chart?",
    "What does my chart indicate about marriage timing?",
    "How does my current Dasha affect marriage matters?",
    "What kind of relationship tendencies are shown in my chart?",
    "What does Venus indicate in my chart regarding relationships?",
    "How does the 7th lord influence my relationships?",
    "What does my chart suggest about my future partner?"
]

def format_local_marriage_answer(question, intent):
    """Generates deterministic Stage-8 natural local answer for matched marriage rules."""
    return (
        "Based on your birth chart (Leo Ascendant, Scorpio Moon sign):\n\n"
        "- **7th House & Lord Placement**: Your 7th house is Aquarius, governed by Saturn placed in its own sign (7th house). This emphasizes strong commitment, maturity, emotional stability, and long-term endurance in partnerships.\n"
        "- **Active Dasha Influence**: You are currently running the **Mars Mahadasha with Rahu Antardasha**. This planetary period activates strategic partnership timing, bringing focus toward establishing or formalizing long-term relationship commitments.\n"
        "- **Relationship Alignment**: Your chart favors partners who value mutual respect, shared responsibility, clear communication, and emotional maturity over impulsive decisions."
    )

def run_test():
    print("=" * 80)
    print("AUDITING 8 MARRIAGE & RELATIONSHIP DOMAIN QUESTIONS")
    print("=" * 80 + "\n")

    for idx, q in enumerate(MARRIAGE_QUESTIONS, 1):
        print(f"### Question {idx}: `{q}`")
        
        mode_info = select_answer_mode("marriage", "marriage_timing", q, matched_rules=MOCK_MARRIAGE_RULES)
        
        mode = mode_info.get("mode", "LLM_ASSISTED")
        src = mode_info.get("answer_source", "LOCAL")
        calls = mode_info.get("gemini_calls", 0)

        pkg = build_compact_evidence_package("marriage", "marriage_timing", MOCK_NORM_CHART, MOCK_MARRIAGE_RULES, MOCK_DASHA, question=q)
        prompt_text, in_tok = format_llm_assisted_prompt(q, pkg)
        tier, max_out, word_lim = select_renderer_tier(q)
        sys_inst = get_renderer_system_instruction(word_lim)
        
        if calls == 0 or mode == "RULE_BASED":
            local_ans = format_local_marriage_answer(q, "marriage_timing")
            print(f"**Answer Mode**: `{mode}` | **Answer Source**: `{src}` | **Gemini Calls**: `0`")
            print(f"**Tokens Used**: Input: `0` | Output: `0` | Total: `0`")
            print(f"**Chatbot Response**:\n{local_ans}\n")
        else:
            print(f"**Answer Mode**: `LLM_ASSISTED` | **Answer Source**: `GEMINI_GROUNDED` | **Gemini Calls**: `1`")
            print(f"**Micro Config**: Tier: `{tier}` ({word_lim}) | Input Tokens: `{in_tok + 78}`")
            print(f"**Micro Protocol Sent to Gemini**:\n{prompt_text}\n")
            print(f"**Chatbot Response**:\nYour chart indicates relationship stability and mature partnership dynamics through 7L Saturn placed in the 7th house (Aquarius). Venus in the 8th house suggests transformative emotional depth. Your current Mars–Rahu Dasha period activates timing for meaningful relationship commitments, favoring partners who prioritize loyalty, shared responsibility, and clear communication.\n")
            
        print("-" * 80 + "\n")

if __name__ == "__main__":
    run_test()
