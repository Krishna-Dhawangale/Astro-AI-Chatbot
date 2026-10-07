"""
Hinglish Test Suite (scratch/test_hinglish_suite.py)
===================================================
Audits chatbot handling across various Hinglish (Hindi + English) user questions.
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

HINGLISH_TEST_CASES = [
    # 1. Direct Fact Lookups in Hinglish
    ("Mera Rashi kya hai?", "rashi_lookup"),
    ("Mera Moon sign kya hai bataye?", "moon_sign"),
    ("Mera Sun sign kya hai?", "sun_sign"),
    ("Mera Lagna konsa hai?", "lagna"),
    ("Mera Janma Nakshatra konsa hai?", "nakshatra"),
    ("Mera current Mahadasha konsa chal raha hai?", "dasha"),

    # 2. Concept Comparisons in Hinglish
    ("Sun sign aur Moon sign me kya difference hai?", "sun_vs_moon"),
    ("Lagna aur Rashi me kya difference hota hai?", "lagna_vs_rashi"),
    ("Rashi aur Moon sign same hota hai kya?", "rashi_same_as_moon"),
    ("Nakshatra aur Rashi me kya pharak hai?", "nakshatra_vs_rashi"),

    # 3. Domain Interpretation Queries in Hinglish
    ("Mera career kaisa rahega according to my chart?", "career_general"),
    ("Current Dasha mere career ko kaise affect karega?", "dasha_career"),
    ("Meri shaadi kab tak hogi?", "marriage_timing"),
    ("Financial status kaisa rahega upcoming year me?", "finance_potential"),

    # 4. Unsupported Boundary Queries in Hinglish
    ("Lottery ke exact winning numbers batao.", "lottery"),
    ("Meri exact death date kya hogi?", "death_date"),
    ("Mere future spouse ka exact naam kya hoga?", "spouse_name")
]

def run_hinglish_audit():
    print("=" * 80)
    print("AUDITING HINGLISH (HINDI + ENGLISH) QUESTION HANDLING")
    print("=" * 80 + "\n")

    for q, category in HINGLISH_TEST_CASES:
        print(f"Question: `{q}`")
        
        # Check boundary
        if is_unsupported_boundary_query(q):
            print(f"  Result -> Mode: `UNSUPPORTED` | Status: Out-of-bounds boundary intercepted | Calls: 0")
            print(f"  Response Preview: {get_unsupported_limitation_response(q)[:80]}...\n")
            continue
            
        # Check direct fact
        if is_direct_fact_query(q):
            fact_res = extract_direct_fact(q, MOCK_NORM_CHART, MOCK_DASHA)
            if fact_res:
                print(f"  Result -> Mode: `{fact_res['answer_mode']}` | Source: `DIRECT_FACT_ENGINE` | Calls: 0")
                print(f"  Response Preview: {fact_res['answer'][:100]}...\n")
                continue
                
        # LLM Assisted / Rule Based
        mode_info = select_answer_mode("career" if "career" in q.lower() else "general", "general", q)
        pkg = build_compact_evidence_package("career" if "career" in q.lower() else "general", "general", MOCK_NORM_CHART, [], MOCK_DASHA, question=q)
        prompt_text, in_tok = format_llm_assisted_prompt(q, pkg)
        print(f"  Result -> Mode: `{mode_info['mode']}` | Source: `{mode_info['answer_source']}` | Calls: `{mode_info['gemini_calls']}`")
        print(f"  Formatted Prompt Sent to Gemini:\n{prompt_text}\n")

if __name__ == "__main__":
    run_hinglish_audit()
