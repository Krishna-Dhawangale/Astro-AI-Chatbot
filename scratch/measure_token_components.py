"""
Token Component Measurement Audit Script
===========================================
Measures exact token component counts for Micro Protocol prompt formatting.
"""

import sys
import os
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.reasoning.llm_renderer import (
    get_renderer_system_instruction,
    select_renderer_tier,
    build_compact_evidence_package,
    format_llm_assisted_prompt,
    estimate_tokens,
    compress_rule_conclusion
)

MOCK_CHART = {
    "ascendant": {"rashi": "Leo", "sign": "Leo"},
    "planets": {
        "Mercury": {"rashi": "Gemini", "sign": "Gemini", "house": 1},
        "Moon": {"rashi": "Scorpio", "sign": "Scorpio"},
        "Sun": {"rashi": "Aries", "sign": "Aries"},
        "Venus": {"rashi": "Pisces", "sign": "Pisces", "house": 8},
        "Jupiter": {"rashi": "Libra", "sign": "Libra", "house": 11}
    }
}

MOCK_DASHA = {
    "current_mahadasha": "Mars",
    "current_antardasha": "Rahu"
}

MOCK_CAREER_RULES = [
    {"rule_id": "CAREER_10TH_LORD_PLACEMENT", "description": "10th lord Mercury in 1st house connects professional identity with personal drive and initiative", "matched": True},
    {"rule_id": "CAREER_DASHA_ACTIVATION", "description": "Active Mars-Rahu period brings high ambition, enterprise, and new professional opportunities", "matched": True}
]

def audit_query(question, domain, intent, matched_rules):
    tier, max_out_tokens, word_limit_str = select_renderer_tier(question, intent=intent)
    sys_inst = get_renderer_system_instruction(word_limit_str)
    
    pkg = build_compact_evidence_package(
        domain=domain,
        intent=intent,
        chart_data=MOCK_CHART,
        matched_rules=matched_rules,
        dasha_hierarchy=MOCK_DASHA,
        question=question
    )
    
    facts = pkg.get("verified_facts", {})
    rules = pkg.get("selected_rules", [])
    
    facts_str = "; ".join(f"{k}={v}" for k, v in facts.items())
    rules_str = "; ".join(compress_rule_conclusion(r) for r in rules)
    
    sys_tokens = estimate_tokens(sys_inst)
    q_tokens = estimate_tokens(f"Q: {question}")
    f_tokens = estimate_tokens(f"F: {facts_str}")
    r_tokens = estimate_tokens(f"R: {rules_str}")
    wrapper_tokens = estimate_tokens("\n\n\n")
    history_tokens = 0
    
    prompt_text, total_user_tokens = format_llm_assisted_prompt(question, pkg)
    total_input_tokens = sys_tokens + total_user_tokens
    
    print("==================================================")
    print(f"QUESTION      : {question}")
    print(f"TIER          : {tier} (Target: {word_limit_str}, Max Out Tokens: {max_out_tokens})")
    print("--------------------------------------------------")
    print(f"System Tokens : {sys_tokens} tokens")
    print(f"Question Toks : {q_tokens} tokens ('Q: {question}')")
    print(f"Facts Tokens  : {f_tokens} tokens ('F: {facts_str}')")
    print(f"Rules Tokens  : {r_tokens} tokens ('R: {rules_str}')")
    print(f"Wrapper Toks  : {wrapper_tokens} tokens")
    print(f"History Toks  : {history_tokens} tokens")
    print("--------------------------------------------------")
    print(f"User Prompt   : {total_user_tokens} tokens")
    print(f"TOTAL INPUT   : {total_input_tokens} tokens")
    print("==================================================")
    print("PROMPT TEXT SENT TO GEMINI:")
    print(prompt_text)
    print("==================================================\n")

if __name__ == "__main__":
    audit_query(
        "What does my chart indicate about career growth?",
        "career",
        "career_general",
        MOCK_CAREER_RULES
    )
    audit_query(
        "What are the strongest career indicators in my chart?",
        "career",
        "career_indicators",
        MOCK_CAREER_RULES[:1]
    )
