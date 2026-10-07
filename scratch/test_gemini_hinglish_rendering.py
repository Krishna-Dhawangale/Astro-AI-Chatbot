"""
Test Script for Gemini Hinglish Rendering
===========================================
Runs complex Hinglish domain queries through Gemini 3.5 Flash Lite using our Micro Protocol payload.
"""

import sys
import os
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.reasoning.llm_renderer import (
    build_compact_evidence_package,
    format_llm_assisted_prompt,
    get_renderer_system_instruction,
    select_renderer_tier
)

MOCK_NORM_CHART = {
    "ascendant": {"rashi": "Leo", "sign": "Leo"},
    "planets": {
        "Moon": {"rashi": "Scorpio", "sign": "Scorpio"},
        "Sun": {"rashi": "Aries", "sign": "Aries"},
        "Mercury": {"rashi": "Gemini", "sign": "Gemini", "house": 1},
        "Venus": {"rashi": "Pisces", "sign": "Pisces", "house": 8},
        "Jupiter": {"rashi": "Libra", "sign": "Libra", "house": 11}
    }
}

MOCK_DASHA = {
    "current_mahadasha": "Mars",
    "current_antardasha": "Rahu"
}

MOCK_RULES = [
    {"rule_id": "CAREER_10TH_LORD_PLACEMENT", "description": "10th lord Mercury in 1st house connects professional identity with personal drive and initiative", "matched": True},
    {"rule_id": "CAREER_DASHA_ACTIVATION", "description": "Active Mars-Rahu period brings high ambition, enterprise, and new professional opportunities", "matched": True}
]

def test_gemini_hinglish():
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[SKIP] GEMINI_API_KEY not found; skipping live Gemini execution.")
        return

    client = genai.Client(api_key=api_key)
    
    hinglish_domain_queries = [
        "Mera career kaisa rahega according to my chart?",
        "Current Dasha mere career ko kaise affect karega?"
    ]

    print("=" * 80)
    print("LIVE GEMINI 3.5 FLASH LITE HINGLISH RESPONSE RENDERING")
    print("=" * 80 + "\n")

    for q in hinglish_domain_queries:
        pkg = build_compact_evidence_package("career", "career_general", MOCK_NORM_CHART, MOCK_RULES, MOCK_DASHA, question=q)
        prompt_text, in_tok = format_llm_assisted_prompt(q, pkg)
        tier, max_out, word_lim = select_renderer_tier(q)
        sys_inst = get_renderer_system_instruction(word_lim)

        print(f"User Query: `{q}`")
        print(f"Micro Prompt Sent:\n{prompt_text}\n")
        
        try:
            config = types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=0.2,
                max_output_tokens=max_out,
            )
            resp = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt_text,
                config=config
            )
            print("Gemini Response:")
            print(resp.text)
            print("-" * 80 + "\n")
        except Exception as e:
            print(f"Gemini execution error: {e}\n")

if __name__ == "__main__":
    test_gemini_hinglish()
