"""
Show Progressive Question Answers
=================================
Script: show_progressive_question_answers.py

Runs the full deterministic pipeline & Stage 8.25 answer synthesizer
on sample natal chart data to print the exact structured text answer
generated for each of the 5 progressive career queries:

1. "Which career suits me?"
2. "Which career suits me based on my chart?"
3. "Which career suits me during my current Dasha?"
4. "Which career should I pursue right now?"
5. "Which career should I pursue in my current Mahadasha and Antardasha?"
"""

import sys
import json
from pathlib import Path

# Force UTF-8 stdout encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.reasoning.pipeline_helper import execute_full_deterministic_pipeline
from backend.reasoning.answer_synthesizer import synthesize_structured_answer
from backend.router.domain import predict_domain_with_confidence
from backend.router.intent import predict_intent_with_confidence

# Sample Natal Chart (Aries Ascendant, Saturn in Capricorn 10th house, Sun+Mercury 10th, Exalted Moon 2nd, Exalted Jupiter 4th)
SAMPLE_NATAL_CHART = {
    "name": "Aries Ascendant with Saturn 10th House (Capricorn)",
    "ascendant": {"longitude": 12.5, "rashi": "Aries", "degree_in_rashi": 12.5},
    "planets": {
        "Sun": {"rashi": "Capricorn", "longitude": 285.0, "house": 10},
        "Moon": {"rashi": "Taurus", "longitude": 45.0, "house": 2},
        "Mars": {"rashi": "Aries", "longitude": 15.0, "house": 1},
        "Mercury": {"rashi": "Capricorn", "longitude": 290.0, "house": 10},
        "Jupiter": {"rashi": "Cancer", "longitude": 95.0, "house": 4},
        "Venus": {"rashi": "Pisces", "longitude": 350.0, "house": 12},
        "Saturn": {"rashi": "Capricorn", "longitude": 280.0, "house": 10},
        "Rahu": {"rashi": "Taurus", "longitude": 50.0, "house": 2},
        "Ketu": {"rashi": "Scorpio", "longitude": 230.0, "house": 8}
    }
}

def print_answers():
    questions = [
        "Which career suits me?",
        "Which career suits me based on my chart?",
        "Which career suits me during my current Dasha?",
        "Which career should I pursue right now?",
        "Which career should I pursue in my current Mahadasha and Antardasha?"
    ]

    pipeline_res = execute_full_deterministic_pipeline(SAMPLE_NATAL_CHART)
    c_rules = pipeline_res["stage_8_15_career_rules"]["rules"]
    matched_rules = [r for r in c_rules if r.get("matched")]

    print("=" * 100)
    print(" DYNAMIC GENERATED ANSWERS ACROSS PROGRESSIVE TIMING QUESTIONS")
    print("=" * 100)
    print(f"Chart: {SAMPLE_NATAL_CHART['name']}\n")

    for idx, q in enumerate(questions, 1):
        dom_res = predict_domain_with_confidence(q)
        intent_res = predict_intent_with_confidence(q)
        
        synth = synthesize_structured_answer(
            domain="career",
            matched_rules=matched_rules,
            timing_data=pipeline_res.get("stage_8_20_dasha_timing", {}),
            question=q
        )

        print("-" * 100)
        print(f"QUESTION #{idx}: \"{q}\"")
        print(f"Domain: {dom_res['label']} | Model: {dom_res.get('selected_model', 'V3')} | Resolved Intent: {intent_res['resolved_intent']} ({intent_res['resolution_reason']})")
        print(f"Source: deterministic_reasoning | Gemini Calls: 0 | Evidence Score: {synth['evidence_score']} | Status: {synth['conflict_status']}")
        print("-" * 100)
        print("GENERATED ANSWER TEXT:")
        print(synth["structured_text"])
        print("\n")

if __name__ == "__main__":
    print_answers()
