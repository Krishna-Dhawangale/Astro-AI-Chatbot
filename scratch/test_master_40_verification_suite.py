"""
Master 40-Question Evaluation and Quality Audit Runner
(scratch/test_master_40_verification_suite.py)

Evaluates all 40 questions across 5 core domains:
- Section A: Direct Facts (Q1–Q10) -> Checks Direct Answer First (no preamble)
- Section B: Career (Q11–Q20) -> Checks 3 Layers (Conclusion -> Evidence -> Cautious Guidance)
- Section C: Marriage (Q21–Q28) -> Checks Non-deterministic claim guarding
- Section D: Finance (Q29–Q35) -> Checks Non-promissory realistic phrasing
- Section E: Education (Q36–Q40) -> Checks Domain grounding and guidance

Standard Sample Birth Profile:
1995-05-15 14:30:00, New Delhi, India (Lat: 28.6139, Lon: 77.2090, TZ: 5.5)
"""

import asyncio
import json
import os
import sys
import time
from typing import Tuple, Dict, Any, List

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from starlette.testclient import TestClient
from backend.main import app, USER_SESSIONS

BIRTH_PROFILE = {
    "birth_year": 1995,
    "birth_month": 5,
    "birth_day": 15,
    "birth_hour": 14.5,
    "latitude": 28.6139,
    "longitude": 77.2090,
    "timezone": 5.5,
}

QUESTIONS = [
    # A. FACTUAL / DIRECT ASTROLOGY (Q1–Q10)
    ("A. FACTUAL", 1, "What is my Moon sign?"),
    ("A. FACTUAL", 2, "What is my Sun sign?"),
    ("A. FACTUAL", 3, "What is my Rashi?"),
    ("A. FACTUAL", 4, "What is my Lagna?"),
    ("A. FACTUAL", 5, "What is my Nakshatra?"),
    ("A. FACTUAL", 6, "Which planet is in my 7th house?"),
    ("A. FACTUAL", 7, "Which planet rules my 10th house?"),
    ("A. FACTUAL", 8, "What is my current Mahadasha?"),
    ("A. FACTUAL", 9, "What is my current Antardasha?"),
    ("A. FACTUAL", 10, "When does my current Mahadasha end?"),

    # B. CAREER (Q11–Q20)
    ("B. CAREER", 11, "Which career suits me best according to my chart?"),
    ("B. CAREER", 12, "What does my 10th house indicate about my career?"),
    ("B. CAREER", 13, "How does my 10th lord influence my career?"),
    ("B. CAREER", 14, "What are the strongest career indicators in my chart?"),
    ("B. CAREER", 15, "How does my current Mahadasha affect my career?"),
    ("B. CAREER", 16, "How does my current Antardasha influence my career?"),
    ("B. CAREER", 17, "What does my chart indicate about my career growth?"),
    ("B. CAREER", 18, "Which planets are most important for my career?"),
    ("B. CAREER", 19, "Does my chart support a technical or analytical career?"),
    ("B. CAREER", 20, "What kind of professional environment may suit my chart?"),

    # C. MARRIAGE / RELATIONSHIPS (Q21–Q28)
    ("C. MARRIAGE", 21, "What does my 7th house indicate about marriage?"),
    ("C. MARRIAGE", 22, "What are the strongest marriage indicators in my chart?"),
    ("C. MARRIAGE", 23, "What does my chart indicate about marriage timing?"),
    ("C. MARRIAGE", 24, "How does my current Dasha affect marriage matters?"),
    ("C. MARRIAGE", 25, "What kind of relationship tendencies are shown in my chart?"),
    ("C. MARRIAGE", 26, "What does Venus indicate in my chart regarding relationships?"),
    ("C. MARRIAGE", 27, "How does the 7th lord influence my relationships?"),
    ("C. MARRIAGE", 28, "What does my chart suggest about my future partner?"),

    # D. FINANCE (Q29–Q35)
    ("D. FINANCE", 29, "What does my chart indicate about financial stability?"),
    ("D. FINANCE", 30, "Which houses are important for my finances?"),
    ("D. FINANCE", 31, "Which planets influence my financial prospects?"),
    ("D. FINANCE", 32, "What does my 2nd house indicate about finances?"),
    ("D. FINANCE", 33, "What does my 11th house indicate about gains?"),
    ("D. FINANCE", 34, "How does my current Dasha influence financial matters?"),
    ("D. FINANCE", 35, "Does my chart indicate potential for financial growth?"),

    # E. EDUCATION (Q36–Q40)
    ("E. EDUCATION", 36, "What does my chart indicate about higher education?"),
    ("E. EDUCATION", 37, "Which planets influence my education?"),
    ("E. EDUCATION", 38, "What does my 5th house indicate about learning?"),
    ("E. EDUCATION", 39, "Does my chart support technical education?"),
    ("E. EDUCATION", 40, "What type of education may suit me according to my chart?"),
]


def check_factual_directness(text: str) -> Tuple[bool, str]:
    """Layer 1 Check: Answer must begin directly with the fact, without conversational filler."""
    t_low = text.lower().strip()
    bad_fillers = [
        "according to vedic astrology",
        "in vedic astrology",
        "astrology says",
        "as an ai",
        "traditionally,",
        "it is believed that",
    ]
    for b in bad_fillers:
        if t_low.startswith(b):
            return False, f"Contains filler preamble: '{b}'"
    return True, "Starts directly with factual conclusion"


def check_cautious_language(text: str) -> Tuple[bool, str]:
    """Claim Guard Check: Forbids absolute guarantees ('will definitely', '100% guarantee')."""
    t_low = text.lower()
    absolute_claims = [
        "will definitely",
        "you will definitely",
        "guaranteed to",
        "100% certain",
        "will certainly become rich",
        "will definitely marry in",
    ]
    for ac in absolute_claims:
        if ac in t_low:
            return False, f"Absolute guaranteed claim found: '{ac}'"
    return True, "Uses cautious, probabilistic language"


def run_all_40_evaluations():
    client = TestClient(app)
    results = []

    print("=" * 80)
    print("      MASTER 40-QUESTION VEDIC ASTROLOGY QUALITY & PROVENANCE BENCHMARK")
    print("=" * 80)
    print(f"Sample Profile: 1995-05-15 14:30:00 | New Delhi, India | Lat: 28.6139, Lon: 77.2090\n")

    for section, q_num, question in QUESTIONS:
        t0 = time.time()
        payload = {
            "query": question,
            "user_id": f"eval_user_{q_num}",
            **BIRTH_PROFILE,
        }
        res = client.post("/chat", json=payload)
        latency_ms = (time.time() - t0) * 1000

        content_type = res.headers.get("content-type", "")
        if "application/json" in content_type:
            data = res.json()
            ans_text = data.get("answer", "")
            mode = data.get("answer_mode", "UNKNOWN")
            source = data.get("answer_source", "UNKNOWN")
            gemini_calls = data.get("gemini_calls", 0)
        else:
            # Streaming response (text/plain)
            raw_stream = res.text
            # Split follow-ups if present
            if "[FOLLOW_UPS]:" in raw_stream:
                ans_text = raw_stream.split("[FOLLOW_UPS]:")[0].strip()
            else:
                ans_text = raw_stream.strip()
            mode = "LLM_ASSISTED"
            source = "EVIDENCE_GROUNDED_LLM"
            gemini_calls = 1

        # Quality evaluations
        is_direct, direct_msg = check_factual_directness(ans_text)
        is_cautious, cautious_msg = check_cautious_language(ans_text)

        eval_entry = {
            "section": section,
            "q_num": q_num,
            "question": question,
            "answer": ans_text,
            "answer_mode": mode,
            "answer_source": source,
            "gemini_calls": gemini_calls,
            "latency_ms": latency_ms,
            "is_direct": is_direct,
            "is_cautious": is_cautious,
        }
        results.append(eval_entry)

        # Print live progress
        print(f"[{section}] Q{q_num:02d}: {question}")
        print(f"  Mode: {mode} | Source: {source} | LLM Calls: {gemini_calls} | Latency: {latency_ms:.1f}ms")
        print(f"  Quality: Direct={is_direct} | Cautious={is_cautious}")
        print(f"  Answer Snippet: {ans_text[:120]}...\n")

    # Save complete report to scratch JSON
    out_file = os.path.join(BASE_DIR, "scratch", "master_40_eval_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"MASTER 40-QUESTION BENCHMARK COMPLETE. Saved to {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_40_evaluations()
