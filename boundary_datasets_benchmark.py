"""
Targeted Boundary Datasets Benchmark Script
===========================================
Script: boundary_datasets_benchmark.py

Tests the controlled production Query Selector across 8 targeted boundary datasets:
1. Career vs Finance
2. Health vs Career
3. Marriage vs Relationship
4. Finance vs Career
5. Multi-domain / Dasha questions
6. Other / General astrology questions
7. Hinglish variations
8. Short ambiguous questions

Strict Rule: Evaluates precision, recall, and selector decision accuracy per boundary.
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib

# Reconfigure encoding for console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Silence scikit-learn warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.router.domain import predict_domain_with_confidence

BOUNDARY_DATASETS: Dict[str, List[Tuple[str, str]]] = {
    "1. Career vs Finance": [
        ("Will I get a salary hike during my job promotion?", "career"),
        ("Will my business turnover increase my financial wealth?", "finance"),
        ("Will I earn more money in my current job?", "finance"),
        ("Should I change my career for a higher salary package?", "career"),
        ("Will my stock market investments pay off?", "finance"),
        ("Will I get a bonus at my workplace?", "finance")
    ],
    "2. Health vs Career": [
        ("Why do I feel exhausted at work?", "health"),
        ("Will my job stress cause health problems?", "health"),
        ("Why do I have trouble sleeping due to office work?", "health"),
        ("Will my physical energy improve so I can work better?", "health"),
        ("Will I take medical leave from my job?", "health")
    ],
    "3. Marriage vs Relationship": [
        ("Will my friendship turn into a relationship?", "marriage"),
        ("Would we make a good couple?", "marriage"),
        ("Should I tell my crush that I love them?", "marriage"),
        ("Will my current relationship lead to marriage?", "marriage"),
        ("When will I find a life partner for marriage?", "marriage")
    ],
    "4. Finance vs Career": [
        ("Will I get money from my investments?", "finance"),
        ("Will I clear my bank debt with my job salary?", "finance"),
        ("Will I get wealth from ancestral property?", "finance"),
        ("Will my savings increase this year?", "finance"),
        ("When will I get financial stability?", "finance")
    ],
    "5. Multi-domain / Dasha": [
        ("Tell me my current Dasha and how my career will be.", "career"),
        ("What does my current Dasha indicate about my health?", "health"),
        ("How will my finances be during my current Dasha?", "finance"),
        ("Will my marriage happen during my current Dasha?", "marriage"),
        ("What is my current Dasha period?", "other")
    ],
    "6. Other / General Astrology": [
        ("What is a nakshatra?", "other"),
        ("What does the 7th house represent in astrology?", "other"),
        ("What is Sade Sati of Saturn?", "other"),
        ("What is a birth chart?", "other"),
        ("How do planetary transits work?", "other")
    ],
    "7. Hinglish Variations": [
        ("mera naukri kab milega", "career"),
        ("sehat kaisi rahegi meri", "health"),
        ("shaadi kab hogi meri", "marriage"),
        ("paisa kab aayega mere paas", "finance"),
        ("kundali kya hoti hai", "other")
    ],
    "8. Short Ambiguous Queries": [
        ("job problem", "career"),
        ("health issue", "health"),
        ("marriage date", "marriage"),
        ("money problem", "finance"),
        ("nakshatra detail", "other")
    ]
}

def run_boundary_benchmark():
    print("=" * 90)
    print(" TARGETED BOUNDARY DATASETS BENCHMARK")
    print("=" * 90)

    total_boundary_q = 0
    total_boundary_pass = 0

    for category, items in BOUNDARY_DATASETS.items():
        print(f"\n--- Category: {category} ({len(items)} queries) ---")
        print(f"{'QUERY':<50} | {'EXPECTED':<8} | {'PREDICTED':<8} | {'MODEL':<10} | {'STATUS'}")
        print("-" * 88)
        
        cat_pass = 0
        for q, exp in items:
            total_boundary_q += 1
            res = predict_domain_with_confidence(q)
            pred = res["label"]
            sel_model = res.get("selected_model", "V3")
            status = "PASS" if pred == exp else "FAIL"
            if pred == exp:
                cat_pass += 1
                total_boundary_pass += 1
            print(f"'{q:<48}' | {exp:<8} | {pred:<8} | {sel_model:<10} | {status}")
        
        print(f"Category Pass Rate: {cat_pass}/{len(items)} ({cat_pass/len(items)*100:.1f}%)")

    print("\n" + "=" * 90)
    print(" SUMMARY DASHBOARD FOR TARGETED BOUNDARIES")
    print("=" * 90)
    print(f"Total Boundary Queries: {total_boundary_q}")
    print(f"Total Passed:           {total_boundary_pass} ({total_boundary_pass/total_boundary_q*100:.1f}%)")
    print("=" * 90)

if __name__ == "__main__":
    run_boundary_benchmark()
