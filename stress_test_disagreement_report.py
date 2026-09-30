"""
Stress-Test & Per-Question Disagreement Verification Script
===========================================================
Script: stress_test_disagreement_report.py

Purpose:
Investigates whether the Evaluation-Driven QUERY_SELECTOR makes genuinely
adaptive model-selection decisions on disagreement questions or collapses to V3.

Tests:
1. Per-question model breakdown for OLD, CONTEXT_V1, V3.
2. Disagreement queries comparison across 5 ensemble/selector strategies.
3. Detailed logging of selected_model, selected_domain, selection_reason,
   reliability scores, margin, agreement, and semantic evidence.
4. Stress-testing across Career, Health, Finance, Marriage, and Mixed/Dasha categories.

Strict Rules: READ-ONLY evaluation. No model files or production code modified.
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

from backend.router.model_selector import (
    load_domain_model_pool,
    predict_single_pair,
    evaluate_and_select_domain,
    detect_query_evidence,
    DOMAIN_CLASSES
)
from benchmark_evaluation_selector import (
    run_majority_voting,
    run_probability_averaging,
    run_weighted_ensemble
)

# ------------------------------------------------------------------
# STRESS-TEST QUERY SUITE (Career, Health, Finance, Marriage, Mixed/Dasha)
# ------------------------------------------------------------------
STRESS_TEST_QUERIES = [
    # --- Career Category ---
    ("Which career suits me?", "career"),
    ("Will I get promoted?", "career"),
    ("Should I change my job?", "career"),
    ("Will my career improve?", "career"),
    ("When will I get a new job?", "career"),
    ("How will my business grow?", "career"),

    # --- Health Category ---
    ("Why do I feel exhausted?", "health"),
    ("Why am I having trouble sleeping?", "health"),
    ("Will my energy improve?", "health"),
    ("Why do I feel weak these days?", "health"),
    ("Why can't I sleep properly at night?", "health"),

    # --- Finance Category ---
    ("Will my income increase?", "finance"),
    ("How will my financial situation be?", "finance"),
    ("Will I gain wealth?", "finance"),
    ("Will I get money from investments?", "finance"),
    ("Will I be rich?", "finance"),

    # --- Marriage Category ---
    ("When will I get married?", "marriage"),
    ("Will my current relationship work out?", "marriage"),
    ("Would we make a good couple?", "marriage"),
    ("I like my friend should I ask her out", "marriage"),
    ("Will my friendship become a relationship", "marriage"),

    # --- Mixed / Dasha Category ---
    ("Tell me my current Dasha and how my career will be.", "career"),
    ("What does my current Dasha indicate about my health?", "health"),
    ("How will my finances be during my current Dasha?", "finance"),
    ("Will my marriage happen during my current Dasha?", "marriage")
]

def format_probs(probs_map: Dict[str, float]) -> str:
    sorted_p = sorted(probs_map.items(), key=lambda x: x[1], reverse=True)[:3]
    return ", ".join([f"{k}:{v:.2f}" for k, v in sorted_p])

def run_stress_test():
    print("=" * 90)
    print(" PER-QUESTION DISAGREEMENT & MODEL-SELECTION STRESS TEST REPORT")
    print("=" * 90)
    print("Models Evaluated: OLD, CONTEXT_V1, V3")
    print("Strategies Evaluated: MAJORITY, PROB_AVG, WEIGHTED, STACKING, QUERY_SELECTOR")
    print("Production Status: READ-ONLY (No models or router files modified)\n")

    pool = load_domain_model_pool()
    if len(pool) < 3:
        print("[ERROR] Could not load all 3 domain models. Aborting.")
        return

    disagreement_count = 0
    unanimous_count = 0
    selected_model_counts = {"OLD": 0, "CONTEXT_V1": 0, "V3": 0, "none": 0}

    print("-" * 90)
    print(f"{'#':<3} | {'QUERY':<48} | {'OLD':<8} | {'V1':<8} | {'V3':<8} | {'SELECTOR':<9} | {'MODEL'}")
    print("-" * 90)

    disagreement_details = []

    for idx, (question, exp_domain) in enumerate(STRESS_TEST_QUERIES, 1):
        old_res = predict_single_pair(pool["OLD"][0], pool["OLD"][1], question)
        v1_res = predict_single_pair(pool["CONTEXT_V1"][0], pool["CONTEXT_V1"][1], question)
        v3_res = predict_single_pair(pool["V3"][0], pool["V3"][1], question)

        preds = [old_res["domain"], v1_res["domain"], v3_res["domain"]]
        probs = [old_res["probabilities"], v1_res["probabilities"], v3_res["probabilities"]]

        maj_domain = run_majority_voting(preds)
        avg_domain = run_probability_averaging(probs)
        weight_domain = run_weighted_ensemble(probs)
        stack_domain = v3_res["domain"] if v3_res["confidence"] > 0.60 else weight_domain

        selector_res = evaluate_and_select_domain(question)
        sel_domain = selector_res["selected_domain"]
        sel_model = selector_res["selected_model"]
        sel_method = selector_res["selection_method"]

        is_disagreement = len(set(preds)) > 1
        if is_disagreement:
            disagreement_count += 1
            selected_model_counts[sel_model] = selected_model_counts.get(sel_model, 0) + 1
        else:
            unanimous_count += 1
            selected_model_counts[sel_model] = selected_model_counts.get(sel_model, 0) + 1

        flag = " [DISAGREE]" if is_disagreement else ""
        print(f"{idx:02d} | '{question:<46}' | {old_res['domain']:<8} | {v1_res['domain']:<8} | {v3_res['domain']:<8} | {sel_domain:<9} | {sel_model}{flag}")

        disagreement_details.append({
            "idx": idx,
            "question": question,
            "expected": exp_domain,
            "old": old_res,
            "v1": v1_res,
            "v3": v3_res,
            "majority": maj_domain,
            "prob_avg": avg_domain,
            "weighted": weight_domain,
            "stacking": stack_domain,
            "selector_domain": sel_domain,
            "selector_model": sel_model,
            "selector_method": sel_method,
            "selector_full": selector_res,
            "is_disagreement": is_disagreement
        })

    # ------------------------------------------------------------------
    # DETAILED DISAGREEMENT AUDIT TRAIL
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" DETAILED AUDIT TRAIL FOR DISAGREEMENT QUESTIONS")
    print("=" * 90)

    for item in disagreement_details:
        if not item["is_disagreement"]:
            continue

        q = item["question"]
        print(f"\n[DISAGREEMENT Q{item['idx']:02d}] \"{q}\" (Expected: {item['expected'].upper()})")
        print("-" * 80)
        print(f"  OLD Model:        Domain = {item['old']['domain']:<10} | Conf = {item['old']['confidence']:.4f} | Margin = {item['old']['margin']:.4f} | Top: [{format_probs(item['old']['probabilities'])}]")
        print(f"  CONTEXT_V1 Model: Domain = {item['v1']['domain']:<10} | Conf = {item['v1']['confidence']:.4f} | Margin = {item['v1']['margin']:.4f} | Top: [{format_probs(item['v1']['probabilities'])}]")
        print(f"  V3 Model:         Domain = {item['v3']['domain']:<10} | Conf = {item['v3']['confidence']:.4f} | Margin = {item['v3']['margin']:.4f} | Top: [{format_probs(item['v3']['probabilities'])}]")
        print("-" * 80)
        print("  STRATEGY DECISIONS:")
        print(f"    - Majority Voting:      {item['majority']}")
        print(f"    - Probability Avg:      {item['prob_avg']}")
        print(f"    - Weighted Ensemble:    {item['weighted']}")
        print(f"    - Stacking Meta:        {item['stacking']}")
        print(f"    - QUERY_SELECTOR:       {item['selector_domain']} (Chosen Model: {item['selector_model']})")
        print(f"    - Selection Reason:     {item['selector_method']}")

        # Print score breakdown from selector
        scores = item["selector_full"].get("score_details", [])
        print("  SELECTOR SCORE BREAKDOWN:")
        for sc in scores:
            print(f"      * {sc['model']:<10} -> Domain: {sc['domain']:<8} | Score: {sc['score']:.4f} | Reliability: {sc['reliability']:.2f} | Evidence Boost: +{sc['evidence_boost']:.2f}")

    # ------------------------------------------------------------------
    # SUMMARY DASHBOARD
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" STRESS-TEST SUMMARY DASHBOARD")
    print("=" * 90)
    print(f"Total Queries Evaluated:          {len(STRESS_TEST_QUERIES)}")
    print(f"Unanimous Agreement Queries:       {unanimous_count} ({unanimous_count/len(STRESS_TEST_QUERIES)*100:.1f}%)")
    print(f"Disagreement Queries:             {disagreement_count} ({disagreement_count/len(STRESS_TEST_QUERIES)*100:.1f}%)")
    print("\nSelected Model Distribution (across all queries):")
    for m_name, count in selected_model_counts.items():
        print(f"  - {m_name:<12}: {count} queries ({count/len(STRESS_TEST_QUERIES)*100:.1f}%)")
    print("=" * 90)

if __name__ == "__main__":
    run_stress_test()
