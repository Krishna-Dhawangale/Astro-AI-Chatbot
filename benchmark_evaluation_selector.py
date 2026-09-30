"""
Multi-System Domain Evaluation & Benchmark Comparison
=====================================================
Script: benchmark_evaluation_selector.py

Evaluates and compares 8 distinct domain classification approaches:
1. OLD model alone
2. CONTEXT_V1 model alone
3. V3 model alone
4. Majority Voting Ensemble
5. Probability Averaging Ensemble
6. Evaluation-Weighted Ensemble
7. Stacking Meta-Classifier
8. Query-Aware Model Selector (model_selector.py)

Outputs comprehensive metrics:
- Overall Accuracy, Macro F1, Weighted F1
- Per-domain Precision, Recall, F1 for Career, Finance, Health, Marriage, Other
- Abstention rate & confusion matrices
- Production Decision Dashboard

Strict Principle: READ-ONLY evaluation. No model files or production router files modified.
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np

# Reconfigure encoding for console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Silence scikit-learn unpickling warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.router.model_selector import (
    load_domain_model_pool,
    predict_single_pair,
    evaluate_and_select_domain,
    DOMAIN_CLASSES
)

# ------------------------------------------------------------------
# COMPREHENSIVE BENCHMARK DATASET (Career, Health, Marriage, Finance, Other)
# ------------------------------------------------------------------
BENCHMARK_DATASET = [
    # --- Health Domain ---
    ("Why do I have low energy lately?", "health"),
    ("Why do I feel weak these days?", "health"),
    ("Why do I keep feeling exhausted?", "health"),
    ("Why am I having trouble sleeping?", "health"),
    ("Why can't I sleep properly at night?", "health"),
    ("Will my sleep improve soon?", "health"),
    ("Why do I wake up several times during the night?", "health"),
    ("Will my energy improve soon?", "health"),

    # --- Marriage / Relationship Domain ---
    ("I like my friend should I ask her out", "marriage"),
    ("Should I tell my crush that I like them", "marriage"),
    ("Will my friendship become a relationship", "marriage"),
    ("Will my current relationship work out", "marriage"),
    ("Would we make a good couple", "marriage"),
    ("When will I get married?", "marriage"),
    ("What does my 7th house say about marriage?", "marriage"),

    # --- Career Domain ---
    ("Will I get promoted?", "career"),
    ("Will my career improve during my current Dasha?", "career"),
    ("When will I get a new job?", "career"),
    ("Should I switch my job?", "career"),
    ("How will my business grow?", "career"),
    ("Which career suits me best?", "career"),
    ("Will I pass my job interview?", "career"),

    # --- Finance Domain ---
    ("Will my income increase?", "finance"),
    ("When will I gain wealth?", "finance"),
    ("How can I improve my financial situation?", "finance"),
    ("Will I get money from investments?", "finance"),
    ("Will I be rich?", "finance"),

    # --- Other / General Domain ---
    ("What is a nakshatra?", "other"),
    ("What is a birth chart?", "other"),
    ("Tell me about astrology", "other"),
    ("What does ascendant mean?", "other"),
]

def run_majority_voting(predictions: List[str]) -> str:
    votes = {}
    for p in predictions:
        votes[p] = votes.get(p, 0) + 1
    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]

def run_probability_averaging(probs_list: List[Dict[str, float]]) -> str:
    avg_probs = {cls: 0.0 for cls in DOMAIN_CLASSES}
    for p_map in probs_list:
        for cls in DOMAIN_CLASSES:
            avg_probs[cls] += p_map.get(cls, 0.0) / len(probs_list)
    return max(avg_probs.keys(), key=lambda cls: avg_probs[cls])

def run_weighted_ensemble(probs_list: List[Dict[str, float]]) -> str:
    # Model weights derived from empirical validation accuracy (OLD: 0.30, CONTEXT_V1: 0.50, V3: 0.90)
    weights = [0.30, 0.50, 0.90]
    total_w = sum(weights)
    weighted_probs = {cls: 0.0 for cls in DOMAIN_CLASSES}
    for p_map, w in zip(probs_list, weights):
        for cls in DOMAIN_CLASSES:
            weighted_probs[cls] += (p_map.get(cls, 0.0) * w) / total_w
    return max(weighted_probs.keys(), key=lambda cls: weighted_probs[cls])

def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total if total > 0 else 0.0

    abstentions = sum(1 for yp in y_pred if yp == "uncertain")
    abstention_rate = abstentions / total if total > 0 else 0.0

    domain_metrics = {}
    for d in DOMAIN_CLASSES:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == d and yp == d)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != d and yp == d)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == d and yp != d)
        support = sum(1 for yt in y_true if yt == d)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        domain_metrics[d] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }

    valid_classes = [d for d in DOMAIN_CLASSES if domain_metrics[d]["support"] > 0]
    macro_f1 = sum(domain_metrics[d]["f1"] for d in valid_classes) / len(valid_classes)
    weighted_f1 = sum(domain_metrics[d]["f1"] * domain_metrics[d]["support"] for d in valid_classes) / total

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "abstention_rate": abstention_rate,
        "domain_metrics": domain_metrics
    }

def run_benchmark():
    print("=" * 80)
    print(" MULTI-SYSTEM DOMAIN EVALUATION & BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"Dataset Size: {len(BENCHMARK_DATASET)} Labeled Questions")
    print("Systems Evaluated: 8 Distinct Domain Classification Strategies")
    print("Production Status: READ-ONLY (No models or router files modified)\n")

    pool = load_domain_model_pool()
    if len(pool) < 3:
        print("[ERROR] Failed to load all 3 domain models. Aborting.")
        return

    systems = [
        "1. OLD",
        "2. CONTEXT_V1",
        "3. V3",
        "4. MAJORITY_VOTING",
        "5. PROBABILITY_AVG",
        "6. WEIGHTED_ENSEMBLE",
        "7. STACKING_META",
        "8. QUERY_SELECTOR"
    ]

    results: Dict[str, Dict[str, List[str]]] = {s: {"y_true": [], "y_pred": []} for s in systems}

    for question, expected in BENCHMARK_DATASET:
        # Base predictions
        old_res = predict_single_pair(pool["OLD"][0], pool["OLD"][1], question)
        ctx_res = predict_single_pair(pool["CONTEXT_V1"][0], pool["CONTEXT_V1"][1], question)
        v3_res = predict_single_pair(pool["V3"][0], pool["V3"][1], question)

        preds_list = [old_res["domain"], ctx_res["domain"], v3_res["domain"]]
        probs_list = [old_res["probabilities"], ctx_res["probabilities"], v3_res["probabilities"]]

        # Ensembles
        maj_domain = run_majority_voting(preds_list)
        avg_domain = run_probability_averaging(probs_list)
        weight_domain = run_weighted_ensemble(probs_list)
        
        # Stacking approximation (Weighted + V3 priority)
        stack_domain = v3_res["domain"] if v3_res["confidence"] > 0.60 else weight_domain

        # Query Selector
        selector_res = evaluate_and_select_domain(question)
        selector_domain = selector_res["selected_domain"]

        # Record
        pairs_mapping = [
            ("1. OLD", old_res["domain"]),
            ("2. CONTEXT_V1", ctx_res["domain"]),
            ("3. V3", v3_res["domain"]),
            ("4. MAJORITY_VOTING", maj_domain),
            ("5. PROBABILITY_AVG", avg_domain),
            ("6. WEIGHTED_ENSEMBLE", weight_domain),
            ("7. STACKING_META", stack_domain),
            ("8. QUERY_SELECTOR", selector_domain)
        ]

        for sys_name, pred in pairs_mapping:
            results[sys_name]["y_true"].append(expected)
            results[sys_name]["y_pred"].append(pred)

    # Calculate metrics
    metrics = {s: calculate_metrics(results[s]["y_true"], results[s]["y_pred"]) for s in systems}

    # 1. OVERALL DASHBOARD
    print("=" * 80)
    print(f"{'SYSTEM / STRATEGY':<24} | {'ACCURACY':<10} | {'MACRO F1':<10} | {'WEIGHTED F1':<12} | {'ABSTAIN %'}")
    print("-" * 75)
    for s in systems:
        m = metrics[s]
        print(f"{s:<24} | {m['accuracy']*100:6.2f}%    | {m['macro_f1']:6.4f}    | {m['weighted_f1']:10.4f}   | {m['abstention_rate']*100:5.1f}%")

    # 2. PER-DOMAIN COMPARISON SUMMARY
    print("\n" + "=" * 80)
    print(" PER-DOMAIN F1-SCORE BREAKDOWN")
    print("=" * 80)
    print(f"{'SYSTEM / STRATEGY':<24} | {'CAREER':<8} | {'HEALTH':<8} | {'MARRIAGE':<9} | {'FINANCE':<8} | {'OTHER':<8}")
    print("-" * 75)

    for s in systems:
        dm = metrics[s]["domain_metrics"]
        print(
            f"{s:<24} | "
            f"{dm['career']['f1']:6.4f}   | "
            f"{dm['health']['f1']:6.4f}   | "
            f"{dm['marriage']['f1']:7.4f}   | "
            f"{dm['finance']['f1']:6.4f}   | "
            f"{dm['other']['f1']:6.4f}"
        )

    # 3. BEST VALIDATED STRATEGY IDENTIFICATION
    best_system = max(systems, key=lambda s: (metrics[s]["accuracy"], metrics[s]["macro_f1"]))
    print("\n" + "=" * 80)
    print(" EVALUATION CONCLUSION")
    print("=" * 80)
    print(f"BEST VALIDATED STRATEGY: {best_system}")
    print(f"Accuracy: {metrics[best_system]['accuracy']*100:.2f}% | Macro F1: {metrics[best_system]['macro_f1']:.4f}")
    print("STATUS: NOT ACTIVATED (Awaiting explicit user approval before modifying router)")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
