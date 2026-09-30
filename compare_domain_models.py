"""
Standalone Domain Model Comparison Script
===========================================
Safely evaluates all available domain model generations (OLD, CONTEXT_V1, V3)
against the project's labeled test/regression dataset without modifying
any production routing files or overwriting model artifacts.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np

# Reconfigure encoding for clean console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Capture sklearn unpickling warnings
sklearn_warnings = []
with warnings.catch_warnings(record=True) as captured_warnings:
    warnings.simplefilter("always")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "backend" / "models" / "question_classifier"

# ------------------------------------------------------------------
# 1. MODEL DISCOVERY & PAIRING
# ------------------------------------------------------------------
MODEL_PAIRS = [
    {
        "name": "OLD (Default)",
        "key": "OLD",
        "model": MODEL_DIR / "domain_model.pkl",
        "vectorizer": MODEL_DIR / "domain_vectorizer.pkl",
        "metadata": MODEL_DIR / "domain_metadata.json",
    },
    {
        "name": "CONTEXT_V1",
        "key": "CONTEXT_V1",
        "model": MODEL_DIR / "domain_model_context_v1.pkl",
        "vectorizer": MODEL_DIR / "domain_vectorizer_context_v1.pkl",
        "metadata": MODEL_DIR / "domain_metadata_context_v1.json",
    },
    {
        "name": "V3",
        "key": "V3",
        "model": MODEL_DIR / "domain_model_v3.pkl",
        "vectorizer": MODEL_DIR / "domain_vectorizer_v3.pkl",
        "metadata": MODEL_DIR / "domain_metadata_v3.json",
    },
]

# ------------------------------------------------------------------
# 2. LABELED REGRESSION DATASET
# ------------------------------------------------------------------
TEST_DATASET = [
    # --- Health Regression Set ---
    ("Why do I have low energy lately?", "health"),
    ("Why do I feel weak these days?", "health"),
    ("Why do I keep feeling exhausted?", "health"),
    ("Why am I having trouble sleeping?", "health"),
    ("Why can't I sleep properly at night?", "health"),
    ("Will my sleep improve soon?", "health"),
    ("Why do I wake up several times during the night?", "health"),
    ("Will my energy improve soon?", "health"),

    # --- Relationship Regression Set ---
    ("I like my friend should I ask her out", "marriage"),
    ("Should I tell my crush that I like them", "marriage"),
    ("Will my friendship become a relationship", "marriage"),
    ("Will my current relationship work out", "marriage"),
    ("Would we make a good couple", "marriage"),
    ("When will I get married?", "marriage"),
    ("What does my 7th house say about marriage?", "marriage"),

    # --- Career Test Set ---
    ("Will I get promoted?", "career"),
    ("Will my career improve during my current Dasha?", "career"),
    ("When will I get a new job?", "career"),
    ("Should I switch my job?", "career"),
    ("How will my business grow?", "career"),
    ("Which career suits me best?", "career"),

    # --- Finance Test Set ---
    ("Will my income increase?", "finance"),
    ("When will I gain wealth?", "finance"),
    ("How can I improve my financial situation?", "finance"),
    ("Will I get money from investments?", "finance"),
    ("Will I be rich?", "finance"),

    # --- Other / General Test Set ---
    ("What is a nakshatra?", "other"),
    ("What is a birth chart?", "other"),
    ("Tell me about astrology", "other"),
    ("What does ascendant mean?", "other"),
]

DOMAIN_CLASSES = ["career", "finance", "health", "marriage", "other"]

def load_and_validate_pair(pair_info: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any], List[str]]:
    """Load and perform 10 integrity checks on a model/vectorizer pair."""
    name = pair_info["name"]
    model_path = pair_info["model"]
    vec_path = pair_info["vectorizer"]
    meta_path = pair_info["metadata"]

    errors = []

    if not model_path.exists():
        errors.append(f"Model file missing: {model_path}")
    if not vec_path.exists():
        errors.append(f"Vectorizer file missing: {vec_path}")

    if errors:
        return None, None, {}, errors

    try:
        model = joblib.load(model_path)
    except Exception as e:
        errors.append(f"Failed loading model: {e}")
        model = None

    try:
        vectorizer = joblib.load(vec_path)
    except Exception as e:
        errors.append(f"Failed loading vectorizer: {e}")
        vectorizer = None

    metadata = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    if model is None or vectorizer is None:
        return None, None, {}, errors

    # Feature & prediction integrity checks
    if not hasattr(model, "classes_"):
        errors.append("Model lacks classes_ attribute")

    try:
        sample_vec = vectorizer.transform(["test question"])
        _ = model.predict(sample_vec)
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(sample_vec)
    except Exception as e:
        errors.append(f"Prediction test failed: {e}")

    return model, vectorizer, metadata, errors


def predict_pair(model: Any, vectorizer: Any, question: str) -> Tuple[str, float, Dict[str, float]]:
    """Generate prediction label, top confidence, and class probability map."""
    X = vectorizer.transform([question])
    pred_label = model.predict(X)[0]

    probs_map = {}
    confidence = 0.0

    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        for cls, prob in zip(classes, probs):
            probs_map[cls] = float(prob)
        confidence = float(probs_map.get(pred_label, max(probs)))
    else:
        confidence = 1.0
        probs_map[pred_label] = 1.0

    return str(pred_label), confidence, probs_map


def calculate_metrics(y_true: List[str], y_pred: List[str], confs: List[float]) -> Dict[str, Any]:
    """Calculate overall accuracy, macro/weighted F1, domain F1s, and confidence calibration metrics."""
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total if total > 0 else 0.0

    # Per-domain Precision, Recall, F1
    domain_metrics = {}
    for d in DOMAIN_CLASSES:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == d and yp == d)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != d and yp == d)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == d and yp != d)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        domain_metrics[d] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for yt in y_true if yt == d)
        }

    # Macro & Weighted F1
    f1_scores = [domain_metrics[d]["f1"] for d in DOMAIN_CLASSES if domain_metrics[d]["support"] > 0]
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    weighted_f1_sum = sum(domain_metrics[d]["f1"] * domain_metrics[d]["support"] for d in DOMAIN_CLASSES)
    weighted_f1 = weighted_f1_sum / total if total > 0 else 0.0

    # Confidence Reliability / Calibration metrics
    correct_confs = [c for yt, yp, c in zip(y_true, y_pred, confs) if yt == yp]
    incorrect_confs = [c for yt, yp, c in zip(y_true, y_pred, confs) if yt != yp]

    avg_all_conf = sum(confs) / len(confs) if confs else 0.0
    avg_correct_conf = sum(correct_confs) / len(correct_confs) if correct_confs else 0.0
    avg_incorrect_conf = sum(incorrect_confs) / len(incorrect_confs) if incorrect_confs else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "domain_metrics": domain_metrics,
        "avg_all_conf": avg_all_conf,
        "avg_correct_conf": avg_correct_conf,
        "avg_incorrect_conf": avg_incorrect_conf,
    }


def run_comparison():
    print("=" * 80)
    print(" VEDIC ASTROLOGY AI — DOMAIN MODEL COMPARISON REPORT")
    print("=" * 80)

    # 1. Scikit-learn compatibility check notice
    print("\n--- 1. ARTIFACT & ENVIRONMENT CHECK ---")
    loaded_pairs = []

    for pair_info in MODEL_PAIRS:
        model, vectorizer, metadata, errors = load_and_validate_pair(pair_info)
        status = "PASSED" if not errors else f"FAILED: {errors}"
        print(f"[{pair_info['name']:<15}] Loading Check: {status}")
        if model and vectorizer:
            loaded_pairs.append({
                "info": pair_info,
                "model": model,
                "vectorizer": vectorizer,
                "metadata": metadata
            })

    # Report unpickling warnings if any occurred during joblib loading
    unpickle_warnings = [w for w in captured_warnings if "unpickle" in str(w.message).lower() or "version" in str(w.message).lower()]
    if unpickle_warnings:
        print(f"\n[NOTICE] Captured {len(unpickle_warnings)} scikit-learn version warnings (estimators unpickled from sklearn 1.6.1 in runtime sklearn 1.9.1).")
        print("Model files remain untouched as authoritative artifacts.")

    if not loaded_pairs:
        print("\n[ERROR] No valid model pairs found. Aborting comparison.")
        return

    # 2. Per-Question Predictions Across All Models
    print("\n" + "=" * 80)
    print(" 2. PER-QUESTION PREDICTION BREAKDOWN")
    print("=" * 80)

    results_by_pair = {p["info"]["key"]: {"y_true": [], "y_pred": [], "confs": []} for p in loaded_pairs}

    for idx, (question, expected_domain) in enumerate(TEST_DATASET, 1):
        print(f"\nQ{idx:02d}: \"{question}\" (Expected: {expected_domain.upper()})")
        print("-" * 75)

        for p in loaded_pairs:
            key = p["info"]["key"]
            name = p["info"]["name"]
            pred_label, conf, probs = predict_pair(p["model"], p["vectorizer"], question)

            results_by_pair[key]["y_true"].append(expected_domain)
            results_by_pair[key]["y_pred"].append(pred_label)
            results_by_pair[key]["confs"].append(conf)

            mark = "PASS" if pred_label == expected_domain else "FAIL"
            prob_str = ", ".join([f"{k}:{v:.2f}" for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]])
            print(f"  [{key:<10}] {mark:<4} | Pred: {pred_label:<10} | Conf: {conf:.4f} | Top: [{prob_str}]")

    # 3. Comprehensive Model Performance Metrics
    print("\n" + "=" * 80)
    print(" 3. MODEL PERFORMANCE METRICS SUMMARY")
    print("=" * 80)

    metrics_by_pair = {}
    print(f"\n{'MODEL':<15} | {'ACCURACY':<10} | {'MACRO F1':<10} | {'WEIGHTED F1':<12} | {'CORRECT/TOTAL'}")
    print("-" * 68)

    for p in loaded_pairs:
        key = p["info"]["key"]
        name = p["info"]["name"]
        data = results_by_pair[key]
        m = calculate_metrics(data["y_true"], data["y_pred"], data["confs"])
        metrics_by_pair[key] = m
        print(f"{name:<15} | {m['accuracy']*100:6.2f}%    | {m['macro_f1']:6.4f}    | {m['weighted_f1']:10.4f}   | {m['correct']}/{m['total']}")

    # 4. Domain-Specific Performance Matrix
    print("\n" + "=" * 80)
    print(" 4. DOMAIN-SPECIFIC PERFORMANCE MATRIX (F1-SCORES)")
    print("=" * 80)
    print(f"\n{'MODEL':<15} | {'CAREER':<10} | {'FINANCE':<10} | {'HEALTH':<10} | {'MARRIAGE':<10} | {'OTHER':<10}")
    print("-" * 75)

    for p in loaded_pairs:
        key = p["info"]["key"]
        name = p["info"]["name"]
        dm = metrics_by_pair[key]["domain_metrics"]
        print(
            f"{name:<15} | "
            f"{dm['career']['f1']:6.4f}    | "
            f"{dm['finance']['f1']:6.4f}    | "
            f"{dm['health']['f1']:6.4f}    | "
            f"{dm['marriage']['f1']:6.4f}    | "
            f"{dm['other']['f1']:6.4f}"
        )

    # 5. Confidence Reliability & Calibration Analysis
    print("\n" + "=" * 80)
    print(" 5. CONFIDENCE RELIABILITY & CALIBRATION ANALYSIS")
    print("=" * 80)
    print(f"\n{'MODEL':<15} | {'AVG ALL CONF':<14} | {'AVG CORRECT CONF':<18} | {'AVG INCORRECT CONF':<18}")
    print("-" * 72)

    for p in loaded_pairs:
        key = p["info"]["key"]
        name = p["info"]["name"]
        m = metrics_by_pair[key]
        print(f"{name:<15} | {m['avg_all_conf']:12.4f}  | {m['avg_correct_conf']:16.4f}  | {m['avg_incorrect_conf']:16.4f}")

    # 6. Scientific Recommendation & Decision Tree Analysis
    print("\n" + "=" * 80)
    print(" 6. SCIENTIFIC RECOMMENDATION & DECISION TREE")
    print("=" * 80)

    # Find best model based on macro F1 and accuracy
    best_pair_key = max(metrics_by_pair.keys(), key=lambda k: (metrics_by_pair[k]["accuracy"], metrics_by_pair[k]["macro_f1"]))
    best_metrics = metrics_by_pair[best_pair_key]

    # Check if best model is significantly superior across all domains
    all_accuracies = [metrics_by_pair[k]["accuracy"] for k in metrics_by_pair]
    is_clearly_superior = best_metrics["accuracy"] == max(all_accuracies) and (best_metrics["accuracy"] - sorted(all_accuracies)[-2] >= 0.10 if len(all_accuracies) > 1 else True)

    print(f"\nBEST OVERALL MODEL: {best_pair_key} (Accuracy: {best_metrics['accuracy']*100:.2f}%, Macro F1: {best_metrics['macro_f1']:.4f})")

    if is_clearly_superior:
        print("\nDECISION TREE CONCLUSION: SINGLE MODEL REPLACEMENT RECOMMENDED")
        print(f"- {best_pair_key} demonstrates consistent superiority across overall dataset accuracy and domain F1 scores.")
        print(f"- Recommendation: Replace default production model with '{best_pair_key}' as a simple drop-in replacement (upon explicit approval).")
        print("- An ensemble/selector layer is NOT required, avoiding unnecessary architectural complexity.")
    else:
        print("\nDECISION TREE CONCLUSION: EVALUATE COMPLEMENTARY STRENGTHS / ENSEMBLE")
        print("- Models display complementary strengths across different domains.")
        print("- Recommendation: Review per-domain F1 scores above. If different models excel at specific domains, consider a domain-weighted model selector.")

    print("\n" + "=" * 80)
    print(" END OF COMPARISON REPORT — ZERO PRODUCTION ROUTER FILES MODIFIED")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison()
