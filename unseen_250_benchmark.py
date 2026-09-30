"""
250-Query Unseen Benchmark & Stress Test Suite
==============================================
Script: unseen_250_benchmark.py

Performs a comprehensive, unseen 250-query benchmark across 5 domain classes (50 queries each):
1. Career (50)
2. Health (50)
3. Marriage / Relationship (50)
4. Finance (50)
5. Other / General (50)

Evaluates:
- OLD alone
- CONTEXT_V1 alone
- V3 alone
- Majority Voting
- Probability Averaging
- Weighted Ensemble
- QUERY_SELECTOR (backend/router/model_selector.py)

Strict Rule: READ-ONLY evaluation. No model files or production routing files modified.
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

# Silence scikit-learn warnings
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
from benchmark_evaluation_selector import (
    run_majority_voting,
    run_probability_averaging,
    run_weighted_ensemble
)

# ------------------------------------------------------------------
# 250 UNSEEN BENCHMARK DATASET (50 Queries per domain)
# ------------------------------------------------------------------
CAREER_QUERIES = [
    ("Which career suits me best?", "career"),
    ("Will I get promoted at my job?", "career"),
    ("Should I switch my current job?", "career"),
    ("When will I get a new job?", "career"),
    ("How will my business grow this year?", "career"),
    ("Is corporate career suitable for me?", "career"),
    ("Will I pass my job interview next week?", "career"),
    ("Will I get transferred to another city in my job?", "career"),
    ("Will I get a government job?", "career"),
    ("Should I join my family business?", "career"),
    ("What is my ideal profession?", "career"),
    ("Will I get a salary hike during promotion?", "career"),
    ("Should I start a new venture or startup?", "career"),
    ("Will my boss appreciate my hard work?", "career"),
    ("How is my workplace environment?", "career"),
    ("Will I face office politics at work?", "career"),
    ("When will I start working?", "career"),
    ("Will I get a job abroad?", "career"),
    ("Is freelancing good for my career?", "career"),
    ("Will my partnership business succeed?", "career"),
    ("Will I get recognition at my workplace?", "career"),
    ("Should I resign from my current position?", "career"),
    ("Will my career improve after Dasha change?", "career"),
    ("Will I get a remote job offer?", "career"),
    ("Should I pursue higher studies for career growth?", "career"),
    ("Will I become a CEO or founder?", "career"),
    ("When will my career stability come?", "career"),
    ("Will I get a promotion in 2026?", "career"),
    ("Should I change my career domain?", "career"),
    ("Will my job contract be renewed?", "career"),
    ("Will I get success in administrative job?", "career"),
    ("Is teaching profession good for me?", "career"),
    ("Will I succeed in IT industry?", "career"),
    ("Will my work stress reduce at office?", "career"),
    ("Will I get a government job in SSC exam?", "career"),
    ("When will my business loss recover?", "career"),
    ("Will I get a managerial position?", "career"),
    ("Should I take up a part time job?", "career"),
    ("Will my client deal close successfully?", "career"),
    ("Will my career peak in Saturn Dasha?", "career"),
    ("mera naukri kab milega", "career"),
    ("promotion kab milega company mein", "career"),
    ("job switch karna chahiye ya nahi", "career"),
    ("vyapar mein safalta kab milegi", "career"),
    ("naukri mein tarakki kab hogi", "career"),
    ("government job kab milegi", "career"),
    ("office politics se kab chhutkara milega", "career"),
    ("boss ke saath problem kab khatam hogi", "career"),
    ("kaam mein safalta kab milegi", "career"),
    ("kaun sa profession mere liye sahi hai", "career")
]

HEALTH_QUERIES = [
    ("Why do I have low energy lately?", "health"),
    ("Why do I feel weak these days?", "health"),
    ("Why do I keep feeling exhausted?", "health"),
    ("Why am I having trouble sleeping?", "health"),
    ("Why can't I sleep properly at night?", "health"),
    ("Will my sleep improve soon?", "health"),
    ("Why do I wake up several times during the night?", "health"),
    ("Will my energy improve soon?", "health"),
    ("Why am I feeling anxious and stressed?", "health"),
    ("How is my health period looking?", "health"),
    ("Why do I feel burnt out at work?", "health"),
    ("Will my physical fitness improve?", "health"),
    ("Why am I feeling depressed and sad?", "health"),
    ("When will I recover from illness?", "health"),
    ("Why do I get frequent headaches?", "health"),
    ("Will my mental peace return?", "health"),
    ("Why am I feeling constantly tired?", "health"),
    ("Will my digestive health improve?", "health"),
    ("Why am I suffering from insomnia?", "health"),
    ("Will my health stay good during Rahu Dasha?", "health"),
    ("Why do I have chronic back pain?", "health"),
    ("Will my medical test reports be normal?", "health"),
    ("Why am I overthinking so much lately?", "health"),
    ("Will my skin health improve?", "health"),
    ("Why am I losing weight unexpectedly?", "health"),
    ("Will my blood pressure stay normal?", "health"),
    ("Why do I feel zero motivation in life?", "health"),
    ("Will my eye health improve after surgery?", "health"),
    ("Why do I feel restless all the time?", "health"),
    ("Will my energy return next month?", "health"),
    ("sehat kaisi rahegi meri", "health"),
    ("tanav aur chinta kab khatam hogi", "health"),
    ("neend nahi aati raat ko", "health"),
    ("hamesha thakaan kyun hoti hai", "health"),
    ("swasthya kab theek hoga mera", "health"),
    ("bimari se kab chutkara milega", "health"),
    ("mental peace kab milega dimag ko", "health"),
    ("body weak feel ho rahi hai", "health"),
    ("depression kab khatam hoga", "health"),
    ("overthinking kaise ruke gi", "health"),
    ("dawai ka asar kab hoga", "health"),
    ("swasthya mein sudhar kab hoga", "health"),
    ("sehat kharab kyun rehti hai", "health"),
    ("swasthya samasya kab door hogi", "health"),
    ("swasthya theek rahega na mera", "health"),
    ("sehat mein sudhar hoga ya nahi", "health"),
    ("thakawat kyun ho rahi hai", "health"),
    ("pet ki bimari kab theek hogi", "health"),
    ("sar dard kab band hoga", "health"),
    ("sehat ki pareshani kab khatam hogi", "health")
]

MARRIAGE_QUERIES = [
    ("When will I get married?", "marriage"),
    ("Will my current relationship work out?", "marriage"),
    ("Would we make a good couple?", "marriage"),
    ("I like my friend should I ask her out", "marriage"),
    ("Should I tell my crush that I like them", "marriage"),
    ("Will my friendship become a relationship", "marriage"),
    ("What does my 7th house say about marriage?", "marriage"),
    ("When will I find true love?", "marriage"),
    ("How will my married life be?", "marriage"),
    ("Are we compatible for marriage?", "marriage"),
    ("Will I have a love marriage or arranged marriage?", "marriage"),
    ("When will I meet my soulmate?", "marriage"),
    ("Will my partner be caring and loving?", "marriage"),
    ("Will my separation end in reconciliation?", "marriage"),
    ("Will my engagement happen this year?", "marriage"),
    ("How will my spouse look and nature be?", "marriage"),
    ("Will my marriage stay happy and long?", "marriage"),
    ("When will my divorce case resolve?", "marriage"),
    ("Will my ex come back to me?", "marriage"),
    ("Will my in-laws be supportive?", "marriage"),
    ("Will I marry someone from abroad?", "marriage"),
    ("Will my relationship convert to marriage?", "marriage"),
    ("When will I meet my future husband?", "marriage"),
    ("When will I meet my future wife?", "marriage"),
    ("Will my commitment issues resolve?", "marriage"),
    ("Will my family accept my love partner?", "marriage"),
    ("Will my late marriage yoga cancel?", "marriage"),
    ("Will my second marriage be successful?", "marriage"),
    ("Will my partner be financially stable?", "marriage"),
    ("Will my Kundali match with my partner?", "marriage"),
    ("shaadi kab hogi meri", "marriage"),
    ("rishta kab pakka hoga", "marriage"),
    ("love marriage hogi ya arranged", "marriage"),
    ("patni kaisi milegi mujhe", "marriage"),
    ("pati kaisa milega mujhe", "marriage"),
    ("jeevan saathi kab milega", "marriage"),
    ("pyaar kab milega life mein", "marriage"),
    ("rishtey mein problem kab theek hogi", "marriage"),
    ("shadi kab tak ho jayegi", "marriage"),
    ("patni se banegi ya nahi", "marriage"),
    ("pati ke saath ladai kab band hogi", "marriage"),
    ("vivah kab hoga mera", "marriage"),
    ("shaadi mein vilamb kyun ho raha hai", "marriage"),
    ("manglik dosh se shaadi ruk rahi hai kya", "marriage"),
    ("shadi ke baad life kaisi rahegi", "marriage"),
    ("kundali milan kaisa hai hamara", "marriage"),
    ("breakup ke baad ex wapas aayega kya", "marriage"),
    ("shaadi ke liye ghar wale manenge kya", "marriage"),
    ("dulha kaisa milega", "marriage"),
    ("dulhan kaisi milegi", "marriage")
]

FINANCE_QUERIES = [
    ("Will my income increase?", "finance"),
    ("When will I gain wealth?", "finance"),
    ("How can I improve my financial situation?", "finance"),
    ("Will I get money from investments?", "finance"),
    ("Will I be rich?", "finance"),
    ("When will I get financial stability?", "finance"),
    ("Will I buy a house in 2026?", "finance"),
    ("Will my debt be cleared soon?", "finance"),
    ("How will my savings look this year?", "finance"),
    ("Will I get money from inheritance?", "finance"),
    ("Will my stock market investments yield profit?", "finance"),
    ("Will my loan get approved by bank?", "finance"),
    ("When will my financial crisis end?", "finance"),
    ("Will I buy a new car this year?", "finance"),
    ("Will I make profit in property investment?", "finance"),
    ("Will my cash flow improve next month?", "finance"),
    ("Will I get unexpected wealth or lottery?", "finance"),
    ("Will my business turnover increase?", "finance"),
    ("How will my 2nd house of wealth perform?", "finance"),
    ("Will I accumulate assets in Jupiter Dasha?", "finance"),
    ("Will I clear my credit card bills?", "finance"),
    ("Will my land investment give good returns?", "finance"),
    ("Will my financial growth be steady?", "finance"),
    ("When will I become financially independent?", "finance"),
    ("Will my mutual funds give good return?", "finance"),
    ("Will I recover my lost money?", "finance"),
    ("Will my crypto investment recover?", "finance"),
    ("Will my gold investment be profitable?", "finance"),
    ("Will I get money from ancestral property?", "finance"),
    ("Will my financial stress end soon?", "finance"),
    ("paisa kab aayega mere paas", "finance"),
    ("paise ki samasya kab door hogi", "finance"),
    ("karz kab khatam hoga mera", "finance"),
    ("dhan mein vriddhi kab hogi", "finance"),
    ("bachat kab hogi paise ki", "finance"),
    ("sampatti kab khareed paunga", "finance"),
    ("nivesh mein laabh hoga kya", "finance"),
    ("makan kab banega mera", "finance"),
    ("karja kab chukega mera", "finance"),
    ("dhan laabh kab hoga", "finance"),
    ("paise ki problem kab khatam hogi", "finance"),
    ("ameer kab banoonga main", "finance"),
    ("kamayi kab badhegi meri", "finance"),
    ("financial stability kab milegi", "finance"),
    ("property kab khareed paunga", "finance"),
    ("loan kab approve hoga bank se", "finance"),
    ("udhaar liya paisa kab wapas milega", "finance"),
    ("dhandhe mein munafa kab hoga", "finance"),
    ("paisa tikta kyun nahi hai pass mein", "finance"),
    ("dhan yog hai ya nahi kundali mein", "finance")
]

OTHER_QUERIES = [
    ("What is a nakshatra?", "other"),
    ("What is a birth chart?", "other"),
    ("Tell me about astrology", "other"),
    ("What does ascendant mean?", "other"),
    ("How do planets affect human life?", "other"),
    ("What is a Mahadasha?", "other"),
    ("What is Sade Sati of Saturn?", "other"),
    ("What is Kundali?", "other"),
    ("What are the 12 houses in astrology?", "other"),
    ("What is Rahu and Ketu?", "other"),
    ("What is a horoscope?", "other"),
    ("What is planetary transit?", "other"),
    ("What is Jupiter Retrograde?", "other"),
    ("What is Mars Dosha or Manglik?", "other"),
    ("What is Navamsha chart?", "other"),
    ("What is Sun sign vs Moon sign?", "other"),
    ("What is Vedic astrology?", "other"),
    ("How does astrology work?", "other"),
    ("What is gemstone astrology?", "other"),
    ("What is birth chart degree?", "other"),
    ("What is Rashi?", "other"),
    ("What is Lagna lord?", "other"),
    ("What is Vimshottari Dasha?", "other"),
    ("What is Kalsarpa Dosha?", "other"),
    ("What is Panchang?", "other"),
    ("kundali kya hoti hai", "other"),
    ("rashi aur lagna mein kya antar hai", "other"),
    ("graha kya hote hain", "other"),
    ("shani ki sade sati kya hoti hai", "other"),
    ("mahadasha kya hoti hai", "other"),
    ("manglik dosh kya hota hai", "other"),
    ("astrology kaise kaam karti hai", "other"),
    ("nakshatra kitne hote hain", "other"),
    ("graho ka prabhav kya hota hai", "other"),
    ("ratna dharan karne se kya hota hai", "other"),
    ("birth chart kya hota hai", "other"),
    ("ascendant kya hota hai", "other"),
    ("dasha system kya hota hai", "other"),
    ("rahu ketu kya hote hain", "other"),
    ("astrology ke baare mein batao", "other"),
    ("astrology real hai kya", "other"),
    ("horoscope kya hota hai", "other"),
    ("rashi phal kya hota hai", "other"),
    ("surya grahan ka asar kya hota hai", "other"),
    ("chandra grahan ka asar kya hota hai", "other"),
    ("drishti kya hoti hai graho ki", "other"),
    ("yuti kya hoti hai graho ki", "other"),
    ("uchh aur neech graha kya hote hain", "other"),
    ("swagrahi graha kya hota hai", "other"),
    ("bhava kya hota hai kundali mein", "other")
]

UNSEEN_250_DATASET = CAREER_QUERIES + HEALTH_QUERIES + MARRIAGE_QUERIES + FINANCE_QUERIES + OTHER_QUERIES

def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total if total > 0 else 0.0

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
        "domain_metrics": domain_metrics
    }

def run_unseen_benchmark():
    print("=" * 95)
    print(" 250-QUERY UNSEEN BENCHMARK & STRESS-TEST EVALUATION")
    print("=" * 95)
    print(f"Total Unseen Questions: {len(UNSEEN_250_DATASET)} (50 Career, 50 Health, 50 Marriage, 50 Finance, 50 Other)")
    print("Models Evaluated: OLD, CONTEXT_V1, V3")
    print("Systems Evaluated: OLD, CONTEXT_V1, V3, Majority Voting, ProbAvg, Weighted, QUERY_SELECTOR")
    print("Production Status: READ-ONLY (No models or router files modified)\n")

    pool = load_domain_model_pool()
    if len(pool) < 3:
        print("[ERROR] Could not load all 3 domain model pairs. Aborting.")
        return

    systems = [
        "1. OLD",
        "2. CONTEXT_V1",
        "3. V3",
        "4. MAJORITY_VOTING",
        "5. PROBABILITY_AVG",
        "6. WEIGHTED_ENSEMBLE",
        "7. QUERY_SELECTOR"
    ]

    results: Dict[str, Dict[str, List[str]]] = {s: {"y_true": [], "y_pred": []} for s in systems}

    selected_counts = {"OLD": 0, "CONTEXT_V1": 0, "V3": 0, "none": 0}
    disagreement_count = 0

    for question, expected in UNSEEN_250_DATASET:
        old_p = predict_single_pair(pool["OLD"][0], pool["OLD"][1], question)
        v1_p = predict_single_pair(pool["CONTEXT_V1"][0], pool["CONTEXT_V1"][1], question)
        v3_p = predict_single_pair(pool["V3"][0], pool["V3"][1], question)

        preds = [old_p["domain"], v1_p["domain"], v3_p["domain"]]
        probs = [old_p["probabilities"], v1_p["probabilities"], v3_p["probabilities"]]

        maj_d = run_majority_voting(preds)
        avg_d = run_probability_averaging(probs)
        weight_d = run_weighted_ensemble(probs)

        selector_res = evaluate_and_select_domain(question)
        sel_d = selector_res["selected_domain"]
        sel_m = selector_res["selected_model"]

        if len(set(preds)) > 1:
            disagreement_count += 1

        selected_counts[sel_m] = selected_counts.get(sel_m, 0) + 1

        mapping = [
            ("1. OLD", old_p["domain"]),
            ("2. CONTEXT_V1", v1_p["domain"]),
            ("3. V3", v3_p["domain"]),
            ("4. MAJORITY_VOTING", maj_d),
            ("5. PROBABILITY_AVG", avg_d),
            ("6. WEIGHTED_ENSEMBLE", weight_d),
            ("7. QUERY_SELECTOR", sel_d)
        ]

        for s_name, pred_d in mapping:
            results[s_name]["y_true"].append(expected)
            results[s_name]["y_pred"].append(pred_d)

    metrics = {s: calculate_metrics(results[s]["y_true"], results[s]["y_pred"]) for s in systems}

    # ==================================================================
    # 1. OVERALL DASHBOARD
    # ==================================================================
    print("=" * 95)
    print(" 1. OVERALL ACCURACY & MACRO F1 DASHBOARD (250 UNSEEN QUERIES)")
    print("=" * 95)
    print(f"{'SYSTEM / STRATEGY':<24} | {'ACCURACY':<10} | {'CORRECT/TOTAL':<14} | {'MACRO F1':<10} | {'WEIGHTED F1'}")
    print("-" * 85)
    for s in systems:
        m = metrics[s]
        print(f"{s:<24} | {m['accuracy']*100:6.2f}%    | {m['correct']:>3}/{m['total']:<3}          | {m['macro_f1']:6.4f}    | {m['weighted_f1']:6.4f}")

    # ==================================================================
    # 2. PER-DOMAIN PERFORMANCE BREAKDOWN
    # ==================================================================
    print("\n" + "=" * 95)
    print(" 2. PER-DOMAIN F1-SCORE BREAKDOWN (250 QUERIES)")
    print("=" * 95)
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

    # ==================================================================
    # 3. PER-DOMAIN RECALL BREAKDOWN (V3 vs QUERY_SELECTOR)
    # ==================================================================
    print("\n" + "=" * 95)
    print(" 3. PER-DOMAIN RECALL COMPARISON (V3 ALONE vs QUERY SELECTOR)")
    print("=" * 95)
    print(f"{'DOMAIN':<14} | {'OLD RECALL':<12} | {'V3 RECALL':<12} | {'QUERY SELECTOR RECALL':<22} | {'RECALL GAIN'}")
    print("-" * 78)

    v3_dm = metrics["3. V3"]["domain_metrics"]
    sel_dm = metrics["7. QUERY_SELECTOR"]["domain_metrics"]
    old_dm = metrics["1. OLD"]["domain_metrics"]

    for d in DOMAIN_CLASSES:
        v3_rec = v3_dm[d]["recall"]
        sel_rec = sel_dm[d]["recall"]
        old_rec = old_dm[d]["recall"]
        diff = sel_rec - v3_rec
        diff_str = f"+{diff*100:.1f}%" if diff > 0 else (f"{diff*100:.1f}%" if diff < 0 else "0.0%")
        print(f"{d.upper():<14} | {old_rec*100:6.1f}%      | {v3_rec*100:6.1f}%      | {sel_rec*100:6.1f}%                 | {diff_str}")

    # ==================================================================
    # 4. MODEL SELECTION DISTRIBUTION
    # ==================================================================
    print("\n" + "=" * 95)
    print(" 4. MODEL SELECTION & DISAGREEMENT SUMMARY")
    print("=" * 95)
    print(f"Total Queries Evaluated:          {len(UNSEEN_250_DATASET)}")
    print(f"Disagreement Queries:             {disagreement_count} ({disagreement_count/len(UNSEEN_250_DATASET)*100:.1f}%)")
    print("-" * 80)
    print("Selected Model Breakdown in QUERY_SELECTOR:")
    for m_name, count in selected_counts.items():
        print(f"  - {m_name:<12}: {count} queries ({count/len(UNSEEN_250_DATASET)*100:.1f}%)")

    # ==================================================================
    # 5. FINAL COMPARISON CONCLUSION
    # ==================================================================
    v3_acc = metrics["3. V3"]["accuracy"]
    sel_acc = metrics["7. QUERY_SELECTOR"]["accuracy"]
    
    print("\n" + "=" * 95)
    print(" 5. FINAL EVALUATION CONCLUSION: QUERY_SELECTOR vs V3 ALONE")
    print("=" * 95)
    print(f"  V3 Model Alone:            Accuracy = {v3_acc*100:.2f}%, Macro F1 = {metrics['3. V3']['macro_f1']:.4f}")
    print(f"  QUERY_SELECTOR:            Accuracy = {sel_acc*100:.2f}%, Macro F1 = {metrics['7. QUERY_SELECTOR']['macro_f1']:.4f}")

    if sel_acc > v3_acc:
        print("\nSUCCESS: QUERY_SELECTOR IS DEMONSTRABLY BETTER THAN V3 ALONE!")
        print(f"  Accuracy Improvement: +{(sel_acc - v3_acc)*100:.2f}% (from {v3_acc*100:.2f}% to {sel_acc*100:.2f}%)")
        print(f"  Finance Recall Gain:  +{(sel_dm['finance']['recall'] - v3_dm['finance']['recall'])*100:.1f}% (from {v3_dm['finance']['recall']*100:.1f}% to {sel_dm['finance']['recall']*100:.1f}%)")
    else:
        print("\nNOTICE: QUERY_SELECTOR performance matches or is lower than V3 alone.")

    print("=" * 95)

if __name__ == "__main__":
    run_unseen_benchmark()
