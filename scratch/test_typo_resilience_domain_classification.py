"""
Typo Resilience & Domain Classification Regression Suite
=========================================================
File: scratch/test_typo_resilience_domain_classification.py

Verifies:
1. Queries with typos (e.g. 'marriead', 'finiacial', 'promtion', 'helth') route to their correct domains.
2. 'when will i get marriead' -> domain: 'marriage' (not career).
3. 'what is my finiacial status in up coming' -> domain: 'finance' / 'wealth' (not career/other).
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.main import normalize_hinglish
from backend.router.domain import predict_domain_with_confidence
from backend.router.pipeline import route_question


def run_typo_resilience_tests():
    print("================================================================================")
    print("STARTING TYPO RESILIENCE & DOMAIN CLASSIFICATION REGRESSION TESTS")
    print("================================================================ crush\n")

    test_queries = [
        ("when will i get marriead", "marriage"),
        ("when will i get marrige", "marriage"),
        ("what is my finiacial status in up coming year", "finance"),
        ("tell me about my finacial growth", "finance"),
        ("when will i get a promtion in my carer", "career"),
        ("how is my helth looking", "health"),
    ]

    for q, expected_domain in test_queries:
        norm_q = normalize_hinglish(q)
        res = route_question(norm_q)
        predicted_domain = res.get("domain")

        print(f"Query           : '{q}'")
        print(f"Normalized      : '{norm_q}'")
        print(f"Predicted Domain: '{predicted_domain}' (Expected: '{expected_domain}')")
        print("-" * 70)

        assert predicted_domain == expected_domain, (
            f"Typo routing failure for '{q}': predicted '{predicted_domain}', expected '{expected_domain}'"
        )

    print("\n================================================================================")
    print("[SUCCESS] ALL TYPO RESILIENCE DOMAIN CLASSIFICATION TESTS PASSED!")
    print("================================================================================\n")


if __name__ == "__main__":
    run_typo_resilience_tests()
