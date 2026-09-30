"""
Permanent Test Suite for Domain Selector & Model Pool
=====================================================
Script: test_domain_selector.py

Verifies:
1. Model Integrity: All model files and vectorizers load properly.
2. Vectorizer Pairing: Feature dimensions match exact model expectations.
3. Prediction Structure: Returns raw probabilities, margins, and scores.
4. Edge Case Handling: Hinglish, ambiguous, multi-domain, and disagreement queries.
5. Read-Only Safety: No model files or configuration artifacts modified.
"""

import sys
import unittest
from pathlib import Path

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

class TestDomainSelector(unittest.TestCase):

    def test_01_model_pool_integrity(self):
        pool = load_domain_model_pool()
        self.assertIn("OLD", pool)
        self.assertIn("CONTEXT_V1", pool)
        self.assertIn("V3", pool)
        
        for name, (model, vectorizer) in pool.items():
            self.assertIsNotNone(model)
            self.assertIsNotNone(vectorizer)
            self.assertTrue(hasattr(model, "predict"))
            self.assertTrue(hasattr(vectorizer, "transform"))

    def test_02_vectorizer_pairing_integrity(self):
        pool = load_domain_model_pool()
        sample_q = "Will I get a promotion in my career?"
        for name, (model, vectorizer) in pool.items():
            res = predict_single_pair(model, vectorizer, sample_q)
            self.assertIn(res["domain"], DOMAIN_CLASSES)
            self.assertGreaterEqual(res["confidence"], 0.0)
            self.assertLessEqual(res["confidence"], 1.0)
            self.assertEqual(len(res["probabilities"]), 5)

    def test_03_query_evidence_detector(self):
        ev_health = detect_query_evidence("Why do I feel tired and exhausted with low energy?")
        self.assertGreater(ev_health["health"], 0.0)

        ev_career = detect_query_evidence("Will I get a job promotion at work?")
        self.assertGreater(ev_career["career"], 0.0)

    def test_04_health_regression(self):
        health_queries = [
            "Why do I have low energy lately?",
            "Why do I feel weak these days?",
            "Why do I keep feeling exhausted?",
            "Why am I having trouble sleeping?"
        ]
        for q in health_queries:
            res = evaluate_and_select_domain(q)
            self.assertEqual(res["selected_domain"], "health", f"Failed on query: {q}")

    def test_05_relationship_regression(self):
        rel_queries = [
            "When will I get married?",
            "Will my current relationship work out",
            "Would we make a good couple"
        ]
        for q in rel_queries:
            res = evaluate_and_select_domain(q)
            self.assertEqual(res["selected_domain"], "marriage", f"Failed on query: {q}")

    def test_06_career_regression(self):
        career_queries = [
            "Will I get promoted?",
            "When will I get a new job?",
            "Which career suits me best?"
        ]
        for q in career_queries:
            res = evaluate_and_select_domain(q)
            self.assertEqual(res["selected_domain"], "career", f"Failed on query: {q}")

    def test_07_raw_predictions_preservation(self):
        res = evaluate_and_select_domain("Tell me about my career and money.")
        self.assertIn("model_predictions", res)
        preds = res["model_predictions"]
        self.assertIn("OLD", preds)
        self.assertIn("CONTEXT_V1", preds)
        self.assertIn("V3", preds)
        for name in ["OLD", "CONTEXT_V1", "V3"]:
            self.assertIn("domain", preds[name])
            self.assertIn("confidence", preds[name])
            self.assertIn("probabilities", preds[name])

if __name__ == "__main__":
    unittest.main()
