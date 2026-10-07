"""
Relevance-Gated Dynamic Architecture Comprehensive Test Suite
================================================================
Validates all 5 Routing Modes & Behavior:
1. DIRECT_API: Direct Factual Lookups (Nakshatra, Moon Sign, Lagna, Dasha) -> Gemini = 0
2. LOCAL_SYNTHESIS / RULE_BASED: Standard Local Rules -> Gemini = 0
3. LLM_ASSISTED: Complex Valid Domain Queries -> Gemini = 1
4. UNSUPPORTED: Boundary Queries -> Gemini = 0 (Reason-specific limitation response)
5. UNRESOLVED: Missing Evidence -> Gemini = 0
"""

import os
import sys
import unittest

# Add backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.reasoning.direct_fact_engine import is_direct_fact_query, extract_direct_fact
from backend.reasoning.rule_coverage import evaluate_rule_coverage, is_unsupported_boundary_query, get_unsupported_reason, get_unsupported_limitation_response
from backend.reasoning.mode_selector import select_answer_mode
from backend.reasoning.llm_renderer import build_compact_evidence_package, format_llm_assisted_prompt


# Sample normalized chart data for Virgo Ascendant, Taurus Moon, Gemini Sun, Mercury in 10th house
MOCK_NORM_CHART = {
    "ascendant": {"rashi": "Virgo", "sign": "Virgo", "degree_in_rashi": 12.5},
    "planets": {
        "Moon": {"rashi": "Taurus", "sign": "Taurus", "nakshatra": "Rohini", "nakshatra_pada": 2, "nakshatra_lord": "Moon", "degree_in_rashi": 15.2, "rashi_lord": "Venus"},
        "Sun": {"rashi": "Gemini", "sign": "Gemini", "degree_in_rashi": 5.4, "rashi_lord": "Mercury"},
        "Mercury": {"rashi": "Gemini", "sign": "Gemini", "house": 10, "rashi_lord": "Mercury"},
        "Venus": {"rashi": "Taurus", "sign": "Taurus", "house": 9, "rashi_lord": "Venus"},
        "Jupiter": {"rashi": "Pisces", "sign": "Pisces", "house": 7, "rashi_lord": "Jupiter"}
    }
}

MOCK_DASHA_HIERARCHY = {
    "current_mahadasha": "Jupiter",
    "current_antardasha": "Saturn",
    "start_date": "2024-01-01",
    "end_date": "2026-05-01"
}

class MockStructuredEvidence:
    def __init__(self, domain="career", intent="general", is_suff=True):
        self.domain = domain
        self.intent = intent
        self._suff = is_suff
        self.facts = {"ascendant": "Virgo", "moon_rashi": "Taurus", "sun_sign": "Gemini"}
        self.evaluations = {}
        self.interpretation = {}
        self.timing = {"active_mahadasha": "Jupiter", "active_antardasha": "Saturn"}
        self.positive_factors = ["CAREER_DASHA_ACTIVATION"]
        self.challenging_factors = []

    def is_sufficient(self):
        return self._suff


class TestRelevanceGatedArchitecture(unittest.TestCase):

    def test_1_direct_api_lookups(self):
        """1. DIRECT_API: Direct Factual Calculations (Nakshatra, Rashi, Lagna, Dasha) -> Gemini = 0"""
        direct_queries = [
            ("What is my Nakshatra?", "nakshatra"),
            ("What is my Moon Rashi?", "moon_sign"),
            ("What is my Lagna?", "lagna"),
            ("What is my current Mahadasha?", "current_dasha"),
            ("What is my Sun sign?", "sun_sign")
        ]
        for q, expected_fact_type in direct_queries:
            with self.subTest(query=q):
                self.assertTrue(is_direct_fact_query(q), f"Query should be recognized as direct fact: {q}")
                res = extract_direct_fact(q, MOCK_NORM_CHART, MOCK_DASHA_HIERARCHY)
                self.assertIsNotNone(res, f"Fact extraction failed for {q}")
                self.assertEqual(res["gemini_calls"], 0)
                self.assertEqual(res["answer_mode"], "DIRECT")
                self.assertTrue("answer" in res)

    def test_2_local_synthesis_rules(self):
        """2. LOCAL_SYNTHESIS / RULE_BASED: Exact Deterministic Rules -> Gemini = 0"""
        local_queries = [
            ("What does my 10th house indicate about my career?", "career", "10th_lord_placement"),
            ("How does my current Dasha affect my career?", "career", "dasha_career_interaction"),
            ("What does my 7th house indicate about marriage?", "marriage", "7th_lord_placement"),
            ("What does my 2nd house indicate about finances?", "finance", "financial_potential")
        ]
        mock_ev = MockStructuredEvidence(domain="career", intent="10th_lord_placement", is_suff=True)
        matched_rules = [{"rule_id": "CAREER_10TH_LORD_PLACEMENT", "matched": True}]

        for q, dom, ent in local_queries:
            with self.subTest(query=q):
                cov = evaluate_rule_coverage(q, dom, ent, mock_ev, matched_rules)
                self.assertTrue(cov["exact_rule_match"], f"Exact rule match should be True for {q}")
                mode = select_answer_mode(dom, ent, q, matched_rules, False, mock_ev)
                self.assertEqual(mode["answer_source"], "LOCAL")
                self.assertEqual(mode["gemini_calls"], 0)

    def test_3_llm_assisted_complex_domain_queries(self):
        """3. LLM_ASSISTED: Complex Valid Domain Queries (Local rule coverage insufficient) -> Gemini = 1"""
        complex_queries = [
            ("Should I move from software engineering into management during my current Dasha?", "career", "career_transition"),
            ("How can my marriage and finances influence each other?", "marriage", "marriage_finance_interaction"),
            ("What kind of business environment would suit me?", "career", "business_suitability"),
            ("How should I approach a career transition according to my chart?", "career", "career_transition")
        ]
        mock_ev = MockStructuredEvidence(domain="career", intent="career_transition", is_suff=True)

        for q, dom, ent in complex_queries:
            with self.subTest(query=q):
                cov = evaluate_rule_coverage(q, dom, ent, mock_ev, [])
                self.assertFalse(cov["exact_rule_match"], f"Exact rule match should be False for complex query {q}")
                self.assertTrue(cov["domain_match"], f"Domain match should be True for {q}")
                mode = select_answer_mode(dom, ent, q, [], False, mock_ev)
                self.assertEqual(mode["answer_source"], "EVIDENCE_GROUNDED_LLM", f"Mode should be EVIDENCE_GROUNDED_LLM for {q}")
                self.assertEqual(mode["gemini_calls"], 1)

    def test_4_unsupported_boundary_queries(self):
        """4. UNSUPPORTED: Out-of-bounds / Deterministic Predictions -> Gemini = 0, Reason-specific limitation"""
        unsupported_queries = [
            ("What was my exact past life?", "PAST_LIFE_IDENTITY"),
            ("Who was I in my previous birth?", "PAST_LIFE_IDENTITY"),
            ("Give me the exact winning lottery numbers.", "LOTTERY_PREDICTION"),
            ("Tell me the exact date I will die.", "DEATH_DATE"),
            ("Tell me the exact name of my future spouse.", "FUTURE_SPOUSE_NAME"),
            ("Tell me exactly what will happen to me tomorrow at 3 PM.", "EXACT_DAILY_EVENT"),
            ("What will be the exact stock price of a company next month?", "EXACT_MARKET_PRICE"),
            ("Can you guarantee that I will become a millionaire?", "GUARANTEED_WEALTH")
        ]
        for q, expected_reason in unsupported_queries:
            with self.subTest(query=q):
                self.assertTrue(is_unsupported_boundary_query(q), f"Query should be recognized as unsupported: {q}")
                reason = get_unsupported_reason(q)
                self.assertEqual(reason, expected_reason, f"Reason mismatch for {q}: got {reason}, expected {expected_reason}")
                lim_resp = get_unsupported_limitation_response(q)
                self.assertTrue(len(lim_resp) > 20, f"Limitation response should be descriptive for {q}")
                mode = select_answer_mode("general", "unsupported", q, [], False, None)
                self.assertEqual(mode["answer_source"], "UNSUPPORTED")
                self.assertEqual(mode["gemini_calls"], 0)

    def test_5_incomplete_evidence_queries(self):
        """5. UNRESOLVED: Missing Evidence -> Gemini = 0"""
        mock_ev_insuff = MockStructuredEvidence(domain="career", intent="general", is_suff=False)
        mode = select_answer_mode("career", "general", "What about my career?", [], False, mock_ev_insuff, evidence_status="UNRESOLVED")
        self.assertEqual(mode["answer_source"], "UNRESOLVED")
        self.assertEqual(mode["gemini_calls"], 0)

    def test_6_compact_evidence_package(self):
        """6. Token Minimization: Compact evidence package contains only verified facts"""
        pkg = build_compact_evidence_package(
            domain="career",
            intent="career_transition",
            chart_data=MOCK_NORM_CHART,
            matched_rules=[],
            dasha_hierarchy=MOCK_DASHA_HIERARCHY
        )
        self.assertEqual(pkg["domain"], "career")
        self.assertIn("10H", pkg["verified_facts"])
        self.assertIn("Dasha", pkg["verified_facts"])
        prompt, est_tokens = format_llm_assisted_prompt("Should I switch careers?", pkg)
        self.assertTrue(est_tokens < 150, f"Prompt token count should be under 150 tokens (got {est_tokens})")


if __name__ == "__main__":
    unittest.main()
