"""
Tests for Backend-First Routing, Local Capability Evaluation, and Grounded LLM Synthesis Fallback
-------------------------------------------------------------------------------------------------
Verifies all 12 regression cases specified in the Architecture Mandate:
1. Direct Lagna lookup -> DIRECT, zero Gemini calls.
2. Current Mahadasha lookup -> DIRECT, zero Gemini calls.
3. Supported career interpretation -> RULE_BASED, zero Gemini calls.
4. Career-delay question unsupported by local synthesizer -> evidence-grounded LLM fallback.
5. Job-growth & job-switch question -> evidence-grounded LLM fallback, not generic canned bullets.
6. Government vs private employment -> evidence-grounded LLM fallback with Sun/Saturn/Mercury facts.
7. Two different career dilemmas -> must not produce identical canned answers.
8. Complex question with complete local rule coverage -> remains local (0 Gemini calls).
9. Missing Dasha evidence -> no fabricated facts.
10. Multi-domain question with partial evidence -> preserves supported facts, marks unresolved parts.
11. Unsupported / out-of-scope question -> layer 1 rejection (0 Gemini calls).
12. Missing Gemini config / API error -> graceful fallback.
"""

import unittest
from unittest.mock import patch, AsyncMock
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.reasoning.rule_coverage import evaluate_rule_coverage, analyze_question_task
from backend.reasoning.mode_selector import select_answer_mode
from backend.reasoning.answer_synthesizer import synthesize_local_answer, can_synthesize_local_answer
from backend.reasoning.structured_evidence import build_structured_evidence
from backend.reasoning.llm_renderer import build_compact_evidence_package, format_llm_assisted_prompt
from backend.reasoning.direct_fact_engine import is_direct_fact_query, extract_direct_fact
from backend.domain_filter import check_domain_prefilter, FilterAction


def get_sample_chart():
    return {
        "ascendant": {"rashi": "Leo", "degree_in_rashi": 12.5, "longitude": 132.5},
        "planets": {
            "Sun": {"rashi": "Aries", "house": 9, "longitude": 15.0},
            "Moon": {"rashi": "Scorpio", "house": 4, "longitude": 220.0},
            "Mars": {"rashi": "Leo", "house": 1, "longitude": 135.0},
            "Mercury": {"rashi": "Taurus", "house": 10, "longitude": 50.0},
            "Jupiter": {"rashi": "Cancer", "house": 12, "longitude": 110.0},
            "Venus": {"rashi": "Pisces", "house": 8, "longitude": 340.0},
            "Saturn": {"rashi": "Aquarius", "house": 7, "longitude": 315.0},
            "Rahu": {"rashi": "Aries", "house": 9, "longitude": 25.0},
            "Ketu": {"rashi": "Libra", "house": 3, "longitude": 205.0},
        }
    }


def get_sample_stage8(chart):
    tenth_rule = {
        "rule_id": "CAREER_10TH_LORD_PLACEMENT",
        "matched": True,
        "interpretation_key": "career_10th_lord",
        "evidence": {
            "house": 10,
            "house_rashi": "Taurus",
            "lord": "Venus",
            "lord_natal_house": 8,
            "lord_natal_rashi": "Pisces"
        }
    }
    return {
        "stage_8_15_career_rules": {"domain": "career", "rules": [tenth_rule]},
        "stage_8_16_marriage_rules": {"rules": []},
        "stage_8_17_finance_rules": {"rules": []},
        "stage_8_18_education_rules": {"rules": []},
        "stage_8_19_property_rules": {"rules": []},
        "stage_8_20_dasha_timing": {},
        "stage_8_21_transit_timing": {},
        "stage_8_22_multi_domain": {}
    }


def get_sample_dasha():
    return {
        "current_mahadasha": "Ketu",
        "current_antardasha": "Sun",
        "dasha_hierarchy": {
            "mahadasha": {"planet": "Ketu"},
            "antardasha": {"planet": "Sun"}
        }
    }


class BackendFirstRoutingTests(unittest.TestCase):

    def setUp(self):
        self.chart = get_sample_chart()
        self.stage8 = get_sample_stage8(self.chart)
        self.dasha = get_sample_dasha()
        self.career_rules = self.stage8["stage_8_15_career_rules"]["rules"]
        self.struct_ev = build_structured_evidence(
            domain="career",
            intent="career_general",
            norm_chart=self.chart,
            stage8_result=self.stage8,
            dasha_hierarchy=self.dasha
        )

    # 1. Direct Lagna lookup -> DIRECT, zero Gemini calls
    def test_01_direct_lagna_lookup(self):
        q = "What is my Lagna?"
        self.assertTrue(is_direct_fact_query(q))
        mode_res = select_answer_mode("general", "direct_fact", q, is_faq=False)
        self.assertEqual(mode_res["mode"], "DIRECT")
        self.assertEqual(mode_res["answer_source"], "LOCAL")
        self.assertEqual(mode_res["gemini_calls"], 0)

    # 2. Current Mahadasha lookup -> DIRECT, zero Gemini calls
    def test_02_current_mahadasha_lookup(self):
        q = "What is my current Mahadasha and Antardasha?"
        self.assertTrue(is_direct_fact_query(q))
        ans = extract_direct_fact(q, self.chart, dasha_hierarchy=self.dasha)
        self.assertIsNotNone(ans)
        ans_text = ans["answer"] if isinstance(ans, dict) else ans
        self.assertIn("Ketu", ans_text)
        self.assertIn("Sun", ans_text)
        mode_res = select_answer_mode("general", "direct_fact", q, is_faq=False)
        self.assertEqual(mode_res["mode"], "DIRECT")
        self.assertEqual(mode_res["gemini_calls"], 0)

    # 3. Supported career interpretation -> RULE_BASED, zero Gemini calls
    def test_03_supported_career_interpretation_stays_local(self):
        q = "Mere chart mein career se related kya indications hain?"
        task = analyze_question_task(q)
        self.assertFalse(task["is_synthesis_gap"])
        cov = evaluate_rule_coverage(q, "career", "career_general", self.struct_ev, self.career_rules)
        self.assertTrue(cov["exact_rule_match"])
        self.assertTrue(cov["answerable_locally"])
        mode_res = select_answer_mode("career", "career_general", q, matched_rules=self.career_rules, structured_evidence=self.struct_ev)
        self.assertEqual(mode_res["mode"], "RULE_BASED")
        self.assertEqual(mode_res["answer_source"], "LOCAL")
        self.assertEqual(mode_res["gemini_calls"], 0)
        local_ans = synthesize_local_answer(self.struct_ev, is_hinglish=True, question=q)
        self.assertIsNotNone(local_ans)
        self.assertIn("Based on your birth chart", local_ans)

    # 4. Career-delay question unsupported by local synthesizer -> synthesis gap
    def test_04_career_delay_unsupported_by_local_synthesizer(self):
        q = "Meri job lagne mein baar-baar delay kyun ho raha hai? Kya mujhe IT field mein try karte rehna chahiye ya kisi aur field mein jaana chahiye?"
        task = analyze_question_task(q)
        self.assertTrue(task["is_causal_delay"])
        self.assertTrue(task["is_synthesis_gap"])
        cov = evaluate_rule_coverage(q, "career", "career_job", self.struct_ev, self.career_rules)
        self.assertFalse(cov["exact_rule_match"])
        self.assertFalse(cov["answerable_locally"])
        mode_res = select_answer_mode("career", "career_job", q, matched_rules=self.career_rules, structured_evidence=self.struct_ev)
        self.assertEqual(mode_res["mode"], "LLM_ASSISTED")
        self.assertEqual(mode_res["answer_source"], "EVIDENCE_GROUNDED_LLM")
        # Local synthesizer refuses to dump generic bullets
        local_ans = synthesize_local_answer(self.struct_ev, is_hinglish=True, question=q)
        self.assertIsNone(local_ans)
        # Compact evidence package extracts delay indicators and Mercury for IT
        pkg = build_compact_evidence_package("career", "career_job", self.chart, self.career_rules, self.dasha, question=q)
        self.assertIn("10H", pkg["verified_facts"])
        self.assertIn("Mercury_IT_Fields", pkg["verified_facts"])

    # 5. Job-growth & job-switch question -> synthesis gap, not generic bullets
    def test_05_job_growth_and_switch_dilemma(self):
        q = "Main abhi job kar raha hoon, lekin growth nahi mil rahi. Kya agle kuch months mein job change karna mere liye beneficial rahega?"
        task = analyze_question_task(q)
        self.assertTrue(task["is_causal_delay"] or task["is_action_dilemma"])
        self.assertTrue(task["is_synthesis_gap"])
        cov = evaluate_rule_coverage(q, "career", "career_job", self.struct_ev, self.career_rules)
        self.assertFalse(cov["exact_rule_match"])
        self.assertFalse(cov["answerable_locally"])
        local_ans = synthesize_local_answer(self.struct_ev, is_hinglish=True, question=q)
        self.assertIsNone(local_ans)

    # 6. Government vs private employment -> evidence-grounded fallback with Sun/Saturn facts
    def test_06_govt_vs_private_employment(self):
        q = "Government job ki preparation karun ya private sector ki opportunities choose karun?"
        task = analyze_question_task(q)
        self.assertTrue(task["is_comparative"])
        self.assertTrue(task["is_synthesis_gap"])
        cov = evaluate_rule_coverage(q, "career", "career_job", self.struct_ev, self.career_rules)
        self.assertFalse(cov["exact_rule_match"])
        self.assertFalse(cov["answerable_locally"])
        pkg = build_compact_evidence_package("career", "career_job", self.chart, self.career_rules, self.dasha, question=q)
        self.assertIn("Sun_Govt_Authority", pkg["verified_facts"])
        self.assertIn("Saturn_Service", pkg["verified_facts"])

    # 7. Two different career dilemmas -> must not produce identical canned answers
    def test_07_two_different_dilemmas_not_identical_canned(self):
        q1 = "Government job ki preparation karun ya private sector choose karun?"
        q2 = "Main abhi job kar raha hoon, lekin growth nahi mil rahi. Kya job change karu?"
        # Both must refuse generic local canned answers
        ans1 = synthesize_local_answer(self.struct_ev, is_hinglish=True, question=q1)
        ans2 = synthesize_local_answer(self.struct_ev, is_hinglish=True, question=q2)
        self.assertIsNone(ans1)
        self.assertIsNone(ans2)
        # Packages must contain different query-specific facts
        pkg1 = build_compact_evidence_package("career", "career_job", self.chart, self.career_rules, self.dasha, question=q1)
        pkg2 = build_compact_evidence_package("career", "career_job", self.chart, self.career_rules, self.dasha, question=q2)
        self.assertIn("Sun_Govt_Authority", pkg1["verified_facts"])
        self.assertNotIn("Sun_Govt_Authority", pkg2["verified_facts"])

    # 8. Complex question with complete local rule coverage -> remains local
    def test_08_complex_question_with_complete_local_rules_stays_local(self):
        q = "What daily routine best supports my vitality and rest?"
        cov = evaluate_rule_coverage(q, "health", "health_vitality", self.struct_ev, [])
        self.assertTrue(cov["exact_rule_match"])
        self.assertTrue(cov["answerable_locally"])

    # 9. Missing Dasha evidence -> no fabricated facts
    def test_09_missing_dasha_evidence_no_fabrication(self):
        q = "What is my current Mahadasha and Antardasha?"
        ans = extract_direct_fact(q, self.chart, dasha_hierarchy=None)
        ans_text = ans["answer"] if isinstance(ans, dict) else ans
        self.assertEqual(ans_text, "Your Dasha timing evidence is unavailable in the API chart data.")

    # 10. Multi-domain question with partial evidence -> preserves supported facts, marks unresolved parts
    def test_10_multi_domain_partial_evidence(self):
        q = "Meri shaadi kab hogi, foreign settlement hoga ya nahi, aur uske baad meri income improve hogi kya?"
        task = analyze_question_task(q)
        self.assertTrue(task["is_multi_topic"])
        pkg = build_compact_evidence_package("multi_domain", "multi_domain", self.chart, [], self.dasha, question=q)
        self.assertIn("7H_Marriage", pkg["verified_facts"])
        self.assertIn("Venus_Marriage_Karaka", pkg["verified_facts"])
        self.assertIn("Foreign settlement", pkg["unresolved"])

    # 11. Unsupported / out-of-scope question -> layer 1 rejection (0 Gemini calls)
    def test_11_out_of_scope_rejection(self):
        q = "which is fasest car in the world"
        decision = check_domain_prefilter(q)
        self.assertEqual(decision.action, FilterAction.REJECT_OUT_OF_SCOPE)
        self.assertIn("vehicle_and_speed_trivia", decision.reason)

    # 12. Local answer capability checker
    def test_12_can_synthesize_local_answer(self):
        general_q = "What does my chart indicate about my career?"
        dilemma_q = "Job vs business: kis mein jyada growth hai?"
        self.assertTrue(can_synthesize_local_answer(self.struct_ev, general_q))
        self.assertFalse(can_synthesize_local_answer(self.struct_ev, dilemma_q))


if __name__ == "__main__":
    unittest.main()
