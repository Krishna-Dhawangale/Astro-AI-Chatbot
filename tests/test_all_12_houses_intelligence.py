# -*- coding: utf-8 -*-
"""
tests/test_all_12_houses_intelligence.py
-----------------------------------------
Comprehensive Acceptance & Unit Test Suite for All 12 Houses (1..12).
Verifies:
1. All 12 Houses House Sign, Lord, Placement, and Occupants resolution.
2. Phrasing Variations (e.g., "occupies", "rules", "cusp", "10h", "who is in").
3. Capability Gate routing (Direct Fact vs Compound/Predictive LLM route).
4. Evidence Builder complete injection for all 12 houses into LLM evidence package.
5. Strict distinction between an empty house and missing chart data.
"""

import sys
import os
import re
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.reasoning.lordship import get_house_details, RASHI_LORDS, RASHI_LIST
from backend.router.semantic_parser import (
    analyze_query_intent,
    extract_all_houses,
    AstrologicalIntent,
)
from backend.router.capability_gate import (
    evaluate_capability,
    ExecutionPath,
    ordinal_house,
)
from backend.reasoning.direct_fact_engine import (
    is_direct_fact_query,
    extract_direct_fact,
)
from backend.reasoning.llm_renderer import (
    build_compact_evidence_package,
    format_llm_assisted_prompt,
)


class AllTwelveHousesIntelligenceTests(unittest.TestCase):
    def setUp(self):
        # Leo Ascendant canonical test chart
        # Leo = 1H, Virgo = 2H, Libra = 3H, Scorpio = 4H, Sagittarius = 5H, Capricorn = 6H,
        # Aquarius = 7H, Pisces = 8H, Aries = 9H, Taurus = 10H, Gemini = 11H, Cancer = 12H
        self.leo_chart = {
            "ascendant": {"rashi": "Leo", "longitude": 130.0},
            "planets": {
                "Sun": {"house": 9, "rashi": "Aries"},
                "Moon": {"house": 10, "rashi": "Taurus"},
                "Mars": {"house": 12, "rashi": "Cancer"},
                "Mercury": {"house": 9, "rashi": "Aries"},
                "Jupiter": {"house": 11, "rashi": "Gemini"},
                "Venus": {"house": 10, "rashi": "Taurus"},
                "Saturn": {"house": 8, "rashi": "Pisces"},
                "Rahu": {"house": 3, "rashi": "Libra"},
                "Ketu": {"house": 9, "rashi": "Aries"},
            },
        }

        # Aries Ascendant test chart
        self.aries_chart = {
            "ascendant": {"rashi": "Aries", "longitude": 15.0},
            "planets": {
                "Sun": {"house": 1, "rashi": "Aries"},
                "Moon": {"house": 2, "rashi": "Taurus"},
                "Mars": {"house": 1, "rashi": "Aries"},
                "Mercury": {"house": 2, "rashi": "Taurus"},
                "Jupiter": {"house": 3, "rashi": "Gemini"},
                "Venus": {"house": 4, "rashi": "Cancer"},
                "Saturn": {"house": 5, "rashi": "Leo"},
                "Rahu": {"house": 6, "rashi": "Virgo"},
                "Ketu": {"house": 12, "rashi": "Pisces"},
            },
        }

        self.dasha = {
            "current_mahadasha": "Rahu",
            "current_antardasha": "Moon",
        }

    # =========================================================================
    # 1. Parameterized Verification for All 12 Houses (Sign, Lord, Lord House)
    # =========================================================================
    def test_all_12_houses_leo_ascendant(self):
        leo_expected = [
            (1, "Leo", "Sun", 9, "Aries"),
            (2, "Virgo", "Mercury", 9, "Aries"),
            (3, "Libra", "Venus", 10, "Taurus"),
            (4, "Scorpio", "Mars", 12, "Cancer"),
            (5, "Sagittarius", "Jupiter", 11, "Gemini"),
            (6, "Capricorn", "Saturn", 8, "Pisces"),
            (7, "Aquarius", "Saturn", 8, "Pisces"),
            (8, "Pisces", "Jupiter", 11, "Gemini"),
            (9, "Aries", "Mars", 12, "Cancer"),
            (10, "Taurus", "Venus", 10, "Taurus"),
            (11, "Gemini", "Mercury", 9, "Aries"),
            (12, "Cancer", "Moon", 10, "Taurus"),
        ]
        for h, exp_sign, exp_lord, exp_lord_h, exp_lord_s in leo_expected:
            with self.subTest(house=h):
                det = get_house_details(self.leo_chart, h)
                self.assertTrue(det["data_available"])
                self.assertEqual(det["sign"], exp_sign)
                self.assertEqual(det["lord"], exp_lord)
                self.assertTrue(det["lord_placement_available"])
                self.assertEqual(det["lord_house"], exp_lord_h)
                self.assertEqual(det["lord_sign"], exp_lord_s)

    # =========================================================================
    # 2. House Occupants Across All 12 Houses (Occupied vs Empty vs Incomplete)
    # =========================================================================
    def test_all_12_houses_occupants_distinction(self):
        # Leo chart occupant distribution:
        # 1H: empty, 2H: empty, 3H: Rahu, 4H: empty, 5H: empty, 6H: empty,
        # 7H: empty, 8H: Saturn, 9H: Ketu, Mercury, Sun, 10H: Moon, Venus,
        # 11H: Jupiter, 12H: Mars
        expected_occupants = {
            1: [],
            2: [],
            3: ["Rahu"],
            4: [],
            5: [],
            6: [],
            7: [],
            8: ["Saturn"],
            9: ["Ketu", "Mercury", "Sun"],
            10: ["Moon", "Venus"],
            11: ["Jupiter"],
            12: ["Mars"],
        }
        for h, exp_occ in expected_occupants.items():
            with self.subTest(house=h):
                det = get_house_details(self.leo_chart, h)
                self.assertTrue(det["occupants_available"])
                self.assertEqual(det["occupants"], exp_occ)
                self.assertEqual(det["is_empty"], len(exp_occ) == 0)

    def test_incomplete_chart_occupants_unavailable(self):
        incomplete_chart = {
            "ascendant": {"rashi": "Leo"},
            "planets": {"Sun": {"house": 9, "rashi": "Aries"}},
        }
        for h in range(1, 13):
            with self.subTest(house=h):
                det = get_house_details(incomplete_chart, h)
                self.assertTrue(det["data_available"])
                self.assertFalse(det["occupants_available"])
                self.assertFalse(det["is_empty"])

    # =========================================================================
    # 3. Phrasing Variations across All 12 Houses
    # =========================================================================
    def test_phrasing_variations_house_sign(self):
        # Questions asking about house sign with different phrasings
        queries = [
            ("Which sign occupies my 10th house?", 10),
            ("Which sign is in my 7th house?", 7),
            ("What sign occupies my first house?", 1),
            ("What zodiac sign is on my 4th house?", 4),
            ("Which rashi occupies my 8th house?", 8),
            ("What sign is in 10h?", 10),
            ("Which sign is on my 12th house?", 12),
        ]
        for q, expected_house in queries:
            with self.subTest(query=q):
                prof = analyze_query_intent(q)
                self.assertEqual(prof.core_intent, AstrologicalIntent.HOUSE_SIGN)
                self.assertEqual(prof.target_houses, [expected_house])
                dec = evaluate_capability(prof, self.leo_chart)
                self.assertEqual(dec.path, ExecutionPath.LOCAL_DIRECT_FACT)
                self.assertIsNotNone(dec.direct_answer)
                self.assertIn("The zodiac sign occupying your", dec.direct_answer["answer"])
                self.assertIsNone(re.search(r"\b1th\b", dec.direct_answer["answer"]))

    def test_phrasing_variations_house_lord(self):
        queries = [
            ("Who rules my 7th house?", 7),
            ("What is my 7th house lord?", 7),
            ("Who is the lord of my 8th house?", 8),
            ("Which planet rules my 1st house?", 1),
            ("Which planet rules my tenth house?", 10),
            ("Who is the ruler of my 12th house?", 12),
        ]
        for q, expected_house in queries:
            with self.subTest(query=q):
                prof = analyze_query_intent(q)
                self.assertEqual(prof.core_intent, AstrologicalIntent.HOUSE_LORD)
                self.assertEqual(prof.target_houses, [expected_house])
                dec = evaluate_capability(prof, self.leo_chart)
                self.assertEqual(dec.path, ExecutionPath.LOCAL_DIRECT_FACT)
                self.assertIsNotNone(dec.direct_answer)
                self.assertIsNone(re.search(r"\b1th\b", dec.direct_answer["answer"]))

    def test_phrasing_variations_house_occupants(self):
        queries = [
            ("Which planet is in my 1st house?", 1),
            ("Who occupies my 10th house?", 10),
            ("What planets are in my 9th house?", 9),
            ("Which planet is sitting in my 8th house?", 8),
            ("What is in my 4th house?", 4),
        ]
        for q, expected_house in queries:
            with self.subTest(query=q):
                prof = analyze_query_intent(q)
                self.assertEqual(prof.core_intent, AstrologicalIntent.HOUSE_OCCUPANTS)
                self.assertEqual(prof.target_houses, [expected_house])
                dec = evaluate_capability(prof, self.leo_chart)
                self.assertEqual(dec.path, ExecutionPath.LOCAL_DIRECT_FACT)
                self.assertIsNotNone(dec.direct_answer)
                # Verify that house 1 says "1st house", NEVER "1th house"
                if expected_house == 1:
                    self.assertIn("1st house", dec.direct_answer["answer"])
                    self.assertIsNone(re.search(r"\b1th\b", dec.direct_answer["answer"]))

    # =========================================================================
    # 4. Capability Gating: Local Direct Fact vs LLM Assisted vs Timing
    # =========================================================================
    def test_compound_multi_house_routes_to_llm(self):
        # Compound queries with multiple houses must route to LLM_ASSISTED
        queries = [
            "What does my 8th house indicate and how does it relate to my 12th house?",
            "What does my 4th house indicate about purchasing a house, and will taking a home loan cause stress?",
            "How does my 9th house connect to studying abroad in my 12th house?",
            "What does my 5th house indicate for education, and how does its lord affect my career?",
        ]
        for q in queries:
            with self.subTest(query=q):
                prof = analyze_query_intent(q)
                self.assertTrue(len(prof.target_houses) > 1 or prof.is_compound)
                dec = evaluate_capability(prof, self.leo_chart)
                self.assertEqual(dec.path, ExecutionPath.LLM_ASSISTED)

    def test_compound_5th_house_education_and_lord_career(self):
        q = "What does my 5th house indicate for education, and how does its lord affect my career?"
        prof = analyze_query_intent(q)
        self.assertIn(5, prof.target_houses)
        self.assertIn(10, prof.target_houses)
        self.assertTrue(prof.is_compound)
        dec = evaluate_capability(prof, self.aries_chart)
        self.assertEqual(dec.path, ExecutionPath.LLM_ASSISTED)

        pkg = build_compact_evidence_package(
            domain="general",
            intent="general",
            chart_data=self.aries_chart,
            matched_rules=[],
            dasha_hierarchy=self.dasha,
            question=q,
        )
        facts = pkg["verified_facts"]
        # Both 5th house and 10th house facts must be present
        self.assertEqual(facts.get("5H"), "Leo")
        self.assertEqual(facts.get("5L"), "Sun@1H")
        self.assertEqual(facts.get("5H_occupants"), "Saturn")
        self.assertEqual(facts.get("10H"), "Capricorn")
        self.assertEqual(facts.get("10L"), "Saturn@5H")
        self.assertEqual(facts.get("10H_occupants"), "None (empty)")
        self.assertIn("Mercury_Intellect_Karaka", facts)
        self.assertIn("Dasha", facts)

    def test_predictive_and_timing_modifiers_route_properly(self):
        # Predictive questions must not be answered with simple static direct facts
        q_pred = "Will my 10th house bring career success this year?"
        prof_pred = analyze_query_intent(q_pred)
        self.assertTrue(prof_pred.has_predictive_modifier or prof_pred.has_timing_modifier)
        dec_pred = evaluate_capability(prof_pred, self.leo_chart)
        self.assertNotEqual(dec_pred.path, ExecutionPath.LOCAL_DIRECT_FACT)

        # Timing question
        q_time = "When will my 7th house manifest marriage according to my dasha?"
        prof_time = analyze_query_intent(q_time)
        dec_time = evaluate_capability(prof_time, self.leo_chart)
        self.assertEqual(dec_time.path, ExecutionPath.TIMING_ENGINE)

    # =========================================================================
    # 5. Evidence Injection into LLM Package for All 12 Houses
    # =========================================================================
    def test_llm_evidence_injection_all_12_houses(self):
        # When ANY house 1..12 is queried, build_compact_evidence_package must inject
        # verified house facts so the LLM never refuses with "insufficient information"
        for h in range(1, 13):
            with self.subTest(house=h):
                q = f"What does my {ordinal_house(h)} house indicate?"
                pkg = build_compact_evidence_package(
                    domain="general",
                    intent="general",
                    chart_data=self.leo_chart,
                    matched_rules=[],
                    dasha_hierarchy=self.dasha,
                    question=q,
                )
                facts = pkg["verified_facts"]
                # Must contain the house sign
                self.assertIn(f"{h}H", facts)
                # Must contain the lord and placement
                self.assertIn(f"{h}L", facts)
                # Must contain occupants or "None (empty)"
                self.assertIn(f"{h}H_occupants", facts)
                # Must contain active Dasha
                self.assertIn("Dasha", facts)

    def test_llm_evidence_injection_multi_house_compound(self):
        # When 8H and 12H are queried together, BOTH houses must be present in facts
        q = "What does my 8th house indicate and how does it relate to my 12th house?"
        pkg = build_compact_evidence_package(
            domain="general",
            intent="general",
            chart_data=self.leo_chart,
            matched_rules=[],
            dasha_hierarchy=self.dasha,
            question=q,
        )
        facts = pkg["verified_facts"]
        self.assertIn("8H", facts)
        self.assertIn("8L", facts)
        self.assertIn("8H_occupants", facts)
        self.assertIn("12H", facts)
        self.assertIn("12L", facts)
        self.assertIn("12H_occupants", facts)

    # =========================================================================
    # 6. Ordinal Helper Correctness (No "1th", "2th", "3th" bugs)
    # =========================================================================
    def test_ordinal_house_helper(self):
        expected_ordinals = {
            1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th",
            6: "6th", 7: "7th", 8: "8th", 9: "9th", 10: "10th",
            11: "11th", 12: "12th"
        }
        for n, exp_str in expected_ordinals.items():
            with self.subTest(num=n):
                self.assertEqual(ordinal_house(n), exp_str)


if __name__ == "__main__":
    unittest.main()
