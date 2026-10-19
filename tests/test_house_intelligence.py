"""
Unit and Integration Tests for House Intelligence & Dynamic Evidence Extraction.
Verifies whole-sign canonical consistency, missing fields, empty houses, all 12 houses,
ordinal variations, compound queries (property + loan), and route preservation.
"""

import unittest
from backend.reasoning.lordship import get_house_details, RASHI_LORDS, RASHI_LIST
from backend.reasoning.direct_fact_engine import (
    is_direct_fact_query,
    extract_direct_fact,
    extract_house_lord_query,
    extract_house_sign_query,
    extract_house_lord_placement_query,
    extract_house_occupants_query,
)
from backend.reasoning.llm_renderer import (
    build_compact_evidence_package,
    format_llm_assisted_prompt,
    get_renderer_system_instruction,
)
from backend.reasoning.rule_coverage import evaluate_rule_coverage


class HouseDetailsCanonicalTests(unittest.TestCase):
    def setUp(self):
        # Leo Ascendant chart
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

    def test_all_twelve_houses_for_leo_ascendant(self):
        expected_signs = [
            (1, "Leo", "Sun"),
            (2, "Virgo", "Mercury"),
            (3, "Libra", "Venus"),
            (4, "Scorpio", "Mars"),
            (5, "Sagittarius", "Jupiter"),
            (6, "Capricorn", "Saturn"),
            (7, "Aquarius", "Saturn"),
            (8, "Pisces", "Jupiter"),
            (9, "Aries", "Mars"),
            (10, "Taurus", "Venus"),
            (11, "Gemini", "Mercury"),
            (12, "Cancer", "Moon"),
        ]
        for h_num, exp_sign, exp_lord in expected_signs:
            with self.subTest(house=h_num):
                det = get_house_details(self.leo_chart, h_num)
                self.assertTrue(det["data_available"])
                self.assertEqual(det["sign"], exp_sign)
                self.assertEqual(det["lord"], exp_lord)

    def test_empty_house_with_sufficient_data(self):
        # In leo_chart, house 4 has NO planets, and 9 planets are present
        det = get_house_details(self.leo_chart, 4)
        self.assertTrue(det["occupants_available"])
        self.assertTrue(det["is_empty"])
        self.assertEqual(det["occupants"], [])

    def test_occupied_house(self):
        # In leo_chart, house 9 has Sun, Mercury, Ketu
        det = get_house_details(self.leo_chart, 9)
        self.assertTrue(det["occupants_available"])
        self.assertFalse(det["is_empty"])
        self.assertEqual(det["occupants"], ["Ketu", "Mercury", "Sun"])

    def test_incomplete_planet_data_does_not_report_empty_house(self):
        # Chart with only Sun
        incomplete_chart = {
            "ascendant": {"rashi": "Leo"},
            "planets": {"Sun": {"house": 9, "rashi": "Aries"}},
        }
        det = get_house_details(incomplete_chart, 4)
        self.assertTrue(det["data_available"])
        self.assertEqual(det["sign"], "Scorpio")
        # Occupants MUST NOT be reported as empty because data is incomplete
        self.assertFalse(det["occupants_available"])
        self.assertFalse(det["is_empty"])

    def test_invalid_house_numbers_and_missing_chart_fields(self):
        # House 0, House 13, negative
        self.assertFalse(get_house_details(self.leo_chart, 0)["data_available"])
        self.assertFalse(get_house_details(self.leo_chart, 13)["data_available"])
        self.assertFalse(get_house_details(self.leo_chart, -1)["data_available"])
        self.assertFalse(get_house_details(None, 4)["data_available"])
        self.assertFalse(get_house_details({}, 4)["data_available"])
        self.assertFalse(get_house_details({"ascendant": {"rashi": "InvalidSign"}}, 4)["data_available"])


class DirectFactHouseLookupsTests(unittest.TestCase):
    def setUp(self):
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

    def test_house_lord_lookups_and_ordinal_variations(self):
        queries = [
            ("What is my 7th house lord?", "7th house", "Aquarius", "Saturn"),
            ("Who is the lord of my 4th house?", "4th house", "Scorpio", "Mars"),
            ("What is my fourth house lord?", "4th house", "Scorpio", "Mars"),
            ("Who rules my 10th house?", "10th house", "Taurus", "Venus"),
            ("what is my 1st lord", "1st house", "Leo", "Sun"),
            ("what is my 12th house lord", "12th house", "Cancer", "Moon"),
        ]
        for q, h_str, exp_sign, exp_lord in queries:
            with self.subTest(query=q):
                self.assertTrue(is_direct_fact_query(q))
                ans = extract_direct_fact(q, self.leo_chart)
                self.assertIsNotNone(ans)
                self.assertEqual(ans["answer_mode"], "DIRECT")
                self.assertEqual(ans["gemini_calls"], 0)
                self.assertIn(exp_sign, ans["answer"])
                self.assertIn(exp_lord, ans["answer"])

    def test_house_sign_lookups(self):
        queries = [
            ("What sign is in my 4th house?", "Scorpio"),
            ("Which rashi is in my 9th house?", "Aries"),
            ("what sign is in my seventh house", "Aquarius"),
        ]
        for q, exp_sign in queries:
            with self.subTest(query=q):
                self.assertTrue(is_direct_fact_query(q))
                ans = extract_direct_fact(q, self.leo_chart)
                self.assertIsNotNone(ans)
                self.assertIn(exp_sign, ans["answer"])

    def test_house_occupants_empty_and_occupied(self):
        # 4th house is empty
        q_empty = "Which planets are in my 4th house?"
        self.assertTrue(is_direct_fact_query(q_empty))
        ans_empty = extract_direct_fact(q_empty, self.leo_chart)
        self.assertIn("There are no natal planets positioned", ans_empty["answer"])

        # 9th house has planets
        q_occ = "Which planets are in my 9th house?"
        self.assertTrue(is_direct_fact_query(q_occ))
        ans_occ = extract_direct_fact(q_occ, self.leo_chart)
        self.assertIn("Sun", ans_occ["answer"])
        self.assertIn("Mercury", ans_occ["answer"])
        self.assertIn("Ketu", ans_occ["answer"])

    def test_all_planetary_placements_lookup(self):
        queries = [
            "which planet is my which house",
            "which planet is in which house",
            "where are my planets placed",
            "planetary placements"
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertTrue(is_direct_fact_query(q))
                ans = extract_direct_fact(q, self.leo_chart)
                self.assertIsNotNone(ans)
                self.assertEqual(ans["answer_mode"], "DIRECT")
                self.assertEqual(ans["gemini_calls"], 0)
                self.assertIn("Sun", ans["answer"])
                self.assertIn("Moon", ans["answer"])
                self.assertIn("Mars", ans["answer"])
                self.assertIn("House 9", ans["answer"])
                self.assertIn("House 10", ans["answer"])

    def test_compound_and_interpretive_queries_are_not_intercepted(self):
        compound_queries = [
            "What does my 4th house indicate about purchasing a house, and will taking a home loan cause stress?",
            "What does my 4th house indicate about purchasing a house?",
            "What does my 11th house indicate about gains?",
            "What does my chart suggest about studying abroad?",
            "Why do I keep experiencing similar relationship patterns?",
        ]
        for q in compound_queries:
            with self.subTest(query=q):
                # Must not be trapped as a 1-line direct fact
                self.assertFalse(is_direct_fact_query(q))


class DynamicEvidenceExtractionTests(unittest.TestCase):
    def setUp(self):
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

    def test_property_and_loan_compound_query_extracts_both_4h_and_6h(self):
        q = "What does my 4th house indicate about purchasing a house, and will taking a home loan cause stress?"
        pkg = build_compact_evidence_package(
            domain="property",
            intent="property_purchase",
            chart_data=self.leo_chart,
            matched_rules=[],
            question=q,
        )
        facts = pkg["verified_facts"]
        # 4H (Property) facts
        self.assertEqual(facts.get("4H"), "Scorpio")
        self.assertEqual(facts.get("4L"), "Mars@12H")
        self.assertIn("Mars_Property_Karaka", facts)

        # 6H (Loan) facts
        self.assertEqual(facts.get("6H_Loan"), "Capricorn")
        self.assertEqual(facts.get("6L_Loan"), "Saturn@8H")

    def test_study_abroad_extracts_both_9h_and_12h(self):
        q = "What does my chart suggest about studying abroad?"
        pkg = build_compact_evidence_package(
            domain="education",
            intent="education_higher",
            chart_data=self.leo_chart,
            matched_rules=[],
            question=q,
        )
        facts = pkg["verified_facts"]
        self.assertEqual(facts.get("9H"), "Aries")
        self.assertEqual(facts.get("9L"), "Mars@12H")
        self.assertEqual(facts.get("12H"), "Cancer")
        self.assertEqual(facts.get("12L"), "Moon@10H")

    def test_gains_query_extracts_11h_and_jupiter(self):
        q = "What does my 11th house indicate about gains?"
        pkg = build_compact_evidence_package(
            domain="finance",
            intent="income",
            chart_data=self.leo_chart,
            matched_rules=[],
            question=q,
        )
        facts = pkg["verified_facts"]
        self.assertEqual(facts.get("11H"), "Gemini")
        self.assertEqual(facts.get("11L"), "Mercury@9H")
        self.assertIn("Jupiter_Gains_Karaka", facts)

    def test_tenth_house_inquiry_extracts_10h_facts(self):
        q = "What does my 10th house indicate?"
        pkg = build_compact_evidence_package(
            domain="other",
            intent="definition",
            chart_data=self.leo_chart,
            matched_rules=[],
            question=q,
        )
        facts = pkg["verified_facts"]
        self.assertEqual(facts.get("10H"), "Taurus")
        self.assertEqual(facts.get("10L"), "Venus@10H")
        self.assertIn("Moon", facts.get("10H_occupants", ""))

    def test_marriage_query_extracts_dynamic_7h_not_hardcoded_pisces(self):
        # In Leo ascendant, 7H is Aquarius (NOT Pisces)
        q = "What does Venus indicate about relationships in my chart?"
        pkg = build_compact_evidence_package(
            domain="marriage",
            intent="marriage_general",
            chart_data=self.leo_chart,
            matched_rules=[],
            question=q,
        )
        facts = pkg["verified_facts"]
        self.assertEqual(facts.get("7H"), "Aquarius")
        self.assertEqual(facts.get("7L"), "Saturn@8H")
        self.assertNotEqual(facts.get("7H"), "Pisces")


class RuleCoverageSufficiencyTests(unittest.TestCase):
    def test_compound_property_plus_loan_escalates_to_hybrid(self):
        q = "What does my 4th house indicate about purchasing a house, and will taking a home loan cause financial stress?"
        cov = evaluate_rule_coverage(
            question=q,
            domain="property",
            intent="property_purchase",
            structured_evidence={"is_sufficient": True},
            matched_rules=[{"rule_id": "PROPERTY_4TH_LORD_PLACEMENT", "matched": True}],
        )
        # Because question combines property and loan, exact_rule_match must be False to escalate to Hybrid
        self.assertFalse(cov["exact_rule_match"])
        self.assertFalse(cov["answerable_locally"])


if __name__ == "__main__":
    unittest.main()
