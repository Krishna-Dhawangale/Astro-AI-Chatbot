import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from llm_router import classify_career_intent, format_career_rule_response
from main import check_qa_database, UserChartProfile, load_qa_database


class TestCareerIntentDiscrimination(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_qa_database()

    def setUp(self):
        self.natal_context = {
            "name": "Arjun",
            "birth_details": "15 May 1995, 14:30",
            "birth_place": "New Delhi",
            "ascendant": "Virgo",
            "moon_sign": "Taurus",
            "sun_sign": "Aries",
            "active_dasha": "Jupiter-Saturn",
            "house_rulerships": {
                "1st_house": "Mercury (Virgo)",
                "6th_house": "Saturn (Aquarius)",
                "7th_house": "Jupiter (Pisces)",
                "10th_house": "Mercury (Gemini)",
                "11th_house": "Moon (Cancer)",
            }
        }
        self.profile = UserChartProfile(
            name="Arjun",
            lagna="Virgo",
            moon_sign="Taurus",
            sun_sign="Aries",
            current_dasha="Jupiter-Saturn",
        )

    # -----------------------------------------------------------------------
    # Requirement 1: INTENT DISCRIMINATION
    # -----------------------------------------------------------------------
    def test_intent_discrimination_classification(self):
        # 1. Gemstones & Remedies
        gemstone_queries = [
            "which gemstones suit my career",
            "gemstones for career",
            "lucky stone for my job",
            "which ratna should I wear for career growth",
            "remedies for career obstacles",
        ]
        for q in gemstone_queries:
            intent = classify_career_intent(q)
            self.assertEqual(
                intent, "gemstone_remedies",
                f"Query '{q}' expected 'gemstone_remedies', got '{intent}'"
            )

        # 2. Business vs Job
        biz_queries = [
            "should I do business or job",
            "business vs job",
            "does my chart favor business or a job",
            "startup vs corporate job",
            "service or business",
        ]
        for q in biz_queries:
            intent = classify_career_intent(q)
            self.assertEqual(
                intent, "business_vs_job",
                f"Query '{q}' expected 'business_vs_job', got '{intent}'"
            )

        # 3. Job Timing & Promotion
        timing_queries = [
            "when will I get a promotion",
            "timing for job change",
            "when should I change my job",
            "job switch timing",
            "when will I get a new job",
        ]
        for q in timing_queries:
            intent = classify_career_intent(q)
            self.assertEqual(
                intent, "job_timing",
                f"Query '{q}' expected 'job_timing', got '{intent}'"
            )

        # 4. Career Path / Vocational Direction
        path_queries = [
            "which career suits me best",
            "what career should I choose",
            "ideal career path for me",
            "career suitability",
            "suitable profession for my chart",
        ]
        for q in path_queries:
            intent = classify_career_intent(q)
            self.assertEqual(
                intent, "career_path",
                f"Query '{q}' expected 'career_path', got '{intent}'"
            )

    # -----------------------------------------------------------------------
    # Requirement 2: DYNAMIC RULE LOOKUP & DISTINCT RESPONSES
    # -----------------------------------------------------------------------
    def test_distinct_local_rule_responses(self):
        """Verify that distinct career questions DO NOT return the same generic text."""
        session_id = "test_distinct_qa_user"

        ans_gemstones = check_qa_database(
            "which gemstones suit my career",
            profile=self.profile,
            user_natal_context=self.natal_context,
            session_id=session_id + "_gem",
        )
        ans_biz = check_qa_database(
            "should I do business or job",
            profile=self.profile,
            user_natal_context=self.natal_context,
            session_id=session_id + "_biz",
        )
        ans_timing = check_qa_database(
            "when will I get a promotion",
            profile=self.profile,
            user_natal_context=self.natal_context,
            session_id=session_id + "_time",
        )
        ans_path = check_qa_database(
            "which career suits me best",
            profile=self.profile,
            user_natal_context=self.natal_context,
            session_id=session_id + "_path",
        )

        # All four must be non-None and completely distinct!
        self.assertIsNotNone(ans_gemstones)
        self.assertIsNotNone(ans_biz)
        self.assertIsNotNone(ans_timing)
        self.assertIsNotNone(ans_path)

        self.assertNotEqual(ans_gemstones, ans_biz, "Gemstones answer must differ from Business answer")
        self.assertNotEqual(ans_gemstones, ans_timing, "Gemstones answer must differ from Timing answer")
        self.assertNotEqual(ans_biz, ans_timing, "Business answer must differ from Timing answer")
        self.assertNotEqual(ans_path, ans_gemstones, "Career path answer must differ from Gemstones answer")

        # Check Contextual Relevance
        self.assertIn("gemstone", ans_gemstones.lower())
        self.assertTrue("business" in ans_biz.lower() and "job" in ans_biz.lower())
        self.assertTrue("promotion" in ans_timing.lower() or "timing" in ans_timing.lower())
        self.assertTrue("vocational" in ans_path.lower() or "career" in ans_path.lower())

    # -----------------------------------------------------------------------
    # Requirement 2 & 3: ROUTE SPECIFIC SUB-INTENTS WITHOUT EXACT MATCH TO LLM
    # -----------------------------------------------------------------------
    def test_unmatched_specific_sub_intent_routes_to_llm(self):
        """
        If a specific sub-intent (like complex gemstone pairing or nuanced timing)
        does not have an exact local rule match, check_qa_database must return None
        so that it gets routed to the LLM with natal context.
        """
        session_id = "test_llm_routing_user"

        # Nuanced gemstone query
        res_gem = check_qa_database(
            "Can I wear a 5-carat Colombian emerald combined with blue sapphire for my quant trading career?",
            profile=self.profile,
            user_natal_context=self.natal_context,
            session_id=session_id,
        )
        self.assertIsNone(
            res_gem,
            "Nuanced gemstone query must return None from check_qa_database to route to LLM!"
        )

        # Nuanced job timing query
        res_timing = check_qa_database(
            "Is Q3 or Q4 of this year better to resign from my current manager and move abroad?",
            profile=self.profile,
            user_natal_context=self.natal_context,
            session_id=session_id,
        )
        self.assertIsNone(
            res_timing,
            "Nuanced timing query must return None from check_qa_database to route to LLM!"
        )

    # -----------------------------------------------------------------------
    # Requirement 3: INTENT-AWARE FALLBACK RULE GENERATOR
    # -----------------------------------------------------------------------
    def test_format_career_rule_response_is_intent_aware(self):
        fb_gem = format_career_rule_response("which gemstones suit my career", self.natal_context)
        fb_biz = format_career_rule_response("should I do business or job", self.natal_context)
        fb_time = format_career_rule_response("when will I get a promotion", self.natal_context)
        fb_path = format_career_rule_response("which career suits me best", self.natal_context)

        self.assertIn("gemstone", fb_gem.lower())
        self.assertIn("business", fb_biz.lower())
        self.assertIn("transition", fb_time.lower())
        self.assertIn("vocational", fb_path.lower())


if __name__ == "__main__":
    unittest.main()
