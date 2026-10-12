import unittest
from backend.llm_router import (
    classify_micro_intent,
    classify_marriage_intent,
    classify_career_intent,
    format_marriage_rule_response,
    format_career_rule_response,
    generate_local_rule_fallback,
    calculate_house_rulerships,
    build_system_prompt,
)
from backend.main import check_qa_database


class TestMicroIntentArchitecture(unittest.TestCase):
    def setUp(self):
        self.natal_aries = {
            "name": "Arjun",
            "birth_details": "14 April 1995, 06:30 AM",
            "birth_place": "New Delhi, India",
            "ascendant": "Aries",
            "moon_sign": "Leo",
            "active_dasha": "Jupiter-Saturn",
            "house_rulerships": calculate_house_rulerships("Aries"),
        }
        self.natal_libra = {
            "name": "Priya",
            "birth_details": "20 October 1998, 08:15 PM",
            "birth_place": "Mumbai, India",
            "ascendant": "Libra",
            "moon_sign": "Gemini",
            "active_dasha": "Rahu-Venus",
            "house_rulerships": calculate_house_rulerships("Libra"),
        }

    # -------------------------------------------------------------------------
    # 1. GRANULAR MICRO-INTENT CLASSIFICATION
    # -------------------------------------------------------------------------
    def test_granular_micro_intent_classification(self):
        # Career sub-intents
        self.assertEqual(classify_micro_intent("which career suits me best"), "career_path")
        self.assertEqual(classify_micro_intent("what field should I choose for my career"), "career_path")
        self.assertEqual(classify_micro_intent("when will I get a promotion"), "job_timing")
        self.assertEqual(classify_micro_intent("when should I change my job"), "job_timing")
        self.assertEqual(classify_micro_intent("should I do business or job"), "business_vs_job")
        self.assertEqual(classify_micro_intent("job vs business"), "business_vs_job")

        # Marriage sub-intents
        self.assertEqual(classify_micro_intent("when will I get married"), "marriage_timing")
        self.assertEqual(classify_micro_intent("meri shadi kab hogi"), "marriage_timing")
        self.assertEqual(classify_micro_intent("marriage timing"), "marriage_timing")

        self.assertEqual(classify_micro_intent("remedies for delay in marriage"), "marriage_remedies")
        self.assertEqual(classify_micro_intent("shadi me deri ke upay"), "marriage_remedies")
        self.assertEqual(classify_micro_intent("marriage remedies"), "marriage_remedies")

        self.assertEqual(classify_micro_intent("what will my future partner's nature be like"), "partner_traits")
        self.assertEqual(classify_micro_intent("how will my spouse be"), "partner_traits")
        self.assertEqual(classify_micro_intent("spouse personality and traits"), "partner_traits")
        self.assertEqual(classify_micro_intent("partner kaisa hoga"), "partner_traits")

    # -------------------------------------------------------------------------
    # 2. UNIQUE QUESTION-SPECIFIC LOCAL RESPONSES & NON-REPETITIVE IN SAME SESSION
    # -------------------------------------------------------------------------
    def test_distinct_answers_for_different_questions_in_same_session(self):
        session_id = "test_multi_intent_session_101"

        # Ask Question 1: Marriage Timing
        ans_timing = check_qa_database(
            "when will I get married",
            user_natal_context=self.natal_aries,
            session_id=session_id,
        )

        # Ask Question 2 in SAME SESSION: Partner Nature
        ans_partner = check_qa_database(
            "what will my future partner's nature be like",
            user_natal_context=self.natal_aries,
            session_id=session_id,
        )

        # Ask Question 3 in SAME SESSION: Marriage Remedies
        ans_remedies = check_qa_database(
            "remedies for delay in marriage",
            user_natal_context=self.natal_aries,
            session_id=session_id,
        )

        # Ask Question 4 in SAME SESSION: Career Path
        ans_career = check_qa_database(
            "which career suits me best",
            user_natal_context=self.natal_aries,
            session_id=session_id,
        )

        # Ask Question 5 in SAME SESSION: Business vs Job
        ans_biz = check_qa_database(
            "should I do business or job",
            user_natal_context=self.natal_aries,
            session_id=session_id,
        )

        # All 5 must be non-None
        self.assertIsNotNone(ans_timing)
        self.assertIsNotNone(ans_partner)
        self.assertIsNotNone(ans_remedies)
        self.assertIsNotNone(ans_career)
        self.assertIsNotNone(ans_biz)

        # All 5 answers must be mutually distinct!
        answers = [ans_timing, ans_partner, ans_remedies, ans_career, ans_biz]
        self.assertEqual(len(answers), len(set(answers)), "All responses in same session must be distinct!")

        # Content relevance validation
        # Partner traits answer MUST describe traits/personality/partner and NOT talk about wedding timing
        self.assertTrue("partner" in ans_partner.lower() or "spouse" in ans_partner.lower())
        self.assertTrue("temperament" in ans_partner.lower() or "nature" in ans_partner.lower() or "personality" in ans_partner.lower() or "equil" in ans_partner.lower() or "disposition" in ans_partner.lower())

        # Remedies answer MUST contain remedy/Lakshmi/Jupiter/mantra
        self.assertTrue("remed" in ans_remedies.lower() or "lakshmi" in ans_remedies.lower() or "mantra" in ans_remedies.lower() or "upay" in ans_remedies.lower())

        # Timing answer MUST contain timing/months/commitment
        self.assertTrue("timing" in ans_timing.lower() or "month" in ans_timing.lower() or "commitment" in ans_timing.lower())

        # Business vs Job answer MUST contain business and job
        self.assertTrue("business" in ans_biz.lower() and "job" in ans_biz.lower())

    # -------------------------------------------------------------------------
    # 3. DYNAMIC USER PERSONALIZATION
    # -------------------------------------------------------------------------
    def test_dynamic_user_personalization_by_chart(self):
        # Aries Lagna has 7th house Libra ruled by Venus
        resp_aries = format_marriage_rule_response(
            "what will my partner's nature be like",
            user_natal_context=self.natal_aries,
            hinglish=False,
        )
        self.assertIn("Libra", resp_aries)
        self.assertIn("Venus", resp_aries)
        self.assertIn("graceful", resp_aries.lower())

        # Libra Lagna has 7th house Aries ruled by Mars
        resp_libra = format_marriage_rule_response(
            "what will my partner's nature be like",
            user_natal_context=self.natal_libra,
            hinglish=False,
        )
        self.assertIn("Aries", resp_libra)
        self.assertIn("Mars", resp_libra)
        self.assertIn("energetic", resp_libra.lower())

        # The two readings must be completely customized to the users' respective charts
        self.assertNotEqual(resp_aries, resp_libra)

    # -------------------------------------------------------------------------
    # 4. LLM PROMPT DIRECTIVE INJECTION
    # -------------------------------------------------------------------------
    def test_llm_prompt_directive_injection(self):
        prompt_partner = build_system_prompt(
            user_natal_context=self.natal_aries,
            query="what is my future partner's nature",
        )
        self.assertIn("PARTNER TRAITS", prompt_partner)
        self.assertIn("DO NOT answer about marriage timing", prompt_partner)

        prompt_timing = build_system_prompt(
            user_natal_context=self.natal_aries,
            query="when will I get married",
        )
        self.assertIn("MARRIAGE TIMING", prompt_timing)

        prompt_biz = build_system_prompt(
            user_natal_context=self.natal_aries,
            query="should I do business or job",
        )
        self.assertIn("BUSINESS VS JOB", prompt_biz)


if __name__ == "__main__":
    unittest.main()
