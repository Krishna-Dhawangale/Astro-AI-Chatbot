import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from router import is_astrology_query, validate_astrology_domain, OUT_OF_SCOPE_RESPONSE
from abuse_guardrail import (
    check_abuse_guardrail,
    reset_abuse_session,
    WARNING_TEMPLATE,
    TERMINATED_MESSAGE,
    is_abusive_language,
)
from llm_router import (
    MAX_TOKENS,
    format_source_badge,
    ensure_two_paragraph_layout,
    format_llm_response,
)


class TestAstroRuleBookAndGuardrails(unittest.TestCase):

    # -----------------------------------------------------------------------
    # Rule 1: DOMAIN RESTRICTION (Astrology Only)
    # -----------------------------------------------------------------------
    def test_domain_restriction_in_scope_queries(self):
        in_scope = [
            "What does my 7th house say about my marriage?",
            "When will I get a promotion in my career?",
            "How will Saturn transit affect my finances and wealth?",
            "What gemstone should I wear for health?",
            "Tell me about my active Rahu Mahadasha",
            "What is my Lagna and Moon sign?",
            "Will I be able to travel abroad for work?",
            "Hi, can you help me read my kundali?",
            "Hello",
            "Namaste",
            "Who are you and what can you do?",
        ]
        for q in in_scope:
            valid, msg = validate_astrology_domain(q)
            self.assertTrue(valid, f"Expected in-scope query to pass: '{q}', but got rejected with '{msg}'")

    def test_domain_restriction_out_of_scope_rejection(self):
        out_of_scope = [
            "Write a python script to reverse a string",
            "How do I debug a NullPointerException in Java?",
            "What is Docker and how do I create a container?",
            "Solve the equation 2x + 5 = 15",
            "What is the square root of 144?",
            "Who won the 2024 US presidential election?",
            "Tell me the recipe for pasta carbonara",
            "Explain quantum physics and theory of relativity",
            "Write an essay about climate change",
            "What is the capital of France?",
        ]
        for q in out_of_scope:
            valid, msg = validate_astrology_domain(q)
            self.assertFalse(valid, f"Expected out-of-scope query to be rejected: '{q}'")
            self.assertEqual(
                msg,
                OUT_OF_SCOPE_RESPONSE,
                f"Expected standard out-of-scope response for '{q}'"
            )

    # -----------------------------------------------------------------------
    # Rule 2: SAFETY & PROFANITY GUARDRAIL (3-Strike Vulgarity Rule)
    # -----------------------------------------------------------------------
    def test_safety_profanity_detection(self):
        # English
        is_bad, _ = is_abusive_language("You are a fucking idiot")
        self.assertTrue(is_bad)
        # Hinglish / Hindi
        is_bad, _ = is_abusive_language("Yeh kya chutiya response hai")
        self.assertTrue(is_bad)
        is_bad, _ = is_abusive_language("bhosdike chup kar")
        self.assertTrue(is_bad)
        # Devanagari
        is_bad, _ = is_abusive_language("मादरचोद")
        self.assertTrue(is_bad)

    def test_three_strike_rule_and_lockout(self):
        test_session = "test_user_strike_system"
        reset_abuse_session(test_session)

        # Strike 1: Warning (1/3)
        res1 = check_abuse_guardrail(test_session, "What the fuck is this?")
        self.assertIsNotNone(res1)
        self.assertEqual(res1["strikes"], 1)
        self.assertFalse(res1["terminated"])
        self.assertIn("Warning (1/3)", res1["message"])
        self.assertEqual(res1["message"], "⚠️ Warning (1/3): Inappropriate or vulgar language is not permitted.")

        # Strike 2: Warning (2/3)
        res2 = check_abuse_guardrail(test_session, "You stupid bitch")
        self.assertIsNotNone(res2)
        self.assertEqual(res2["strikes"], 2)
        self.assertFalse(res2["terminated"])
        self.assertIn("Warning (2/3)", res2["message"])
        self.assertEqual(res2["message"], "⚠️ Warning (2/3): Inappropriate or vulgar language is not permitted.")

        # Strike 3: Termination
        res3 = check_abuse_guardrail(test_session, "Saale bhosdike")
        self.assertIsNotNone(res3)
        self.assertEqual(res3["strikes"], 3)
        self.assertTrue(res3["terminated"])
        self.assertEqual(res3["message"], TERMINATED_MESSAGE)
        self.assertEqual(res3["message"], "🚫 Session Terminated: You have exceeded the limit for inappropriate language.")

        # Strike 4 / Subsequent input: Session must remain TERMINATED and locked!
        res4 = check_abuse_guardrail(test_session, "When will I get married?")
        self.assertIsNotNone(res4, "Terminated session must block subsequent clean queries!")
        self.assertTrue(res4["terminated"])
        self.assertEqual(res4["message"], TERMINATED_MESSAGE)

        # Reset session allows clean messages again
        reset_abuse_session(test_session)
        res5 = check_abuse_guardrail(test_session, "When will I get married?")
        self.assertIsNone(res5, "Clean query after reset should be allowed!")

    # -----------------------------------------------------------------------
    # Rule 3: TOKEN & OUTPUT BUDGET CONSTRAINT
    # -----------------------------------------------------------------------
    def test_max_tokens_budget(self):
        self.assertEqual(MAX_TOKENS, 140, "MAX_TOKENS must be strictly 140.")

    def test_provenance_source_badge(self):
        # Local Database / API
        b_local = format_source_badge("Local Database")
        self.assertEqual(b_local.strip(), "[Local Database / FreeAstrologyAPI]")

        b_api = format_source_badge("FreeAstrologyAPI")
        self.assertEqual(b_api.strip(), "[Local Database / FreeAstrologyAPI]")

        # LLMs
        b_llama = format_source_badge("llama-3.3-70b-versatile")
        self.assertEqual(b_llama.strip(), "[LLM - Llama 3.3 70B]")

        b_gemini = format_source_badge("gemini-3.5-flash")
        self.assertEqual(b_gemini.strip(), "[LLM - Gemini 3.5 Flash]")

    def test_structured_two_paragraph_layout(self):
        sample_output = (
            "Your 10th house is governed by Mercury, indicating sharp professional communication. "
            "The active Rahu period encourages technological ventures and career pivots. "
            "Focus on structured planning during the upcoming Jupiter transit. "
            "Wear an emerald after consulting your charts and maintain consistent daily focus."
        )
        formatted = ensure_two_paragraph_layout(sample_output)
        self.assertIn("**Astrological Insight:**", formatted)
        self.assertIn("**Actionable Advice:**", formatted)
        parts = formatted.split("\n\n")
        self.assertEqual(len(parts), 2, "Must contain exactly two paragraphs separated by double newline.")

        # Full formatting with provenance suffix
        full = format_llm_response(sample_output, model_id="llama-3.3-70b-versatile", include_suffix=True)
        self.assertIn("**Astrological Insight:**", full)
        self.assertIn("**Actionable Advice:**", full)
        self.assertIn("[LLM - Llama 3.3 70B]", full)


if __name__ == "__main__":
    unittest.main()
