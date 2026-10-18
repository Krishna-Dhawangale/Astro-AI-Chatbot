import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from fastapi.testclient import TestClient
from main import app, USER_STORE, calculate_house_rulerships
from llm_router import PROFILE_REQUIRED_PROMPT

client = TestClient(app)


class TestSessionIsolationAndContextBinding(unittest.TestCase):

    def setUp(self):
        # Configure User 1 "Krishna": Libra Lagna, Moon Gemini, 10H Cancer (Moon ruled)
        USER_STORE.set_profile("user_krishna_token", {
            "user_id": "user_krishna_token",
            "name": "Krishna",
            "ascendant": "Libra",
            "moon_sign": "Gemini",
            "active_dasha": "Rahu-Mars",
            "user_natal_context": {
                "name": "Krishna",
                "ascendant": "Libra",
                "moon_sign": "Gemini",
                "active_dasha": "Rahu-Mars",
                "house_rulerships": calculate_house_rulerships("Libra"),
            }
        })

        # Configure User 2 "Harshad": Scorpio Lagna, Moon Taurus, 10H Leo (Sun ruled)
        USER_STORE.set_profile("user_harshad_token", {
            "user_id": "user_harshad_token",
            "name": "Harshad",
            "ascendant": "Scorpio",
            "moon_sign": "Taurus",
            "active_dasha": "Jupiter-Saturn",
            "user_natal_context": {
                "name": "Harshad",
                "ascendant": "Scorpio",
                "moon_sign": "Taurus",
                "active_dasha": "Jupiter-Saturn",
                "house_rulerships": calculate_house_rulerships("Scorpio"),
            }
        })

    # -------------------------------------------------------------------------
    # 1. DYNAMIC SESSION USER BINDING VIA AUTHORIZATION HEADER & COOKIES
    # -------------------------------------------------------------------------
    def test_authorization_header_binding_distinct_outputs(self):
        """User 1 (Krishna) and User 2 (Harshad) ask identical question with Bearer tokens."""
        query = "which career suits me best"

        # Krishna with Bearer token
        resp_krishna = client.post(
            "/chat",
            headers={"Authorization": "Bearer user_krishna_token"},
            json={"query": query},
        )
        self.assertEqual(resp_krishna.status_code, 200)
        data_k = resp_krishna.json()
        ans_k = data_k.get("answer") or data_k.get("response") or ""
        self.assertIn("Libra", ans_k)
        self.assertIn("Moon", ans_k)

        # Harshad with Bearer token
        resp_harshad = client.post(
            "/chat",
            headers={"Authorization": "Bearer user_harshad_token"},
            json={"query": query},
        )
        self.assertEqual(resp_harshad.status_code, 200)
        data_h = resp_harshad.json()
        ans_h = data_h.get("answer") or data_h.get("response") or ""
        self.assertIn("Scorpio", ans_h)
        self.assertIn("Sun", ans_h)

        # Strictly distinct personalized answers
        self.assertNotEqual(ans_k, ans_h)

    def test_cookie_session_binding(self):
        """Test extraction of user identity via session cookies."""
        query = "which career suits me best"

        resp_cookie = client.post(
            "/chat",
            cookies={"session_token": "user_harshad_token"},
            json={"query": query},
        )
        self.assertEqual(resp_cookie.status_code, 200)
        ans = resp_cookie.json().get("answer", "")
        self.assertIn("Scorpio", ans)
        self.assertIn("Sun", ans)

    # -------------------------------------------------------------------------
    # 2. PREVENT GLOBAL FALLBACKS (CLEAN PROFILE PROMPT FOR MISSING DATA)
    # -------------------------------------------------------------------------
    def test_missing_profile_returns_profile_required_prompt(self):
        """An unauthenticated or new user with no profile receives a clean setup prompt."""
        query = "which career suits me best"

        resp_unregistered = client.post(
            "/chat",
            headers={"Authorization": "Bearer brand_new_user_without_chart"},
            json={"query": query},
        )
        self.assertEqual(resp_unregistered.status_code, 200)
        data = resp_unregistered.json()
        self.assertEqual(data.get("status"), "profile_required")
        self.assertTrue(data.get("missing_profile"))
        self.assertIn("complete your profile", data.get("answer", "").lower())
        # Must NEVER contain Krishna's Libra default chart
        self.assertNotIn("Libra Lagna", data.get("answer", ""))

    def test_api_chat_missing_profile_prevention(self):
        """Test /api/chat blocks fake chart generation when user profile is missing."""
        resp = client.post(
            "/api/chat",
            headers={"Authorization": "Bearer anonymous_user_999"},
            json={"prompt": "What does my 7th house say about my marriage?"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "profile_required")
        self.assertTrue(data.get("missing_profile"))

    # -------------------------------------------------------------------------
    # 3. NON-PERSONAL GENERAL FAQS STILL ANSWER WITHOUT REQUIRING PROFILE
    # -------------------------------------------------------------------------
    def test_general_faq_without_profile(self):
        """General non-personal concepts (like Sade Sati or Greetings) do not require a profile."""
        resp = client.post(
            "/chat",
            headers={"Authorization": "Bearer anonymous_user_999"},
            json={"query": "Namaste"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertNotEqual(data.get("status"), "profile_required")


if __name__ == "__main__":
    unittest.main()
