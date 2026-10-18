import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from fastapi.testclient import TestClient
from main import app
from abuse_guardrail import reset_abuse_session, TERMINATED_MESSAGE
from router import OUT_OF_SCOPE_RESPONSE

client = TestClient(app)


class TestFastApiMiddlewareAndEndpoints(unittest.TestCase):

    def setUp(self):
        self.session_id = "integration_test_user_99"
        reset_abuse_session(self.session_id)

    # -----------------------------------------------------------------------
    # Rule 1: HTTP Domain Restriction via /chat and /api/chat
    # -----------------------------------------------------------------------
    def test_out_of_scope_rejection_http_chat(self):
        payload = {
            "user_id": self.session_id,
            "query": "Write a python function to compute factorial",
            "birth_year": 1995,
            "birth_month": 5,
            "birth_day": 15,
            "birth_hour": 14.5,
            "latitude": 28.6139,
            "longitude": 77.2090,
            "timezone": 5.5,
        }
        resp = client.post("/chat", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "out_of_scope")
        self.assertEqual(data.get("answer"), OUT_OF_SCOPE_RESPONSE)
        self.assertEqual(data.get("response"), OUT_OF_SCOPE_RESPONSE)

    def test_out_of_scope_rejection_http_api_chat(self):
        payload = {
            "prompt": "Explain quantum physics and how black holes work",
            "user_profile": {"user_id": self.session_id, "name": "Aarav"},
        }
        resp = client.post("/api/chat", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "out_of_scope")
        self.assertEqual(data.get("answer"), OUT_OF_SCOPE_RESPONSE)

    # -----------------------------------------------------------------------
    # Rule 2: HTTP 3-Strike Safety & Profanity Guardrail via /chat
    # -----------------------------------------------------------------------
    def test_three_strike_safety_http_chat(self):
        base_payload = {
            "user_id": self.session_id,
            "birth_year": 1995,
            "birth_month": 5,
            "birth_day": 15,
            "birth_hour": 14.5,
            "latitude": 28.6139,
            "longitude": 77.2090,
            "timezone": 5.5,
        }

        # Strike 1
        p1 = dict(base_payload, query="Fuck you assistant")
        r1 = client.post("/chat", json=p1)
        self.assertEqual(r1.status_code, 200)
        d1 = r1.json()
        self.assertEqual(d1["status"], "warning")
        self.assertEqual(d1["abuse_strikes"], 1)
        self.assertFalse(d1["terminated"])
        self.assertEqual(d1["answer"], "⚠️ Warning (1/3): Inappropriate or vulgar language is not permitted.")

        # Strike 2
        p2 = dict(base_payload, query="Chutiya bot")
        r2 = client.post("/chat", json=p2)
        self.assertEqual(r2.status_code, 200)
        d2 = r2.json()
        self.assertEqual(d2["status"], "warning")
        self.assertEqual(d2["abuse_strikes"], 2)
        self.assertFalse(d2["terminated"])
        self.assertEqual(d2["answer"], "⚠️ Warning (2/3): Inappropriate or vulgar language is not permitted.")

        # Strike 3: Terminated
        p3 = dict(base_payload, query="Bhosdike chup kar")
        r3 = client.post("/chat", json=p3)
        self.assertEqual(r3.status_code, 200)
        d3 = r3.json()
        self.assertEqual(d3["status"], "terminated")
        self.assertEqual(d3["abuse_strikes"], 3)
        self.assertTrue(d3["terminated"])
        self.assertEqual(d3["answer"], TERMINATED_MESSAGE)
        self.assertEqual(d3["answer"], "🚫 Session Terminated: You have exceeded the limit for inappropriate language.")

        # Strike 4 / Subsequent input on terminated session: LOCKED!
        p4 = dict(base_payload, query="When will I get married?")
        r4 = client.post("/chat", json=p4)
        self.assertEqual(r4.status_code, 200)
        d4 = r4.json()
        self.assertEqual(d4["status"], "terminated")
        self.assertTrue(d4["terminated"])
        self.assertEqual(d4["answer"], TERMINATED_MESSAGE)


if __name__ == "__main__":
    unittest.main()
