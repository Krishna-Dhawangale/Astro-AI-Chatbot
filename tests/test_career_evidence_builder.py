import json
import os
import sys
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.reasoning.llm_renderer import (
    build_compact_evidence_package,
    format_llm_assisted_prompt,
)
from backend.reasoning.pipeline_helper import execute_full_deterministic_pipeline
from backend.reasoning.structured_evidence import build_structured_evidence


CAREER_QUESTIONS = {
    "local": "How does my 10th lord influence my career?",
    "llm": "What does my 10th house indicate about my career?",
}


def make_chart(ascendant, tenth_lord, tenth_lord_house, tenth_lord_rashi):
    planets = {
        "Sun": {"rashi": "Aries", "longitude": 10.0, "house": 1},
        "Moon": {"rashi": "Taurus", "longitude": 40.0, "house": 2},
        "Mercury": {"rashi": "Gemini", "longitude": 70.0, "house": 3},
        "Venus": {"rashi": "Cancer", "longitude": 100.0, "house": 4},
        "Mars": {"rashi": "Leo", "longitude": 130.0, "house": 5},
        "Jupiter": {"rashi": "Virgo", "longitude": 160.0, "house": 6},
        "Saturn": {"rashi": "Libra", "longitude": 190.0, "house": 7},
        "Rahu": {"rashi": "Scorpio", "longitude": 220.0, "house": 8},
        "Ketu": {"rashi": "Sagittarius", "longitude": 250.0, "house": 9},
    }
    planets[tenth_lord] = {
        "rashi": tenth_lord_rashi,
        "longitude": 280.0,
        "house": tenth_lord_house,
    }
    return {
        "ascendant": {
            "longitude": 10.0,
            "rashi": ascendant,
            "degree_in_rashi": 10.0,
        },
        "planets": planets,
    }


def make_api_chart(chart):
    output = {
        "Ascendant": {
            "zodiac_sign_name": chart["ascendant"]["rashi"],
            "fullDegree": chart["ascendant"]["longitude"],
            "normDegree": chart["ascendant"]["degree_in_rashi"],
        }
    }
    for name, planet in chart["planets"].items():
        output[name] = {
            "zodiac_sign_name": planet["rashi"],
            "fullDegree": planet["longitude"],
            "normDegree": planet["longitude"] % 30,
            "house_number": planet["house"],
        }
    return {"output": output}


class CareerEvidenceBuilderTests(unittest.TestCase):
    def setUp(self):
        self.charts = [
            make_chart("Aries", "Saturn", 10, "Capricorn"),
            make_chart("Leo", "Venus", 10, "Taurus"),
            make_chart("Scorpio", "Sun", 10, "Leo"),
        ]

    def get_career_rules(self, chart):
        result = execute_full_deterministic_pipeline(
            chart,
            dt=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        return result, result["stage_8_15_career_rules"]["rules"]

    def test_multiple_ascendants_use_career_rule_evidence(self):
        expected = [
            ("Capricorn", "Saturn"),
            ("Taurus", "Venus"),
            ("Leo", "Sun"),
        ]

        for chart, (house_sign, lord) in zip(self.charts, expected):
            with self.subTest(ascendant=chart["ascendant"]["rashi"]):
                stage8_result, rules = self.get_career_rules(chart)
                matched_rules = [rule for rule in rules if rule["matched"]]
                package = build_compact_evidence_package(
                    "career",
                    "career_general",
                    chart,
                    matched_rules,
                    question=CAREER_QUESTIONS["llm"],
                )

                self.assertEqual(package["verified_facts"]["10H"], house_sign)
                self.assertEqual(package["verified_facts"]["10L"], lord)
                self.assertEqual(
                    package["verified_facts"]["10L_placement"],
                    f"10H ({chart['planets'][lord]['rashi']})",
                )

                rule = next(
                    rule for rule in matched_rules
                    if rule["rule_id"] == "CAREER_10TH_LORD_PLACEMENT"
                )
                evidence = rule["evidence"]
                self.assertEqual(evidence["house"], 10)
                self.assertEqual(evidence["house_rashi"], house_sign)
                self.assertEqual(evidence["lord"], lord)

                local_evidence = build_structured_evidence(
                    "career",
                    "career_general",
                    chart,
                    stage8_result,
                    {},
                )
                self.assertEqual(local_evidence.facts["tenth_house_sign"], house_sign)
                self.assertEqual(local_evidence.facts["tenth_lord"], lord)
                self.assertEqual(local_evidence.facts["tenth_lord_house"], 10)

    def test_missing_career_rule_does_not_invent_tenth_house_facts(self):
        chart = self.charts[0]
        package = build_compact_evidence_package(
            "career",
            "career_general",
            chart,
            [],
            question=CAREER_QUESTIONS["llm"],
        )
        prompt, _ = format_llm_assisted_prompt(CAREER_QUESTIONS["llm"], package)

        self.assertNotIn("10H", package["verified_facts"])
        self.assertNotIn("10L", package["verified_facts"])
        self.assertNotIn("Gemini", prompt)
        self.assertNotIn("Mercury", prompt)
        self.assertTrue("No verified chart facts available" in prompt or "Lagna=" in prompt)
        self.assertIn("No matched rules available", prompt)


        incomplete_evidence = build_structured_evidence(
            "career",
            "career_general",
            chart,
            {},
            {},
        )
        self.assertIsNone(incomplete_evidence.facts["tenth_house_sign"])
        self.assertIsNone(incomplete_evidence.facts["tenth_lord"])
        self.assertFalse(incomplete_evidence.is_sufficient())


class CareerChatPathTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from backend import main

        self.main = main
        self.user_id = "career-evidence-builder-test"
        main.USER_SESSIONS.pop(self.user_id, None)
        self.chart = make_chart("Aries", "Saturn", 10, "Capricorn")
        self.api_chart = make_api_chart(self.chart)
        self.career_rule = {
            "rule_id": "CAREER_10TH_LORD_PLACEMENT",
            "matched": True,
            "interpretation_key": "career_10th_lord",
            "evidence": {
                "house": 10,
                "house_rashi": "Capricorn",
                "lord": "Saturn",
                "lord_natal_house": 10,
                "lord_natal_rashi": "Capricorn",
            },
        }
        self.stage8_result = {
            "stage_8_15_career_rules": {
                "domain": "career",
                "rules": [self.career_rule],
            },
            "stage_8_16_marriage_rules": {"rules": []},
            "stage_8_17_finance_rules": {"rules": []},
            "stage_8_18_education_rules": {"rules": []},
            "stage_8_19_property_rules": {"rules": []},
            "stage_8_20_dasha_timing": {},
            "stage_8_21_transit_timing": {},
            "stage_8_22_multi_domain": {},
        }

    def tearDown(self):
        self.main.USER_SESSIONS.pop(self.user_id, None)

    def make_patches(self, question, local_answerable):
        async def fake_run_in_threadpool(func, *_args):
            if func.__name__ == "get_chart_from_api":
                return [0] * 10, self.api_chart
            return {"output": {}}

        def fake_evidence_availability(*_args):
            return {
                "required_evidence": [],
                "available_evidence": [],
                "missing_evidence": [],
                "evidence_status": "COMPLETE",
            }

        def fake_intent(*_args):
            return {
                "raw_intent": "career_general",
                "raw_confidence": 1.0,
                "resolved_intent": "10th_lord_placement" if local_answerable else "career_general",
                "resolution_reason": "test",
            }

        patches = [
            patch.object(self.main, "run_in_threadpool", side_effect=fake_run_in_threadpool),
            patch("backend.router.intent.predict_intent_with_confidence", side_effect=fake_intent),
            patch.object(
                self.main,
                "route_question",
                return_value={"domain": "career", "complexity": "normal"},
            ),
            patch.object(
                self.main,
                "execute_full_deterministic_pipeline",
                return_value=self.stage8_result,
            ),
            patch.object(
                self.main,
                "build_interpretation_analysis",
                return_value={
                    "domains": {
                        "career": {
                            "matched_interpretations": [],
                            "timing": {},
                        }
                    }
                },
            ),
            patch.object(
                self.main,
                "evaluate_evidence_availability",
                side_effect=fake_evidence_availability,
            ),
            patch.object(
                self.main,
                "check_intent_evidence_relevance",
                return_value=(local_answerable, [self.career_rule] if local_answerable else []),
            ),
            patch.object(
                self.main,
                "format_local_interpretation_answer",
                return_value="Local career answer.",
            ),
            patch.object(self.main, "generate_follow_up_questions", return_value=[]),
            patch.object(self.main, "print_evidence_observability_report"),
            patch.object(self.main, "record_production_decision_trace"),
            patch.object(self.main, "print_no_llm_call_report"),
            patch.object(self.main, "print_llm_call_report", return_value={}),
            patch.object(self.main, "print_request_total_report"),
            patch.object(self.main, "load_dotenv"),
        ]
        return patches

    async def test_local_answerable_question_does_not_call_gemini(self):
        from backend.main import ChatRequest

        patches = self.make_patches(CAREER_QUESTIONS["local"], local_answerable=True)
        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            with patch.object(
                self.main,
                "stream_cascading_router",
                new=AsyncMock(side_effect=AssertionError("Gemini must not be called")),
            ) as gemini_router:
                response = await self.main._handle_chat_response(
                    ChatRequest(
                        user_id=self.user_id,
                        query=CAREER_QUESTIONS["local"],
                    )
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)["answer_source"], "LOCAL")
        gemini_router.assert_not_awaited()

    async def test_exact_dynamic_prompt_is_passed_to_gemini(self):
        from backend.main import ChatRequest

        captured = {}

        async def fake_gemini_router(**kwargs):
            captured.update(kwargs)
            yield "Grounded answer.", {
                "provider": "gemini",
                "model": "test-model",
            }

        patches = self.make_patches(CAREER_QUESTIONS["llm"], local_answerable=False)
        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            with (
                patch.object(self.main, "gemini_client", object()),
                patch.dict(
                    "os.environ",
                    {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""},
                ),
                patch.object(
                    self.main,
                    "stream_cascading_router",
                    new=fake_gemini_router,
                ),
            ):
                response = await self.main._handle_chat_response(
                    ChatRequest(
                        user_id=self.user_id,
                        query=CAREER_QUESTIONS["llm"],
                    )
                )
                chunks = [chunk async for chunk in response.body_iterator]

        self.assertEqual(response.status_code, 200)
        self.assertIn("10H=Capricorn", captured["prompt"])
        self.assertIn("10L=Saturn", captured["prompt"])
        self.assertIn("10L_placement=10H (Capricorn)", captured["prompt"])
        self.assertIn("Saturn rules the Capricorn 10th house", captured["prompt"])
        self.assertNotIn("10H=Gemini", captured["prompt"])
        self.assertNotIn("Mercury 10L@1H", captured["prompt"])
        self.assertTrue(chunks)


if __name__ == "__main__":
    unittest.main()
