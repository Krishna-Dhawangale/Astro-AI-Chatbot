"""
Stage 15 Verification — Multi-Domain Answer Demonstration
===========================================================
Demonstrates dynamic, planet-grounded interpretations for all supported domains:
Career, Marriage, Finance, Education, Property, and Health.
"""

import json
from backend.reasoning.answer_synthesizer import synthesize_structured_answer

# Sample test cases representing extracted Stage 8 evidence for different domains
TEST_CASES = {
    "career": [
        {
            "rule_id": "CAREER_10TH_LORD_PLACEMENT",
            "matched": True,
            "evidence": {"house": 10, "lord": "Saturn", "lord_natal_house": 10, "lord_natal_rashi": "Capricorn"}
        },
        {
            "rule_id": "CAREER_KARAKA_DIGNITY",
            "matched": True,
            "evidence": [
                {"planet": "Saturn", "rashi": "Capricorn", "dignity": "own_sign"},
                {"planet": "Sun", "rashi": "Aries", "dignity": "exalted"}
            ]
        }
    ],
    "marriage": [
        {
            "rule_id": "MARRIAGE_7TH_LORD_PLACEMENT",
            "matched": True,
            "evidence": {"house": 7, "lord": "Venus", "lord_natal_house": 7, "lord_natal_rashi": "Taurus"}
        },
        {
            "rule_id": "MARRIAGE_KARAKA_DIGNITY",
            "matched": True,
            "evidence": [
                {"planet": "Venus", "rashi": "Taurus", "dignity": "own_sign"},
                {"planet": "Jupiter", "rashi": "Cancer", "dignity": "exalted"}
            ]
        }
    ],
    "finance": [
        {
            "rule_id": "FINANCE_2ND_LORD_PLACEMENT",
            "matched": True,
            "evidence": {"house": 2, "lord": "Mercury", "lord_natal_house": 11, "lord_natal_rashi": "Gemini"}
        },
        {
            "rule_id": "FINANCE_KARAKA_DIGNITY",
            "matched": True,
            "evidence": [
                {"planet": "Mercury", "rashi": "Gemini", "dignity": "own_sign"},
                {"planet": "Jupiter", "rashi": "Cancer", "dignity": "exalted"}
            ]
        }
    ],
    "education": [
        {
            "rule_id": "EDUCATION_4TH_LORD_PLACEMENT",
            "matched": True,
            "evidence": {"house": 4, "lord": "Jupiter", "lord_natal_house": 5, "lord_natal_rashi": "Sagittarius"}
        },
        {
            "rule_id": "EDUCATION_KARAKA_DIGNITY",
            "matched": True,
            "evidence": [
                {"planet": "Jupiter", "rashi": "Sagittarius", "dignity": "own_sign"},
                {"planet": "Mercury", "rashi": "Virgo", "dignity": "exalted"}
            ]
        }
    ],
    "property": [
        {
            "rule_id": "PROPERTY_4TH_LORD_PLACEMENT",
            "matched": True,
            "evidence": {"house": 4, "lord": "Mars", "lord_natal_house": 4, "lord_natal_rashi": "Aries"}
        },
        {
            "rule_id": "PROPERTY_KARAKA_DIGNITY",
            "matched": True,
            "evidence": [
                {"planet": "Mars", "rashi": "Aries", "dignity": "own_sign"},
                {"planet": "Saturn", "rashi": "Libra", "dignity": "exalted"}
            ]
        }
    ]
}

def main():
    print("=" * 80)
    print("STAGE 15 MULTI-DOMAIN SEMANTIC INTERPRETATION DEMO")
    print("=" * 80)

    for domain, rules in TEST_CASES.items():
        res = synthesize_structured_answer(domain, rules)
        print(f"\n" + "#" * 80)
        print(f" DOMAIN: {domain.upper()}")
        print(f" RECOMMENDED THEMES: {res['recommended_fields']}")
        print(f" CONFLICT STATUS: {res['conflict_status']} | SCORE: {res['evidence_score']}")
        print("#" * 80)
        print(res["structured_text"])
        print("\n")

if __name__ == "__main__":
    main()
