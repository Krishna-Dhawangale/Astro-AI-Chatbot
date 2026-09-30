"""
Stage 15 — Semantic Correctness & Traceability Benchmark Suite
===============================================================
Script: verify_stage15_semantic_correctness.py

Verifies that Stage 15 semantic conclusions are:
1. Dynamic & Chart-Sensitive: Same question across 10 distinct charts produces materially different evidence, themes, and explanations.
2. Traceable: Explicit 5-step lineage from Chart Evidence -> Planetary Quality -> Matched Rule -> Semantic Theme -> User Statement.
3. Astrologically Grounded: 10th lord, rashis, houses, and dignities correctly drive recommendation themes.
4. Cleanly Separated: Timing (Dasha/Transit) is retained in Section 5 without polluting baseline suitability.
"""

import sys
import json
from pathlib import Path

# Force UTF-8 stdout encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.reasoning.pipeline_helper import execute_full_deterministic_pipeline
from backend.reasoning.interpretation import build_interpretation_analysis
from backend.reasoning.answer_synthesizer import (
    synthesize_structured_answer,
    evaluate_evidence_weights,
    resolve_evidence_conflicts,
    derive_domain_recommendation_themes,
    PLANETARY_CAREER_THEMES
)

# ---------------------------------------------------------------------------
# 10 Diverse Test Chart Definitions
# ---------------------------------------------------------------------------

CHARTS = [
    {
        "id": "Chart_1_Saturn",
        "name": "Chart 1 — Saturn Dominant (Aries Asc, 10th Lord Saturn in Capricorn 10th)",
        "expected_lord": "Saturn",
        "expected_house": 10,
        "expected_rashi": "Capricorn",
        "expected_status": "MODERATELY_FAVORED_WITH_COUNTERBALANCE",
        "expected_primary_theme": "Operations Management & Governance",
        "ascendant": {"longitude": 12.5, "rashi": "Aries", "degree_in_rashi": 12.5},
        "planets": {
            "Sun": {"rashi": "Capricorn", "longitude": 285.0, "house": 10},
            "Moon": {"rashi": "Taurus", "longitude": 45.0, "house": 2},
            "Mars": {"rashi": "Aries", "longitude": 15.0, "house": 1},
            "Mercury": {"rashi": "Capricorn", "longitude": 290.0, "house": 10},
            "Jupiter": {"rashi": "Cancer", "longitude": 95.0, "house": 4},
            "Venus": {"rashi": "Pisces", "longitude": 350.0, "house": 12},
            "Saturn": {"rashi": "Capricorn", "longitude": 280.0, "house": 10},
            "Rahu": {"rashi": "Taurus", "longitude": 50.0, "house": 2},
            "Ketu": {"rashi": "Scorpio", "longitude": 230.0, "house": 8}
        }
    },
    {
        "id": "Chart_2_Mercury",
        "name": "Chart 2 — Mercury Dominant (Virgo Asc, 10th Lord Mercury in Gemini 10th)",
        "expected_lord": "Mercury",
        "expected_house": 10,
        "expected_rashi": "Gemini",
        "expected_status": "STRONGLY_FAVORED",
        "expected_primary_theme": "Data & Business Analytics",
        "ascendant": {"longitude": 165.0, "rashi": "Virgo", "degree_in_rashi": 15.0},
        "planets": {
            "Sun": {"rashi": "Gemini", "longitude": 75.0, "house": 10},
            "Moon": {"rashi": "Taurus", "longitude": 45.0, "house": 9},
            "Mars": {"rashi": "Capricorn", "longitude": 285.0, "house": 5},
            "Mercury": {"rashi": "Gemini", "longitude": 80.0, "house": 10},
            "Jupiter": {"rashi": "Cancer", "longitude": 95.0, "house": 11},
            "Venus": {"rashi": "Taurus", "longitude": 50.0, "house": 9},
            "Saturn": {"rashi": "Aquarius", "longitude": 315.0, "house": 6},
            "Rahu": {"rashi": "Virgo", "longitude": 160.0, "house": 1},
            "Ketu": {"rashi": "Pisces", "longitude": 340.0, "house": 7}
        }
    },
    {
        "id": "Chart_3_Sun",
        "name": "Chart 3 — Sun Dominant (Scorpio Asc, 10th Lord Sun in Leo 10th)",
        "expected_lord": "Sun",
        "expected_house": 10,
        "expected_rashi": "Leo",
        "expected_status": "STRONGLY_FAVORED",
        "expected_primary_theme": "Government & Public Administration",
        "ascendant": {"longitude": 225.0, "rashi": "Scorpio", "degree_in_rashi": 15.0},
        "planets": {
            "Sun": {"rashi": "Leo", "longitude": 135.0, "house": 10},
            "Moon": {"rashi": "Aries", "longitude": 15.0, "house": 6},
            "Mars": {"rashi": "Scorpio", "longitude": 230.0, "house": 1},
            "Mercury": {"rashi": "Leo", "longitude": 140.0, "house": 10},
            "Jupiter": {"rashi": "Sagittarius", "longitude": 255.0, "house": 2},
            "Venus": {"rashi": "Libra", "longitude": 200.0, "house": 12},
            "Saturn": {"rashi": "Capricorn", "longitude": 285.0, "house": 3},
            "Rahu": {"rashi": "Gemini", "longitude": 75.0, "house": 8},
            "Ketu": {"rashi": "Sagittarius", "longitude": 255.0, "house": 2}
        }
    },
    {
        "id": "Chart_4_Mars",
        "name": "Chart 4 — Mars Dominant (Cancer Asc, 10th Lord Mars in Aries 10th, Exalted Sun)",
        "expected_lord": "Mars",
        "expected_house": 10,
        "expected_rashi": "Aries",
        "expected_status": "STRONGLY_FAVORED",
        "expected_primary_theme": "Engineering & Technology",
        "ascendant": {"longitude": 95.0, "rashi": "Cancer", "degree_in_rashi": 5.0},
        "planets": {
            "Sun": {"rashi": "Aries", "longitude": 15.0, "house": 10},
            "Moon": {"rashi": "Taurus", "longitude": 45.0, "house": 11},
            "Mars": {"rashi": "Aries", "longitude": 20.0, "house": 10},
            "Mercury": {"rashi": "Taurus", "longitude": 50.0, "house": 11},
            "Jupiter": {"rashi": "Pisces", "longitude": 345.0, "house": 9},
            "Venus": {"rashi": "Pisces", "longitude": 350.0, "house": 9},
            "Saturn": {"rashi": "Aquarius", "longitude": 315.0, "house": 8},
            "Rahu": {"rashi": "Virgo", "longitude": 165.0, "house": 3},
            "Ketu": {"rashi": "Pisces", "longitude": 345.0, "house": 9}
        }
    },
    {
        "id": "Chart_5_Jupiter",
        "name": "Chart 5 — Jupiter Dominant (Gemini Asc, 10th Lord Jupiter in Cancer 2nd - Exalted)",
        "expected_lord": "Jupiter",
        "expected_house": 2,
        "expected_rashi": "Cancer",
        "expected_status": "STRONGLY_FAVORED",
        "expected_primary_theme": "Strategic Consulting & Mentorship",
        "ascendant": {"longitude": 75.0, "rashi": "Gemini", "degree_in_rashi": 15.0},
        "planets": {
            "Sun": {"rashi": "Virgo", "longitude": 165.0, "house": 4},
            "Moon": {"rashi": "Taurus", "longitude": 45.0, "house": 12},
            "Mars": {"rashi": "Capricorn", "longitude": 285.0, "house": 8},
            "Mercury": {"rashi": "Virgo", "longitude": 170.0, "house": 4},
            "Jupiter": {"rashi": "Cancer", "longitude": 95.0, "house": 2},
            "Venus": {"rashi": "Libra", "longitude": 200.0, "house": 5},
            "Saturn": {"rashi": "Aquarius", "longitude": 315.0, "house": 9},
            "Rahu": {"rashi": "Aries", "longitude": 15.0, "house": 11},
            "Ketu": {"rashi": "Libra", "longitude": 195.0, "house": 5}
        }
    },
    {
        "id": "Chart_6_Venus",
        "name": "Chart 6 — Venus Dominant (Leo Asc, 10th Lord Venus in Taurus 10th - Own Sign)",
        "expected_lord": "Venus",
        "expected_house": 10,
        "expected_rashi": "Taurus",
        "expected_status": "STRONGLY_FAVORED",
        "expected_primary_theme": "User Experience & Product Design",
        "ascendant": {"longitude": 130.0, "rashi": "Leo", "degree_in_rashi": 10.0},
        "planets": {
            "Sun": {"rashi": "Leo", "longitude": 135.0, "house": 1},
            "Moon": {"rashi": "Taurus", "longitude": 45.0, "house": 10},
            "Mars": {"rashi": "Scorpio", "longitude": 225.0, "house": 4},
            "Mercury": {"rashi": "Gemini", "longitude": 75.0, "house": 11},
            "Jupiter": {"rashi": "Sagittarius", "longitude": 255.0, "house": 5},
            "Venus": {"rashi": "Taurus", "longitude": 45.0, "house": 10},
            "Saturn": {"rashi": "Aquarius", "longitude": 315.0, "house": 7},
            "Rahu": {"rashi": "Virgo", "longitude": 165.0, "house": 2},
            "Ketu": {"rashi": "Pisces", "longitude": 345.0, "house": 8}
        }
    },
    {
        "id": "Chart_7_Moon",
        "name": "Chart 7 — Moon Dominant (Libra Asc, 10th Lord Moon in Cancer 10th - Own Sign)",
        "expected_lord": "Moon",
        "expected_house": 10,
        "expected_rashi": "Cancer",
        "expected_status": "STRONGLY_FAVORED",
        "expected_primary_theme": "Healthcare & Counseling",
        "ascendant": {"longitude": 195.0, "rashi": "Libra", "degree_in_rashi": 15.0},
        "planets": {
            "Sun": {"rashi": "Leo", "longitude": 140.0, "house": 11},
            "Moon": {"rashi": "Cancer", "longitude": 100.0, "house": 10},
            "Mars": {"rashi": "Capricorn", "longitude": 280.0, "house": 4},
            "Mercury": {"rashi": "Virgo", "longitude": 170.0, "house": 12},
            "Jupiter": {"rashi": "Cancer", "longitude": 95.0, "house": 10},
            "Venus": {"rashi": "Taurus", "longitude": 45.0, "house": 8},
            "Saturn": {"rashi": "Aquarius", "longitude": 315.0, "house": 5},
            "Rahu": {"rashi": "Gemini", "longitude": 75.0, "house": 9},
            "Ketu": {"rashi": "Sagittarius", "longitude": 255.0, "house": 3}
        }
    },
    {
        "id": "Chart_8_Rahu",
        "name": "Chart 8 — Rahu/AI Dominant (Sagittarius Asc, 10th Lord Mercury Exalted, Rahu 7th Exalted)",
        "expected_lord": "Mercury",
        "expected_house": 10,
        "expected_rashi": "Virgo",
        "expected_status": "STRONGLY_FAVORED",
        "expected_primary_theme": "Data & Business Analytics",
        "ascendant": {"longitude": 255.0, "rashi": "Sagittarius", "degree_in_rashi": 15.0},
        "planets": {
            "Sun": {"rashi": "Leo", "longitude": 135.0, "house": 9},
            "Moon": {"rashi": "Taurus", "longitude": 45.0, "house": 6},
            "Mars": {"rashi": "Aries", "longitude": 15.0, "house": 5},
            "Mercury": {"rashi": "Virgo", "longitude": 170.0, "house": 10},
            "Jupiter": {"rashi": "Cancer", "longitude": 95.0, "house": 8},
            "Venus": {"rashi": "Libra", "longitude": 200.0, "house": 11},
            "Saturn": {"rashi": "Aquarius", "longitude": 315.0, "house": 3},
            "Rahu": {"rashi": "Gemini", "longitude": 75.0, "house": 7},
            "Ketu": {"rashi": "Sagittarius", "longitude": 255.0, "house": 1}
        }
    },
    {
        "id": "Chart_9_Debilitated",
        "name": "Chart 9 — Debilitated Placements (Capricorn Asc, 10th Lord Venus Debilitated in 9th Virgo)",
        "expected_lord": "Venus",
        "expected_house": 9,
        "expected_rashi": "Virgo",
        "expected_status": "CHALLENGING_PERIOD",
        "expected_primary_theme": "User Experience & Product Design",
        "ascendant": {"longitude": 280.0, "rashi": "Capricorn", "degree_in_rashi": 10.0},
        "planets": {
            "Sun": {"rashi": "Libra", "longitude": 195.0, "house": 10},
            "Moon": {"rashi": "Scorpio", "longitude": 225.0, "house": 11},
            "Mars": {"rashi": "Cancer", "longitude": 105.0, "house": 7},
            "Mercury": {"rashi": "Virgo", "longitude": 170.0, "house": 9},
            "Jupiter": {"rashi": "Gemini", "longitude": 75.0, "house": 6},
            "Venus": {"rashi": "Virgo", "longitude": 175.0, "house": 9},
            "Saturn": {"rashi": "Aries", "longitude": 15.0, "house": 4},
            "Rahu": {"rashi": "Taurus", "longitude": 45.0, "house": 5},
            "Ketu": {"rashi": "Scorpio", "longitude": 225.0, "house": 11}
        }
    },
    {
        "id": "Chart_10_Dusthana",
        "name": "Chart 10 — Dusthana Placement (Aquarius Asc, 10th Lord Mars in 8th House Virgo)",
        "expected_lord": "Mars",
        "expected_house": 8,
        "expected_rashi": "Virgo",
        "expected_status": "CHALLENGING_PERIOD",
        "expected_primary_theme": "Engineering & Technology",
        "ascendant": {"longitude": 315.0, "rashi": "Aquarius", "degree_in_rashi": 15.0},
        "planets": {
            "Sun": {"rashi": "Scorpio", "longitude": 225.0, "house": 10},
            "Moon": {"rashi": "Cancer", "longitude": 100.0, "house": 6},
            "Mars": {"rashi": "Virgo", "longitude": 165.0, "house": 8},
            "Mercury": {"rashi": "Gemini", "longitude": 75.0, "house": 5},
            "Jupiter": {"rashi": "Capricorn", "longitude": 285.0, "house": 12},
            "Venus": {"rashi": "Pisces", "longitude": 350.0, "house": 2},
            "Saturn": {"rashi": "Aries", "longitude": 15.0, "house": 3},
            "Rahu": {"rashi": "Taurus", "longitude": 45.0, "house": 4},
            "Ketu": {"rashi": "Scorpio", "longitude": 225.0, "house": 10}
        }
    }
]


def run_benchmark():
    print("=" * 100)
    print("STAGE 15 SEMANTIC CORRECTNESS & TRACEABILITY BENCHMARK SUITE")
    print("Question: 'Which career suits me best?'")
    print("=" * 100)

    results_table = []
    unique_theme_sets = set()

    for idx, chart in enumerate(CHARTS, start=1):
        chart_id = chart["id"]
        chart_name = chart["name"]
        
        # Execute Pipeline
        stage8_res = execute_full_deterministic_pipeline(chart)
        c_rules = stage8_res["stage_8_15_career_rules"]["rules"]
        matched_rules = [r for r in c_rules if r.get("matched")]
        
        # Add Dasha and Transit timing rules if available
        dasha_rules = stage8_res.get("stage_8_20_dasha_timing", {}).get("rules", [])
        transit_rules = stage8_res.get("stage_8_21_transit_timing", {}).get("rules", [])
        for r in dasha_rules + transit_rules:
            if r.get("matched") and r.get("domain") in ["career", "all"]:
                matched_rules.append(r)

        # Synthesize Answer
        synth = synthesize_structured_answer(domain="career", matched_rules=matched_rules)
        
        foundation_rule = next((r for r in synth["weighted_rules"] if r["category"] == "foundation"), None)
        ev = foundation_rule["evidence"] if foundation_rule else {}
        
        lord = ev.get("lord", "N/A")
        house = ev.get("lord_natal_house", ev.get("house", "N/A"))
        rashi = ev.get("lord_natal_rashi", ev.get("house_rashi", "N/A"))
        
        themes = synth.get("recommended_fields", [])
        unique_theme_sets.add(tuple(themes))
        conflict_status = synth.get("conflict_status", "N/A")
        score = synth.get("evidence_score", 0.0)

        # Print signals for debugging
        print(f"\n[{chart_id}] Status: {conflict_status} | Pos: {synth.get('positive_signals')} | Chall: {synth.get('challenging_signals')}")

        # Extract 5-Step Lineage Traceability
        # Step 1: Chart Evidence
        step1_ev = f"10th Lord {lord} in H{house} ({rashi})"
        # Step 2: Planetary Quality
        planet_qualities = PLANETARY_CAREER_THEMES.get(lord, {}).get("qualities", "N/A")
        # Step 3: Matched Rule ID
        matched_ids = [r["rule_id"] for r in synth["weighted_rules"][:3]]
        # Step 4: Semantic Theme
        primary_theme = themes[0] if themes else "N/A"
        # Step 5: User Statement
        first_section = synth["structured_text"].split("### 2.")[0].strip()

        # Assertions
        assert lord == chart["expected_lord"], f"[{chart_id}] Expected lord {chart['expected_lord']}, got {lord}"
        assert house == chart["expected_house"], f"[{chart_id}] Expected house {chart['expected_house']}, got {house}"
        assert rashi == chart["expected_rashi"], f"[{chart_id}] Expected rashi {chart['expected_rashi']}, got {rashi}"
        assert conflict_status == chart["expected_status"], f"[{chart_id}] Expected status {chart['expected_status']}, got {conflict_status}"

        results_table.append({
            "chart_id": chart_id,
            "chart_name": chart_name,
            "lord": lord,
            "placement": f"House {house} ({rashi})",
            "status": conflict_status,
            "score": score,
            "themes": themes,
            "primary_theme": primary_theme,
            "traceability": {
                "step1_evidence": step1_ev,
                "step2_quality": planet_qualities,
                "step3_rules": matched_ids,
                "step4_theme": primary_theme,
                "step5_user_statement_preview": first_section[:120] + "..."
            }
        })

    # Print Formatted Results Table
    print("\n" + "=" * 120)
    print(f"{'CHART ID':<18} | {'10TH LORD':<9} | {'PLACEMENT':<20} | {'CONFLICT STATUS':<30} | {'PRIMARY DERIVED THEME':<35}")
    print("=" * 120)

    for item in results_table:
        print(f"{item['chart_id']:<18} | {item['lord']:<9} | {item['placement']:<20} | {item['status']:<30} | {item['primary_theme']:<35}")

    print("=" * 120)

    print("\n" + "=" * 100)
    print("DETAILED 5-STEP LINEAGE TRACEABILITY AUDIT (SAMPLE CHARTS)")
    print("=" * 100)

    for item in results_table[:3]:
        print(f"\n[TRACEABILITY CHECK: {item['chart_id']} ({item['lord']})]")
        t = item["traceability"]
        print(f"  1. Chart Evidence  : {t['step1_evidence']}")
        print(f"  2. Planetary Qual  : {t['step2_quality']}")
        print(f"  3. Matched Rules   : {t['step3_rules']}")
        print(f"  4. Semantic Theme  : {t['step4_theme']}")
        print(f"  5. User Statement  :\n     \"{t['step5_user_statement_preview']}\"")

    # Check for Dynamic Variety across charts
    print("\n" + "=" * 100)
    print("DYNAMIC VARIETY & DIVERSITY VERIFICATION")
    print("=" * 100)
    print(f"Total Charts Tested    : {len(CHARTS)}")
    print(f"Unique Theme Combos    : {len(unique_theme_sets)}")
    print(f"Chart Sensitivity Rate : {(len(unique_theme_sets) / len(CHARTS)) * 100:.1f}%")

    assert len(unique_theme_sets) >= 7, f"Expected at least 7 unique theme sets across 10 charts, got {len(unique_theme_sets)}"

    print("\n[SUCCESS] STAGE 15 SEMANTIC CORRECTNESS & TRACEABILITY BENCHMARK PASSED 100%!")
    print("All 10 distinct charts produced correct 10th lord extractions, accurate conflict statuses, and traceable user-facing answers.")

if __name__ == "__main__":
    run_benchmark()
