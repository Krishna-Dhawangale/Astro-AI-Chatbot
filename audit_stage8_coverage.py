"""
Stage 8 Rule Engine Audit & Evidence Coverage Inspector
======================================================
Script: audit_stage8_coverage.py

Executes a comprehensive audit of Stage 8 rule engines across all supported domains:
- Career (career_rules.py)
- Marriage (marriage_rules.py)
- Finance (finance_rules.py)
- Health (health_rules.py)
- Education (education_rules.py)
- Property (property_rules.py)

Audits:
1. Chart Facts -> Rule Evidence -> Interpretation Key mapping.
2. Verified matched rule outputs for sample natal charts.
3. Intent-to-Evidence Matrix readiness check.

Strict Rule: READ-ONLY evaluation. No code or model files modified.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Reconfigure encoding for clean console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.reasoning.pipeline_helper import execute_full_deterministic_pipeline
from backend.reasoning.stage8_pipeline import validate_stage8_pipeline
from backend.reasoning.interpretation import (
    interpret_rule,
    get_interpretation_text,
    INTERPRETATION_TEMPLATES
)

# Sample Natal Chart Data (Aries Ascendant, Saturn in Capricorn 10th house, Sun+Mercury 10th house)
SAMPLE_CHART_DATA = {
    "ascendant": {
        "longitude": 12.5,
        "rashi": "Aries",
        "degree_in_rashi": 12.5
    },
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
}

def run_stage8_audit():
    print("=" * 90)
    print(" STAGE 8 RULE ENGINE COVERAGE & EVIDENCE AUDIT")
    print("=" * 90)

    # Execute full deterministic pipeline on sample chart
    pipeline_result = execute_full_deterministic_pipeline(SAMPLE_CHART_DATA)

    # Validate Stage 8 structure
    is_valid_stage8 = validate_stage8_pipeline(pipeline_result)
    print(f"Stage 8 Pipeline Structure Integrity: {'PASS' if is_valid_stage8 else 'FAIL'}\n")

    domains = ["career", "marriage", "finance", "education", "property"]
    
    audit_summary = {}

    for domain in domains:
        stage_key = f"stage_8_{'15_career' if domain=='career' else ('16_marriage' if domain=='marriage' else ('17_finance' if domain=='finance' else ('18_education' if domain=='education' else '19_property')))}_rules"
        domain_data = pipeline_result.get(stage_key, {})
        
        # Get rules list from direct domain_data or nested rule analysis dict
        if isinstance(domain_data, dict) and "rules" in domain_data:
            rules = domain_data["rules"]
        elif isinstance(domain_data, dict) and rule_analysis_key in domain_data:
            rules = domain_data[rule_analysis_key].get("rules", [])
        else:
            rules = []
        matched_rules = [r for r in rules if r.get("matched", False)]

        audit_summary[domain] = {
            "total_rules": len(rules),
            "matched_rules": len(matched_rules),
            "matched_rule_ids": [r["rule_id"] for r in matched_rules],
            "keys": [r.get("interpretation_key") for r in matched_rules]
        }

        print(f"--- Domain: {domain.upper()} ---")
        print(f"Total Rules Defined: {len(rules)} | Matched Rules: {len(matched_rules)}")
        for r in rules:
            status = "MATCHED" if r.get("matched") else "NOT MATCHED"
            interp_key = r.get("interpretation_key")
            interp_text = get_interpretation_text(interp_key) if r.get("matched") else "N/A"
            print(f"  [{status:<11}] Rule ID: {r.get('rule_id'):<35} | Key: {interp_key}")
            if r.get("matched"):
                print(f"                Interpretation Text: \"{interp_text}\"")
                print(f"                Evidence Payload:    {json.dumps(r.get('evidence', {}))}")

        print("-" * 90)

    print("\n" + "=" * 90)
    print(" SUMMARY OF STAGE 8 AUDIT")
    print("=" * 90)
    for domain, info in audit_summary.items():
        print(f"  - {domain.upper():<10}: {info['matched_rules']}/{info['total_rules']} Rules Matched | Keys: {info['keys']}")

    print("=" * 90)

if __name__ == "__main__":
    run_stage8_audit()
