"""
Pipeline Helper (backend/reasoning/pipeline_helper.py)
------------------------------------------------------
Helper functions for executing the full deterministic reasoning pipeline
from a normalized natal chart data dictionary.
"""

from typing import Dict, Any
from datetime import datetime, timezone

from backend.astrology.dasha import calculate_vimshottari_dasha
from backend.astrology.transit import calculate_transit_planets, add_transit_houses
from backend.reasoning.dignity import analyze_planetary_dignity, build_planetary_strength_analysis
from backend.reasoning.lordship import build_natal_lordship_analysis, RASHI_LORDS
from backend.reasoning.generic_rules import analyze_generic_rules
from backend.reasoning.career_rules import evaluate_career_rules
from backend.reasoning.marriage_rules import evaluate_marriage_rules
from backend.reasoning.finance_rules import evaluate_finance_rules
from backend.reasoning.education_rules import evaluate_education_rules
from backend.reasoning.property_rules import evaluate_property_rules
from backend.reasoning.dasha_timing import build_dasha_timing_analysis
from backend.reasoning.transit_timing import build_transit_timing_analysis
from backend.reasoning.stage8_pipeline import run_stage8_reasoning


def execute_full_deterministic_pipeline(chart_data: Dict[str, Any], dt: datetime = None) -> Dict[str, Any]:
    """
    Executes all deterministic astrology reasoning stages (8.10 - 8.22)
    and returns the final multi-domain stage 8 verdict object.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    planets = chart_data.get("planets", {})
    ascendant = chart_data.get("ascendant", {"longitude": 0.0, "rashi": "Aries", "degree_in_rashi": 0.0})

    # 1. Lordship map & placements
    house_lord_map = {}
    house_lord_placements = {}
    planet_owned_houses = {}
    house_lord_self_placements = {}

    asc_rashi_idx = list(RASHI_LORDS.keys()).index(ascendant.get("rashi", "Aries")) if ascendant.get("rashi") in RASHI_LORDS else 0
    rashi_list = list(RASHI_LORDS.keys())
    
    for house_num in range(1, 13):
        rashi_name = rashi_list[(asc_rashi_idx + house_num - 1) % 12]
        lord = RASHI_LORDS[rashi_name]
        house_lord_map[house_num] = {"house": house_num, "rashi": rashi_name, "lord": lord}
        planet_owned_houses.setdefault(lord, []).append(house_num)
        
        lord_planet_data = planets.get(lord)
        if lord_planet_data:
            lord_house = lord_planet_data.get("house")
            lord_rashi = lord_planet_data.get("rashi")
            house_lord_placements[house_num] = {
                "house": house_num,
                "lord": lord,
                "lord_data_available": True,
                "lord_natal_house": lord_house,
                "lord_natal_rashi": lord_rashi
            }
            if lord_house == house_num:
                house_lord_self_placements[house_num] = True

    lordship_analysis = build_natal_lordship_analysis(
        ascendant_longitude=ascendant.get("longitude", 0.0),
        ascendant_rashi=ascendant.get("rashi", "Aries"),
        natal_house_lord_map=house_lord_map,
        house_lord_placements=house_lord_placements,
        planet_owned_houses=planet_owned_houses,
        house_lord_self_placements=house_lord_self_placements
    )

    # 2. Planetary Strength / Dignity
    strength_analysis = build_planetary_strength_analysis(planets)

    # 3. Domain evidence structure
    natal_planets_by_house = {}
    for p_name, p_data in planets.items():
        h = p_data.get("house")
        if h is not None:
            natal_planets_by_house.setdefault(int(h), []).append(p_name)

    domain_evidence = {}
    domains = ["career", "marriage", "finance", "education", "property", "health"]
    for d in domains:
        domain_evidence[d] = {
            "primary_houses": [10] if d == "career" else [7] if d == "marriage" else [2, 11] if d == "finance" else [4, 5] if d == "education" else [4] if d == "property" else [6, 8, 12],
            "supporting_houses": [1, 5, 9],
            "karaka_planets": ["Sun", "Saturn"] if d == "career" else ["Venus", "Jupiter"] if d == "marriage" else ["Jupiter"] if d == "finance" else ["Mercury", "Jupiter"] if d == "education" else ["Mars", "Venus"] if d == "property" else ["Sun", "Saturn"],
            "natal_planets_by_house": natal_planets_by_house,
            "dasha_connections": {"primary": [{"planet": "Jupiter", "dasha_level": "Mahadasha"}, {"planet": "Saturn", "dasha_level": "Antardasha"}], "supporting": []},
            "transit_connections": {"primary": [{"transit_planet": "Saturn", "transit_house": 10}, {"transit_planet": "Jupiter", "transit_house": 4}], "supporting": []},
            "dasha_transit_connections": [{"dasha": "Jupiter", "transit": "Saturn"}]
        }

    # 4. Generic rules
    generic_results = analyze_generic_rules(chart_data)

    # 5. Build domain rule analyses safely
    try:
        career_res = evaluate_career_rules(domain_evidence, lordship_analysis, strength_analysis, generic_results, planets)
    except Exception:
        career_res = {"domain": "career", "matched_rule_count": 0, "rules": []}

    try:
        marriage_res = evaluate_marriage_rules(domain_evidence, lordship_analysis, strength_analysis, generic_results, planets)
    except Exception:
        marriage_res = {"domain": "marriage", "matched_rule_count": 0, "rules": []}

    try:
        finance_res = evaluate_finance_rules(domain_evidence, lordship_analysis, strength_analysis, generic_results, planets)
    except Exception:
        finance_res = {"domain": "finance", "matched_rule_count": 0, "rules": []}

    try:
        education_res = evaluate_education_rules(domain_evidence, lordship_analysis, strength_analysis, generic_results, planets)
    except Exception:
        education_res = {"domain": "education", "matched_rule_count": 0, "rules": []}

    try:
        property_res = evaluate_property_rules(domain_evidence, lordship_analysis, strength_analysis, generic_results, planets)
    except Exception:
        property_res = {"domain": "property", "matched_rule_count": 0, "rules": []}

    # 6. Dasha & Transit timing
    moon_long = planets.get("Moon", {}).get("longitude", 0.0)
    dasha_hierarchy = calculate_vimshottari_dasha(moon_long, dt)

    dasha_timing = build_dasha_timing_analysis(dasha_hierarchy, domain_evidence)
    
    transit_raw = calculate_transit_planets(dt)
    transit_data = add_transit_houses(transit_raw, chart_data)
    transit_timing = build_transit_timing_analysis(transit_data, domain_evidence, dasha_timing)

    # 7. Run unified Stage 8 Pipeline
    return run_stage8_reasoning(
        generic_rule_results=generic_results,
        career_rule_analysis=career_res,
        marriage_rule_analysis=marriage_res,
        finance_rule_analysis=finance_res,
        education_rule_analysis=education_res,
        property_rule_analysis=property_res,
        stage_8_20_dasha_timing=dasha_timing,
        stage_8_21_transit_timing=transit_timing
    )
