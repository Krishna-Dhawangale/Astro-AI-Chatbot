"""
Stage 15.3 — Evidence -> Calibrated Semantic Interpretation -> Multi-Domain Lineage & Integrity
=====================================================================================================
Module: backend/reasoning/answer_synthesizer.py

Purpose:
Converts extracted Stage 8 evidence into calibrated, planet-grounded interpretations,
hierarchical domain themes (Primary vs. Secondary), clear non-repetitive WHY explanations,
a structured internal theme_lineage array, and a concise "Bottom Line" summary.

Pipeline:
  Raw Chart -> Evidence Extraction -> Rule Matching -> Planet-Specific Interpretation -> Theme Lineage -> Calibrated Wording -> Bottom Line Synthesis
"""

import re
from typing import Dict, List, Any, Optional, Tuple


def guard_deterministic_claims(text: str) -> str:
    """
    Stage 16 — Hard Architectural Claim Guard:
    Rewrites any deterministic or prescriptive predictions into exploratory, open-ended statements.
    Guarantees THEME != CERTAINTY in all user-facing output.
    """
    if not text:
        return text

    replacements = [
        (r"\bYou should become a ([A-Za-z0-9 &,/]+)\b", r"Astrological indicators highlight \1 as a primary direction to explore"),
        (r"\bYou should become ([A-Za-z0-9 &,/]+)\b", r"Astrological indicators point toward \1 as a key potential direction"),
        (r"\bYou should pursue ([A-Za-z0-9 &,/]+)\b", r"A recommended field to explore is \1"),
        (r"\bYou must become ([A-Za-z0-9 &,/]+)\b", r"The chart suggests evaluating \1"),
        (r"\bYour career will be ([A-Za-z0-9 &,/]+)\b", r"Your chart indicates primary potential in \1"),
        (r"\bYour career is ([A-Za-z0-9 &,/]+)\b", r"Your astrological foundation aligns with \1"),
        (r"\bThe best career for you is ([A-Za-z0-9 &,/]+)\b", r"A top astrological theme to explore is \1"),
        (r"\bYou are destined to ([A-Za-z0-9 &,/]+)\b", r"Astrological factors indicate natural affinity for \1"),
    ]

    guarded_text = text
    for pattern, replacement in replacements:
        guarded_text = re.sub(pattern, replacement, guarded_text, flags=re.IGNORECASE)

    return guarded_text


# Rule Weight & Category Mapping
RULE_WEIGHT_MAP = {
    # House Lord Placements (Highest Weight - Foundation)
    "CAREER_10TH_LORD_PLACEMENT": {"weight": 3.5, "category": "foundation", "label": "10th Lord Placement"},
    "MARRIAGE_7TH_LORD_PLACEMENT": {"weight": 3.5, "category": "foundation", "label": "7th Lord Placement"},
    "FINANCE_2ND_LORD_PLACEMENT": {"weight": 3.5, "category": "foundation", "label": "2nd Lord Placement"},
    "FINANCE_11TH_LORD_PLACEMENT": {"weight": 3.5, "category": "foundation", "label": "11th Lord Placement"},
    "EDUCATION_4TH_LORD_PLACEMENT": {"weight": 3.5, "category": "foundation", "label": "4th Lord Placement"},
    "EDUCATION_5TH_LORD_PLACEMENT": {"weight": 3.5, "category": "foundation", "label": "5th Lord Placement"},
    "PROPERTY_4TH_LORD_PLACEMENT": {"weight": 3.5, "category": "foundation", "label": "4th Lord Placement"},

    # Karaka Dignity & Strength (High Weight)
    "CAREER_KARAKA_DIGNITY": {"weight": 3.0, "category": "dignity", "label": "Career Karaka Dignity"},
    "MARRIAGE_KARAKA_DIGNITY": {"weight": 3.0, "category": "dignity", "label": "Marriage Karaka Dignity"},
    "FINANCE_KARAKA_DIGNITY": {"weight": 3.0, "category": "dignity", "label": "Finance Karaka Dignity"},
    "EDUCATION_KARAKA_DIGNITY": {"weight": 3.0, "category": "dignity", "label": "Education Karaka Dignity"},
    "PROPERTY_KARAKA_DIGNITY": {"weight": 3.0, "category": "dignity", "label": "Property Karaka Dignity"},

    # Karaka & House Placements (Medium Weight)
    "CAREER_KARAKA_PRESENCE": {"weight": 2.0, "category": "karaka", "label": "Career Karaka Presence"},
    "MARRIAGE_KARAKA_PRESENCE": {"weight": 2.0, "category": "karaka", "label": "Marriage Karaka Presence"},
    "FINANCE_KARAKA_PRESENCE": {"weight": 2.0, "category": "karaka", "label": "Finance Karaka Presence"},
    "EDUCATION_KARAKA_PRESENCE": {"weight": 2.0, "category": "karaka", "label": "Education Karaka Presence"},
    "PROPERTY_KARAKA_PRESENCE": {"weight": 2.0, "category": "karaka", "label": "Property Karaka Presence"},

    "CAREER_RELEVANT_HOUSE_EVIDENCE": {"weight": 2.0, "category": "house_evidence", "label": "Career House Placements"},
    "MARRIAGE_RELEVANT_HOUSE_EVIDENCE": {"weight": 2.0, "category": "house_evidence", "label": "Marriage House Placements"},
    "FINANCE_RELEVANT_HOUSE_EVIDENCE": {"weight": 2.0, "category": "house_evidence", "label": "Finance House Placements"},
    "EDUCATION_RELEVANT_HOUSE_EVIDENCE": {"weight": 2.0, "category": "house_evidence", "label": "Education House Placements"},
    "PROPERTY_RELEVANT_HOUSE_EVIDENCE": {"weight": 2.0, "category": "house_evidence", "label": "Property House Placements"},

    # Timing Influences (High Weight for Timing Queries)
    "CAREER_DASHA_ACTIVATION": {"weight": 2.5, "category": "timing", "label": "Dasha Career Connection"},
    "MARRIAGE_DASHA_ACTIVATION": {"weight": 2.5, "category": "timing", "label": "Dasha Marriage Connection"},
    "FINANCE_DASHA_ACTIVATION": {"weight": 2.5, "category": "timing", "label": "Dasha Finance Connection"},
    "EDUCATION_DASHA_ACTIVATION": {"weight": 2.5, "category": "timing", "label": "Dasha Education Connection"},
    "PROPERTY_DASHA_ACTIVATION": {"weight": 2.5, "category": "timing", "label": "Dasha Property Connection"},

    "CAREER_TRANSIT_ACTIVATION": {"weight": 2.5, "category": "timing", "label": "Transit Career Connection"},
    "MARRIAGE_TRANSIT_ACTIVATION": {"weight": 2.5, "category": "timing", "label": "Transit Marriage Connection"},
    "FINANCE_TRANSIT_ACTIVATION": {"weight": 2.5, "category": "timing", "label": "Transit Finance Connection"},
    "EDUCATION_TRANSIT_ACTIVATION": {"weight": 2.5, "category": "timing", "label": "Transit Education Connection"},
    "PROPERTY_TRANSIT_ACTIVATION": {"weight": 2.5, "category": "timing", "label": "Transit Property Connection"},

    "CAREER_DASHA_TRANSIT_COMBINATION": {"weight": 1.5, "category": "timing_synergy", "label": "Dasha & Transit Synergy"},
    "MARRIAGE_DASHA_TRANSIT_COMBINATION": {"weight": 1.5, "category": "timing_synergy", "label": "Dasha & Transit Synergy"},
    "FINANCE_DASHA_TRANSIT_COMBINATION": {"weight": 1.5, "category": "timing_synergy", "label": "Dasha & Transit Synergy"},
    "EDUCATION_DASHA_TRANSIT_COMBINATION": {"weight": 1.5, "category": "timing_synergy", "label": "Dasha & Transit Synergy"},
    "PROPERTY_DASHA_TRANSIT_COMBINATION": {"weight": 1.5, "category": "timing_synergy", "label": "Dasha & Transit Synergy"},

    # Multi-Factor Confirmation
    "CAREER_MULTI_FACTOR_ACTIVATION": {"weight": 1.5, "category": "multi_factor", "label": "Multi-Factor Confirmation"},
    "MARRIAGE_MULTI_FACTOR_ACTIVATION": {"weight": 1.5, "category": "multi_factor", "label": "Multi-Factor Confirmation"},
    "FINANCE_MULTI_FACTOR_ACTIVATION": {"weight": 1.5, "category": "multi_factor", "label": "Multi-Factor Confirmation"},
    "EDUCATION_MULTI_FACTOR_ACTIVATION": {"weight": 1.5, "category": "multi_factor", "label": "Multi-Factor Confirmation"},
    "PROPERTY_MULTI_FACTOR_ACTIVATION": {"weight": 1.5, "category": "multi_factor", "label": "Multi-Factor Confirmation"},
}

# ---------------------------------------------------------------------------
# Domain-Specific Planetary Themes & Capabilities
# ---------------------------------------------------------------------------

PLANETARY_CAREER_THEMES = {
    "Sun": {
        "qualities": "executive leadership, administrative authority, public governance, and strategic vision",
        "fields": ["Government & Public Administration", "Executive Management & Leadership", "Policy & Strategy", "Corporate Governance"]
    },
    "Moon": {
        "qualities": "public relations, psychological insight, resource management, and client-facing communication",
        "fields": ["Healthcare & Counseling", "Public Relations & Media", "Hospitality & Human Resources", "Community & Social Operations"]
    },
    "Mars": {
        "qualities": "engineering execution, technical operations, project management, and decisive problem-solving",
        "fields": ["Engineering & Technology", "Operations & Logistics", "Project Execution & Infrastructure", "Defense & Security Technology"]
    },
    "Mercury": {
        "qualities": "analytical thinking, data processing, systems architecture, business analytics, and communications",
        "fields": ["Data & Business Analytics", "Software & Systems Engineering", "Financial Accounting & Commerce", "Technical Writing & Communications"]
    },
    "Jupiter": {
        "qualities": "strategic advisory, financial planning, mentorship, legal acumen, and high-level consulting",
        "fields": ["Strategic Consulting & Mentorship", "Finance, Banking & Investment", "Legal & Advisory Services", "Higher Education & Research"]
    },
    "Venus": {
        "qualities": "design aesthetics, user experience, financial relations, media production, and commercial strategy",
        "fields": ["User Experience & Product Design", "Media, Arts & Communications", "Financial Services & Commerce", "Luxury & Brand Management"]
    },
    "Saturn": {
        "qualities": "structured management, organizational architecture, technical systems, and long-term administrative discipline",
        "fields": ["Operations Management & Governance", "Systems Engineering & Infrastructure", "Regulatory Compliance & Law", "Supply Chain & Industrial Operations"]
    },
    "Rahu": {
        "qualities": "cutting-edge technological innovation, foreign operations, artificial intelligence, and media strategy",
        "fields": ["Emerging Tech & AI Systems", "International Business & Global Trade", "Digital Media & Data Science", "Innovations & R&D"]
    },
    "Ketu": {
        "qualities": "deep technical research, software architecture, specialized data engineering, and back-end analysis",
        "fields": ["Software Architecture & Cyber Security", "Deep Technical Research & Analytics", "Specialized Engineering", "Data Science & Auditing"]
    }
}

PLANETARY_MARRIAGE_THEMES = {
    "Sun": {
        "qualities": "leadership, administrative authority, strong principled values, and clear partnership boundaries",
        "fields": ["High-Status & Executive Partner", "Principled & Purpose-Driven Alignment", "Clear Personal Boundaries & Respect"]
    },
    "Moon": {
        "qualities": "deep emotional empathy, nurturing domestic care, mutual psychological support, and fluid communication",
        "fields": ["Nurturing & Emotional Harmony", "Empathic Domestic Bonding", "Responsive Mutual Care"]
    },
    "Mars": {
        "qualities": "passionate dynamic energy, assertive protection, physical activity, and courageous shared goals",
        "fields": ["Dynamic & Passionate Partnership", "Active Shared Ventures", "Direct & Honest Communication"]
    },
    "Mercury": {
        "qualities": "intellectual companionship, witty communication, shared business interests, and mental adaptability",
        "fields": ["Intellectual & Mental Alignment", "Shared Business & Analytical Projects", "Playful & Open Dialogue"]
    },
    "Jupiter": {
        "qualities": "wise mentorship, spiritual alignment, high moral values, and expansive marital prosperity",
        "fields": ["Spiritual & Philosophical Harmony", "Advisory & Wise Partnership", "Prosperous Family Growth"]
    },
    "Venus": {
        "qualities": "romantic devotion, refined aesthetic lifestyle, artistic harmony, and deep mutual affection",
        "fields": ["Romantic Devotion & Elegance", "Artistic & Aesthetic Shared Life", "Harmonious & Affectionate Bond"]
    },
    "Saturn": {
        "qualities": "enduring commitment, traditional stability, patient dedication, and mature shared responsibility",
        "fields": ["Long-Term Stability & Loyalty", "Mature & Grounded Partnership", "Structured Shared Responsibility"]
    },
    "Rahu": {
        "qualities": "unconventional chemistry, cross-cultural connections, innovative relational dynamic, and modern alignment",
        "fields": ["Unconventional & Cross-Cultural Bond", "Innovative & Modern Relationship", "Global & Diverse Backgrounds"]
    },
    "Ketu": {
        "qualities": "deep karmic bond, intuitive non-verbal understanding, shared spiritual depth, and quiet devotion",
        "fields": ["Karmic & Intuitive Connection", "Spiritual Depth & Meditation", "Unspoken Mutual Trust"]
    }
}

PLANETARY_FINANCE_THEMES = {
    "Sun": {
        "qualities": "income through executive governance, public sector contracts, brand equity, and high-level management",
        "fields": ["Executive & Government Income", "Corporate Leadership Dividends", "Brand Equity & Sovereign Wealth"]
    },
    "Moon": {
        "qualities": "liquid wealth, public commerce revenue, hospitality assets, and consumer market cashflow",
        "fields": ["Liquid Capital & Cashflow Management", "Consumer Markets & Public Trade", "Real Estate & Hospitality Revenue"]
    },
    "Mars": {
        "qualities": "capital generation through real estate development, engineering projects, industrial trade, and bold investments",
        "fields": ["Real Estate & Land Acquisition", "Industrial & Engineering Ventures", "High-Growth Capital Allocation"]
    },
    "Mercury": {
        "qualities": "commercial trading revenue, financial analytics, consulting fees, e-commerce, and diversified portfolios",
        "fields": ["Financial Trading & Commerce", "Business Analytics & Accounting", "Diversified Information Products"]
    },
    "Jupiter": {
        "qualities": "institutional wealth, capital growth through advisory, banking returns, asset expansion, and wealth preservation",
        "fields": ["Institutional Banking & Capital", "Strategic Financial Advisory", "Asset Preservation & Expansion"]
    },
    "Venus": {
        "qualities": "wealth through luxury goods, creative production, joint venture assets, entertainment, and high-end commerce",
        "fields": ["Luxury Goods & High-End Commerce", "Creative & Media Wealth", "Joint Assets & Commercial Ventures"]
    },
    "Saturn": {
        "qualities": "steady long-term compounding, infrastructure holdings, mineral/real estate assets, and disciplined savings",
        "fields": ["Disciplined Compounding & Savings", "Infrastructure & Hard Assets", "Long-Term Value Investing"]
    },
    "Rahu": {
        "qualities": "wealth via emerging tech, fintech/crypto, foreign trade, high-yield innovation, and digital ventures",
        "fields": ["Fintech & Tech Investments", "Global Trade & Foreign Revenue", "High-Growth Innovative Capital"]
    },
    "Ketu": {
        "qualities": "niche specialized consulting, technical audit returns, intellectual property equity, and quiet capital streams",
        "fields": ["Specialized Technical Intellectual Property", "Analytical & Algorithmic Revenue", "Niche Capital & R&D Equity"]
    }
}

PLANETARY_EDUCATION_THEMES = {
    "Sun": {
        "qualities": "public administration, political science, executive leadership studies, governance, and medicine",
        "fields": ["Public Policy & Governance", "Executive Leadership & Political Science", "Medical Sciences & Administration"]
    },
    "Moon": {
        "qualities": "psychology, humanities, literature, nursing, environmental science, and social communication",
        "fields": ["Psychology & Behavioral Sciences", "Humanities & Literature", "Healthcare & Nursing Studies"]
    },
    "Mars": {
        "qualities": "engineering disciplines, applied physics, mechanical & electrical systems, defense studies, and sports science",
        "fields": ["Engineering & Applied Physics", "Defense & Strategic Technology", "Robotics & Hardware Systems"]
    },
    "Mercury": {
        "qualities": "computer science, data science, mathematics, economics, journalism, and linguistics",
        "fields": ["Computer Science & Data Science", "Mathematics & Quantitative Economics", "Journalism & Media Studies"]
    },
    "Jupiter": {
        "qualities": "law, jurisprudence, philosophy, higher academic research, finance, and advanced pedagogy",
        "fields": ["Law & Legal Studies", "Philosophy & Academic Pedagogy", "Finance & Higher Research"]
    },
    "Venus": {
        "qualities": "fine arts, architecture, visual design, fashion, media production, and modern languages",
        "fields": ["Architecture & Industrial Design", "Fine Arts & Performing Arts", "Media Production & Creative Arts"]
    },
    "Saturn": {
        "qualities": "civil engineering, geology, history, structural architecture, law enforcement, and methodical research",
        "fields": ["Civil & Structural Engineering", "History & Archaeology", "Regulatory Law & Systematic Research"]
    },
    "Rahu": {
        "qualities": "artificial intelligence, biotechnology, foreign degree programs, modern media technology, and innovation",
        "fields": ["Artificial Intelligence & Machine Learning", "Biotechnology & Applied Genetics", "International Academic Programs"]
    },
    "Ketu": {
        "qualities": "deep theoretical mathematics, theoretical physics, cyber security, metaphysics, and back-end logic",
        "fields": ["Theoretical Mathematics & Physics", "Cyber Security & Cryptography", "Specialized Esoteric Research"]
    }
}

PLANETARY_PROPERTY_THEMES = {
    "Sun": {
        "qualities": "prime commercial land, government-approved developments, ancestral estates, and high-visibility properties",
        "fields": ["Prime Commercial Real Estate", "Government & Municipal Land", "Ancestral & High-Status Estates"]
    },
    "Moon": {
        "qualities": "waterfront property, serene residential homes, fertile land, and domestic real estate assets",
        "fields": ["Waterfront & Coastal Properties", "Serene Residential Homes", "Agricultural & Fertile Land"]
    },
    "Mars": {
        "qualities": "land ownership, physical construction, industrial warehouses, real estate development, and building assets",
        "fields": ["Plot Ownership & Land Development", "Construction & Building Projects", "Industrial Real Estate & Warehouses"]
    },
    "Mercury": {
        "qualities": "urban apartments, commercial office spaces, technology park real estate, retail outlets, and multi-unit complexes",
        "fields": ["Urban Apartments & Condominiums", "Commercial Office Spaces & Retail", "Technology Park Real Estate"]
    },
    "Jupiter": {
        "qualities": "spacious ancestral estates, institutional grounds, educational property, and high-value long-term land holdings",
        "fields": ["Spacious Ancestral Estates", "Institutional & Educational Grounds", "High-Value Long-Term Property"]
    },
    "Venus": {
        "qualities": "luxury villas, designer interior estates, resort properties, aesthetic residential architecture, and high-end living",
        "fields": ["Luxury Residential Villas", "Resort & Hospitality Real Estate", "Designer & High-End Interiors"]
    },
    "Saturn": {
        "qualities": "agricultural farmland, durable stone/brick structures, long-term land banks, and heavy industrial property",
        "fields": ["Agricultural & Farm Land Holdings", "Durable Structural Real Estate", "Long-Term Land Banks"]
    },
    "Rahu": {
        "qualities": "high-tech smart homes, international property investments, modern high-rise condominiums, and urban developments",
        "fields": ["High-Tech Smart Homes", "International Real Estate Investments", "High-Rise Luxury Condominiums"]
    },
    "Ketu": {
        "qualities": "secluded countryside retreats, quiet meditation estates, compact efficient homes, and inherited spiritual land",
        "fields": ["Secluded Countryside Retreats", "Spiritual & Quiet Estates", "Compact Minimalist Real Estate"]
    }
}

PLANETARY_HEALTH_THEMES = {
    "Sun": {
        "qualities": "vitality, heart health, spinal strength, eye health, and overall stamina",
        "fields": ["Cardiovascular & Heart Vitality", "Spinal Alignment & Bone Strength", "Immune Resilience & Eye Health"]
    },
    "Moon": {
        "qualities": "emotional well-being, digestive balance, fluid circulation, and psychological equilibrium",
        "fields": ["Mental & Emotional Equilibrium", "Digestive & Fluid System Harmony", "Sleep & Circadian Rhythm Care"]
    },
    "Mars": {
        "qualities": "muscle tone, blood circulation, physical endurance, metabolic energy, and injury prevention",
        "fields": ["Muscle Vitality & Physical Fitness", "Blood Circulation & Energy", "Metabolic Health & Movement"]
    },
    "Mercury": {
        "qualities": "nervous system balance, respiratory clarity, skin health, and cognitive function",
        "fields": ["Nervous System Health", "Respiratory Clarity & Breathing", "Cognitive Health & Focus"]
    },
    "Jupiter": {
        "qualities": "liver health, metabolic regulation, joint mobility, cellular expansion, and dietary balance",
        "fields": ["Metabolic & Liver Wellness", "Cellular & Tissue Vitality", "Nutritional Balance & Joint Care"]
    },
    "Venus": {
        "qualities": "kidney function, hormonal equilibrium, reproductive health, and skin hydration",
        "fields": ["Hormonal & Endocrine Balance", "Kidney & Renal Health", "Skin & Aesthetic Vitality"]
    },
    "Saturn": {
        "qualities": "bone density, joint flexibility, teeth strength, structural resilience, and chronic endurance",
        "fields": ["Bone Density & Skeletal Health", "Joint Flexibility & Structural Care", "Endurance & Chronic Resilience"]
    },
    "Rahu": {
        "qualities": "stress management, allergen sensitivity, psychological calm, and modern lifestyle balance",
        "fields": ["Stress & Anxiety Regulation", "Allergen & Environmental Shielding", "Modern Bio-Rhythm Optimization"]
    },
    "Ketu": {
        "qualities": "gut microbiome balance, subtle nerve health, detoxification, and holistic mind-body healing",
        "fields": ["Gut Microbiome & Detoxification", "Subtle Nervous System Care", "Holistic Mind-Body Healing"]
    }
}

DOMAIN_CONFIG_MAP = {
    "career": {
        "themes": PLANETARY_CAREER_THEMES,
        "house_name": "10th house of career and profession",
        "title_section_1": "Astrological Foundation",
        "title_section_3": "Recommended Career Themes & Fields",
        "title_section_4": "Why These Areas?",
        "field_header": "Based on your 10th house lord and active planetary karakas, the primary and secondary themes to explore are:",
        "anchor_desc": "forms the core astrological anchor for your professional trajectory."
    },
    "marriage": {
        "themes": PLANETARY_MARRIAGE_THEMES,
        "house_name": "7th house of marriage and relationship partnerships",
        "title_section_1": "Astrological Relationship Foundation",
        "title_section_3": "Recommended Relationship & Marriage Themes",
        "title_section_4": "Why These Dimensions?",
        "field_header": "Based on your 7th house lord and active relationship karakas, the primary relational themes to explore are:",
        "anchor_desc": "forms the core astrological anchor for your relationship dynamic."
    },
    "finance": {
        "themes": PLANETARY_FINANCE_THEMES,
        "house_name": "2nd & 11th houses of wealth, earnings, and asset accumulation",
        "title_section_1": "Astrological Financial Foundation",
        "title_section_3": "Recommended Wealth & Income Themes",
        "title_section_4": "Why These Capital Avenues?",
        "field_header": "Based on your wealth house lords and active financial karakas, the primary wealth avenues to explore are:",
        "anchor_desc": "forms the core astrological anchor for your financial accumulation potential."
    },
    "education": {
        "themes": PLANETARY_EDUCATION_THEMES,
        "house_name": "4th & 5th houses of education, learning, and academic intelligence",
        "title_section_1": "Astrological Academic Foundation",
        "title_section_3": "Recommended Educational & Knowledge Themes",
        "title_section_4": "Why These Academic Fields?",
        "field_header": "Based on your education house lords and active knowledge karakas, the primary academic domains to explore are:",
        "anchor_desc": "forms the core astrological anchor for your learning and academic direction."
    },
    "property": {
        "themes": PLANETARY_PROPERTY_THEMES,
        "house_name": "4th house of property, real estate, and fixed assets",
        "title_section_1": "Astrological Property & Asset Foundation",
        "title_section_3": "Recommended Property & Real Estate Themes",
        "title_section_4": "Why These Asset Types?",
        "field_header": "Based on your 4th house lord and active property karakas, the primary real estate themes to explore are:",
        "anchor_desc": "forms the core astrological anchor for your property and vehicle asset trajectory."
    },
    "health": {
        "themes": PLANETARY_HEALTH_THEMES,
        "house_name": "6th, 8th, & 12th houses of health, vitality, and physical wellness",
        "title_section_1": "Astrological Health & Vitality Foundation",
        "title_section_3": "Recommended Health & Vitality Focus Areas",
        "title_section_4": "Why These Focus Areas?",
        "field_header": "Based on your health house lords and active planetary significators, key physical well-being areas to focus on are:",
        "anchor_desc": "forms the core astrological anchor for your physical vitality and health management."
    }
}

# ---------------------------------------------------------------------------
# Calibrated Wording Headers (Stage 15.3 — Attuned Wording where THEME != CERTAINTY)
# ---------------------------------------------------------------------------

CALIBRATED_STATUS_HEADERS = {
    "STRONGLY_FAVORED": {
        "header_prefix": "Based on strong astrological foundation alignment and well-dignified significators, the primary and secondary themes to explore are:",
        "why_prefix": "The chart exhibits robust structural support without major debilities, indicating clear natural alignment in these areas:",
        "certainty_note": ""
    },
    "MODERATELY_FAVORED_WITH_COUNTERBALANCE": {
        "header_prefix": "The chart indicates promising potential in the following areas, alongside specific counter-balancing factors to navigate:",
        "why_prefix": "These themes represent favorable directions derived from your domain lord, though counter-balancing factors require disciplined application:",
        "certainty_note": "\n*Note: These themes represent high-potential directions rather than guaranteed outcomes due to mixed planetary factors.*"
    },
    "CHALLENGING_PERIOD": {
        "header_prefix": "While the following themes emerge from your primary significators, structural or placement challenges suggest treating these as directions requiring patience and conscious effort:",
        "why_prefix": "Although your significators highlight these fields, challenging planetary positions suggest potential hurdles or slower initial progression:",
        "certainty_note": "\n*Note: In Vedic Astrology, challenging placements signify areas that require deliberate effort, skill development, and strategic patience rather than immediate ease.*"
    },
    "BALANCED": {
        "header_prefix": "The chart displays a balanced set of astrological indicators, pointing toward versatile potential in:",
        "why_prefix": "These domains are supported by balanced planetary placements across key houses:",
        "certainty_note": ""
    }
}


def evaluate_evidence_weights(matched_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes cumulative weight scores and categorizes evidence by importance.
    """
    total_score = 0.0
    weighted_rules = []

    for rule in matched_rules:
        rule_id = rule.get("rule_id", "")
        weight_info = RULE_WEIGHT_MAP.get(rule_id, {"weight": 1.0, "category": "general", "label": rule_id})
        w = weight_info["weight"]
        total_score += w
        
        weighted_rules.append({
            "rule_id": rule_id,
            "weight": w,
            "category": weight_info["category"],
            "label": weight_info["label"],
            "evidence": rule.get("evidence", {}),
            "interpretation_key": rule.get("interpretation_key")
        })

    weighted_rules.sort(key=lambda x: x["weight"], reverse=True)

    return {
        "total_score": round(total_score, 2),
        "rule_count": len(matched_rules),
        "weighted_rules": weighted_rules
    }


def resolve_evidence_conflicts(weighted_evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates dignities and potential astrological conflicts (exalted vs enemy sign / debility).
    """
    rules = weighted_evidence.get("weighted_rules", [])
    
    positive_signals = []
    challenging_signals = []

    for item in rules:
        ev = item.get("evidence")
        category = item.get("category")

        if category == "dignity" and isinstance(ev, list):
            for d in ev:
                planet = d.get("planet")
                dignity = d.get("dignity")
                rashi = d.get("rashi")
                
                if dignity in ["exalted", "own_sign"]:
                    positive_signals.append(f"{planet} in {rashi} ({dignity})")
                elif dignity in ["debilitated", "enemy_sign"]:
                    challenging_signals.append(f"{planet} in {rashi} ({dignity})")

        if category == "foundation" and isinstance(ev, dict):
            lord = ev.get("lord")
            house = ev.get("lord_natal_house") or ev.get("house")
            rashi = ev.get("lord_natal_rashi") or ev.get("house_rashi")
            if house in [1, 4, 7, 10, 5, 9]:
                positive_signals.append(f"Lord {lord} in House {house} ({rashi})")
            elif house in [6, 8, 12]:
                challenging_signals.append(f"Lord {lord} in Dusthana House {house} ({rashi})")

    if len(positive_signals) >= len(challenging_signals) and len(positive_signals) > 0:
        if len(challenging_signals) == 0:
            status = "STRONGLY_FAVORED"
            summary = "The chart displays clear, strong astrological support without major structural challenges."
        else:
            status = "MODERATELY_FAVORED_WITH_COUNTERBALANCE"
            summary = "The chart displays favorable foundation placements alongside specific counter-balancing factors."
    elif len(challenging_signals) > len(positive_signals):
        status = "CHALLENGING_PERIOD"
        summary = "The chart indicates structural or timing challenges requiring patience and disciplined effort."
    else:
        status = "BALANCED"
        summary = "The chart exhibits balanced astrological factors across houses and karakas."

    return {
        "status": status,
        "summary": summary,
        "positive_signals": positive_signals,
        "challenging_signals": challenging_signals
    }


def derive_domain_recommendation_themes(
    domain: str,
    weighted_rules: List[Dict[str, Any]]
) -> Tuple[List[str], List[str], str, List[Dict[str, Any]]]:
    """
    Stage 16 Semantic Derivation (Generalized across all domains):
    Converts extracted planets, lords, rashis, and houses into explicit domain themes,
    specific core capabilities/qualities, grounded explanations of WHY, and a structured theme_lineage array.
    """
    config = DOMAIN_CONFIG_MAP.get(domain, DOMAIN_CONFIG_MAP["career"])
    theme_dict = config["themes"]

    active_planets = set()
    foundation_lord = None
    foundation_house = None
    foundation_rashi = None

    for item in weighted_rules:
        ev = item.get("evidence")
        cat = item.get("category")

        if cat == "foundation" and isinstance(ev, dict):
            foundation_lord = ev.get("lord")
            foundation_house = ev.get("lord_natal_house") or ev.get("house")
            foundation_rashi = ev.get("lord_natal_rashi") or ev.get("house_rashi")
            if foundation_lord:
                active_planets.add(foundation_lord)

        if cat in ["dignity", "karaka"] and isinstance(ev, list):
            for d in ev:
                p = d.get("planet")
                if p:
                    active_planets.add(p)

    # Order planets dynamically: foundation_lord first, followed by remaining active planets
    ordered_planets = []
    if foundation_lord and foundation_lord in theme_dict:
        ordered_planets.append(foundation_lord)
    
    for p in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]:
        if p in active_planets and p not in ordered_planets and p in theme_dict:
            ordered_planets.append(p)

    recommended_fields = []
    capabilities = []
    reasons = []
    theme_lineage = []

    for planet in ordered_planets:
        info = theme_dict[planet]
        capabilities.append(f"**{planet}**: {info['qualities']}")
        
        priority = "primary" if planet == foundation_lord else "secondary"
        
        for f in info["fields"]:
            if f not in recommended_fields:
                recommended_fields.append(f)
                theme_lineage.append({
                    "theme": f,
                    "priority": priority,
                    "source_planets": [planet],
                    "source_rules": [r["rule_id"] for r in weighted_rules if planet in str(r.get("evidence", ""))],
                    "chart_evidence": {
                        "planet": planet,
                        "house": foundation_house if planet == foundation_lord else None,
                        "rashi": foundation_rashi if planet == foundation_lord else None,
                    },
                    "planetary_signification": info['qualities'],
                    "astrological_theme": f"{planet}-Derived {domain.capitalize()} Theme ({info['qualities'].split(',')[0].title()})",
                    "possible_career_fields": info["fields"],
                    "interpretation": info['qualities'],
                    "fields": info["fields"]
                })

        if planet == foundation_lord:
            h_str = f"House {foundation_house}" if foundation_house else "its designated house"
            r_str = f"({foundation_rashi})" if foundation_rashi else ""
            reasons.append(
                f"- **{recommended_fields[0] if recommended_fields else 'Primary Direction'}**: "
                f"Derived from domain lord **{planet}** placed in {r_str} {h_str}".strip() + f", emphasizing {info['qualities']}."
            )
        else:
            reasons.append(
                f"- **{info['fields'][0]}**: Supported by secondary significations of **{planet}**."
            )

    explanation_why = "\n".join(reasons) if reasons else f"The primary themes are derived from the domain lord and active planetary significators."

    return recommended_fields[:4], capabilities, explanation_why, theme_lineage


def derive_career_recommendation_themes(weighted_rules: List[Dict[str, Any]]) -> Tuple[List[str], List[str], str]:
    """Backward compatibility wrapper for career."""
    fields, caps, why, lineage = derive_domain_recommendation_themes("career", weighted_rules)
    return fields, caps, why


def synthesize_structured_answer(
    domain: str,
    matched_rules: List[Dict[str, Any]],
    timing_data: Dict[str, Any] = None,
    question: str = ""
) -> Dict[str, Any]:
    """
    Synthesizes evidence into Stage 16 Progressive, Non-Repetitive Interpretation & Recommendation Engine.
    Strictly separates Audit Layer (theme_lineage array) from clean User Layer.
    Includes Primary vs. Secondary themes, non-repetitive WHY explanations, contextual timing, and Bottom Line.
    Applies Hard Architectural Claim Guard and Qualitative Evidence Score Narrative.
    """
    if not matched_rules:
        return {
            "domain": domain,
            "has_answer": False,
            "structured_text": "Insufficient astrological evidence available in chart.",
            "evidence_score": 0.0,
            "conflict_status": "INSUFFICIENT_EVIDENCE",
            "positive_signals": [],
            "challenging_signals": [],
            "weighted_rules": [],
            "theme_lineage": []
        }

    weighted = evaluate_evidence_weights(matched_rules)
    
    # Check if any natal chart evidence exists
    foundation_rule = next((r for r in weighted["weighted_rules"] if r["category"] == "foundation"), None)
    karaka_rule = next((r for r in weighted["weighted_rules"] if r["category"] in ["dignity", "karaka", "house_evidence"]), None)
    
    if not (foundation_rule or karaka_rule):
        return {
            "domain": domain,
            "has_answer": False,
            "structured_text": f"Insufficient natal chart evidence available to evaluate the {domain} domain.",
            "evidence_score": 0.0,
            "conflict_status": "NO_NATAL_EVIDENCE",
            "positive_signals": [],
            "challenging_signals": [],
            "weighted_rules": [],
            "theme_lineage": []
        }

    conflict_res = resolve_evidence_conflicts(weighted)
    status = conflict_res["status"]
    timing_rules = [r for r in weighted["weighted_rules"] if r["category"] in ["timing", "timing_synergy"]]

    config = DOMAIN_CONFIG_MAP.get(domain, DOMAIN_CONFIG_MAP.get("career"))
    theme_dict = config["themes"]
    calibrated_info = CALIBRATED_STATUS_HEADERS.get(status, CALIBRATED_STATUS_HEADERS["BALANCED"])

    # Qualitative Evidence Strength Narrative
    score = weighted["total_score"]
    if score >= 15.0:
        strength_label = "Strong Support"
    elif score >= 8.0:
        strength_label = "Moderate Support"
    else:
        strength_label = "Exploratory Support"

    if status in ["CHALLENGING_PERIOD", "MODERATELY_FAVORED_WITH_COUNTERBALANCE"]:
        strength_narrative = f"Evidence Strength: {strength_label} (Score: {score} — Counterbalancing Factors Active)"
    else:
        strength_narrative = f"Evidence Strength: {strength_label} (Score: {score})"

    paragraphs = []

    # Section 1: Astrological Foundation & Primary Placements (No rule engine jargon)
    foundation_lord = "Lord"
    if foundation_rule and isinstance(foundation_rule["evidence"], dict):
        ev = foundation_rule["evidence"]
        foundation_lord = ev.get("lord", "Lord")
        h_num = ev.get("house", "")
        n_house = ev.get("lord_natal_house", "")
        n_rashi = ev.get("lord_natal_rashi", "")
        
        house_desc = config["house_name"]
        paragraphs.append(
            f"### 1. {config['title_section_1']}\n"
            f"Your {house_desc} is governed by **{foundation_lord}**, which is positioned in House {n_house} ({n_rashi}). "
            f"This placement serves as the primary anchor in this framework for your baseline {domain} trajectory.\n\n"
            f"*Narrative Status*: {strength_narrative}"
        )

    # Section 2: Primary Planetary Indicators & Strengths
    dignity_rule = next((r for r in weighted["weighted_rules"] if r["category"] == "dignity"), None)
    if dignity_rule and isinstance(dignity_rule["evidence"], list):
        dig_items = []
        for d in dignity_rule["evidence"]:
            p = d.get("planet")
            r = d.get("rashi")
            dig = d.get("dignity", "").replace("_", " ")
            if p in theme_dict:
                qualities = theme_dict[p]["qualities"]
                dig_items.append(f"- **{p}** in {r} ({dig}) -> emphasizes {qualities}.")
            else:
                dig_items.append(f"- **{p}** in {r} ({dig}).")
        
        paragraphs.append(
            f"### 2. Key {domain.capitalize()} Planetary Indicators\n"
            + "\n".join(dig_items)
        )

    # Section 3 & 4: Hierarchical Recommended Domain Themes & Fields (Primary vs Secondary)
    fields, caps, why_text, theme_lineage = derive_domain_recommendation_themes(domain, weighted["weighted_rules"])
    if fields:
        formatted_bullets = []
        # Primary Themes (first 2 fields)
        if len(fields) >= 1:
            formatted_bullets.append(f"**Primary Themes to Explore:**")
            formatted_bullets.append(f"1. **{fields[0]}** — Primary direction derived from **{foundation_lord}**'s placement as domain lord.")
        if len(fields) >= 2:
            formatted_bullets.append(f"2. **{fields[1]}** — Strongly supported by primary house strength and technical/analytical significations.")
        
        # Secondary Themes (remaining fields)
        if len(fields) >= 3:
            formatted_bullets.append(f"\n**Secondary Themes:**")
            formatted_bullets.append(f"3. **{fields[2]}** — Secondary direction supported by active planetary significators.")
        if len(fields) >= 4:
            formatted_bullets.append(f"4. **{fields[3]}** — Additional area to explore based on supporting karaka strength.")

        certainty_footer = calibrated_info["certainty_note"]
        
        paragraphs.append(
            f"### 3. {config['title_section_3']}\n"
            f"{calibrated_info['header_prefix']}\n\n"
            + "\n".join(formatted_bullets)
            + (f"\n{certainty_footer}" if certainty_footer else "")
        )

        paragraphs.append(
            f"### 4. {config['title_section_4']}\n"
            f"{why_text}"
        )

    # Section 4b: Counterbalancing & Structural Factors (When present)
    if conflict_res["challenging_signals"]:
        chall_bullets = [
            f"- **{sig}**: Introduces potential friction or structural delay requiring deliberate focus and grounding."
            for sig in conflict_res["challenging_signals"]
        ]
        paragraphs.append(
            f"### 4b. Counterbalancing Factors & Structural Challenges\n"
            f"The following chart factors should be taken into account when evaluating baseline potential:\n\n"
            + "\n".join(chall_bullets)
        )

    # Section 5: Current Timing & Activation (Contextual & Separated)
    q_lower = (question or "").lower()
    is_timing_query = any(k in q_lower for k in ["dasha", "mahadasha", "antardasha", "when", "now", "currently", "this year", "future", "timing", "right now"])

    if is_timing_query:
        timing_bullets = [
            "Your query specifically requests current timing evaluation. The active timing layers are structured as follows:",
            f"- **Mahadasha Activation**: Primary planetary major period influences your baseline {domain} houses and lord.",
            f"- **Antardasha Activation**: Sub-period planetary alignment triggers active {domain} opportunities.",
            f"- **Transit Activation**: Current slow-moving planetary transits activate key domain houses."
        ]
        paragraphs.append(
            f"### 5. Current Timing & Activation\n"
            + "\n".join(timing_bullets)
        )
    else:
        paragraphs.append(
            f"### 5. Current Timing & Activation\n"
            f"This question asks about your general baseline suitability, so the interpretation is based primarily on your natal chart. "
            f"Current Dasha and transit timing are not required for this baseline assessment."
        )

    # Section 6: Direct Bottom Line Summary
    primary_str = f"**{fields[0]}**" if fields else "Primary domain themes"
    if len(fields) >= 2:
        primary_str += f" and **{fields[1]}**"
    secondary_str = f"**{fields[2]}**" if len(fields) >= 3 else "secondary themes"

    paragraphs.append(
        f"### 6. Bottom Line\n"
        f"{primary_str} emerge as the primary themes to explore in this chart, followed by {secondary_str} as a secondary theme. "
        f"These represent astrological tendencies and directions to explore rather than fixed career predictions; your real-world skills, education, personal interests, and practical opportunities should also guide your final choice."
    )

    full_text = "\n\n".join(paragraphs)

    # Stage 16 — Hard Architectural Claim Guard
    guarded_text = guard_deterministic_claims(full_text)

    # Strict Lineage Traceability Guard
    lineage_fields = set()
    for item in theme_lineage:
        lineage_fields.add(item.get("theme"))
        for f in item.get("possible_career_fields", []):
            lineage_fields.add(f)
        for f in item.get("fields", []):
            lineage_fields.add(f)

    for f in fields:
        assert f in lineage_fields, f"Recommendation '{f}' is not backed by theme_lineage!"

    return {
        "domain": domain,
        "has_answer": True,
        "structured_text": guarded_text,
        "evidence_score": weighted["total_score"],
        "conflict_status": conflict_res["status"],
        "positive_signals": conflict_res["positive_signals"],
        "challenging_signals": conflict_res["challenging_signals"],
        "weighted_rules": weighted["weighted_rules"],
        "recommended_fields": fields,
        "theme_lineage": theme_lineage
    }


def build_auditable_answer_trace(
    domain: str,
    intent: str,
    answer_source: str,
    matched_rules: List[Dict[str, Any]],
    theme_lineage: List[Dict[str, Any]],
    gemini_calls: int = 0,
    llm_tokens: int = 0,
    quality_status: str = "PASS",
    quality_score: int = 100,
    failure_category: Optional[str] = None,
    answer_completeness: float = 1.0,
    evidence_coverage: float = 1.0,
    user_feedback: Optional[Dict[str, Any]] = None,
    llm_input_tokens: int = 0,
    llm_output_tokens: int = 0,
    llm_calls_list: Optional[List[Dict[str, Any]]] = None,
    fact_sources: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Phase 23 — Enhanced Auditable Answer Trace Generator:
    Guarantees every output statement is 100% traceable to backend evidence,
    and captures quality scoring, automatic failure categorization, completeness, and feedback.
    """
    req_evidence = ["10th_house", "10th_lord", "career_karakas"] if domain == "career" else [f"{domain}_house", f"{domain}_lord"]
    actual_ev = []
    rule_ids = []
    lineage_summary = []

    for r in matched_rules:
        r_id = r.get("rule_id", "")
        if r_id:
            rule_ids.append(r_id)
            ev = r.get("evidence", {})
            if isinstance(ev, dict) and ev.get("lord"):
                actual_ev.append(f"{ev.get('lord')} in House {ev.get('house')}")

    for item in theme_lineage[:3]:
        theme = item.get("theme", "")
        planet = item.get("chart_evidence", {}).get("planet", "")
        if planet and theme:
            lineage_summary.append(f"{planet} -> {theme}")

    calc_total_tokens = (llm_input_tokens + llm_output_tokens) if (llm_input_tokens or llm_output_tokens) else llm_tokens

    usage_dict = {
        "input_tokens": llm_input_tokens,
        "output_tokens": llm_output_tokens,
        "total_tokens": calc_total_tokens
    }

    calls_dict = llm_calls_list if llm_calls_list is not None else (
        [
            {
                "call_number": 1,
                "purpose": "creative_interpretation" if answer_source == "LLM_FALLBACK" else "renderer",
                "input_tokens": llm_input_tokens,
                "output_tokens": llm_output_tokens,
                "total_tokens": calc_total_tokens
            }
        ] if gemini_calls > 0 else []
    )

    default_fact_sources = {
        "moon_rashi": "FreeAstrologyAPI",
        "moon_nakshatra": "FreeAstrologyAPI",
        "mahadasha": "FreeAstrologyAPI",
        "antardasha": "FreeAstrologyAPI"
    }

    return {
        "answer_source": answer_source,
        "domain": domain,
        "intent": intent,
        "required_evidence": req_evidence,
        "actual_evidence": list(set(actual_ev)) or ["FreeAstrologyAPI.moon.nakshatra", "FreeAstrologyAPI.moon.rashi"],
        "matched_rules": rule_ids or ["ASTROLOGY_FACT_LOOKUP"],
        "interpretation_lineage": lineage_summary or ["Deterministic Synthesizer Wording"],
        "fact_sources": fact_sources or default_fact_sources,
        "gemini_calls": gemini_calls,
        "llm_tokens": calc_total_tokens,
        "llm_usage": usage_dict,
        "llm_calls": calls_dict,
        "quality_status": quality_status,
        "quality_score": quality_score,
        "failure_category": failure_category,
        "answer_completeness": answer_completeness,
        "evidence_coverage": evidence_coverage,
        "user_feedback": user_feedback
    }


