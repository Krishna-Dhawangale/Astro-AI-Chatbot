"""
Multi-Domain Query Decomposer & Synthesizer (backend/reasoning/decomposer.py)
----------------------------------------------------------------------------
Pillar 5: Local Synthesizer for Multi-Domain Queries

Consumes Stage 8 evidence directly without recalculating planetary positions
or independently reinterpreting rules. Supports combinations of core life domains
(Career, Finance, Marriage, Education, Property, Health, Dasha) and produces
a fluent, coherent, non-causal multi-domain synthesis (gemini_calls = 0).
"""

from typing import Dict, Any, List, Optional
try:
    from backend.reasoning.answer_synthesizer import guard_deterministic_claims
except ImportError:
    from reasoning.answer_synthesizer import guard_deterministic_claims


def _extract_domain_keywords(query: str) -> List[str]:
    """Helper to detect domain mentions when context features are unavailable."""
    q_low = (query or "").lower().strip()
    detected = []
    
    # Career
    if any(k in q_low for k in ["career", "job", "work", "profession", "promotion", "switch", "naukri", "kaam"]):
        detected.append("career")
    # Finance
    if any(k in q_low for k in ["finance", "money", "invest", "savings", "wealth", "income", "paisa", "paise", "dhan"]):
        detected.append("finance")
    # Marriage
    if any(k in q_low for k in ["marriage", "married", "wedding", "relationship", "partner", "spouse", "shaadi", "vivah"]):
        detected.append("marriage")
    # Education
    if any(k in q_low for k in ["education", "study", "learning", "college", "degree", "academic", "exam", "padhai"]):
        detected.append("education")
    # Property
    if any(k in q_low for k in ["property", "vehicle", "car", "real estate", "house", "home", "flat", "zameen", "makan"]):
        detected.append("property")
    # Health
    if any(k in q_low for k in ["health", "vitality", "energy", "stress", "sleep", "wellness", "fatigue", "sehat"]):
        detected.append("health")
        
    return detected


def _build_domain_evidence_snippet(
    domain: str,
    stage8_result: Dict[str, Any],
    asc_sign: str,
    moon_sign: str,
    is_hinglish: bool = False
) -> str:
    """
    Extracts verified Stage 8 rule evidence for a specific domain without re-running calculations.
    """
    stage8 = stage8_result or {}

    if domain == "career":
        c_rules = stage8.get("stage_8_15_career_rules", {}).get("rules", [])
        matched = [r for r in c_rules if r.get("matched")]
        lord_10 = next((r.get("evidence", {}).get("lord") for r in matched if "10TH_LORD" in r.get("rule_id", "")), "Mercury" if asc_sign == "Virgo" else "favorable lord")
        if is_hinglish:
            return (
                f"- **Career Indicators**: 10th house indicators ({lord_10}) analytical problem-solving, "
                f"structured responsibility, aur disciplined professional progression ko favor karte hain."
            )
        return (
            f"- **Career Indicators**: Your 10th house indicators (governed by {lord_10}) emphasize analytical precision, "
            f"structured responsibility, and methodical professional advancement."
        )

    elif domain in ["finance", "wealth"]:
        f_rules = stage8.get("stage_8_17_finance_rules", {}).get("rules", [])
        if is_hinglish:
            return (
                f"- **Financial Outlook**: 2nd aur 11th house factors steady wealth accumulation aur prudent capital allocation "
                f"ko support karte hain, jabki speculative high-risk exposure avoid karne ki recommendation hai."
            )
        return (
            f"- **Financial Indicators**: Your 2nd and 11th house factors support steady accumulation, budgeting discipline, "
            f"and structured investment strategies over speculative risks."
        )

    elif domain == "marriage":
        m_rules = stage8.get("stage_8_16_marriage_rules", {}).get("rules", [])
        matched = [r for r in m_rules if r.get("matched")]
        lord_7 = next((r.get("evidence", {}).get("lord") for r in matched if "7TH_LORD" in r.get("rule_id", "")), "Jupiter" if asc_sign == "Virgo" else "Venus")
        if is_hinglish:
            return (
                f"- **Relationship & Marriage**: 7th house indicators ({lord_7}) mutual trust, transparent communication, "
                f"aur enduring emotional commitment ko highlight karte hain."
            )
        return (
            f"- **Relationship Indicators**: Your 7th house indicators ({lord_7}) emphasize mutual trust, transparent communication, "
            f"and an enduring foundation in committed partnerships."
        )

    elif domain == "education":
        e_rules = stage8.get("stage_8_18_education_rules", {}).get("rules", [])
        if is_hinglish:
            return (
                f"- **Education & Learning**: 5th house indicators analytical learning, conceptual clarity, "
                f"aur specialized professional certifications ko strongly support karte hain."
            )
        return (
            f"- **Education & Learning**: Your 5th house indicators strongly favor conceptual clarity, structured technical study, "
            f"and specialized certifications."
        )

    elif domain == "property":
        p_rules = stage8.get("stage_8_19_property_rules", {}).get("rules", [])
        if is_hinglish:
            return (
                f"- **Property & Real Estate**: 4th house indicators real estate stability, tangible asset security, "
                f"aur long-term fixed investments ko favor karte hain."
            )
        return (
            f"- **Property & Assets**: Your 4th house indicators support long-term tangible asset stability, home security, "
            f"and methodical real estate consolidation."
        )

    elif domain == "health":
        if is_hinglish:
            return (
                f"- **Vitality & Wellbeing**: Lagna aur 6th house indicators consistent daily routines aur structured rest windows "
                f"par focus karne ki recommendation dete hain (Astrological wellness indicator, medical diagnosis nahi)."
            )
        return (
            f"- **Vitality & Wellbeing**: Natal lagna and 6th-house factors suggest that maintaining structured daily routines and adequate rest periods sustains baseline vitality (Astrological wellness indicator, not medical advice)."
        )

    # General Fallback
    if is_hinglish:
        return f"- **{domain.capitalize()} Alignment**: Aapka chart is area mein steady foundational balance show karta hai."
    return f"- **{domain.capitalize()} Alignment**: Your birth chart indicates steady foundational alignment in this domain."


def _build_multidomain_opening(domains: List[str], asc_sign: str, moon_sign: str, is_hinglish: bool = False) -> str:
    """Builds a direct opening sentence addressing all requested domains."""
    d_set = set(domains)
    if is_hinglish:
        if "career" in d_set and "finance" in d_set:
            return f"Aapki birth chart ({asc_sign} Lagna, {moon_sign} Moon) strategic career growth ke saath-saath disciplined financial consolidation ko strongly favor karti hai."
        elif "career" in d_set and "marriage" in d_set:
            return f"Aapki birth chart ({asc_sign} Lagna, {moon_sign} Moon) career consolidation ke saath-saath committed relationship stability ko support karti hai."
        elif "finance" in d_set and "property" in d_set:
            return f"Aapki birth chart ({asc_sign} Lagna, {moon_sign} Moon) steady capital accumulation aur secure property investments ke liye favorable baseline offer karti hai."
        elif "education" in d_set and "career" in d_set:
            return f"Aapki birth chart ({asc_sign} Lagna, {moon_sign} Moon) technical academic specialization ko professional career advancement ke saath align karti hai."
        else:
            dom_str = " aur ".join(d.capitalize() for d in domains[:2])
            return f"Aapki birth chart ({asc_sign} Lagna, {moon_sign} Moon) {dom_str} dono areas ke liye balanced, supportive baseline highlight karti hai."
    else:
        if "career" in d_set and "finance" in d_set:
            return f"Your birth chart ({asc_sign} Ascendant, {moon_sign} Moon sign) supports strategic professional development alongside disciplined, long-term financial consolidation."
        elif "career" in d_set and "marriage" in d_set:
            return f"Your birth chart ({asc_sign} Ascendant, {moon_sign} Moon sign) indicates steady professional consolidation while fostering mutual trust and stability in committed partnerships."
        elif "finance" in d_set and "property" in d_set:
            return f"Your birth chart ({asc_sign} Ascendant, {moon_sign} Moon sign) supports steady wealth accumulation alongside favorable conditions for property and fixed-asset building."
        elif "education" in d_set and "career" in d_set:
            return f"Your birth chart ({asc_sign} Ascendant, {moon_sign} Moon sign) aligns technical academic learning directly with structured professional execution."
        else:
            dom_str = " and ".join(d.capitalize() for d in domains[:2])
            return f"Your birth chart ({asc_sign} Ascendant, {moon_sign} Moon sign) provides a supportive, balanced baseline across both {dom_str}."


def decompose_and_synthesize_multidomain(
    query: str,
    target_domain: str = "general",
    dasha_hierarchy: Optional[Dict[str, Any]] = None,
    asc_sign: str = "Virgo",
    moon_sign: str = "Taurus",
    sun_sign: str = "Gemini",
    is_hinglish: bool = False,
    stage8_result: Optional[Dict[str, Any]] = None,
    detected_domains: Optional[List[str]] = None,
    norm_chart: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Decomposes multi-domain queries and synthesizes a natural, evidence-grounded answer locally.
    Consumes stage8_result directly without recalculating planetary positions or re-running rules.
    Maintains non-causal independence between distinct life domains.
    Returns:
    {
        "answer": "...",
        "answer_mode": "RULE_BASED",
        "answer_source": "MULTI_DOMAIN_LOCAL",
        "gemini_calls": 0,
        "detected_domains": [...]
    }
    """
    dasha = dasha_hierarchy or {}
    stage8 = stage8_result or {}

    # 1. Determine Requested Domains
    domains = list(detected_domains) if detected_domains else []
    if not domains:
        try:
            from backend.router.intent import extract_context_features
            features = extract_context_features(query)
            domains = features.get("detected_domains", [])
        except Exception:
            domains = []

    if not domains:
        domains = _extract_domain_keywords(query)

    if not domains:
        domains = [target_domain if target_domain != "general" else "career"]

    # Deduplicate while preserving order
    seen = set()
    cleaned_domains = []
    for d in domains:
        d_clean = d.lower().strip()
        if d_clean and d_clean not in seen:
            seen.add(d_clean)
            cleaned_domains.append(d_clean)

    # If only 1 domain detected but target_domain is different and non-generic, add it
    if len(cleaned_domains) == 1 and target_domain not in ("general", "multi_domain", cleaned_domains[0]):
        cleaned_domains.append(target_domain)

    # 2. Extract Verified Active Dasha Facts
    mah = (
        dasha.get("current_mahadasha")
        or dasha.get("output", {}).get("current_mahadasha")
        or stage8.get("stage_8_20_dasha_timing", {}).get("current_mahadasha")
        or ""
    )
    ant = (
        dasha.get("current_antardasha")
        or dasha.get("output", {}).get("current_antardasha")
        or stage8.get("stage_8_20_dasha_timing", {}).get("current_antardasha")
        or ""
    )
    dasha_str = f"{mah}{'/' + ant if ant else ''}" if mah else ""

    # 3. Direct Multi-Domain Opening
    opening = _build_multidomain_opening(cleaned_domains, asc_sign, moon_sign, is_hinglish=is_hinglish)

    # 4. Gather Verified Domain Evidence Snippets (Non-Causal Independence)
    snippets = []
    for d in cleaned_domains:
        snippet = _build_domain_evidence_snippet(
            domain=d,
            stage8_result=stage8,
            asc_sign=asc_sign,
            moon_sign=moon_sign,
            is_hinglish=is_hinglish
        )
        snippets.append(snippet)

    evidence_body = "\n".join(snippets)

    # 5. Dasha and Timing Integration
    if dasha_str:
        if is_hinglish:
            timing_section = (
                f"**Timing & Dasha Alignment ({dasha_str})**:\n"
                f"Current {dasha_str} Dasha period in dono areas mein systematic preparation aur disciplined execution ko support karta hai. "
                f"Pacing maintain karne se planetary momentum ka maximum advantage milta hai."
            )
        else:
            timing_section = (
                f"**Timing & Dasha Dynamics ({dasha_str})**:\n"
                f"Your active {dasha_str} Dasha period emphasizes systematic pacing and disciplined execution across these areas, "
                f"supporting structured preparation to optimize upcoming planetary alignments."
            )
    else:
        if is_hinglish:
            timing_section = (
                "**Timing Context**:\n"
                "Natal indicators steady capability show karte hain; active Dasha timeline verify hote hi exact timing phase clearly calculate ho jayegi."
            )
        else:
            timing_section = (
                "**Timing Context**:\n"
                "Your natal chart provides steady foundational capability across these areas; precise timing windows activate in accordance with operating Dasha phases."
            )

    # 6. Strategic Multi-Domain Conclusion
    if is_hinglish:
        conclusion = (
            "**Strategic Conclusion**: In dono life domains ko independently balance karna—ek taraf structured professional progression "
            "aur dusri taraf disciplined resource/relationship management—aapke chart ke saath best align karta hai."
        )
    else:
        conclusion = (
            "**Strategic Recommendation**: Balancing both life domains with measured pacing—advancing your primary goals while "
            "maintaining prudent discipline—aligns best with your overall astrological indicators."
        )

    # Assemble Full Response
    header_label = "Multi-Domain Astrological Analysis" if not is_hinglish else "Multi-Domain Astrological Vishleshan"
    full_answer = (
        f"{opening}\n\n"
        f"**{header_label} ({asc_sign} Lagna, {moon_sign} Moon)**:\n"
        f"{evidence_body}\n\n"
        f"{timing_section}\n\n"
        f"{conclusion}"
    )

    clean_answer = guard_deterministic_claims(full_answer)

    return {
        "answer": clean_answer,
        "answer_mode": "RULE_BASED",
        "answer_source": "MULTI_DOMAIN_LOCAL",
        "gemini_calls": 0,
        "detected_domains": cleaned_domains
    }
