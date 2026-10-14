"""
tests/test_phase1_semantic_routing.py
Phase 1 Semantic Routing & Non-Destructive Context Feature Test Suite
=====================================================================
Covers:
1. Five Documented Routing Failures (Business Partner, Job Change + Marriage, Switch Jobs + Invest Savings,
   Career Field vs Car, Rahu-Saturn Career + Finance).
2. Polysemy Disambiguation (partner, invest, switch, property).
3. Explicit User-Stated Temporal Horizons (extracted intent metadata without astrological validity claims).
4. Negation Detection & Suppression.
5. Multi-part and Comparison Query Attributes.
6. Hinglish Paraphrases.
7. Single-Domain Controls (Ensures pure single-domain queries are not erroneously marked multi-domain).
"""

import unittest
from backend.router.intent import (
    extract_context_features,
    predict_intent_with_confidence,
    predict_intent
)
from backend.qa_router import (
    classify_multi_domain_micro_intents,
    MICRO_INTENT_REGISTRY
)


class TestPhase1FiveDocumentedFailures(unittest.TestCase):
    """
    Validates that all five historically documented routing failures are permanently resolved.
    """

    def test_failure_1_business_partner(self):
        """
        Failure 1: "Should I take a business partner for my startup?"
        Expected: Disambiguated to business/career, NOT marriage.
        """
        query = "Should I take a business partner for my startup?"
        features = extract_context_features(query)
        self.assertEqual(features["disambiguated_terms"].get("partner"), "business")
        self.assertIn("career_signal", features["active_domain_signals"])
        self.assertNotIn("marriage_signal", features["active_domain_signals"])

        micro_intents = classify_multi_domain_micro_intents(query)
        self.assertIn("career_domain_fit", micro_intents)
        self.assertNotIn("marriage_compatibility", micro_intents)
        self.assertNotIn("marriage_timing", micro_intents)

    def test_failure_2_job_change_and_marriage(self):
        """
        Failure 2: "Can I focus on a job change while planning my marriage?"
        Expected: Both Career and Marriage domains detected.
        """
        query = "Can I focus on a job change while planning my marriage?"
        features = extract_context_features(query)
        self.assertIn("career_signal", features["active_domain_signals"])
        self.assertIn("marriage_signal", features["active_domain_signals"])
        self.assertTrue(features["is_multi_part"])

        intent_res = predict_intent_with_confidence(query)
        self.assertEqual(intent_res["resolved_intent"], "multi_domain")

        micro_intents = classify_multi_domain_micro_intents(query)
        domains = {MICRO_INTENT_REGISTRY[it].domain for it in micro_intents if it in MICRO_INTENT_REGISTRY}
        self.assertIn("career", domains, f"Career domain missing from micro_intents: {micro_intents}")
        self.assertIn("marriage", domains, f"Marriage domain missing from micro_intents: {micro_intents}")

    def test_failure_3_switch_jobs_and_invest_savings(self):
        """
        Failure 3: "Should I switch jobs and invest my savings?"
        Expected: Both Career and Finance domains detected.
        """
        query = "Should I switch jobs and invest my savings?"
        features = extract_context_features(query)
        self.assertIn("career_signal", features["active_domain_signals"])
        self.assertIn("finance_signal", features["active_domain_signals"])
        self.assertEqual(features["disambiguated_terms"].get("switch"), "career_job_change")
        self.assertEqual(features["disambiguated_terms"].get("investment"), "finance")

        intent_res = predict_intent_with_confidence(query)
        self.assertEqual(intent_res["resolved_intent"], "multi_domain")

        micro_intents = classify_multi_domain_micro_intents(query)
        domains = {MICRO_INTENT_REGISTRY[it].domain for it in micro_intents if it in MICRO_INTENT_REGISTRY}
        self.assertIn("career", domains, f"Career missing from: {micro_intents}")
        self.assertIn("finance", domains, f"Finance missing from: {micro_intents}")

    def test_failure_4_career_field_not_car(self):
        """
        Failure 4: "What career field is best for my chart?"
        Expected: career_domain_fit, NOT property_vehicle_timing (substring 'car' in 'career').
        """
        query = "What career field is best for my chart?"
        features = extract_context_features(query)
        self.assertIn("career_signal", features["active_domain_signals"])
        self.assertNotIn("property_signal", features["active_domain_signals"])

        micro_intents = classify_multi_domain_micro_intents(query)
        self.assertIn("career_domain_fit", micro_intents)
        self.assertNotIn("property_vehicle_timing", micro_intents)

    def test_failure_5_dasha_career_stabilize_invest(self):
        """
        Failure 5: "Will my career stabilize and is it safe to invest under Rahu-Saturn dasha?"
        Expected: Dasha (Rahu-Saturn) + Career + Finance all detected.
        """
        query = "Will my career stabilize and is it safe to invest under Rahu-Saturn dasha?"
        features = extract_context_features(query)
        self.assertTrue(features["has_dasha_signal"])
        self.assertIn("career_signal", features["active_domain_signals"])
        self.assertIn("finance_signal", features["active_domain_signals"])

        micro_intents = classify_multi_domain_micro_intents(query)
        self.assertIn("dasha_impact_rahu_saturn", micro_intents)
        domains = {MICRO_INTENT_REGISTRY[it].domain for it in micro_intents if it in MICRO_INTENT_REGISTRY}
        self.assertIn("career", domains, f"Career missing in: {micro_intents}")
        self.assertIn("finance", domains, f"Finance missing in: {micro_intents}")


class TestPolysemyDisambiguation(unittest.TestCase):
    """
    Validates non-destructive disambiguation of polysemous words:
    - partner (business vs life/marriage)
    - investment (real estate vs business vs financial savings)
    - switch (job change vs team change vs relocation)
    - property (real estate vs astrological attributes)
    """

    def test_partner_polysemy(self):
        biz = extract_context_features("Should I choose a partner for my tech firm?")
        self.assertEqual(biz["disambiguated_terms"].get("partner"), "business")
        self.assertIn("career_signal", biz["active_domain_signals"])
        self.assertNotIn("marriage_signal", biz["active_domain_signals"])

        life = extract_context_features("When will I meet my life partner?")
        self.assertEqual(life["disambiguated_terms"].get("partner"), "marriage")
        self.assertIn("marriage_signal", life["active_domain_signals"])

    def test_investment_polysemy(self):
        prop_inv = extract_context_features("Should I invest in property this year?")
        self.assertEqual(prop_inv["disambiguated_terms"].get("investment"), "property")
        self.assertIn("property_signal", prop_inv["active_domain_signals"])

        biz_inv = extract_context_features("Should I invest in a startup venture?")
        self.assertEqual(biz_inv["disambiguated_terms"].get("investment"), "career")
        self.assertIn("career_signal", biz_inv["active_domain_signals"])

        fin_inv = extract_context_features("Is it safe to invest in mutual funds and shares?")
        self.assertEqual(fin_inv["disambiguated_terms"].get("investment"), "finance")
        self.assertIn("finance_signal", fin_inv["active_domain_signals"])

    def test_switch_polysemy(self):
        job_sw = extract_context_features("Can I switch jobs in the next 6 months?")
        self.assertEqual(job_sw["disambiguated_terms"].get("switch"), "career_job_change")
        self.assertIn("career_signal", job_sw["active_domain_signals"])

        team_sw = extract_context_features("Can I switch teams at work?")
        self.assertEqual(team_sw["disambiguated_terms"].get("switch"), "career_team_change")
        self.assertIn("career_signal", team_sw["active_domain_signals"])

        reloc_sw = extract_context_features("Should I switch cities next year?")
        self.assertEqual(reloc_sw["disambiguated_terms"].get("switch"), "relocation")
        self.assertIn("travel_signal", reloc_sw["active_domain_signals"])

    def test_property_polysemy(self):
        real_estate = extract_context_features("When is the right time to buy a property?")
        self.assertEqual(real_estate["disambiguated_terms"].get("property"), "real_estate")
        self.assertIn("property_signal", real_estate["active_domain_signals"])

        astro_attr = extract_context_features("What are the astrological properties of Saturn?")
        self.assertEqual(astro_attr["disambiguated_terms"].get("property"), "astrological_attribute")
        self.assertNotIn("property_signal", astro_attr["active_domain_signals"])
        # Micro intent should NOT be property_vehicle_timing
        micro_intents = classify_multi_domain_micro_intents("What are the astrological properties of Saturn?")
        self.assertNotIn("property_vehicle_timing", micro_intents)


class TestExplicitUserHorizons(unittest.TestCase):
    """
    Validates explicit user-stated time horizons:
    Only captured when the user explicitly provides them; no astrological claims made.
    """

    def test_explicit_horizon_months(self):
        feat = extract_context_features("Will I get a promotion in the next 6 months?")
        self.assertIsNotNone(feat["explicit_user_horizon"])
        self.assertIn("6 months", feat["explicit_user_horizon"])

    def test_explicit_horizon_this_year(self):
        feat = extract_context_features("Can I change my job this year?")
        self.assertEqual(feat["explicit_user_horizon"], "this year")

    def test_no_explicit_horizon(self):
        feat = extract_context_features("Will my career improve?")
        self.assertIsNone(feat["explicit_user_horizon"])


class TestNegationHandling(unittest.TestCase):
    """
    Validates that negated domains are pruned from active domain signals
    and micro-intents.
    """

    def test_negate_marriage_ask_career(self):
        query = "I am not interested in marriage, tell me about my career growth"
        features = extract_context_features(query)
        self.assertIn("marriage", features["negated_domains"])
        self.assertNotIn("marriage_signal", features["active_domain_signals"])
        self.assertIn("career_signal", features["active_domain_signals"])

        micro_intents = classify_multi_domain_micro_intents(query)
        domains = {MICRO_INTENT_REGISTRY[it].domain for it in micro_intents if it in MICRO_INTENT_REGISTRY}
        self.assertNotIn("marriage", domains)
        self.assertIn("career", domains)

    def test_negate_job_ask_property(self):
        query = "Don't tell me about jobs, when will I buy a car?"
        features = extract_context_features(query)
        self.assertIn("career", features["negated_domains"])
        self.assertNotIn("career_signal", features["active_domain_signals"])
        self.assertIn("property_signal", features["active_domain_signals"])

        micro_intents = classify_multi_domain_micro_intents(query)
        domains = {MICRO_INTENT_REGISTRY[it].domain for it in micro_intents if it in MICRO_INTENT_REGISTRY}
        self.assertNotIn("career", domains)
        self.assertIn("property", domains)


class TestHinglishParaphrases(unittest.TestCase):
    """
    Validates accurate routing on common Hinglish phrasing.
    """

    def test_hinglish_career_and_business_invest(self):
        query = "Mera career kab stable hoga aur kya business mein invest karu?"
        micro_intents = classify_multi_domain_micro_intents(query)
        domains = {MICRO_INTENT_REGISTRY[it].domain for it in micro_intents if it in MICRO_INTENT_REGISTRY}
        self.assertIn("career", domains)
        self.assertIn("career_timing", micro_intents)

    def test_hinglish_trapped_money(self):
        query = "Paisa fas gaya hai nikalne ke yog kab hain?"
        micro_intents = classify_multi_domain_micro_intents(query)
        self.assertIn("finance_timing", micro_intents)

    def test_hinglish_shaadi_timing(self):
        query = "Shaadi ke chances kab tak ban rahe hain?"
        micro_intents = classify_multi_domain_micro_intents(query)
        self.assertIn("marriage_timing", micro_intents)

    def test_hinglish_naukri_ya_business(self):
        query = "Naukri badlu ya business shuru karu?"
        micro_intents = classify_multi_domain_micro_intents(query)
        self.assertIn("career_domain_fit", micro_intents)


class TestSingleDomainControls(unittest.TestCase):
    """
    Ensures single domain queries are not accidentally flagged as multi-domain.
    """

    def test_pure_career_query(self):
        query = "When will my career stabilize?"
        features = extract_context_features(query)
        self.assertEqual(len(features["active_domain_signals"]), 1)
        self.assertEqual(features["active_domain_signals"][0], "career_signal")
        self.assertFalse(features["is_multi_part"])

    def test_pure_marriage_query(self):
        query = "When will I get married?"
        features = extract_context_features(query)
        self.assertEqual(len(features["active_domain_signals"]), 1)
        self.assertEqual(features["active_domain_signals"][0], "marriage_signal")
        self.assertFalse(features["is_multi_part"])

    def test_pure_finance_query(self):
        query = "What is my financial stability?"
        features = extract_context_features(query)
        self.assertEqual(len(features["active_domain_signals"]), 1)
        self.assertEqual(features["active_domain_signals"][0], "finance_signal")
        self.assertFalse(features["is_multi_part"])

    def test_pure_travel_query(self):
        query = "When will I travel abroad?"
        features = extract_context_features(query)
        self.assertEqual(len(features["active_domain_signals"]), 1)
        self.assertEqual(features["active_domain_signals"][0], "travel_signal")
        self.assertFalse(features["is_multi_part"])


if __name__ == "__main__":
    unittest.main()
