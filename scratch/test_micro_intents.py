import sys
sys.path.insert(0, 'backend')

from qa_router import classify_micro_intent, resolve_micro_intent_local_template
from response_builder import build_user_chart_context

test_queries = [
    # Career
    ("kab milegi job", "career_timing", "career"),
    ("promotion kab hoga", "career_timing", "career"),
    ("when will I change jobs", "career_timing", "career"),
    ("job milegi ya nahi", "career_probability", "career"),
    ("career mein success ke asar", "career_probability", "career"),
    ("business karu ya job", "career_domain_fit", "career"),
    ("which industry suits me", "career_domain_fit", "career"),

    # Finance
    ("paisa kab aayega", "finance_timing", "finance"),
    ("financial stability kab hogi", "finance_timing", "finance"),
    ("loan lena chahiye kya", "finance_risk", "finance"),
    ("stock market mein loss hoga kya", "finance_risk", "finance"),
    ("wealth prospects", "wealth_accumulation", "finance"),
    ("savings kaise badhaye", "wealth_accumulation", "finance"),

    # Health
    ("health kab theek hogi", "health_timing", "health"),
    ("recovery period", "health_timing", "health"),
    ("konsa roag ya stress aane wala hai", "health_mitigation", "health"),
    ("remedies for mental peace", "health_mitigation", "health"),

    # Marriage
    ("meri shadi kab hogi", "marriage_timing", "marriage"),
    ("mere shadi hone ke asar hai ya nhi", "marriage_probability", "marriage"),
    ("mai shadi karu ya nhi", "marriage_decision", "marriage"),
    ("shaadi kab hogi", "marriage_timing", "marriage"),
    ("life partner kaisa hoga", "partner_profile", "marriage"),
    ("love or arranged marriage", "relationship_compatibility", "marriage"),
    ("shaadi mein delay ke upay", "relationship_mitigation", "marriage"),

    # Education
    ("which field of higher education", "education_field_fit", "education"),
    ("competitive exam clear hoga kya", "exam_timing_success", "education"),
    ("studying abroad", "study_abroad", "education"),

    # Travel
    ("when will I travel abroad", "foreign_travel_timing", "travel"),
    ("permanent foreign settlement", "foreign_settlement_fit", "travel"),
    ("visa problem remedies", "travel_obstacle_mitigation", "travel"),

    # Spirituality
    ("who is my ishta devata", "ishta_devata_guidance", "spirituality"),
    ("pitra dosha remedies", "karmic_dosha_mitigation", "spirituality"),
    ("spiritual growth path", "spiritual_growth_dharma", "spirituality"),

    # Gemstones
    ("kaunsa ratna pahanu", "gemstone_recommendation", "gemstones"),
    ("gemstone wearing rules", "gemstone_wearing_rules", "gemstones"),
]

print("=" * 60)
print("TESTING MICRO-INTENT CLASSIFICATION")
print("=" * 60)

passed = 0
for q, expected_intent, expected_domain in test_queries:
    res = classify_micro_intent(q)
    intent = res["micro_intent"]
    domain = res["domain"]
    is_ok = (intent == expected_intent and domain == expected_domain)
    status = "PASS" if is_ok else "FAIL"
    if is_ok:
        passed += 1
    print(f"[{status}] '{q}' -> Intent: {intent} (expected: {expected_intent}) | Domain: {domain} | Conf: {res['confidence']}")

print(f"\nResult: {passed}/{len(test_queries)} passed.")

# Test Ambiguity Fallback Safety
print("\n" + "=" * 60)
print("TESTING AMBIGUITY DETECTION (FALLBACK SAFETY)")
print("=" * 60)

ambiguous_queries = [
    "paisa kab aayega aur share market mein invest karu ya nahi",
    "job milegi ya nahi aur promotion kab hoga",
]

for aq in ambiguous_queries:
    res = classify_micro_intent(aq)
    print(f"Query: '{aq}'")
    print(f"  is_ambiguous: {res['is_ambiguous']}")
    print(f"  candidate_intents: {res['candidate_intents']}")
    print(f"  routing_action: {res['routing_action']}")
    print(f"  scored_candidates: {res.get('scored_candidates')}\n")

# Test Local QA Template Resolution & Hyper-Personalization
print("=" * 60)
print("TESTING DYNAMIC CHART INJECTION FOR MICRO-INTENTS")
print("=" * 60)

chart_ctx = build_user_chart_context(
    user_name="Vikram",
    lagna="Scorpio",
    moon_sign="Cancer",
    sun_sign="Aries",
    current_dasha="Jupiter Mahadasha and Saturn Antardasha",
    astro_features=[7, 0, 3, 2, 8, 0, 7, 0, 3, 9]
)

test_resolve_q = "kab milegi job"
ans_tuple = resolve_micro_intent_local_template(test_resolve_q, chart_ctx)
assert ans_tuple is not None, "Failed to resolve template"
ans_text, f_ups, dom = ans_tuple

print(f"Query: '{test_resolve_q}'")
print(f"Resolved Answer:\n{ans_text}\n")
print(f"Follow-ups: {f_ups}")
print(f"Domain: {dom}")

assert "Vikram" in ans_text, "Missing user_name"
assert "Scorpio" in ans_text, "Missing lagna"
assert "Jupiter Mahadasha" in ans_text, "Missing current_dasha"
assert "<ElicitationsGroup>" not in ans_text, "Should not contain ElicitationsGroup XML"
print("\n>>> ALL TESTS COMPLETED!")
