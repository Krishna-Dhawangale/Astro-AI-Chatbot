"""
Quick validation: tests that resolve_domain_locked_followup correctly handles
short keys, conversational prefixes, typos, and hinglish queries.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from response_builder import (
    build_user_chart_context,
    resolve_domain_locked_followup,
    normalize_qa_concept,
    load_qa_database_dict,
)

# Fake user chart context (Leo Lagna, Aquarius Moon, Rohini Nakshatra)
ctx = build_user_chart_context(
    user_name="Rahul",
    lagna="Leo",
    moon_sign="Aquarius",
    sun_sign="Leo",
    current_dasha="Jupiter Mahadasha and Saturn Antardasha",
    nakshatra="Rohini",
    nakshatra_pada="Pada 2",
    nakshatra_lord="Moon",
)

db = load_qa_database_dict()

test_queries = [
    "for my nakshtra",
    "what is my nakshatra",
    "my nakshatra",
    "mera nakshatra kya hai",
    "for my nakshatra",
    "mera moon sign",
    "what is my lagna",
    "my lagna",
    "meri dasha",
    "what is my 10th house",
    "for my 7th house",
    "my dasha",
    "nakshatra",
    "moon sign",
    "lagna",
    "10th house",
]

print("=" * 70)
print(f"{'QUERY':<40} {'CONCEPT':<20} MATCH?")
print("=" * 70)

for q in test_queries:
    concept = normalize_qa_concept(q)
    result = resolve_domain_locked_followup(q, ctx, qa_db=db, is_first_turn=False)
    if result:
        ans, followups, domain = result
        # Show first 80 chars of answer
        preview = ans[:80].replace("\n", " ")
        print(f"[OK] {q:<40} concept='{concept}', domain='{domain}'")
        print(f"    Preview: {preview}...")
    else:
        print(f"[--] {q:<40} concept='{concept}' -> NO MATCH")
    print()
