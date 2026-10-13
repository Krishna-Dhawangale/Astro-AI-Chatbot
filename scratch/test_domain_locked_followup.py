import asyncio
import os
import sys
import httpx
from dotenv import load_dotenv

backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

load_dotenv(os.path.join(backend_dir, ".env"))

from response_builder import (
    DOMAIN_LOCKED_FOLLOWUPS,
    detect_query_primary_domain,
    get_domain_locked_followups,
    render_elicitations_group,
    build_user_chart_context,
    resolve_domain_locked_followup,
    attach_domain_locked_elicitations,
)


def test_domain_mapping():
    print("=" * 60)
    print("TEST 1: Domain-Locked Follow-Up Mapping Verification")
    print("=" * 60)
    for domain in ["career", "marriage", "finance", "health", "gemstones", "education", "travel", "spirituality"]:
        followups = get_domain_locked_followups(domain)
        print(f"Domain: [{domain.upper()}] ({len(followups)} questions)")
        for i, q in enumerate(followups, 1):
            print(f"  {i}. {q}")
        assert 2 <= len(followups) <= 3, f"Expected 2-3 follow-ups, got {len(followups)}"
        assert len(followups) == len(set(followups)), "Duplicate follow-ups detected"

    # Specific assertions for requested domain questions
    career_q = get_domain_locked_followups("career")
    assert "When is the right time for a job change or promotion?" in career_q
    assert "Which specific work environment or industry suits my Lagna lord?" in career_q
    assert "Which gemstone strengthens my 10th house career lord?" in career_q

    marriage_q = get_domain_locked_followups("marriage")
    assert "What are the personality traits of my future life partner?" in marriage_q
    assert "When will my 7th house marriage activation period begin?" in marriage_q
    assert "Which astrological remedies can resolve relationship delays?" in marriage_q

    finance_q = get_domain_locked_followups("finance")
    assert "What are my long-term wealth accumulation prospects in my 2nd and 11th house?" in finance_q
    assert "Which planetary period indicates high financial gains?" in finance_q
    assert "What remedies should I perform to prevent financial losses?" in finance_q

    print(">>> TEST 1 PASSED: Strict Domain Mapping Verified!\n")


def test_hyper_personalized_local_resolution():
    print("=" * 60)
    print("TEST 2: Dynamic QA Database Resolution (User A vs User B)")
    print("=" * 60)
    # User A: Leo Lagna, Arjun, Jupiter Mahadasha
    ctx_a = build_user_chart_context(
        user_name="Arjun",
        lagna="Leo",
        moon_sign="Aquarius",
        sun_sign="Aries",
        current_dasha="Jupiter Mahadasha and Saturn Antardasha",
    )

    # User B: Aries Lagna, Priya, Venus Mahadasha
    ctx_b = build_user_chart_context(
        user_name="Priya",
        lagna="Aries",
        moon_sign="Cancer",
        sun_sign="Scorpio",
        current_dasha="Venus Mahadasha and Mars Antardasha",
    )

    test_followup = "When is the right time for a job change or promotion?"

    res_a = resolve_domain_locked_followup(test_followup, ctx_a)
    assert res_a is not None, "Failed to resolve follow-up for User A"
    ans_a, next_a, dom_a = res_a

    res_b = resolve_domain_locked_followup(test_followup, ctx_b)
    assert res_b is not None, "Failed to resolve follow-up for User B"
    ans_b, next_b, dom_b = res_b

    print("--- User A (Arjun, Leo Lagna) Response ---")
    print(ans_a)
    print("Follow-ups:", next_a)

    print("\n--- User B (Priya, Aries Lagna) Response ---")
    print(ans_b)
    print("Follow-ups:", next_b)

    # Assertions for hyper-personalization
    assert "Arjun" in ans_a and "Leo" in ans_a, "User A chart tokens missing"
    assert "Taurus" in ans_a and "Venus" in ans_a, "User A 10th house/lord missing"
    assert "Jupiter Mahadasha" in ans_a, "User A dasha missing"

    assert "Priya" in ans_b and "Aries" in ans_b, "User B chart tokens missing"
    assert "Capricorn" in ans_b and "Saturn" in ans_b, "User B 10th house/lord missing"
    assert "Venus Mahadasha" in ans_b, "User B dasha missing"

    assert ans_a != ans_b, "CRITICAL: Responses must NOT be identical!"
    print(">>> TEST 2 PASSED: Zero Hardcoding! Both charts produced unique, hyper-personalized readings.\n")


def test_elicitations_ui_component():
    print("=" * 60)
    print("TEST 3: <ElicitationsGroup> UI Component Output")
    print("=" * 60)
    followups = [
        "When is the right time for a job change or promotion?",
        "Which gemstone strengthens my 10th house career lord?",
    ]
    ui_component = render_elicitations_group(followups)
    print("Rendered Component:\n" + ui_component)

    assert "<ElicitationsGroup>" in ui_component
    assert "</ElicitationsGroup>" in ui_component
    assert '<Elicitation query="When is the right time for a job change or promotion?">' in ui_component
    assert '<Elicitation query="Which gemstone strengthens my 10th house career lord?">' in ui_component

    # Test clean text attachment
    raw_answer = "**Astrological Insight:** Favorable career phase."
    attached, f_list = attach_domain_locked_elicitations(raw_answer, "my career growth", domain="career")
    assert "<ElicitationsGroup>" not in attached
    assert attached.startswith(raw_answer)
    print(">>> TEST 3 PASSED: <ElicitationsGroup> UI component stripped and clean text verified!\n")


def test_marriage_and_finance_followups():
    print("=" * 60)
    print("TEST 4: Marriage & Finance Follow-Up Personalization")
    print("=" * 60)
    ctx = build_user_chart_context(
        user_name="Rohit",
        lagna="Gemini",
        moon_sign="Libra",
        sun_sign="Taurus",
        current_dasha="Mercury Mahadasha and Jupiter Antardasha",
    )

    # Marriage follow-up: 7th house for Gemini is Sagittarius (Jupiter)
    m_res = resolve_domain_locked_followup("What are the personality traits of my future life partner?", ctx)
    assert m_res is not None
    m_ans, m_f, _ = m_res
    print("Marriage Answer:", m_ans)
    assert "Sagittarius" in m_ans and "Jupiter" in m_ans
    assert "Rohit" in m_ans

    # Finance follow-up: 2nd house Cancer (Moon), 11th house Aries (Mars)
    f_res = resolve_domain_locked_followup("What are my long-term wealth accumulation prospects in my 2nd and 11th house?", ctx)
    assert f_res is not None
    f_ans, f_f, _ = f_res
    print("\nFinance Answer:", f_ans)
    assert "Cancer" in f_ans and "Moon" in f_ans
    assert "Aries" in f_ans and "Mars" in f_ans
    assert "Rohit" in f_ans
    print(">>> TEST 4 PASSED: Marriage & Finance follow-ups dynamically resolved!\n")


async def test_live_fastapi_endpoint():
    print("=" * 60)
    print("TEST 5: Live FastAPI /chat Interception Integration")
    print("=" * 60)
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=15.0) as client:
        # Step 1: User asks career primary question (Streaming LLM response)
        payload1 = {
            "query": "Tell me about my career trajectory.",
            "name": "Kavita",
            "birth_year": 1998,
            "birth_month": 7,
            "birth_day": 15,
            "birth_hour": 14.5,
            "latitude": 28.6139,
            "longitude": 77.2090,
            "timezone": 5.5,
            "user_id": "test_user_kavita",
        }
        res1 = await client.post("/chat", json=payload1)
        assert res1.status_code == 200, f"Error {res1.status_code}: {res1.text}"
        body1 = res1.text
        print("Response 1 stream length:", len(body1))
        assert "<ElicitationsGroup>" not in body1, "<ElicitationsGroup> should not be in stream text"
        assert "[FOLLOW_UPS]:" in body1, "[FOLLOW_UPS] marker missing from stream"

        # Step 2: User clicks follow-up chip: "When is the right time for a job change or promotion?"
        payload2 = {
            "query": "When is the right time for a job change or promotion?",
            "name": "Kavita",
            "birth_year": 1998,
            "birth_month": 7,
            "birth_day": 15,
            "birth_hour": 14.5,
            "latitude": 28.6139,
            "longitude": 77.2090,
            "timezone": 5.5,
            "user_id": "test_user_kavita",
        }
        res2 = await client.post("/chat", json=payload2)
        assert res2.status_code == 200, f"Error {res2.status_code}: {res2.text}"
        data2 = res2.json()
        print("\nResponse 2 (Follow-up click) Answer:\n", data2.get("answer"))
        print("Response 2 source:", data2.get("source"))
        print("Response 2 related_questions:", data2.get("related_questions"))

        assert data2.get("source") == "qa_database_followup", "Expected qa_database_followup intercept"
        assert data2.get("answer_source") == "LOCAL"
        assert "Kavita" in data2.get("answer", "")
        assert "Libra" in data2.get("answer", "")
        assert "<ElicitationsGroup>" not in data2.get("answer", ""), "<ElicitationsGroup> should not be in answer text"
        assert len(data2.get("related_questions", [])) in (2, 3)
        print(">>> TEST 5 PASSED: Live FastAPI Endpoint Intercept and Clean Text verified!\n")


async def main():
    test_domain_mapping()
    test_hyper_personalized_local_resolution()
    test_elicitations_ui_component()
    test_marriage_and_finance_followups()
    await test_live_fastapi_endpoint()
    print("ALL 5 DOMAIN-LOCKED FOLLOW-UP TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(main())
