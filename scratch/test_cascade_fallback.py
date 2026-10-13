import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch
import httpx
from dotenv import load_dotenv

# Ensure backend directory is in sys.path
backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

load_dotenv(os.path.join(backend_dir, ".env"))

from llm_router import (
    call_llm_with_cascade,
    CascadeLLMResult,
    get_configured_groq_models,
    AsyncGroqRotationPool,
    GroqAccountClient,
    get_groq_accounts,
    is_rate_limit_error,
)


async def run_tests():
    print("=" * 60)
    print("TEST 1: Live call_llm_with_cascade (Active Production Groq Model)")
    print("=" * 60)
    models = get_configured_groq_models()
    print("Configured Groq models:", models)
    assert "openai/gpt-oss-120b" in models
    assert "openai/gpt-oss-20b" in models
    assert "qwen/qwen3.8-27b" in models

    prompt = "Synthesize astrological career prediction for Aries Ascendant with Jupiter in 10th house."
    res = await call_llm_with_cascade(prompt=prompt, max_tokens=100)
    print("Result String:\n", str(res))
    print("\nResult Meta:", res.meta)
    assert "[Local Database + LLM - Groq-Account-1]" in str(res), "Provenance badge [Local Database + LLM - Groq-Account-1] missing!"
    assert res.meta["tier"] == 1, "Expected successful Tier 1 execution"
    assert res.meta["account"] == "Groq-Account-1"
    print(">>> TEST 1 PASSED: Live Cascade Execution Successful with [Local Database + LLM - Groq-Account-1]!\n")

    print("=" * 60)
    print("TEST 2: Simulated 404 Model Not Found Dynamic Failover")
    print("=" * 60)
    req_dummy = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp_404 = httpx.Response(
        status_code=404,
        text='{"error":{"message":"The model does not exist","type":"invalid_request_error","code":"model_not_found"}}',
        request=req_dummy
    )
    err_404 = httpx.HTTPStatusError("model_not_found", request=req_dummy, response=resp_404)

    resp_200 = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "**Astrological Insight:** Success after dynamic 404 failover."}}]},
        request=req_dummy,
    )

    mock_client_404 = AsyncMock()
    call_count_404 = 0

    async def side_effect_404(*args, **kwargs):
        nonlocal call_count_404
        call_count_404 += 1
        # First 2 attempts return 404 model_not_found
        if call_count_404 <= 2:
            raise err_404
        # 3rd attempt succeeds
        return resp_200

    mock_client_404.post = AsyncMock(side_effect=side_effect_404)

    res_404 = await call_llm_with_cascade(
        prompt="Tell me about Jupiter in 10th house.",
        max_tokens=60,
        custom_httpx_client=mock_client_404
    )
    print("Result String:\n", str(res_404))
    print("Result Meta:", res_404.meta)
    assert res_404.meta["status"] == "success"
    print(">>> TEST 2 PASSED: Successfully caught 404 and dynamically recovered without crashing!\n")

    print("=" * 60)
    print("TEST 3: Simulated Tier 1 429 Rate Limit -> Millisecond Failover to Tier 2")
    print("=" * 60)
    resp_429 = httpx.Response(status_code=429, text="Rate limit exceeded: TPM limit reached", request=req_dummy)
    err_429 = httpx.HTTPStatusError("Rate limit exceeded", request=req_dummy, response=resp_429)

    mock_client_429 = AsyncMock()
    call_count_429 = 0

    async def side_effect_tier1_429(*args, **kwargs):
        nonlocal call_count_429
        call_count_429 += 1
        # Attempt 1 on Account 1 hits 429 rate limit
        if call_count_429 == 1:
            raise err_429
        # Attempt 2 is on Account 2 and succeeds immediately
        return resp_200

    mock_client_429.post = AsyncMock(side_effect=side_effect_tier1_429)

    res_t2 = await call_llm_with_cascade(
        prompt="Tell me about financial growth.",
        max_tokens=60,
        custom_httpx_client=mock_client_429
    )
    print("Result String:\n", str(res_t2))
    print("Result Meta:", res_t2.meta)
    assert res_t2.meta["tier"] == 2, f"Expected Tier 2, got {res_t2.meta.get('tier')}"
    assert res_t2.meta["account"] in ("Groq-Account-2", "Groq-2"), "Expected Groq-Account-2 account"
    assert "[Local Database + LLM - Groq-Account-2]" in str(res_t2), "Provenance badge [Local Database + LLM - Groq-Account-2] missing!"
    assert call_count_429 == 2, f"Expected immediate 2-attempt failover, got {call_count_429}"
    print(">>> TEST 3 PASSED: Millisecond Seamless Failover from Tier 1 (429) to Tier 2!\n")

    print("=" * 60)
    print("TEST 4: Simulated Tier 1 & 2 Failure -> Tier 3 (Gemini) Fallback")
    print("=" * 60)
    mock_client_all_fail = AsyncMock()
    mock_client_all_fail.post = AsyncMock(side_effect=err_429)

    class MockGeminiResponse:
        text = "**Astrological Insight:** Gemini backup synthesis guidance."

    class MockGeminiAioModels:
        async def generate_content(self, *args, **kwargs):
            return MockGeminiResponse()

    class MockGeminiAio:
        models = MockGeminiAioModels()

    class MockGeminiClient:
        aio = MockGeminiAio()

    res_t3 = await call_llm_with_cascade(
        prompt="Tell me about health remedies.",
        max_tokens=60,
        custom_httpx_client=mock_client_all_fail,
        custom_gemini_client=MockGeminiClient()
    )
    print("Result String:\n", str(res_t3))
    print("Result Meta:", res_t3.meta)
    assert res_t3.meta["tier"] == 3, f"Expected Tier 3, got {res_t3.meta.get('tier')}"
    assert res_t3.meta["provider"] == "gemini", "Expected provider gemini"
    assert "[Local Database + LLM - Gemini-1.5-Flash]" in str(res_t3), "Provenance badge [Local Database + LLM - Gemini-1.5-Flash] missing!"
    print(">>> TEST 4 PASSED: Seamless Failover from Tier 1 & 2 to Tier 3 (Gemini) with proper badge!\n")

    print("=" * 60)
    print("TEST 5: Simulated All 3 Network Tiers Failing -> Tier 4 Safety Net")
    print("=" * 60)
    class MockGeminiFailingModels:
        async def generate_content(self, *args, **kwargs):
            raise Exception("429 Resource Exhausted: Gemini rate limit exceeded")

    class MockGeminiFailingAio:
        models = MockGeminiFailingModels()

    class MockGeminiFailingClient:
        aio = MockGeminiFailingAio()

    res_t4 = await call_llm_with_cascade(
        prompt="Tell me about gemstones.",
        max_tokens=60,
        custom_httpx_client=mock_client_all_fail,
        custom_gemini_client=MockGeminiFailingClient()
    )
    print("Result String:\n", str(res_t4))
    print("Result Meta:", res_t4.meta)
    assert res_t4.meta["tier"] == 4, f"Expected Tier 4, got {res_t4.meta.get('tier')}"
    assert "[Local Database + LLM - deterministic_synthesis]" in str(res_t4)
    print(">>> TEST 5 PASSED: Graceful Safety Net without session crash!\n")

    print("=" * 60)
    print("TEST 6: Multi-Account Groq Rotation Pool & Client Manager Verification")
    print("=" * 60)
    pool = AsyncGroqRotationPool.from_env()
    accounts = pool.get_accounts()
    print("Configured Groq Accounts in Pool:", [a.account_id for a in accounts])
    assert len(accounts) >= 2, f"Expected at least 2 accounts, found {len(accounts)}"
    assert accounts[0].account_id == "Groq-Account-1"
    assert accounts[1].account_id == "Groq-Account-2"

    # Test round-robin dispatch
    rr1 = await pool.get_next_round_robin_account()
    rr2 = await pool.get_next_round_robin_account()
    print(f"Round-Robin Rotation: {rr1.account_id} -> {rr2.account_id}")
    assert rr1.account_id != rr2.account_id, "Round robin should rotate between accounts"

    # Test rate limit marking and cool-off
    accounts[0].mark_rate_limited(cooloff_seconds=2.0)
    assert accounts[0].is_rate_limited is True
    avail = await pool.get_next_round_robin_account()
    assert avail.account_id == "Groq-Account-2", "Should skip rate-limited Account 1"
    accounts[0].reset_limit()
    assert accounts[0].is_rate_limited is False
    print(">>> TEST 6 PASSED: Rotation pool and client manager verified!\n")

    print("=" * 60)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_tests())
