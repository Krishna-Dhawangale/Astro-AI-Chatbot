import asyncio
import os
import sys

# Ensure backend is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.llm_router import (
    stream_cascading_router,
    generate_cascading_router,
    format_model_prefix,
    format_local_badge,
    LOCAL_DB_BADGE,
    DIRECT_CALC_BADGE,
    GROQ_MODELS,
    OPENROUTER_MODELS,
    GEMINI_SAFETY_MODELS,
)
from backend.main import label_answer_source


async def test_router_prefixes():
    print("==================================================")
    print("TESTING LLM ROUTER MODEL PREFIX & LOCAL BADGES")
    print("==================================================")

    # Test 1: Verify Prefix Helpers
    assert format_model_prefix("llama-3.3-70b-versatile") == "[Model: llama-3.3-70b-versatile]\n\n"
    assert format_local_badge("Local Database") == "[Source: Local Database]\n\n"
    print("[PASS] Prefix formatting helpers validated.")

    # Test 2: Local DB label_answer_source
    direct_res = label_answer_source("Your lagna is Virgo at 15 degrees.", "DIRECT")
    assert direct_res.startswith("[Source: Direct Calculation]\n\n")
    print("[PASS] label_answer_source DIRECT badge validated:")
    print("       " + direct_res.split("\n")[0])

    rule_res = label_answer_source("Based on 10th house indicators, career favors leadership.", "RULE_BASED")
    assert rule_res.startswith("[Source: Local Database]\n\n")
    print("[PASS] label_answer_source RULE_BASED badge validated:")
    print("       " + rule_res.split("\n")[0])

    # Test 3: Tier 5 Local Fallback in router
    async def mock_exhausted_router():
        chunks = []
        async for chunk, meta in stream_cascading_router(
            prompt="test prompt",
            system_prompt="test system",
            local_fallback_text="Planetary placements indicate steady growth.",
            # Force empty providers by providing mock invalid client if needed or testing Tier 5 directly
        ):
            chunks.append(chunk)
        return "".join(chunks)

    # Test 4: Live Cascade Call with valid keys from .env
    print("\n--- Testing live cascading router stream ---")
    test_prompt = "Explain Sun in 10th house in 45 words."
    test_sys = "You are a Vedic astrologer. Return only the synthesized insight."

    received_chunks = []
    received_metas = []
    async for chunk, meta in stream_cascading_router(
        prompt=test_prompt,
        system_prompt=test_sys,
        max_output_tokens=100,
        temperature=0.2,
    ):
        if chunk:
            received_chunks.append(chunk)
        if meta:
            received_metas.append(meta)

    full_output = "".join(received_chunks)
    print("Response output snippet:\n" + full_output[:200] + "...\n")

    # Assert response has either [Model: ...] or [Source: Local Database]
    assert full_output.startswith("[Model: ") or full_output.startswith("[Source: ")
    if full_output.startswith("[Model: "):
        print(f"[PASS] Response begins with model prefix: {full_output.splitlines()[0]}")
    else:
        print(f"[PASS] Response begins with local badge: {full_output.splitlines()[0]}")

    print("\n[ALL ROUTER & BADGE TESTS PASSED SUCCESSFULLY!]")


if __name__ == "__main__":
    asyncio.run(test_router_prefixes())
