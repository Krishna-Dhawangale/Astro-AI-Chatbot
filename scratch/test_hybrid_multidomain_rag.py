import asyncio
import os
import sys

# Ensure backend directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.llm_router import (
    detect_query_domains,
    is_multi_domain_query,
    extract_modular_domain_evidence,
    format_multidomain_evidence_context,
    synthesize_hybrid_multidomain_rag,
    stream_hybrid_multidomain_rag,
    load_qa_database,
)


async def run_hybrid_multidomain_tests():
    print("=================================================================")
    print("TESTING HYBRID MULTI-DOMAIN RAG PIPELINE (MODULAR DB + LLM)")
    print("=================================================================")

    # Test 1: Multi-domain Detection
    q1 = "Can you analyze my career, finance, health, and recommended gemstones all at once?"
    detected = detect_query_domains(q1)
    print(f"\n[Test 1] Query: '{q1}'")
    print(f"         Detected domains: {detected}")
    assert "career" in detected
    assert "finance" in detected
    assert "health" in detected
    assert "gemstones" in detected
    assert is_multi_domain_query(q1) is True
    print("         [PASS] Multi-domain detection correctly identified 4 domains.")

    # Test 2: Modular Local DB Extraction
    qa_db = load_qa_database()
    assert len(qa_db) > 0, "QA Database failed to load"
    evidence = extract_modular_domain_evidence(q1, detected, qa_db)
    print("\n[Test 2] Modular Domain Rules Extracted:")
    for dom, rules in evidence.items():
        print(f"         - {dom.upper()}: {len(rules)} rules retrieved")
        assert len(rules) > 0, f"No rules retrieved for domain {dom}"
        print(f"           Sample: '{rules[0][:70]}...'")

    context_str = format_multidomain_evidence_context(evidence)
    assert "### Domain: CAREER" in context_str
    assert "### Domain: FINANCE" in context_str
    assert "### Domain: HEALTH" in context_str
    assert "### Domain: GEMSTONES" in context_str
    print("         [PASS] Modular local database extraction verified.")

    # Test 3: LLM Synthesis with Live Fallback / API Call
    print("\n[Test 3] Executing synthesize_hybrid_multidomain_rag...")
    final_output, meta = await synthesize_hybrid_multidomain_rag(q1)
    print("\n--- Generated Hybrid Multi-Domain Response ---")
    print(final_output)
    print("----------------------------------------------")

    # Assert layout requirements:
    assert "**Astrological Insight:**" in final_output, "Missing **Astrological Insight:**"
    assert "**Actionable Advice:**" in final_output, "Missing **Actionable Advice:**"

    # Assert provenance badge at end:
    assert "[Local Database + LLM - " in final_output, "Missing hybrid provenance badge"
    assert final_output.strip().endswith("]"), "Response does not end with hybrid badge"

    badge_line = [line for line in final_output.splitlines() if line.startswith("[Local Database + LLM -")][-1]
    print(f"\n[PASS] Hybrid provenance badge verified: {badge_line}")
    print(f"       Active Model: {meta.get('active_model')}")

    # Test 4: Streaming Generator
    print("\n[Test 4] Testing stream_hybrid_multidomain_rag...")
    stream_chunks = []
    async for chunk, s_meta in stream_hybrid_multidomain_rag(q1):
        if chunk:
            stream_chunks.append(chunk)

    assembled_stream = "".join(stream_chunks).strip()
    assert "**Astrological Insight:**" in assembled_stream
    assert "**Actionable Advice:**" in assembled_stream
    assert "[Local Database + LLM -" in assembled_stream
    print("         [PASS] Streaming hybrid generator produces complete valid output.")

    print("\n=================================================================")
    print("ALL HYBRID MULTI-DOMAIN RAG TESTS PASSED SUCCESSFULLY (100%)!")
    print("=================================================================")


if __name__ == "__main__":
    asyncio.run(run_hybrid_multidomain_tests())
