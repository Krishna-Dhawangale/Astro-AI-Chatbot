"""
backend/intent_classifier.py
Interface Module for Multi-Domain & Micro-Intent Classification
================================================================
Exports the micro-intent NLP classifier and retrieval functions from backend/qa_router.py.
"""

from backend.qa_router import (
    MICRO_INTENT_REGISTRY,
    MicroIntentDefinition,
    VALID_PLANETS,
    classify_multi_domain_micro_intents,
    retrieve_aggregated_local_context,
    build_llm_refiner_prompts,
    append_multidomain_attribution_footer,
    execute_multidomain_hybrid_synthesis,
)
from backend.answer_synthesizer import (
    build_multidomain_system_prompt,
    build_context_payload,
    format_domains_title,
    format_multidomain_footer,
    synthesize_multidomain_answer,
)

__all__ = [
    "MICRO_INTENT_REGISTRY",
    "MicroIntentDefinition",
    "VALID_PLANETS",
    "classify_multi_domain_micro_intents",
    "retrieve_aggregated_local_context",
    "build_llm_refiner_prompts",
    "append_multidomain_attribution_footer",
    "execute_multidomain_hybrid_synthesis",
    "build_multidomain_system_prompt",
    "build_context_payload",
    "format_domains_title",
    "format_multidomain_footer",
    "synthesize_multidomain_answer",
]

