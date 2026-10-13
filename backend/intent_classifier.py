"""
Intent Classifier Interface Module
==================================
Module: backend/intent_classifier.py

Exports the Multi-Domain Micro-Intent NLP Router APIs and registry from backend/qa_router.py.
Provides full backwards compatibility for intent_classifier imports.
"""

from .qa_router import (
    MICRO_INTENT_REGISTRY,
    MicroIntentDefinition,
    classify_micro_intent,
    resolve_micro_intent_local_template,
    load_qa_database_dict,
    compute_intent_match_score,
    normalize_text_nlp,
)

__all__ = [
    "MICRO_INTENT_REGISTRY",
    "MicroIntentDefinition",
    "classify_micro_intent",
    "resolve_micro_intent_local_template",
    "load_qa_database_dict",
    "compute_intent_match_score",
    "normalize_text_nlp",
]
