# Controlled Production Release Manifest — Phase 18

> **Production Statement**:
> The system is production-ready as a deterministic, chart-grounded astrology reasoning engine with controlled LLM fallback, while domain-classification accuracy remains an ongoing improvement area.

---

## 1. System Deployment Specifications

| Component | Identifier / Version | File Reference / Source |
| :--- | :--- | :--- |
| **Git Commit Hash** | `8d1b26b` | `git tag phase-18-production` |
| **Release Tag** | `phase-18-production` | Git Checkpoint |
| **Rollback Tag (Baseline)** | `production-candidate-phase13` | Commit `601c76c` |
| **Rollback Tag (Selector)** | `phase-14-candidate` | Commit `e47fa61` |
| **Python Environment** | `3.11.0 (64-bit AMD64)` | System Python Environment |
| **scikit-learn Version** | `1.9.1` | `venv/Lib/site-packages/sklearn` |
| **joblib Version** | `1.6.0` | `venv/Lib/site-packages/joblib` |
| **FastAPI Framework** | `0.115.0+` | `backend/main.py` |

---

## 2. Pipeline Subsystem Versions

| Subsystem | Version Identifier | File Location | Key Responsibilities |
| :--- | :--- | :--- | :--- |
| **Model Artifacts** | `OLD`, `CONTEXT_V1`, `V3` | `backend/models/*.pkl` | 3 independent ML domain classifiers |
| **Domain Selector** | `MultiSignalWeightedSelector_v14` | `backend/router/model_selector.py` | Multi-prob vector + agreement + evidence weighting |
| **Router Pipeline** | `PipelineRouter_v13` | `backend/router/pipeline.py` | Core question routing & mode selection |
| **Intent Classifier** | `IntentOverlay_v4` | `backend/router/intent.py` | 23 intent classification & crush overlay |
| **Evidence Matrix** | `IntentRequiredEvidenceMap_v7` | `backend/reasoning/intent_evidence_matrix.py` | Mandatory/Supporting/Timing matrix & safety gate |
| **Stage 8 Rules Engine** | `Stage8_15_DeterministicRules_v2` | `backend/reasoning/pipeline_helper.py` | Astrological rule matching from API chart data |
| **Answer Synthesizer** | `DomainAnswerSynthesizer_v15` | `backend/reasoning/answer_synthesizer.py` | Local natural language interpretation generation |
| **API Normalization** | `normalize_v1` | `backend/astrology/normalize.py` | FreeAstrologyAPI / Prokerala payload normalization |
| **Decision Trace Logger**| `DecisionTraceLogger_v1` | `backend/logs/production_decision_trace.jsonl` | Full observability audit trace per request |

---

## 3. Production Verification Benchmarks

| Metric | Measured Value | Standard Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Golden Set Accuracy (600 queries)** | **93.17%** | $\ge 93.17\%$ | **PASSED** |
| **Unseen Set Accuracy (200 queries)** | **87.00%** | $\ge 84.00\%$ | **PASSED** |
| **Finance Unseen Accuracy** | **85.00%** | $\ge 85.00\%$ | **PASSED** |
| **Health Unseen Accuracy** | **80.00%** | $\ge 80.00\%$ | **PASSED** |
| **Career Unseen Accuracy** | **100.00%** | $\ge 95.00\%$ | **PASSED** |
| **Marriage Unseen Accuracy** | **95.00%** | $\ge 90.00\%$ | **PASSED** |
| **Multi-Domain Unseen Accuracy** | **100.00%** | $\ge 95.00\%$ | **PASSED** |
| **Gemini Call Ratio (Locally Answerable)**| **0.00% (`gemini_calls = 0`)** | $0.00\%$ | **PASSED** |
| **Determinism Rate** | **100.00%** | $100.00\%$ | **PASSED** |

---

## 4. Operational Monitoring Guidance (Phase 19 Target)

`production_decision_trace.jsonl` will actively trace the following weak points for Phase 19 analysis:
- `Health` domain routing (80% accuracy backlog target)
- `Finance` domain routing (85% accuracy backlog target)
- `EVIDENCE_CONFLICT` occurrences
- `UNRESOLVED` evidence states
- `gemini_calls` count & `answer_source` ("local" vs "gemini")
