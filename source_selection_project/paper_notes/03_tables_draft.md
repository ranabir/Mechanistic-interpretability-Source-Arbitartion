# Tables Draft
> These are paper-ready table drafts. Update numbers from JSON after each phase.
> Source paths noted under each table.

---

## Table 1 — Dataset and Filtering Funnel

| Stage | Count | Notes |
|---|---|---|
| Raw facts generated | 501 | 10 categories, all real-world |
| Prompt families | 4 | explicit_context, explicit_memory, ambiguous, naturalistic |
| Total prompts | 2,004 | 501 × 4 |
| Single-token valid (Qwen) | 189 / 501 | 37.7% per regime |
| Single-token valid (Gemma) | 243 / 501 | 48.5% per regime |
| Multi-token subset (Qwen) | 312 / 501 | Kept for RAG eval |
| Multi-token subset (Gemma) | 258 / 501 | Kept for RAG eval |
| Valid causal pairs — Qwen-3B (explicit) | 133 | 70.4% of single-token |
| Valid causal pairs — Qwen-1.5B (explicit) | 149 | 78.8% |
| Valid causal pairs — Gemma-2B (explicit) | 227 | 93.4% |

*Source*: `data/manifests/dataset_generation_manifest.json`, `dataset_funnel_summary.csv`, `pair_construction_funnel.json`

---

## Table 2 — Baseline Source Preference (Single-Token, explicit_context)

| Model | ctx_win% | mem_win% | Mean Score | n |
|---|---|---|---|---|
| Qwen-3B | 73.0% | 27.0% | +3.37 | 189 |
| Qwen-1.5B | 95.2% | 4.8% | +5.26 | 189 |
| Gemma-2B | 91.8% | 8.2% | +5.66 | 243 |

*Source*: `results/latest/{model}/baseline_summary.json` → `regimes.explicit_context.single_token`

---

## Table 3 — Baseline Source Preference (Single-Token, ambiguous_conflict)

| Model | ctx_win% | mem_win% | Mean Score | n |
|---|---|---|---|---|
| Qwen-3B | [TBD] | [TBD] | [TBD] | 189 |
| Qwen-1.5B | [TBD] | [TBD] | [TBD] | 189 |
| Gemma-2B | [TBD] | [TBD] | [TBD] | 243 |

*Source*: `results/latest/{model}/baseline_summary.json` → `regimes.ambiguous_conflict.single_token`

---

## Table 4 — Matched Pair Funnel (Survivorship Reporting)

| Model | Regime | Input | Valid | corrupt_weak | clean_weak | clean_ctx_won |
|---|---|---|---|---|---|---|
| Qwen-3B | explicit_context | 189 | **133** (70%) | 29 | 6 | 21 |
| Qwen-3B | ambiguous | 189 | **136** (72%) | 27 | 10 | 16 |
| Qwen-1.5B | explicit_context | 189 | **149** (79%) | 9 | 7 | 24 |
| Qwen-1.5B | ambiguous | 189 | **153** (81%) | 5 | 9 | 22 |
| Gemma-2B | explicit_context | 243 | **227** (93%) | 4 | 4 | 8 |
| Gemma-2B | ambiguous | 243 | **170** (70%) | **63** | 4 | 6 |

*Source*: `data/causal_tracing_pairs/{model}/pairs_summary.json`

---

## Table 5 — Residual Stream Patching Results [PENDING Phase 8]

| Model | Regime | Peak layer | Peak effect (ΔSPS) | Location |
|---|---|---|---|---|
| Qwen-3B | explicit_context | [TBD] | [TBD] | [TBD] |
| Qwen-1.5B | explicit_context | [TBD] | [TBD] | [TBD] |
| Gemma-2B | explicit_context | [TBD] | [TBD] | [TBD] |

---

## Table 6 — Top Context-Override Heads per Model [PENDING Phase 9]

| Model | Head | Layer | ΔSPS | Attention pattern |
|---|---|---|---|---|
| Qwen-3B | [TBD] | [TBD] | [TBD] | [TBD] |
| Qwen-1.5B | [TBD] | [TBD] | [TBD] | [TBD] |
| Gemma-2B | [TBD] | [TBD] | [TBD] | [TBD] |

---

## Table 7 — Cross-Model Head Alignment [PENDING Phase 18]

| Metric | Qwen-3B vs 1.5B | Qwen-3B vs Gemma | Qwen-1.5B vs Gemma |
|---|---|---|---|
| Top-5 head overlap | [TBD] | [TBD] | [TBD] |
| Relative layer position | [TBD] | [TBD] | [TBD] |
| DLA cosine similarity | [TBD] | [TBD] | [TBD] |

---

## Table 8 — Naturalistic RAG Generalization [PENDING Phase 17]

| Model | ctx_win (single-token) | ctx_win (multi-token RAG) | Drop |
|---|---|---|---|
| Qwen-3B | 73.0% | [TBD] | [TBD] |
| Qwen-1.5B | 95.2% | [TBD] | [TBD] |
| Gemma-2B | 91.8% | [TBD] | [TBD] |
