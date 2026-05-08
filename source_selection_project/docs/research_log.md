# Research Log — Instruction-Conditioned Source Selection Under Context-Memory Conflict

> **Purpose**: Living document. Updated after every experimental phase.
> All numbers here are sourced directly from `results/latest/**/*.json` and `data/manifests/*.json`.
> This document is the primary reference for writing the paper.

---

## Project Overview

**Research Question**: When an instruction-tuned LLM is given a document containing a fact that conflicts with its training memory, *which internal mechanisms determine whether the model follows the document or its memory?*

**Key Distinction from Prior Work**:
- Prior work conflates "what does the model output" with "what mechanism caused it"
- We explicitly separate:
  - **Explicit-Instruction regime (A/B)**: prompt tells the model which source to use → measures instruction-following circuitry
  - **Ambiguous regime (C)**: no instruction → measures genuine arbitration between competing sources
  - **Naturalistic RAG regime (D)**: realistic search-result-style context → measures generalization

**Models tested**:
| Model | HF ID | Parameters | Family |
|---|---|---|---|
| Qwen 3B | `Qwen/Qwen2.5-3B-Instruct` | 3B | Qwen2.5 |
| Qwen 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Qwen2.5 |
| Gemma 2B | `google/gemma-2-2b-it` | 2B | Gemma-2 |

---

## Phase 3: Dataset

**Script**: `scripts/00_generate_dataset.py`
**Output**: `data/raw/facts.jsonl`, `data/generated/*.jsonl`

### Design Principles
- Every fact is **real-world** — no synthetic filler
- Each fact has a `memory_answer` (what the model should know) and a `context_answer` (the wrong claim placed in the document)
- The `context_answer` is a plausible distractor from the same semantic category (e.g., another capital city, not a random word)
- 4 prompt families generated per fact — one per regime

### Dataset Statistics
| Category | Count | Answer type | Token notes |
|---|---|---|---|
| Geography | 137 | City names, currencies | Mixed (Paris=1tok, Buenos Aires=2tok) |
| Chemistry | 90 | Symbols (H, Fe), atomic numbers | ✅ Mostly single-token |
| Biology | 66 | Mammal/fish, legs, speeds | Mostly single-token |
| History | 50 | Years (1776), inventor names | Mixed |
| Sports | 33 | Host cities, years, team sizes | Mixed |
| Literature | 30 | Author names | Multi-token |
| Mathematics | 26 | Integers, shapes | ✅ Single-token |
| Physics | 25 | SI unit names | Mixed |
| Astronomy | 22 | Integers, classifications | ✅ Single-token |
| Music | 22 | string/brass/woodwind | ✅ Single-token |
| **TOTAL** | **501** | — | — |

**Prompt families** (501 facts × 4 regimes = 2,004 total prompts):
- `explicit_context_prompts.jsonl` — "Based only on the document..."
- `explicit_memory_prompts.jsonl` — "Ignoring the document, factually..."
- `ambiguous_conflict_prompts.jsonl` — "What is the X of Y?" (no instruction)
- `naturalistic_rag_prompts.jsonl` — Search result / travel blog / academic snippet formats

---

## Phase 4: Token Variant Filtering

**Script**: `scripts/01_validate_token_variants.py`
**Output**: `data/filtered/{model}/single_token_*.jsonl`, `data/filtered/{model}/multi_token_*.jsonl`

### Design
For each answer, we generate semantic variants (e.g., "Paris", " Paris", "paris", " paris") and check which tokenize to exactly 1 token under each model's tokenizer.

**Two-track output** (key design decision — no examples discarded):
- **Single-token subset** → used for causal tracing (logit margin is precise)
- **Multi-token subset** → used for naturalistic RAG eval (first-subword approximation)

### Filtering Results
| Model | Single-token | Multi-token | Single-token % |
|---|---|---|---|
| Qwen 3B | 189 / 501 | 312 / 501 | 37.7% |
| Qwen 1.5B | 189 / 501 | 312 / 501 | 37.7% |
| Gemma 2B | 243 / 501 | 258 / 501 | 48.5% |

**Note for paper**: Qwen uses the same tokenizer for 3B and 1.5B (Qwen2.5 vocab), so filtering results are identical. Gemma has a wider single-token yield because its tokenizer handles more word-initial forms as single tokens.

Survivorship report: `data/manifests/excluded_examples_summary.csv` (3,528 excluded examples with reasons).

---

## Phase 5: Baseline Source Preference Scoring

**Script**: `scripts/02_compute_baselines.py`
**Output**: `results/latest/{model}/baseline_source_preference_{regime}.json`, `baseline_summary.json`

### Metric
```
Score(S) = max_{t ∈ T_context} logit(t) − max_{t ∈ T_memory} logit(t)

Score > 0  → model prefers context (document) answer
Score < 0  → model prefers memory (training) answer
Score ≈ 0  → ambiguous / tied
```

### Results: `explicit_context` regime (model told to follow document)

| Model | ctx_win (single-token) | ctx_win (multi-token) | mean_score (single) |
|---|---|---|---|
| Qwen 3B | **73.0%** | 47.4% | +3.37 |
| Qwen 1.5B | **95.2%** | 57.7% | +5.26 |
| Gemma 2B | **91.8%** | 49.6% | +5.66 |

### Key Observations (for paper)

1. **Qwen 3B follows context less than Qwen 1.5B** (73% vs 95%), despite being the larger model. This is counterintuitive — larger models are often assumed to be better instruction followers. This suggests Qwen 3B has *stronger parametric memory* that resists context override, not weaker instruction-following.

2. **Multi-token ctx_win drops sharply vs single-token** for all models (~47-57% vs 73-95%). This is expected: multi-token answers involve multi-step decoding dynamics not captured by the first-subword logit margin. The single-token metric is more precise for our purposes.

3. **Gemma 2B explicit_context ctx_win = 91.8%** — Gemma is highly compliant with document instructions. This makes it a useful cross-family comparison: if the same mechanisms we find in Qwen also appear in Gemma, the circuit is likely a general property of instruction-tuning, not model-family-specific.

---

## Phase 6: Matched Pair Construction

**Script**: `scripts/03_build_matched_pairs.py`
**Output**: `data/causal_tracing_pairs/{model}/pairs_{regime}.jsonl`

### What a matched pair contains
```json
{
  "clean_prompt":  "Question: What is the capital of France?\nAnswer:",
  "clean_score":   -4.2,        // memory wins on clean (Paris > Rome)
  "corrupt_prompt": "Document: The capital of France is Rome.\nQuestion: Based only on the document...\nAnswer:",
  "corrupt_score":  +3.1,       // context wins on corrupt (Rome > Paris)
  "clean_to_corrupt_flip": 7.3  // magnitude of the flip caused by inserting the document
}
```

### Validity Criteria
- `corrupt_score ≥ +0.5` (context wins clearly on the corrupt prompt)
- `clean_score ≤ -0.5` (memory wins clearly on the clean prompt)
- Both sides have single-token valid variants

### Pair Construction Funnel

| Model | Regime | Input | Valid Pairs | Yield |
|---|---|---|---|---|
| Qwen 3B | explicit_context | 189 | **133** | 70.4% |
| Qwen 3B | ambiguous | 189 | **136** | 72.0% |
| Qwen 1.5B | explicit_context | 189 | **149** | 78.8% |
| Qwen 1.5B | ambiguous | 189 | **153** | 81.0% |
| Gemma 2B | explicit_context | 243 | **227** | 93.4% |
| Gemma 2B | ambiguous | 243 | **170** | 70.0% |

### Exclusion Breakdown (for paper — anti-survivorship-bias reporting)

| Model | Regime | Corrupt weak | Clean weak | Clean ctx won |
|---|---|---|---|---|
| Qwen 3B | explicit_context | 29 | 6 | 21 |
| Qwen 1.5B | explicit_context | 9 | 7 | 24 |
| Gemma 2B | explicit_context | 4 | 4 | 8 |
| Gemma 2B | ambiguous | **63** | 4 | 6 |

**Key observation**: Gemma ambiguous has `corrupt_weak=63` — on 63 prompts with no explicit instruction, Gemma's memory resisted the context (corrupt score < 0.5). This is the opposite of its explicit_context behavior (corrupt_weak=4). **This is the first evidence of genuine ambiguity-conditional source selection in Gemma — a core paper claim.**

---

## Upcoming Phases

| Phase | Goal | Status |
|---|---|---|
| 7 | Lock model configs + reproducibility metadata | 🔲 |
| 8 | Residual stream patching | 🔲 |
| 9 | Attention head patching | 🔲 |
| 10 | Ablation controls | 🔲 |
| 11 | Attention pattern baselines | 🔲 |
| 12 | Direct Logit Attribution (DLA) | 🔲 |
| 13 | Path / edge patching | 🔲 |
| 14 | Probing | 🔲 |
| 15 | Steering experiments | 🔲 |
| 16 | Ambiguous conflict eval | 🔲 |
| 17 | Naturalistic RAG eval | 🔲 |
| 18 | Cross-model comparison | 🔲 |
| 19–22 | Plots, tables, paper draft, claim check | 🔲 |

---

## Full Dataset Funnel (for paper Table 1)

```
501 real-world facts (10 categories)
  × 4 prompt regimes
= 2,004 raw prompts

→ Single-token filtering:
    Qwen:   189 / 501 (37.7%) per regime
    Gemma:  243 / 501 (48.5%) per regime

→ Matched pair construction (explicit_context, single-token):
    Qwen 3B:   133 / 189 valid pairs (70.4%)
    Qwen 1.5B: 149 / 189 valid pairs (78.8%)
    Gemma 2B:  227 / 243 valid pairs (93.4%)

These are the causal tracing subjects.
Multi-token (312 Qwen / 258 Gemma) → naturalistic RAG eval.
```
