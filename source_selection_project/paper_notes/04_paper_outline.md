# Paper Outline
> Working title: "Instruction-Conditioned Source Selection Under Context-Memory Conflict: A Mechanistic Study Across Three Open-Weight LLMs"

---

## Abstract (draft)
When instruction-tuned LLMs are presented with documents containing facts that conflict with their training memory, their source selection behavior varies substantially by model size, family, and instruction type. We present a reproducible, token-variant-aware mechanistic interpretability pipeline that explicitly separates instruction-conditioned source selection (where the prompt specifies which source to use) from ambiguous arbitration (where no instruction is given). Across three open-weight models — Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct, and Gemma-2-2b-it — we construct matched pairs of clean (no-document) and corrupt (conflicting-document) prompts, apply residual stream patching and attention head patching, and identify the late-layer circuit components causally responsible for context-override behavior. We report full dataset funnels to address survivorship bias, and evaluate generalization to multi-token and naturalistic RAG settings.

---

## Section 1 — Introduction
- The knowledge-conflict problem in RAG systems
- Prior work: context heads, memory heads, late-layer attention (cite Shi 2024, Wolfson 2025, Jin 2024)
- Gap: prior work does not separate instruction-conditioned selection from autonomous arbitration
- Our contribution: token-variant-aware matched pairs, regime separation, cross-family replication, transparent funnel reporting
- **Tables here**: Table 2 (teaser: ctx_win across models and regimes)

## Section 2 — Related Work
- Knowledge conflict in LLMs
- Mechanistic interpretability of factual recall (Rome circuit, ROME, MEMIT)
- Context-preference heads (Shi et al. 2024)
- RAG faithfulness analysis
- **Novelty table**: `02_findings_and_claims.md` → Novelty Claims

## Section 3 — Methodology
### 3.1 Dataset Design
- 501 real-world facts, 10 categories, 4 regimes
- Plausible distractors, not random noise
- **Table**: Table 1 (funnel)

### 3.2 Token-Variant-Aware Filtering
- Why single-token subset for causal tracing
- How variants are generated and filtered
- Two-track design (single vs multi-token)

### 3.3 Source Preference Score
- Definition: logit margin metric
- Clean vs corrupt prompts
- Matched pair validity criteria (MIN_MARGIN = 0.5)

### 3.4 Causal Tracing Protocol
- Residual stream patching
- Attention head patching
- Direct Logit Attribution (DLA)
- Path/edge patching (if applicable)

## Section 4 — Baseline Results
- **Table 2**: ctx_win by model and regime (single-token)
- **Table 3**: ambiguous regime ctx_win
- **Finding F1**: Qwen-3B less compliant than Qwen-1.5B
- **Finding F2**: Gemma instruction-conditional gating
- **Table 4**: Matched pair funnel (survivorship)

## Section 5 — Residual Stream Patching [PENDING]
- Layer-by-layer ΔSPS curves for each model
- **Table 5**: Peak layers
- **Figure**: Patching curves (plots from Phase 8)

## Section 6 — Attention Head Analysis [PENDING]
- Top context-override heads per model
- **Table 6**: Head × layer × ΔSPS
- Attention patterns on clean vs corrupt prompts
- **Figure**: Head heatmaps

## Section 7 — Direct Logit Attribution [PENDING]
- Which heads write context vs memory tokens to residual stream
- Ablation: zeroing context-override heads → does ctx_win drop?

## Section 8 — Cross-Model Comparison [PENDING]
- **Table 7**: Head alignment across models
- Do the same relative layer positions host context-override heads?
- Qwen within-family vs Qwen-Gemma cross-family

## Section 9 — Generalization: Ambiguous Arbitration and RAG [PENDING]
- Ambiguous regime: does the same circuit activate?
- **Finding F2** extended: Gemma gating is mechanistically explained
- **Table 8**: Naturalistic RAG ctx_win

## Section 10 — Discussion
- Why Qwen-3B resists context more than 1.5B (memory strength hypothesis)
- Implications for RAG system design
- Limitations: single-token subset, static facts, no multi-hop

## Section 11 — Conclusion

## Appendix
- Full token variant analysis
- All excluded examples (survivorship)
- Prompt templates (all 4 regimes)
- Reproducibility checklist

---

## Writing Schedule
| Section | Status | Blocker |
|---|---|---|
| 1, 2 | Draft possible now | None |
| 3 | Draft possible now | None |
| 4 | Draft possible now | None |
| 5 | After Phase 8 | Causal tracing |
| 6 | After Phase 9 | Head patching |
| 7 | After Phase 12 | DLA |
| 8 | After Phase 18 | Cross-model |
| 9 | After Phase 16-17 | Ambiguous + RAG eval |
| 10, 11 | Last | All results |
