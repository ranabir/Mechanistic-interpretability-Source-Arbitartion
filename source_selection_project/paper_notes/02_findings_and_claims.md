# Findings and Claims
> Each claim is tagged: [STRONG] = ready to assert | [PENDING] = needs causal tracing | [HYPOTHESIS] = untested

---

## Behavioural Findings (from Phases 5–6)

### F1 — Qwen-3B is less context-compliant than Qwen-1.5B [STRONG]
**Evidence**: explicit_context ctx_win = 73.0% (3B) vs 95.2% (1.5B).
**Implication**: Scaling within the Qwen2.5 family does not monotonically increase instruction compliance. 3B may have stronger parametric knowledge that competes with document instructions.
**Paper location**: Section 4 (Baseline Results), Figure 1.

### F2 — Gemma-2B shows instruction-conditional source gating [STRONG]
**Evidence**: explicit_context yield = 93.4% (nearly always follows doc); ambiguous yield = 70.0% with `corrupt_weak=63` (memory resists when no instruction).
**Implication**: Gemma has a gating mechanism sensitive to whether an instruction is present. When told to follow the document, it almost always does. When no instruction is given, ~26% of the time it follows memory instead.
**Paper location**: Section 4, Table 2.

### F3 — Multi-token answers have noisier source preference signals [STRONG]
**Evidence**: Explicit_context ctx_win drops from ~85% (single-token avg) to ~52% (multi-token avg).
**Implication**: Logit margin at position 1 is a poor proxy for full multi-token generation. Causal tracing experiments must use single-token subset for clean results.
**Paper location**: Appendix (token variant analysis).

### F4 — 21-29 examples where Qwen-3B memory overrides explicit "follow document" instruction [STRONG]
**Evidence**: `corrupt_weak=29` for Qwen-3B explicit_context.
**Implication**: There exists a non-trivial set of facts where Qwen-3B's parametric memory is strong enough to override explicit document-following instructions. These are the "hard conflict" cases.
**Paper location**: Section 4, discussion.

---

## Mechanistic Claims (PENDING causal tracing)

### H1 — Late-layer attention heads mediate context-override [HYPOTHESIS → PENDING]
**Background**: Prior work (Shi et al. 2024; Wolfson et al. 2025) found context-preference heads in late layers.
**Our claim**: We will show which specific heads in Qwen-3B/1.5B are causally responsible for the clean→corrupt flip, using residual stream patching (Phase 8) + head patching (Phase 9).

### H2 — The same mechanism operates differently under ambiguous vs explicit instruction [HYPOTHESIS → PENDING]
**Background**: Finding F2 shows behavioural difference. The mechanistic question is whether this is mediated by different heads being activated or the same heads with different magnitudes.
**Our claim**: Compare causal tracing results on `explicit_context` pairs vs `ambiguous_conflict` pairs for the same facts.

### H3 — Gemma exhibits a qualitatively similar but architecturally distinct mechanism [HYPOTHESIS → PENDING]
**Background**: Gemma uses a different attention architecture (multi-query attention, different layer counts).
**Our claim**: If we find the same late-layer context-override heads in Gemma that we find in Qwen, this provides cross-family evidence for a general mechanism.

---

## Novelty Claims (vs prior work)

| Prior work | What they did | What we add |
|---|---|---|
| Shi et al. 2024 | Found context-heads in GPT-2/Llama via attention ablation | We use **logit-margin matched pairs** + **token-variant-aware** filtering; test on instruction-tuned models |
| Wolfson et al. 2025 | Taming conflicts with intervention | We **separate instruction-conditioned vs ambiguous** arbitration |
| Jin et al. 2024 | Where Knowledge Collides | We track the **full dataset funnel** to address survivorship bias; cross-family replication |

**Narrow claim (safe to make)**: "We provide the first token-variant-aware, regime-separated mechanistic analysis of instruction-conditioned source selection across three open-weight instruction-tuned LLMs, with transparent survivorship reporting."
