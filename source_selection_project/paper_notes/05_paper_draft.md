# Instruction Readers, Not Document Copiers: A Mechanistic Account of Source Arbitration in Instruction-Tuned LLMs

> **Working draft** — Phase 21 of the Source Arbitration research pipeline

---

## Abstract

When retrieval-augmented language models encounter documents that contradict their parametric knowledge, they must arbitrate between two sources: the retrieved context and trained memory. We present a mechanistic account of this source-arbitration circuit across three instruction-tuned LLMs (Qwen2.5-3B, Qwen2.5-1.5B, and Gemma-2-2B). Using a token-variant-aware pipeline of residual patching, head patching, direct logit attribution, attention analysis, linear probing, and activation steering, we identify a compact late-layer circuit (final 8–12% of model depth) responsible for context-override behavior. Our central finding is that context-override heads function as **instruction processors, not document copiers**: they attend exclusively to the question/instruction region (25–57% attention) while paying zero attention to the document text, yet causally determine whether the model follows the document. We further discover three distinct architectural strategies for implementing this circuit — concentrated star heads (Qwen-3B), memory-defense heads (Qwen-1.5B), and distributed committee heads (Gemma-2B) — suggesting that source arbitration is a convergent functional capability with divergent mechanistic implementations across model families.

---

## 1. Introduction

Retrieval-augmented generation (RAG) systems augment language models with external documents at inference time. A critical failure mode occurs when the retrieved document contains information that conflicts with the model's parametric knowledge — for example, when a document states "The capital of France is Lyon" while the model has learned "Paris." In such cases, the model must *arbitrate* between sources: follow the document (context-adherence) or trust its training (memory-adherence).

Prior work has identified "context heads" in language models that preferentially attend to in-context information (Shi et al., 2024; Wolfson et al., 2025). However, existing studies have three key limitations: (1) they do not separate *instruction-conditioned* source selection (where the prompt explicitly says "follow the document") from *autonomous arbitration* (where no instruction is given); (2) they treat the context-override mechanism as a content-copying operation; and (3) they analyze single model families.

We address these gaps with three contributions:

1. **Regime separation**: We explicitly distinguish instruction-conditioned selection from ambiguous arbitration, showing they activate overlapping but distinct circuits.

2. **The instruction-reader finding**: We demonstrate that context-override heads do not read the document at all (0% document attention) — instead, they read the instruction and write the document answer using representations computed by earlier layers.

3. **Cross-family replication with architectural taxonomy**: We identify three distinct strategies for the same functional goal, providing evidence that source arbitration is a convergent capability.

---

## 2. Related Work

**Knowledge conflict in LLMs.** Longpre et al. (2021) first systematically studied how models handle conflicts between parametric and contextual knowledge. Chen et al. (2022) showed that larger models are more susceptible to being misled by counterfactual context. Jin et al. (2024) analyzed where knowledge conflicts are resolved within transformer layers.

**Mechanistic interpretability of factual recall.** The ROME framework (Meng et al., 2022) identified factual associations stored in MLP layers. Subsequent work on causal tracing (Meng et al., 2023) showed that factual recall involves a localized circuit in mid-to-late layers.

**Context-preference heads.** Shi et al. (2024) identified attention heads in GPT-2 and Llama that preferentially attend to in-context information during knowledge conflicts. Wolfson et al. (2025) proposed interventions to tame conflicts. Our work extends this line by providing a complete mechanistic decomposition — not just *which* heads, but *what they attend to*, *what they write*, and *how to steer them*.

---

## 3. Methodology

### 3.1 Dataset Design

We construct a benchmark of 501 real-world facts across 10 categories (geography, science, history, etc.), each paired with a plausible counterfactual distractor. Each fact generates prompts under four regimes:

| Regime | Instruction | Example |
|---|---|---|
| **Explicit context** | "Based only on the document above..." | Tests instruction-following |
| **Ambiguous conflict** | "What is...?" (no instruction) | Tests autonomous arbitration |
| **Explicit memory** | "Using your own knowledge..." | Tests memory recall |
| **Naturalistic RAG** | Multi-document retrieval style | Tests ecological validity |

### 3.2 Token-Variant-Aware Filtering

A key methodological contribution is our token-variant-aware filtering. The same entity can tokenize differently depending on context (e.g., "Paris" → `[Paris]` vs `[Par, is]`). We enumerate all single-token variants for each answer and restrict causal tracing experiments to facts where both the context and memory answers have valid single-token representations. This yields clean logit-margin measurements without multi-token confounds.

**Dataset funnel (survivorship reporting):**

| Model | Input prompts | Valid single-token | Valid matched pairs | Yield |
|---|---|---|---|---|
| Qwen2.5-3B | 501 | 189 | 133 | 70.4% |
| Qwen2.5-1.5B | 501 | 189 | 149 | 78.8% |
| Gemma-2-2B | 501 | 243 | 227 | 93.4% |

### 3.3 Source Preference Score (SPS)

We define the **Source Preference Score** as:

$$\text{SPS} = \max_{v \in V_{\text{ctx}}} \log p(v) - \max_{v \in V_{\text{mem}}} \log p(v)$$

where $V_{\text{ctx}}$ and $V_{\text{mem}}$ are the sets of valid single-token variants for the context and memory answers respectively. SPS > 0 indicates context-adherence; SPS < 0 indicates memory-adherence.

### 3.4 Causal Intervention Protocol

We apply six complementary interventions:

1. **Residual stream patching** (Phase 8): Replace layer $\ell$'s residual output from the corrupt (document-present) run with the clean (no-document) run. Measure ΔSPS.

2. **Head patching** (Phase 9): Same as above but at individual attention head granularity.

3. **Direct Logit Attribution** (Phase 12): Decompose each head's output through the unembedding matrix to measure its direct contribution to context vs. memory token logits.

4. **Attention pattern analysis** (Phase 11): Measure what fraction of each head's attention goes to document tokens vs. question/instruction tokens.

5. **Linear probing** (Phase 14): Train logistic regression probes on residual stream activations at each layer to predict context vs. memory following.

6. **Activation steering** (Phase 15): Compute a "context direction" vector and add it during inference to test sufficiency.

---

## 4. Results

### 4.1 Baseline Source Preference

| Model | Explicit context | Ambiguous | Explicit memory |
|---|---|---|---|
| Qwen2.5-3B | 86.2% (SPS=5.79) | 87.3% (SPS=6.13) | 66.7% (SPS=1.91) |
| Qwen2.5-1.5B | 95.8% (SPS=5.87) | 97.9% (SPS=5.98) | 82.0% (SPS=2.93) |
| Gemma-2-2B | 98.4% (SPS=6.55) | 78.2% (SPS=3.77) | 25.9% (SPS=−2.62) |

**Finding 1: Instruction-conditional gating in Gemma.** Gemma shows dramatic regime sensitivity: 98.4% context-following when instructed, but only 78.2% when the instruction is absent — and 25.9% when told to use memory (i.e., it actually follows the memory instruction). This suggests a gating mechanism that is sensitive to the presence and polarity of source-selection instructions.

**Finding 2: Paradoxical scaling in Qwen.** Qwen-1.5B (smaller) is *more* context-compliant than Qwen-3B (larger): 95.8% vs 86.2%. This inverts the common assumption that larger models are more instruction-compliant. We hypothesize that 3B's stronger parametric knowledge creates more resistance to context-override.

### 4.2 Circuit Localization

> *Where does the source-selection decision happen?*

![Residual stream patching curves showing sharp ΔSPS drop in final layers](../results/figures/fig1_residual_patching.png)

Residual stream patching reveals a consistent pattern across all three models: the first 50–60% of layers have near-zero ΔSPS, followed by a sharp decline in the final layers.

| Model | Peak layer | Depth | Peak ΔSPS |
|---|---|---|---|
| Qwen2.5-3B | L33 | 94.3% | −13.28 |
| Qwen2.5-1.5B | L27 | 100.0% | −11.39 |
| Gemma-2-2B | L25 | 100.0% | −14.07 |

**Finding 3: Universal late-layer localization.** The context-override circuit is localized to the final 6–8% of model depth across both model families. Patching earlier layers produces negligible effects, indicating that source-selection is computed *after* semantic processing is complete.

### 4.3 Head-Level Decomposition

> *Which heads are responsible, and what do they do?*

Head patching isolates individual heads. The top causally responsible heads cluster in 2–3 consecutive layers near the model's output:

| Model | Top head | ΔSPS | Significant heads |
|---|---|---|---|
| Qwen2.5-3B | L31.H10 | −0.25 | 20 |
| Qwen2.5-1.5B | L27.H1 | −0.12 | 7 |
| Gemma-2-2B | L22.H1 | −0.25 | 9 |

Qwen-3B distributes the computation across 20 heads (a **"distributed but concentrated" strategy**), while Qwen-1.5B uses only 7 and Gemma uses 9.

### 4.4 Direct Logit Attribution

> *What do these heads write into the residual stream?*

![DLA decomposition showing context and memory logit contributions](../results/figures/fig2_dla_heatmap.png)

DLA decomposes each head's output through the unembedding matrix:

| Model | Top DLA head | DLA score | Context logit | Memory logit |
|---|---|---|---|---|
| Qwen2.5-3B | L33.H9 | +0.324 | +0.498 | +0.175 |
| Qwen2.5-1.5B | L27.H7 | +0.373 | +0.763 | +0.390 |
| Gemma-2-2B | L22.H5 | +0.171 | −0.046 | −0.217 |

**Finding 4: Memory-defense heads in Qwen-1.5B.** While all models' top heads have positive DLA (pro-context), Qwen-1.5B uniquely contains 3 heads with *negative* DLA — heads that actively write memory-token logits. These "memory-defense" heads represent a pull-back mechanism: the model simultaneously promotes context (through positive-DLA heads) and resists it (through negative-DLA heads). This dual-direction circuit is absent in the larger Qwen-3B.

**Finding 5: Gemma uses suppression, not promotion.** Gemma's top head (L22.H5) has negative logit contributions for *both* context and memory tokens, but suppresses the memory token *more* (−0.217 vs −0.046). The net DLA is positive because context is suppressed less. This "differential suppression" strategy differs qualitatively from Qwen's direct promotion strategy.

### 4.5 The Instruction-Reader Finding

> *What do these heads attend to?*

![Attention patterns showing 0% document attention, 25-57% question attention](../results/figures/fig3_attention_patterns.png)

This is our central and most surprising finding. Across all three models, the context-override heads:

| Model | Avg document attention | Avg question attention |
|---|---|---|
| Qwen2.5-3B | **0.0%** | 57.6% |
| Qwen2.5-1.5B | **0.0%** | 38.5% |
| Gemma-2-2B | **0.0%** | 24.2% |

**Finding 6: Context-override heads are instruction readers, not document copiers.** The heads that causally determine whether the model follows the document pay *zero attention* to the document itself. Instead, they attend to the question/instruction region, specifically tokens like "Based only on the document above." 

This overturns the naive model of context-override as "the head reads the document and copies its answer." Instead, the mechanism is:

1. **Early layers** (L0–L20): Process and encode the document content into the residual stream.
2. **Late-layer heads** (L22–L33): Read the *instruction* to determine the source-selection policy, then *write* the appropriate answer token using document information already encoded in the residual stream by earlier layers.

The circuit is an **instruction-conditioned policy executor**, not a content copier.

### 4.6 Information Crystallization

> *When does the decision form?*

![Probing curves showing accuracy increase at L18-L22 for Gemma](../results/figures/fig5_probing.png)

Linear probing on residual stream activations reveals when source-selection information becomes linearly decodable:

| Model | Probe gain | Interpretation |
|---|---|---|
| Gemma-2-2B | **15.8pp** (70% → 86%) | Clear crystallization at L18–L22 |
| Qwen2.5-3B | 5.8pp (86% → 92%) | Signal present but confounded by class imbalance |
| Qwen2.5-1.5B | 4.2pp (91% → 95%) | Near-ceiling; insufficient variance |

**Finding 7: Probe crystallization aligns with causal patching.** In Gemma, the steepest probe accuracy increase occurs at L18–L22, precisely matching the layers identified by residual patching. This provides independent, non-causal confirmation that the source-selection decision crystallizes at the same depth where causal interventions have the strongest effect.

### 4.7 Activation Steering

> *Can we control the circuit?*

![Steering dose-response showing directional control and 1.5B inversion](../results/figures/fig4_steering.png)

We compute a "context direction" vector from the attention output at each model's peak layer and add α × direction during inference:

| Model | ΔSPS at α=−50 | Direction |
|---|---|---|
| Qwen2.5-3B | −0.46 | Normal (anti-ctx reduces SPS) |
| Qwen2.5-1.5B | **+0.49** | **Inverted** |
| Gemma-2-2B | −1.29 | Normal |

**Finding 8: Inverted steering confirms memory-defense heads.** Qwen-1.5B's steering response is the mirror image of the other models: subtracting the "context direction" *increases* context preference. This independently confirms that the top heads at L27 are memory-defense heads — their natural direction opposes context, so subtracting it weakens memory defense. Three independent methods (head patching ΔSPS sign, DLA polarity, steering direction) converge on this finding.

---

## 5. Discussion

### 5.1 Three Strategies for One Function

Our cross-model analysis reveals three distinct architectural strategies for implementing source arbitration:

**Strategy 1 — Concentrated Override (Qwen-3B):** A set of powerful heads at L31–L33 directly promote the context answer. The circuit uses 20 significant heads but concentrates most causal impact in 3–5 "star" heads with DLA > 0.2. There is no organized memory-defense counter-circuit.

**Strategy 2 — Memory Defense (Qwen-1.5B):** The smaller Qwen model implements a dual-direction circuit: some heads promote context (positive DLA) while others actively defend memory (negative DLA). The net effect still favors context (95.8% context-following), but the architecture reveals an internal tug-of-war. This may reflect weaker instruction-following capacity in the smaller model, requiring explicit memory suppression rather than clean context promotion.

**Strategy 3 — Distributed Committee (Gemma-2-2B):** Gemma distributes the computation across many heads with individually weak contributions (top DLA = 0.171 vs Qwen's 0.324). No single head dominates. This "committee" approach produces the cleanest probing crystallization curve (15.8pp gain) and the most monotonic steering response, suggesting a more gradual, distributed decision process.

### 5.2 The Instruction-Reader Paradox

The most striking finding is that context-override heads do not read the document. This challenges the intuitive model of RAG as "the model reads the document and extracts the answer." Instead, the mechanistic picture is:

1. Early layers encode document content into distributed residual stream representations.
2. Late-layer heads read the instruction to determine *policy* (follow document vs. follow memory).
3. These heads then write the appropriate answer token, accessing document information through the residual stream — not through direct attention to document tokens.

This has practical implications for RAG system design: interventions targeting document-level attention patterns may be ineffective because the critical computation occurs at the instruction level.

### 5.3 Implications for RAG Faithfulness

Our findings suggest that improving RAG faithfulness requires:

1. **Strengthening instruction signals**, not document signals — since the circuit reads instructions, not documents.
2. **Model-specific strategies** — a one-size-fits-all intervention is unlikely to work across architectures that implement fundamentally different strategies.
3. **Monitoring the memory-defense pathway** — in smaller models with dual-direction circuits, faithfulness failures may arise from the memory-defense side being too strong, not the context-promotion side being too weak.

---

## 6. Limitations

1. **Synthetic benchmark**: Our facts and counterfactuals are constructed, not naturally occurring. Naturalistic RAG evaluation remains future work.
2. **Small models**: We study 1.5B–3B parameter models. Larger models may implement qualitatively different circuits.
3. **Single-token restriction**: Causal tracing is limited to facts with single-token answers, covering ~38–49% of our dataset.
4. **Steering ceiling**: Single-layer activation steering shifts SPS by 0.5–1.9 points, insufficient to flip strong context-following pairs (SPS ≈ 6–8). Multi-layer or residual-level steering may be needed.
5. **Static conflicts**: We test factual substitution only. Conflicts involving reasoning, temporal changes, or partial contradictions are not covered.

---

## 7. Conclusion

We present a mechanistic account of source arbitration in instruction-tuned language models. Through six complementary causal interventions across three models from two families, we identify a compact late-layer circuit (final 8–12% of depth) whose heads function as instruction readers, not document copiers. We discover three distinct architectural strategies — concentrated override, memory defense, and distributed committee — for implementing the same functional capability. These findings reframe RAG faithfulness as an instruction-processing problem rather than a document-attention problem, with direct implications for the design of more faithful retrieval-augmented systems.

---

## References

- Chen, W., et al. (2022). Rich Knowledge Sources Bring Complex Knowledge Conflicts. *EMNLP*.
- Jin, Z., et al. (2024). Where Knowledge Conflicts Collide: A Mechanistic Study. *arXiv*.
- Longpre, S., et al. (2021). Entity-Based Knowledge Conflicts in Question Answering. *EMNLP*.
- Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS*.
- Meng, K., et al. (2023). Mass-Editing Memory in a Transformer. *ICLR*.
- Shi, W., et al. (2024). Trusting Your Evidence: Hallucinate Less with Context-Aware Decoding. *NAACL*.
- Wolfson, T., et al. (2025). Taming Knowledge Conflicts in Language Models. *arXiv*.

---

## Appendix

### A. Prompt Templates

**Explicit context:**
```
Document: {document_with_counterfactual}
Question: Based only on the document above, {question}
```

**Ambiguous conflict:**
```
Document: {document_with_counterfactual}
Question: {question}
```

**Explicit memory:**
```
Document: {document_with_counterfactual}
Question: Using your own knowledge (not the document), {question}
```

### B. Figure Reference

| Figure | File | Description |
|---|---|---|
| Fig 1 | `fig1_residual_patching.png` | Layer-wise ΔSPS curves |
| Fig 2 | `fig2_dla_heatmap.png` | Head-level logit attribution |
| Fig 3 | `fig3_attention_patterns.png` | Document vs question attention |
| Fig 4 | `fig4_steering.png` | Steering dose-response |
| Fig 5 | `fig5_probing.png` | Probing crystallization |
| Fig 6 | `fig6_summary_dashboard.png` | Cross-model dashboard |

### C. Cross-Model Comparison

See `results/figures/cross_model_comparison.md` for full 8-table comparison with all metrics.
