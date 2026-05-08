# Reproducibility Report

> **Paper**: "Instruction Readers, Not Document Copiers: Mechanistic Interpretability of Source Arbitration Across Instruction-Tuned Language Models"

---

## 1. Environment

### Hardware
- **Tested on**: Apple M4 Pro (36 GB unified memory)
- **Accelerator**: MPS (Metal Performance Shaders)
- **Estimated VRAM usage**: ~8 GB per model (peak during patching)
- **Alternative**: Any CUDA GPU with ≥16 GB VRAM (adjust `device` in scripts)

### Software
| Package | Version | Purpose |
|---|---|---|
| Python | 3.11.5 | Runtime |
| PyTorch | 2.11.0 | Model inference, hooks |
| Transformers | 5.8.0 | Model loading, tokenizer |
| NumPy | 2.4.4 | Numerical computation |
| SciPy | 1.17.1 | Statistical tests |
| scikit-learn | 1.8.0 | Linear probing |
| Matplotlib | 3.10.9 | Plotting |

### Setup
```bash
git clone <repo_url>
cd source_selection_project
python -m venv .venv
source .venv/bin/activate
pip install torch transformers numpy scipy scikit-learn matplotlib
```

---

## 2. Models

All models are loaded from HuggingFace Hub. No fine-tuning required.

| Model ID | Parameters | Layers | Heads |
|---|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | 3B | 36 | 16 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 28 | 12 |
| `google/gemma-2-2b-it` | 2B | 26 | 8 |

**Note**: Models are downloaded automatically on first run (~6 GB per model).

---

## 3. Full Pipeline Execution

### Step-by-step commands

Each script reads from the previous step's output. Run in order:

```bash
# Step 1: Generate dataset (501 facts × 4 regimes)
python scripts/00_generate_dataset.py

# Step 2: Validate token variants per model
python scripts/01_validate_token_variants.py

# Step 3: Compute baseline source preference scores
python scripts/02_compute_baselines.py

# Step 4: Build matched pairs for causal tracing
python scripts/03_build_matched_pairs.py

# Step 5: Lock model configs (architecture metadata)
python scripts/04_lock_model_configs.py

# Step 6: Residual stream patching (GPU, ~20 min per model)
python scripts/05_residual_stream_patching.py

# Step 7: Head patching (GPU, ~30 min per model)
python scripts/06_head_patching.py

# Step 8: Attention pattern analysis (GPU, ~10 min per model)
python scripts/08_attention_patterns.py

# Step 9: Direct Logit Attribution (GPU, ~15 min per model)
python scripts/09_dla.py

# Step 10: Linear probing (~5 min per model)
python scripts/10_probing.py

# Step 11: Activation steering (GPU, ~15 min per model)
python scripts/11_steering.py

# Step 12: Generate publication figures (no GPU)
python scripts/12_plots.py

# Step 13: Cross-model comparison tables (no GPU)
python scripts/13_cross_model.py

# Step 14: Cumulative ablation + bootstrap CIs (no GPU)
python scripts/14_cumulative_ablation_ci.py
```

### Estimated total runtime
| Phase | Time per model | GPU required |
|---|---|---|
| Dataset + baselines (Steps 1–4) | ~10 min | Yes (inference) |
| Residual patching (Step 6) | ~20 min | Yes |
| Head patching (Step 7) | ~30 min | Yes |
| Attention + DLA (Steps 8–9) | ~25 min | Yes |
| Probing + steering (Steps 10–11) | ~20 min | Yes |
| Plots + analysis (Steps 12–14) | ~5 min | No |
| **Total (3 models)** | **~5.5 hours** | |

---

## 4. Expected Output Structure

```
results/latest/
├── qwen2_5_3b_instruct/
│   ├── model_config.json
│   ├── baseline_summary.json
│   ├── residual_patching_explicit_context.json
│   ├── head_patching_explicit_context.json
│   ├── attention_patterns_explicit_context.json
│   ├── dla_explicit_context.json
│   ├── probing_ambiguous_conflict.json
│   ├── steering_explicit_context.json
│   ├── cumulative_ablation.json
│   └── bootstrap_cis.json
├── qwen2_5_1_5b_instruct/
│   └── (same structure)
├── gemma_2_2b_it/
│   └── (same structure)
results/figures/
├── fig1_residual_patching.{png,pdf}
├── fig2_dla_heatmap.{png,pdf}
├── fig3_attention_patterns.{png,pdf}
├── fig4_steering.{png,pdf}
├── fig5_probing.{png,pdf}
├── fig6_summary_dashboard.{png,pdf}
├── cross_model_comparison.md
└── figure_guide.md
```

---

## 5. Expected Metric Ranges

These ranges define what a successful reproduction should look like. Minor numerical differences are expected due to floating-point non-determinism.

### Table 2: Baseline context-win rates
| Model | Expected | Acceptable range |
|---|---|---|
| Qwen-3B (explicit ctx) | 86.2% | 80–92% |
| Qwen-1.5B (explicit ctx) | 95.8% | 90–100% |
| Gemma-2B (explicit ctx) | 98.4% | 94–100% |

### Table 3: Circuit localization
| Model | Peak layer | Expected ΔSPS | Acceptable |
|---|---|---|---|
| Qwen-3B | L33 (±1) | −13.3 | −8 to −18 |
| Qwen-1.5B | L27 (±1) | −11.4 | −7 to −16 |
| Gemma-2B | L25 (±1) | −14.1 | −9 to −19 |

### Table 4: DLA scores
| Model | Top DLA | Expected | Acceptable |
|---|---|---|---|
| Qwen-3B | L33.H9 | 0.324 | 0.15–0.50 |
| Qwen-1.5B | L27.H7 | 0.373 | 0.15–0.55 |
| Gemma-2B | L22.H5 | 0.171 | 0.08–0.30 |

### Key invariants (should hold exactly)
1. **Document attention = 0%** for all top-5 heads across all models
2. **Peak layer depth > 85%** for all models
3. **Qwen-1.5B steering direction is inverted** (ΔSPS > 0 at α < 0)
4. **Qwen-1.5B has ≥ 1 negative-DLA head**

---

## 6. Random Seeds

All scripts use `seed=42` by default. The following parameters affect stochasticity:
- Pair sampling: `max_pairs=40` (random subset of valid pairs)
- Probe training: sklearn `LogisticRegression(random_state=42)`
- Steering direction: computed from mean activations (deterministic given pairs)

To reproduce exact numbers, ensure the same seed and pair count.

---

## 7. Known Variability Sources

1. **MPS vs CUDA**: MPS and CUDA produce slightly different float32 results due to different reduction orders. Expect ±5% variation in continuous metrics.
2. **Model version**: HuggingFace model weights may be updated. Pin to specific commit hashes if exact reproduction is needed.
3. **Pair sampling**: With `max_pairs=40`, different seeds will select different fact subsets. The qualitative findings (late-layer localization, 0% doc attention, 1.5B inversion) are robust across seeds.
4. **Tokenizer updates**: Transformers library updates may change tokenization, affecting the valid-pairs funnel.

---

## 8. Verification Checklist

After running the full pipeline, verify:

- [ ] `baseline_summary.json` exists for all 3 models
- [ ] `residual_patching_explicit_context.json` shows peak at >85% depth
- [ ] `head_patching_explicit_context.json` has `top_heads` list
- [ ] `attention_patterns_explicit_context.json` shows doc_attn ≈ 0 for top heads
- [ ] `dla_explicit_context.json` shows positive DLA for top heads
- [ ] `steering_explicit_context.json` shows inverted direction for Qwen-1.5B
- [ ] `probing_*.json` shows accuracy > majority baseline for Gemma
- [ ] All 6 figures generated in `results/figures/`
- [ ] `cross_model_comparison.md` contains 8 tables
