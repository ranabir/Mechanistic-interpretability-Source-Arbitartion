# Instruction Readers, Not Document Copiers

### Mechanistic Interpretability of Source Arbitration Across Instruction-Tuned Language Models


When a retrieval-augmented language model encounters a document that contradicts its parametric knowledge, it must decide whether to follow the document or rely on memory. This repository contains the code, data, and manuscript for a mechanistic investigation of this **source-arbitration circuit** across three instruction-tuned models.

---

## Key Findings

<table>
<tr>
<td width="50%">

**🔬 The Instruction-Reader Finding**

Context-override heads do not read the document. They attend primarily to the question/instruction region (25–57%) while assigning negligible attention to document tokens—yet they substantially influence whether the model follows the document answer.

</td>
<td width="50%">

**🧠 Three Distinct Strategies**

| Strategy | Model | Mechanism |
|---|---|---|
| Concentrated Override | Qwen-3B | 3–5 high-DLA "star" heads |
| Memory Defense | Qwen-1.5B | Opposing heads that defend memory |
| Distributed Committee | Gemma-2B | Many heads with modest individual contributions |

</td>
</tr>
</table>

### Summary of Results

| Finding | Evidence |
|---|---|
| Late-layer localization | Context-override peaks at 92–96% depth across all models |
| Instruction reading | 0% document attention, 25–57% question attention |
| Memory-defense heads | Qwen-1.5B has 3 negative-DLA heads that write memory logits |
| Differential suppression | Gemma suppresses both tokens but memory more (net positive DLA) |
| Cumulative head-set controls | Top-10 heads produce 11–30× more effect than random late-layer heads |
| Inverted steering | Subtracting the "context direction" in Qwen-1.5B *increases* context preference |
| Probe crystallization | Gemma's linear probe accuracy rises 15.8pp at the circuit layer |

---

## Models Studied

| Model | Parameters | Layers | Heads | Family |
|---|---|---|---|---|
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3B | 36 | 16 | Qwen |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B | 28 | 12 | Qwen |
| [Gemma-2-2B-IT](https://huggingface.co/google/gemma-2-2b-it) | 2B | 26 | 8 | Google |

---

## Repository Structure

```
.
├── paper_ref_source_arb/          # LaTeX manuscript (ICML 2026 format)
│   ├── main.tex                   # Main paper source
│   ├── tables/                    # All tables (with 95% CIs)
│   ├── figures/                   # Publication figures (PDF + PNG)
│   └── bib/references.bib        # Bibliography
│
├── source_selection_project/      # Experimental code & results
│   ├── scripts/                   # Numbered pipeline scripts (00–14)
│   ├── src/source_selection/      # Reusable library modules
│   ├── data/                      # Generated datasets & matched pairs
│   ├── results/latest/            # Per-model JSON result artifacts
│   ├── results/figures/           # Generated plots + figure guide
│   ├── configs/                   # Model & experiment configs
│   ├── tests/                     # Unit tests
│   ├── paper_notes/               # Research log & draft notes
│   └── REPRODUCE.md               # Full reproducibility guide
│
└── .gitignore
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- GPU with ≥8 GB VRAM (MPS or CUDA)
- ~18 GB disk for model weights (downloaded automatically)

### Setup

```bash
git clone https://github.com/ranabir/Mechanistic-interpretability-Source-Arbitartion.git
cd Mechanistic-interpretability-Source-Arbitartion/source_selection_project

python -m venv .venv
source .venv/bin/activate
pip install torch transformers numpy scipy scikit-learn matplotlib
```

### Run the Full Pipeline

```bash
# 1. Dataset generation (501 facts × 4 regimes)
python scripts/00_generate_dataset.py

# 2. Token variant validation
python scripts/01_validate_token_variants.py

# 3. Baseline source preference scores
python scripts/02_compute_baselines.py

# 4. Construct matched pairs for causal tracing
python scripts/03_build_matched_pairs.py

# 5. Lock model architecture configs
python scripts/04_lock_model_configs.py

# 6. Residual stream patching (~20 min/model)
python scripts/05_residual_stream_patching.py

# 7. Head patching (~30 min/model)
python scripts/06_head_patching.py

# 8. Attention pattern decomposition (~10 min/model)
python scripts/08_attention_patterns.py

# 9. Direct Logit Attribution (~15 min/model)
python scripts/09_dla.py

# 10. Linear probing (~5 min/model)
python scripts/10_probing.py

# 11. Activation steering (~15 min/model)
python scripts/11_steering.py

# 12. Generate publication figures
python scripts/12_plots.py

# 13. Cross-model comparison
python scripts/13_cross_model.py

# 14. Cumulative head-set patching + bootstrap CIs
python scripts/14_cumulative_ablation_ci.py
```

**Total runtime**: ~5.5 hours for all 3 models on Apple M4 Pro.

---

## Methods Overview

```
                    ┌──────────────────────────────────────┐
                    │     501 Facts × 4 Prompt Regimes     │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │   Token-Variant-Aware Filtering      │
                    │   (Single-token answer validation)    │
                    └──────────────┬───────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
     ┌────────▼────────┐  ┌───────▼────────┐  ┌────────▼────────┐
     │   Qwen 2.5-3B   │  │  Qwen 2.5-1.5B │  │   Gemma-2-2B    │
     └────────┬────────┘  └───────┬────────┘  └────────┬────────┘
              │                   │                     │
              └───────────────────┼─────────────────────┘
                                  │
                    ┌─────────────▼─────────────────────┐
                    │    Six Causal Interventions        │
                    │                                    │
                    │  1. Residual stream patching       │
                    │  2. Attention head patching        │
                    │  3. Direct Logit Attribution       │
                    │  4. Attention decomposition        │
                    │  5. Linear probing                 │
                    │  6. Activation steering            │
                    └─────────────┬─────────────────────┘
                                  │
                    ┌─────────────▼─────────────────────┐
                    │  Cumulative Controls + Bootstrap   │
                    │  CIs + Cross-Model Comparison      │
                    └───────────────────────────────────┘
```

### Source Preference Score (SPS)

The core metric quantifies how strongly a model favors the context answer over its memorized answer:

```
SPS = max log p(v_ctx) − max log p(v_mem)
       v∈V_ctx           v∈V_mem
```

where V_ctx and V_mem are sets of valid single-token variants. SPS > 0 indicates context-adherence.

---

## Results at a Glance

### Circuit Localization

| Model | Peak Layer | Depth | Top Head ΔSPS | 95% CI |
|---|---|---|---|---|
| Qwen-3B | L33/36 | 91.7% | −0.25 | [−0.38, −0.12] |
| Qwen-1.5B | L27/28 | 96.4% | −0.12 | [−0.17, −0.07] |
| Gemma-2B | L25/26 | 96.2% | −0.25 | [−0.32, −0.19] |

### Cumulative Head-Set Patching

| Head Set | Qwen-3B | Qwen-1.5B | Gemma-2B |
|---|---|---|---|
| Top-10 candidates | −2.09 ± 0.34 | −0.46 ± 0.15 | −1.54 ± 0.17 |
| Random-3 (late layer) | −0.07 ± 0.22 | +0.00 ± 0.07 | −0.14 ± 0.09 |
| **Separation factor** | **30×** | **∞** | **11×** |

---

## Reproducibility

See [`source_selection_project/REPRODUCE.md`](source_selection_project/REPRODUCE.md) for:
- Exact environment and dependency versions
- Step-by-step pipeline commands
- Expected metric ranges for all key findings
- Random seed documentation
- Verification checklist

All experiments use `seed=42` with `max_pairs=40–60` matched pairs per causal intervention.

---

## Paper

The LaTeX source for the ICML 2026 submission is in [`paper_ref_source_arb/`](paper_ref_source_arb/). To compile:

```bash
cd paper_ref_source_arb
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or upload the entire `paper_ref_source_arb/` directory to [Overleaf](https://www.overleaf.com/).

---

## Environment

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| PyTorch | 2.11+ |
| Transformers | 5.8+ |
| NumPy | 2.4+ |
| SciPy | 1.17+ |
| scikit-learn | 1.8+ |
| Matplotlib | 3.10+ |

Tested on **Apple M4 Pro** (36 GB unified memory) with MPS acceleration.
Compatible with CUDA GPUs (≥16 GB VRAM).

---

## Citation

```bibtex
@inproceedings{anonymous2026instruction,
  title={Instruction Readers, Not Document Copiers: Mechanistic Interpretability of Source Arbitration Across Instruction-Tuned Language Models},
  author={Anonymous},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML), Mechanistic Interpretability Workshop},
  year={2026}
}
```

---

## License

This project is released for academic research purposes. See individual model licenses for model weight usage terms.
