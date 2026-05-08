# Project Clean Start

**Timestamp:** 2026-05-07 15:15:00 UTC (approx)

## Action Taken
The project was completely wiped and restarted to establish a fully reproducible causal tracing pipeline. The goal is to study source selection under context-memory conflict across three models (Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct, google/gemma-2-2b-it) and to address reviewer concerns surrounding novelty, survivorship bias, and instruction-conditioned routing.

## Backed Up
- The entire `Source-Arbitration Mech Interp` previous active directory was backed up to `../source_selection_backup_*.tar.gz`.

## Preserved
- `legacy_latex_reference/`: A copy of the previous LaTeX directory for style and bibliography reference ONLY. 
- All previous code, result JSONs, plots, and tables were cleared out of the active path to ensure NO old data leaks into the new experiments.

## New Structure
- Established `source_selection_project/` containing empty scaffolding for `configs/`, `data/`, `docs/`, `scripts/`, `src/`, `tests/`, and the new LaTeX output directory `paper/icml_mi_source_selection/`.
