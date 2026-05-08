"""
Phase 5: Baseline Source Preference Scoring.

For each model × regime:
  - Runs both the single-token causal-tracing subset
  - Runs the multi-token naturalistic eval subset (first-subword approximation)
  - Computes source preference score per prompt
  - Saves full per-prompt results + summary statistics

Outputs:
  results/latest/{model}/baseline_source_preference_{regime}.json
  results/latest/{model}/baseline_summary.json
  results/latest/{model}/run_metadata.json
"""
import sys, os, json, datetime, random
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from source_selection.data import load_jsonl, save_json
from source_selection.model_loading import load_model_and_tokenizer, save_run_metadata, get_device
from source_selection.scoring import get_answer_position_logits, compute_source_score
from source_selection.token_variants import generate_variants

MODELS = {
    "qwen2_5_3b_instruct":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_5_1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma_2_2b_it":         "google/gemma-2-2b-it",
}

REGIMES = [
    "explicit_context",
    "explicit_memory",
    "ambiguous_conflict",
    "naturalistic_rag",
]

SEED = 42
TOP_K_FILTER = 5          # answer must be in top-K predictions to "pass" (config)
MIN_ABS_MARGIN = 0.5      # minimum |score| for matched pair eligibility (config)


def run_baselines_for_model(model_name, model_id, base_dir):
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    res_dir = os.path.join(base_dir, "results", "latest", model_name)
    filtered_dir = os.path.join(base_dir, "data", "filtered", model_name)
    os.makedirs(res_dir, exist_ok=True)

    try:
        model, tokenizer, device, dtype = load_model_and_tokenizer(model_id)
    except Exception as e:
        print(f"  [SKIP] Could not load {model_name}: {e}")
        return

    save_run_metadata(
        model_id=model_id, device=device, dtype=dtype,
        config_file="experiment_defaults.yaml", seed=SEED,
        out_path=os.path.join(res_dir, "run_metadata.json")
    )

    all_summaries = {}

    for regime in REGIMES:
        # ---- single-token subset ----
        single_file = os.path.join(filtered_dir, f"single_token_{regime}_prompts.jsonl")
        multi_file  = os.path.join(filtered_dir, f"multi_token_{regime}_prompts.jsonl")

        results_st = score_prompts(
            model, tokenizer, device,
            load_jsonl(single_file), regime, model_name,
            single_token_only=True, subset="single_token"
        )
        results_mt = score_prompts(
            model, tokenizer, device,
            load_jsonl(multi_file), regime, model_name,
            single_token_only=False, subset="multi_token"
        )

        combined = results_st + results_mt
        save_json(
            os.path.join(res_dir, f"baseline_source_preference_{regime}.json"),
            {"model": model_name, "regime": regime,
             "single_token_results": results_st,
             "multi_token_results": results_mt}
        )

        # Summary statistics
        def summarize(results, label):
            valid = [r for r in results if r["source_score"] is not None]
            if not valid:
                return {"n": 0, "label": label}
            scores = [r["source_score"] for r in valid]
            n_ctx  = sum(1 for r in valid if r["predicted_source"] == "context")
            n_mem  = sum(1 for r in valid if r["predicted_source"] == "memory")
            margins= [abs(s) for s in scores]
            return {
                "label": label, "n": len(valid),
                "context_win_rate": round(n_ctx / len(valid), 4),
                "memory_win_rate":  round(n_mem / len(valid), 4),
                "mean_score": round(sum(scores)/len(scores), 4),
                "mean_abs_margin": round(sum(margins)/len(margins), 4),
                "passes_margin_filter": sum(1 for m in margins if m >= MIN_ABS_MARGIN),
            }

        all_summaries[regime] = {
            "single_token": summarize(results_st, "single_token"),
            "multi_token":  summarize(results_mt, "multi_token"),
        }

        st = all_summaries[regime]["single_token"]
        mt = all_summaries[regime]["multi_token"]
        print(f"  [{regime}]")
        if st["n"] > 0:
            print(f"    single-token: n={st['n']}, ctx_win={st['context_win_rate']:.2%}, "
                  f"mem_win={st['memory_win_rate']:.2%}, mean_score={st['mean_score']:.3f}")
        if mt["n"] > 0:
            print(f"    multi-token:  n={mt['n']}, ctx_win={mt['context_win_rate']:.2%}, "
                  f"mem_win={mt['memory_win_rate']:.2%}, mean_score={mt['mean_score']:.3f}")

    save_json(os.path.join(res_dir, "baseline_summary.json"), {
        "model": model_name, "model_id": model_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "top_k_filter": TOP_K_FILTER,
        "min_abs_margin": MIN_ABS_MARGIN,
        "seed": SEED,
        "regimes": all_summaries,
    })

    # Free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Saved baseline results -> {res_dir}")


def score_prompts(model, tokenizer, device, prompts, regime, model_name,
                  single_token_only, subset):
    results = []
    if not prompts:
        return results

    for p in tqdm(prompts, desc=f"  {subset}/{regime}", leave=False):
        try:
            logits = get_answer_position_logits(model, tokenizer, p["prompt"], device)
        except Exception as e:
            results.append({**p, "source_score": None, "error": str(e)})
            continue

        # Use pre-computed variants if available, else generate
        ctx_vars = p.get("context_variants_all") or generate_variants(p["context_answer"])
        mem_vars = p.get("memory_variants_all") or generate_variants(p["memory_answer"])

        score_info = compute_source_score(
            logits, tokenizer, ctx_vars, mem_vars,
            single_token_only=single_token_only
        )

        top5_ids = {t["token_id"] for t in score_info.get("top_5_tokens", [])}
        ctx_ids = set(score_info.get("context_token_ids", []))
        mem_ids = set(score_info.get("memory_token_ids", []))
        passes_topk = bool(ctx_ids & top5_ids or mem_ids & top5_ids)
        passes_margin = (
            score_info["source_score"] is not None and
            abs(score_info["source_score"]) >= MIN_ABS_MARGIN
        )

        expected = "context" if "context" in regime else \
                   ("memory" if "memory" in regime else "none")

        results.append({
            "prompt_id":     p["prompt_id"],
            "fact_id":       p["fact_id"],
            "model":         model_name,
            "regime":        regime,
            "subset":        subset,
            "category":      p.get("category", ""),
            "context_answer": p["context_answer"],
            "memory_answer":  p["memory_answer"],
            "source_score":   score_info["source_score"],
            "context_max_logit": score_info["context_max_logit"],
            "memory_max_logit":  score_info["memory_max_logit"],
            "predicted_source":  score_info["predicted_source"],
            "expected_source":   expected,
            "top_5_tokens":      score_info["top_5_tokens"],
            "passes_top_k_filter": passes_topk,
            "passes_margin_filter": passes_margin,
        })

    return results


def main():
    random.seed(SEED)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for model_name, model_id in MODELS.items():
        run_baselines_for_model(model_name, model_id, base_dir)
    print("\nBaseline scoring complete.")

if __name__ == "__main__":
    main()
