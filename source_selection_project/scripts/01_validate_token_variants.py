"""
Token variant validation script.
For each model × regime, determines which prompts have single-token answers
(suitable for causal tracing) and saves both:
  - single_token subset  -> data/filtered/{model}/single_token_{regime}.jsonl
  - multi_token subset   -> data/filtered/{model}/multi_token_{regime}.jsonl

Multi-token examples are NOT discarded. They are used for:
  - naturalistic RAG evaluation
  - broader ablation behavioral analysis
  - survivorship bias reporting
"""
import sys, os, json, csv, datetime
from collections import Counter
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from source_selection.data import load_jsonl, save_json, save_jsonl
from source_selection.token_variants import generate_variants, filter_single_token_variants

MODELS = {
    "qwen2_5_3b_instruct":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_5_1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma_2_2b_it":         "google/gemma-2-2b-it",
}

REGIMES = {
    "explicit_context":  "explicit_context_prompts.jsonl",
    "explicit_memory":   "explicit_memory_prompts.jsonl",
    "ambiguous_conflict":"ambiguous_conflict_prompts.jsonl",
    "naturalistic_rag":  "naturalistic_rag_prompts.jsonl",
}

def load_tokenizer(model_name, model_id):
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        print(f"  [OK] Loaded tokenizer: {model_id}")
        return tok
    except Exception as e:
        if "gated" in str(e).lower() or "token" in str(e).lower():
            print(f"  [SKIP] {model_name}: requires HuggingFace login.")
            print(f"         Run: huggingface-cli login")
            print(f"         Then accept the license at: https://huggingface.co/{model_id}")
        else:
            print(f"  [SKIP] {model_name}: {e}")
        return None

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    res_dir_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "latest"))
    man_dir = os.path.join(data_dir, "manifests")
    os.makedirs(man_dir, exist_ok=True)

    all_excluded = []
    funnel_rows = []

    for model_name, model_id in MODELS.items():
        print(f"\n=== {model_name} ===")
        tokenizer = load_tokenizer(model_name, model_id)
        if tokenizer is None:
            continue

        res_dir = os.path.join(res_dir_base, model_name)
        filtered_dir = os.path.join(data_dir, "filtered", model_name)
        os.makedirs(res_dir, exist_ok=True)
        os.makedirs(filtered_dir, exist_ok=True)

        for regime, filename in REGIMES.items():
            prompts = load_jsonl(os.path.join(data_dir, "generated", filename))
            if not prompts:
                print(f"  [WARN] No prompts found for {regime}")
                continue

            single_token_prompts = []
            multi_token_prompts = []

            cat_before = Counter()
            cat_after_single = Counter()
            cat_after_multi = Counter()
            ans_len_before = Counter()
            ans_len_after_single = Counter()
            excluded_sample = []

            stats = {
                "model": model_name, "regime": regime,
                "total_prompts": len(prompts),
                "single_token_both_valid": 0,
                "multi_token_at_least_one": 0,
                "excluded_no_valid_variants_either_side": 0,
                "excluded_no_context_single_token": 0,
                "excluded_no_memory_single_token": 0,
                "excluded_neither_single_token": 0,
                "category_distribution_before": {},
                "category_distribution_after_single": {},
                "category_distribution_after_multi": {},
                "answer_length_distribution_before": {},
                "answer_length_distribution_after_single": {},
                "excluded_examples_sample": [],
            }

            for p in prompts:
                cat = p.get("category", "unknown")
                ctx_ans = p["context_answer"]
                mem_ans = p["memory_answer"]

                cat_before[cat] += 1
                ans_len_before[len(ctx_ans)] += 1
                ans_len_before[len(mem_ans)] += 1

                ctx_vars = generate_variants(ctx_ans)
                mem_vars = generate_variants(mem_ans)
                valid_ctx = filter_single_token_variants(tokenizer, ctx_vars)
                valid_mem = filter_single_token_variants(tokenizer, mem_vars)

                p_aug = {**p,
                    "context_variants_all": ctx_vars,
                    "memory_variants_all": mem_vars,
                    "valid_context_single_token_variants": valid_ctx,
                    "valid_memory_single_token_variants": valid_mem,
                    "is_single_token_valid": len(valid_ctx) > 0 and len(valid_mem) > 0,
                }

                if len(valid_ctx) > 0 and len(valid_mem) > 0:
                    # Both sides have single-token variants -> causal tracing
                    single_token_prompts.append(p_aug)
                    cat_after_single[cat] += 1
                    ans_len_after_single[len(ctx_ans)] += 1
                    stats["single_token_both_valid"] += 1
                else:
                    # At least one side is multi-token -> behavioral eval only
                    # But still keep it for naturalistic RAG eval
                    multi_token_prompts.append(p_aug)
                    cat_after_multi[cat] += 1
                    stats["multi_token_at_least_one"] += 1

                    reason = "neither" if len(valid_ctx) == 0 and len(valid_mem) == 0 \
                        else ("no_context_single_token" if len(valid_ctx) == 0 else "no_memory_single_token")
                    if reason == "neither":
                        stats["excluded_neither_single_token"] += 1
                    elif reason == "no_context_single_token":
                        stats["excluded_no_context_single_token"] += 1
                    else:
                        stats["excluded_no_memory_single_token"] += 1

                    if len(excluded_sample) < 10:
                        excluded_sample.append({
                            "prompt_id": p["prompt_id"], "fact_id": p["fact_id"],
                            "category": cat, "context_answer": ctx_ans,
                            "memory_answer": mem_ans, "reason": reason,
                        })
                    all_excluded.append({
                        "model": model_name, "regime": regime,
                        "prompt_id": p["prompt_id"], "fact_id": p["fact_id"],
                        "category": cat, "context_answer": ctx_ans,
                        "memory_answer": mem_ans, "reason": reason,
                    })

            stats["category_distribution_before"] = dict(cat_before)
            stats["category_distribution_after_single"] = dict(cat_after_single)
            stats["category_distribution_after_multi"] = dict(cat_after_multi)
            stats["answer_length_distribution_before"] = dict(ans_len_before)
            stats["answer_length_distribution_after_single"] = dict(ans_len_after_single)
            stats["excluded_examples_sample"] = excluded_sample

            # Save JSON stats
            save_json(os.path.join(res_dir, f"token_variant_validation_{regime}.json"), stats)

            # Save both subsets
            save_jsonl(os.path.join(filtered_dir, f"single_token_{filename}"), single_token_prompts)
            save_jsonl(os.path.join(filtered_dir, f"multi_token_{filename}"), multi_token_prompts)

            pct = 100 * stats["single_token_both_valid"] / max(1, len(prompts))
            print(f"  {regime}: {len(prompts)} prompts -> "
                  f"{stats['single_token_both_valid']} single-token ({pct:.1f}%), "
                  f"{stats['multi_token_at_least_one']} multi-token")

            funnel_rows.append({
                "model": model_name, "regime": regime,
                "total": len(prompts),
                "single_token": stats["single_token_both_valid"],
                "multi_token": stats["multi_token_at_least_one"],
                "pct_single": round(pct, 1),
            })

    # Save excluded examples summary
    if all_excluded:
        csv_path = os.path.join(man_dir, "excluded_examples_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_excluded[0].keys())
            writer.writeheader()
            writer.writerows(all_excluded)
        print(f"\nSaved {len(all_excluded)} excluded examples -> {csv_path}")

    # Save funnel summary
    if funnel_rows:
        funnel_path = os.path.join(man_dir, "dataset_funnel_summary.csv")
        with open(funnel_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=funnel_rows[0].keys())
            writer.writeheader()
            writer.writerows(funnel_rows)
        print(f"Saved funnel summary -> {funnel_path}")

    print("\nToken variant validation complete.")

if __name__ == "__main__":
    main()
