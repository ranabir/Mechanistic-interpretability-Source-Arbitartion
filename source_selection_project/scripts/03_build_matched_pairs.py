"""
Phase 6: Matched Pair Construction.

For each fact in the single-token causal tracing subset, we build a matched pair:
  clean_prompt:   no document → model should answer from memory
  corrupt_prompt: document with wrong fact → model should follow context

A pair is VALID for causal tracing if:
  1. clean   → memory_answer wins    (score ≤ -MIN_MARGIN)
  2. corrupt → context_answer wins   (score ≥ +MIN_MARGIN)

The margin threshold ensures both sides have clear, non-ambiguous preference.
Valid pairs are saved to:
  data/causal_tracing_pairs/{model}/pairs_{regime}.jsonl
  data/causal_tracing_pairs/{model}/pairs_summary.json

The funnel statistics (how many facts survive) are saved to:
  data/manifests/pair_construction_funnel.json
"""
import sys, os, json, datetime, random
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from source_selection.data import load_jsonl, save_json, save_jsonl
from source_selection.model_loading import load_model_and_tokenizer, get_device
from source_selection.scoring import get_answer_position_logits, compute_source_score
from source_selection.token_variants import generate_variants
from source_selection.prompt_templates import CLEAN_PROMPT_TEMPLATES

MODELS = {
    "qwen2_5_3b_instruct":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_5_1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma_2_2b_it":         "google/gemma-2-2b-it",
}

# We build pairs only from explicit_context prompts (the core intervention regime)
# and ambiguous (for the arbitration analysis).
PAIR_REGIMES = ["explicit_context", "ambiguous_conflict"]

SEED = 42
MIN_MARGIN = 0.5       # minimum |score| for a prompt to pass
random.seed(SEED)


def build_pairs_for_model(model_name, model_id, base_dir):
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    res_dir      = os.path.join(base_dir, "results", "latest", model_name)
    filtered_dir = os.path.join(base_dir, "data", "filtered", model_name)
    pairs_dir    = os.path.join(base_dir, "data", "causal_tracing_pairs", model_name)
    os.makedirs(pairs_dir, exist_ok=True)

    try:
        model, tokenizer, device, dtype = load_model_and_tokenizer(model_id)
    except Exception as e:
        print(f"  [SKIP] {model_name}: {e}")
        return None

    # Load all facts for clean prompt generation
    facts_raw = load_jsonl(os.path.join(base_dir, "data", "raw", "facts.jsonl"))
    facts_by_id = {f["id"]: f for f in facts_raw}

    model_funnel = {"model": model_name, "regimes": {}}

    for regime in PAIR_REGIMES:
        filename = f"single_token_{regime}_prompts.jsonl"
        corrupt_prompts = load_jsonl(os.path.join(filtered_dir, filename))
        if not corrupt_prompts:
            print(f"  [SKIP] No single-token prompts found for {regime}")
            continue

        # Load pre-computed baseline scores
        baseline_file = os.path.join(res_dir, f"baseline_source_preference_{regime}.json")
        corrupt_scores_by_pid = {}
        if os.path.exists(baseline_file):
            baseline_data = json.load(open(baseline_file))
            for r in baseline_data.get("single_token_results", []):
                corrupt_scores_by_pid[r["prompt_id"]] = r
        
        valid_pairs = []
        excluded = []
        funnel = {
            "total_corrupt_prompts": len(corrupt_prompts),
            "corrupt_passes_margin": 0,
            "clean_generated": 0,
            "clean_passes_margin": 0,
            "valid_pairs": 0,
            "excluded_corrupt_weak": 0,
            "excluded_clean_weak": 0,
            "excluded_clean_context_won": 0,
        }

        for p in tqdm(corrupt_prompts, desc=f"  {regime}", leave=False):
            fact = facts_by_id.get(p["fact_id"])
            if not fact:
                continue

            # --- Step 1: Check corrupt prompt passes margin ---
            pre_score = corrupt_scores_by_pid.get(p["prompt_id"])
            if pre_score and pre_score["source_score"] is not None:
                corrupt_score = pre_score["source_score"]
            else:
                # Re-score if not cached
                logits = get_answer_position_logits(model, tokenizer, p["prompt"], device)
                ctx_vars = p.get("context_variants_all") or generate_variants(p["context_answer"])
                mem_vars = p.get("memory_variants_all") or generate_variants(p["memory_answer"])
                info = compute_source_score(logits, tokenizer, ctx_vars, mem_vars, single_token_only=True)
                corrupt_score = info["source_score"]

            if corrupt_score is None or corrupt_score < MIN_MARGIN:
                # Context didn't win strongly enough on corrupt
                funnel["excluded_corrupt_weak"] += 1
                excluded.append({
                    "fact_id": p["fact_id"], "reason": "corrupt_weak",
                    "corrupt_score": corrupt_score, "threshold": MIN_MARGIN
                })
                continue
            funnel["corrupt_passes_margin"] += 1

            # --- Step 2: Build and score the clean prompt ---
            template = random.choice(CLEAN_PROMPT_TEMPLATES)
            clean_prompt_text = template.format(
                relation=fact["relation"], subject=fact["subject"]
            )
            funnel["clean_generated"] += 1

            clean_logits = get_answer_position_logits(model, tokenizer, clean_prompt_text, device)
            ctx_vars = p.get("valid_context_single_token_variants") or generate_variants(p["context_answer"])
            mem_vars = p.get("valid_memory_single_token_variants") or generate_variants(p["memory_answer"])
            clean_info = compute_source_score(clean_logits, tokenizer, ctx_vars, mem_vars, single_token_only=True)
            clean_score = clean_info["source_score"]

            if clean_score is None:
                funnel["excluded_clean_weak"] += 1
                excluded.append({"fact_id": p["fact_id"], "reason": "clean_no_score"})
                continue

            # --- Step 3: Validate memory wins on clean ---
            if clean_score >= -MIN_MARGIN:
                # Memory didn't win clearly on clean (context might have leaked, or flat)
                if clean_score > 0:
                    funnel["excluded_clean_context_won"] += 1
                    reason = "clean_context_won"
                else:
                    funnel["excluded_clean_weak"] += 1
                    reason = "clean_margin_weak"
                excluded.append({
                    "fact_id": p["fact_id"], "reason": reason,
                    "clean_score": clean_score, "corrupt_score": corrupt_score
                })
                continue

            # --- Valid pair ---
            funnel["clean_passes_margin"] += 1
            funnel["valid_pairs"] += 1

            valid_pairs.append({
                "pair_id":            f"pair_{p['prompt_id']}",
                "fact_id":            p["fact_id"],
                "model":              model_name,
                "regime":             regime,
                "category":           p.get("category", fact.get("category", "")),
                "difficulty":         p.get("difficulty", fact.get("difficulty", "")),

                # Fact info
                "subject":            fact["subject"],
                "relation":           fact["relation"],
                "memory_answer":      p["memory_answer"],
                "context_answer":     p["context_answer"],
                "conflict_sentence":  fact["conflict_sentence"],

                # Clean (no-doc) prompt
                "clean_prompt":       clean_prompt_text,
                "clean_score":        round(clean_score, 6),
                "clean_top5":         clean_info["top_5_tokens"],

                # Corrupt (with-doc) prompt
                "corrupt_prompt":     p["prompt"],
                "corrupt_score":      round(corrupt_score, 6),
                "corrupt_top5":       pre_score.get("top_5_tokens") if pre_score else [],

                # Token info
                "valid_context_single_token_variants": ctx_vars,
                "valid_memory_single_token_variants":  mem_vars,

                # Pair margin summary
                "clean_to_corrupt_flip": round(corrupt_score - clean_score, 6),
            })

        save_jsonl(os.path.join(pairs_dir, f"pairs_{regime}.jsonl"), valid_pairs)
        save_jsonl(os.path.join(pairs_dir, f"excluded_{regime}.jsonl"), excluded)

        model_funnel["regimes"][regime] = funnel
        pct = 100 * funnel["valid_pairs"] / max(1, funnel["total_corrupt_prompts"])
        print(f"  [{regime}] {funnel['total_corrupt_prompts']} → "
              f"{funnel['valid_pairs']} valid pairs ({pct:.1f}%)")
        print(f"    corrupt weak: {funnel['excluded_corrupt_weak']}, "
              f"clean weak: {funnel['excluded_clean_weak']}, "
              f"clean ctx won: {funnel['excluded_clean_context_won']}")

    save_json(os.path.join(pairs_dir, "pairs_summary.json"), {
        "model": model_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "min_margin": MIN_MARGIN,
        "funnel": model_funnel,
    })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_funnel


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    all_funnels = {}
    for model_name, model_id in MODELS.items():
        result = build_pairs_for_model(model_name, model_id, base_dir)
        if result:
            all_funnels[model_name] = result

    # Save global funnel manifest
    save_json(
        os.path.join(base_dir, "data", "manifests", "pair_construction_funnel.json"),
        {"timestamp": datetime.datetime.now().isoformat(),
         "min_margin": MIN_MARGIN, "models": all_funnels}
    )
    print("\nMatched pair construction complete.")

if __name__ == "__main__":
    main()
