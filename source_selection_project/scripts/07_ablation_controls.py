"""
Phase 10: Ablation Controls.

Three ablation conditions per model:
  1. ABLATE_TOP: Zero out the top-3 context-override heads → SPS should drop
  2. ABLATE_RANDOM_MID: Zero out 3 random mid-layer heads → SPS should NOT drop
  3. ABLATE_OTHER_LATE: Zero out 3 other late-layer heads (not top) → SPS should NOT drop

This validates that the identified heads are NECESSARY and SPECIFIC,
not just any late-layer head or generic disruption.

Output: results/latest/{model}/ablation_controls_{regime}.json
"""
import sys, os, json, datetime, random
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from source_selection.data import load_jsonl, save_json
from source_selection.model_loading import load_model_and_tokenizer
from source_selection.scoring import compute_source_score
from source_selection.token_variants import generate_variants

MODELS = {
    "qwen2_5_3b_instruct":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_5_1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma_2_2b_it":         "google/gemma-2-2b-it",
}

REGIME = "explicit_context"
MAX_PAIRS = 60
SEED = 42


def get_ablation_groups(model_name, res_dir):
    """
    Build the three ablation groups from head patching results.
    Returns dict with keys: top, random_mid, other_late
    Each value is a list of (layer, head) tuples.
    """
    cfg = json.load(open(os.path.join(res_dir, "model_config.json")))
    hp = json.load(open(os.path.join(res_dir, f"head_patching_{REGIME}.json")))

    num_layers = cfg["num_layers"]
    num_heads = cfg["num_heads"]
    top_heads = [(h["layer"], h["head"]) for h in hp["top_heads"][:3]]

    # Mid-layer random heads (layers 20-50% depth)
    mid_start = num_layers // 5
    mid_end = num_layers // 2
    random.seed(SEED + 1)
    mid_candidates = [(l, h) for l in range(mid_start, mid_end) for h in range(num_heads)]
    random_mid = random.sample(mid_candidates, 3)

    # Other late-layer heads (same layers as top heads, but different heads)
    top_layers = set(l for l, h in top_heads)
    top_set = set(top_heads)
    late_candidates = [(l, h) for l in top_layers for h in range(num_heads)
                       if (l, h) not in top_set]
    random.seed(SEED + 2)
    other_late = random.sample(late_candidates, min(3, len(late_candidates)))

    return {
        "top": top_heads,
        "random_mid": random_mid,
        "other_late": other_late,
    }, cfg


def ablate_heads_and_score(model, tokenizer, device, pairs, heads_to_zero, num_heads, head_dim):
    """
    Run all pairs with specified heads zeroed out.
    Returns list of SPS scores and ctx_win count.
    """
    layers_obj = model.model.layers

    def make_zero_hook(layer_idx, head_indices):
        """Create a hook that zeros specified heads in the attention output."""
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            h_new = h.clone()
            for head_idx in head_indices:
                start = head_idx * head_dim
                end = start + head_dim
                h_new[0, -1, start:end] = 0.0
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new
        return hook

    # Group heads by layer
    layer_heads = {}
    for l, h in heads_to_zero:
        layer_heads.setdefault(l, []).append(h)

    scores = []
    ctx_wins = 0

    for pair in pairs:
        ctx_vars = pair.get("valid_context_single_token_variants") or \
                   generate_variants(pair["context_answer"])
        mem_vars = pair.get("valid_memory_single_token_variants") or \
                   generate_variants(pair["memory_answer"])

        corrupt_ids = tokenizer(pair["corrupt_prompt"], return_tensors="pt").input_ids.to(device)

        # Register hooks
        handles = []
        for l, head_indices in layer_heads.items():
            attn = layers_obj[l].self_attn
            handles.append(attn.register_forward_hook(make_zero_hook(l, head_indices)))

        with torch.no_grad():
            outputs = model(corrupt_ids)

        for hh in handles:
            hh.remove()

        logits = outputs.logits[0, -1, :].float().cpu()
        info = compute_source_score(logits, tokenizer, ctx_vars, mem_vars, single_token_only=True)
        s = info["source_score"]
        if s is not None:
            scores.append(s)
            if s > 0:
                ctx_wins += 1

    return scores, ctx_wins


def run_ablation_for_model(model_name, model_id, base_dir):
    print(f"\n{'='*60}")
    print(f"  Phase 10: Ablation Controls — {model_name}")
    print(f"{'='*60}")

    res_dir = os.path.join(base_dir, "results", "latest", model_name)
    pairs_dir = os.path.join(base_dir, "data", "causal_tracing_pairs", model_name)

    # Skip if done
    out_path = os.path.join(res_dir, f"ablation_controls_{REGIME}.json")
    if os.path.exists(out_path):
        d = json.load(open(out_path))
        print(f"  [SKIP] Already done.")
        for cond, r in d["conditions"].items():
            print(f"    {cond}: ctx_win={r['ctx_win_pct']:.1f}%, mean_sps={r['mean_sps']:.2f}")
        return

    groups, cfg = get_ablation_groups(model_name, res_dir)
    print(f"  Ablation groups:")
    for name, heads in groups.items():
        print(f"    {name}: {heads}")

    model, tokenizer, device, dtype = load_model_and_tokenizer(model_id)

    pairs = load_jsonl(os.path.join(pairs_dir, f"pairs_{REGIME}.jsonl"))
    random.seed(SEED)
    if MAX_PAIRS and len(pairs) > MAX_PAIRS:
        pairs = random.sample(pairs, MAX_PAIRS)

    # Baseline (no ablation)
    print(f"  Running baseline (no ablation)...")
    baseline_scores = []
    baseline_ctx = 0
    for pair in tqdm(pairs, desc="  baseline", leave=False):
        ctx_vars = pair.get("valid_context_single_token_variants") or generate_variants(pair["context_answer"])
        mem_vars = pair.get("valid_memory_single_token_variants") or generate_variants(pair["memory_answer"])
        corrupt_ids = tokenizer(pair["corrupt_prompt"], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(corrupt_ids)
        logits = out.logits[0, -1, :].float().cpu()
        info = compute_source_score(logits, tokenizer, ctx_vars, mem_vars, single_token_only=True)
        if info["source_score"] is not None:
            baseline_scores.append(info["source_score"])
            if info["source_score"] > 0:
                baseline_ctx += 1

    results = {
        "baseline": {
            "heads": [],
            "n": len(baseline_scores),
            "ctx_win": baseline_ctx,
            "ctx_win_pct": round(100 * baseline_ctx / max(1, len(baseline_scores)), 1),
            "mean_sps": round(sum(baseline_scores) / max(1, len(baseline_scores)), 4),
        }
    }
    print(f"  Baseline: ctx_win={results['baseline']['ctx_win_pct']:.1f}%, "
          f"mean_sps={results['baseline']['mean_sps']:.2f}")

    # Run each ablation condition
    for cond_name, heads in groups.items():
        print(f"  Running ablation: {cond_name} → zeroing {heads}...")
        scores, ctx_wins = ablate_heads_and_score(
            model, tokenizer, device, pairs, heads,
            cfg["num_heads"], cfg["head_dim"]
        )
        results[cond_name] = {
            "heads": [{"layer": l, "head": h} for l, h in heads],
            "n": len(scores),
            "ctx_win": ctx_wins,
            "ctx_win_pct": round(100 * ctx_wins / max(1, len(scores)), 1),
            "mean_sps": round(sum(scores) / max(1, len(scores)), 4),
            "delta_ctx_win_pct": round(
                100 * ctx_wins / max(1, len(scores)) - results["baseline"]["ctx_win_pct"], 1),
            "delta_mean_sps": round(
                sum(scores) / max(1, len(scores)) - results["baseline"]["mean_sps"], 4),
        }
        print(f"    ctx_win={results[cond_name]['ctx_win_pct']:.1f}% "
              f"(Δ={results[cond_name]['delta_ctx_win_pct']:+.1f}%), "
              f"mean_sps={results[cond_name]['mean_sps']:.2f} "
              f"(Δ={results[cond_name]['delta_mean_sps']:+.2f})")

    save_json(out_path, {
        "model": model_name,
        "regime": REGIME,
        "num_pairs": len(pairs),
        "seed": SEED,
        "timestamp": datetime.datetime.now().isoformat(),
        "conditions": results,
    })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for model_name, model_id in MODELS.items():
        run_ablation_for_model(model_name, model_id, base_dir)
    print("\nAblation controls complete.")

if __name__ == "__main__":
    main()
