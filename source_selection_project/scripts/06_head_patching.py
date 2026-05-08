"""
Phase 9: Attention Head Patching.

For each model, using the top-5 layers from Phase 8:
  - For each matched pair:
    1. Capture clean attention output at target layer
    2. For each head in that layer, patch the clean head into the corrupt run
    3. Measure ΔSPS per head
  - Average across pairs to rank heads by causal importance

Outputs:
  results/latest/{model}/head_patching_{regime}.json
    Per-layer, per-head average ΔSPS + top heads ranked

This identifies the specific "context-override heads."
"""
import sys, os, json, datetime, random
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from source_selection.data import load_jsonl, save_json
from source_selection.model_loading import load_model_and_tokenizer
from source_selection.scoring import compute_source_score
from source_selection.token_variants import generate_variants
from source_selection.patching import HeadPatcher

MODELS = {
    "qwen2_5_3b_instruct":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_5_1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma_2_2b_it":         "google/gemma-2-2b-it",
}

PAIR_REGIMES = ["explicit_context", "ambiguous_conflict"]
MAX_PAIRS = 40   # fewer pairs since we iterate heads × layers
TOP_N_LAYERS = 5 # only patch heads in the top-N layers from Phase 8
SEED = 42


def run_head_patching(model_name, model_id, base_dir):
    print(f"\n{'='*60}")
    print(f"  Phase 9: Head Patching — {model_name}")
    print(f"{'='*60}")

    res_dir   = os.path.join(base_dir, "results", "latest", model_name)
    pairs_dir = os.path.join(base_dir, "data", "causal_tracing_pairs", model_name)

    # Load architecture
    cfg = json.load(open(os.path.join(res_dir, "model_config.json")))
    num_layers = cfg["num_layers"]
    num_heads  = cfg["num_heads"]
    head_dim   = cfg["head_dim"]
    print(f"  Architecture: {num_layers}L × {num_heads}H, head_dim={head_dim}")

    try:
        model, tokenizer, device, dtype = load_model_and_tokenizer(model_id)
    except Exception as e:
        print(f"  [SKIP] {model_name}: {e}")
        return

    patcher = HeadPatcher(model, num_heads, head_dim)

    for regime in PAIR_REGIMES:
        # Skip if done
        out_path = os.path.join(res_dir, f"head_patching_{regime}.json")
        if os.path.exists(out_path):
            d = json.load(open(out_path))
            print(f"  [SKIP] {regime} already done. Top head: L{d['top_heads'][0]['layer']}.H{d['top_heads'][0]['head']}")
            continue

        # Get top layers from Phase 8
        resid_path = os.path.join(res_dir, f"residual_patching_{regime}.json")
        if not os.path.exists(resid_path):
            print(f"  [SKIP] No residual patching results for {regime}")
            continue
        resid = json.load(open(resid_path))
        target_layers = [t["layer"] for t in resid["top5_layers"][:TOP_N_LAYERS]]
        print(f"  [{regime}] Target layers: {target_layers}")

        # Load pairs
        pairs = load_jsonl(os.path.join(pairs_dir, f"pairs_{regime}.jsonl"))
        if not pairs:
            print(f"  [SKIP] No pairs for {regime}")
            continue

        random.seed(SEED)
        if MAX_PAIRS and len(pairs) > MAX_PAIRS:
            pairs = random.sample(pairs, MAX_PAIRS)

        total_ops = len(pairs) * len(target_layers) * num_heads
        print(f"  Patching {len(pairs)} pairs × {len(target_layers)} layers × {num_heads} heads = {total_ops} ops")

        # Accumulator: (layer, head) -> list of ΔSPS
        head_deltas = {}
        for l in target_layers:
            for h in range(num_heads):
                head_deltas[(l, h)] = []

        for pair in tqdm(pairs, desc=f"  {regime}", leave=False):
            ctx_vars = pair.get("valid_context_single_token_variants") or \
                       generate_variants(pair["context_answer"])
            mem_vars = pair.get("valid_memory_single_token_variants") or \
                       generate_variants(pair["memory_answer"])

            clean_ids   = tokenizer(pair["clean_prompt"], return_tensors="pt").input_ids.to(device)
            corrupt_ids = tokenizer(pair["corrupt_prompt"], return_tensors="pt").input_ids.to(device)

            corrupt_score = pair.get("corrupt_score")
            if corrupt_score is None:
                with torch.no_grad():
                    out = model(corrupt_ids)
                c_logits = out.logits[0, -1, :].float().cpu()
                info = compute_source_score(c_logits, tokenizer, ctx_vars, mem_vars, single_token_only=True)
                corrupt_score = info["source_score"] or 0.0

            for l in target_layers:
                # Capture clean attention output at this layer
                clean_attn_out = patcher.capture_attn_output(clean_ids, l)

                for h in range(num_heads):
                    patched_logits = patcher.patch_single_head(
                        corrupt_ids, clean_attn_out, l, h
                    )
                    patched_info = compute_source_score(
                        patched_logits, tokenizer, ctx_vars, mem_vars, single_token_only=True
                    )
                    ps = patched_info["source_score"]
                    if ps is not None:
                        delta = ps - corrupt_score
                        head_deltas[(l, h)].append(delta)

        # Average per head
        avg_head_delta = {}
        for (l, h), vals in head_deltas.items():
            if vals:
                avg_head_delta[f"L{l}.H{h}"] = {
                    "layer": l, "head": h,
                    "avg_delta_sps": round(sum(vals)/len(vals), 6),
                    "n_samples": len(vals),
                    "std": round((sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals))**0.5, 6),
                }

        # Rank by |ΔSPS|
        ranked = sorted(avg_head_delta.values(), key=lambda x: abs(x["avg_delta_sps"]), reverse=True)

        print(f"  [{regime}] Top-10 heads:")
        for r in ranked[:10]:
            direction = "←memory" if r["avg_delta_sps"] > 0 else "→context"
            print(f"    L{r['layer']:2d}.H{r['head']:2d}: ΔSPS={r['avg_delta_sps']:+.4f} (±{r['std']:.2f}) {direction}")

        save_json(out_path, {
            "model":        model_name,
            "regime":       regime,
            "num_pairs":    len(pairs),
            "target_layers": target_layers,
            "num_heads":    num_heads,
            "head_dim":     head_dim,
            "max_pairs":    MAX_PAIRS,
            "seed":         SEED,
            "timestamp":    datetime.datetime.now().isoformat(),
            "all_heads":    avg_head_delta,
            "top_heads":    ranked[:20],
        })
        print(f"  Saved -> {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for model_name, model_id in MODELS.items():
        run_head_patching(model_name, model_id, base_dir)
    print("\nHead patching complete.")

if __name__ == "__main__":
    main()
