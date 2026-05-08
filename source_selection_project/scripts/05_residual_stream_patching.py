"""
Phase 8: Residual Stream Patching.

For each matched pair (clean, corrupt):
  - Capture residual stream activations from the clean (memory) run at every layer
  - For each layer l, patch clean activations into the corrupt (context) run
  - Measure ΔSPS = patched_score - corrupt_score at each layer
  - Average ΔSPS across all pairs per layer

Outputs:
  results/latest/{model}/residual_patching_{regime}.json
    Contains per-pair results + layer-averaged ΔSPS curve

This gives us a "patching curve" identifying which layers are causally
responsible for the context-override behavior.
"""
import sys, os, json, datetime
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from source_selection.data import load_jsonl, save_json
from source_selection.model_loading import load_model_and_tokenizer
from source_selection.scoring import compute_source_score
from source_selection.token_variants import generate_variants
from source_selection.patching import ResidualStreamPatcher

MODELS = {
    "qwen2_5_3b_instruct":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_5_1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma_2_2b_it":         "google/gemma-2-2b-it",
}

PAIR_REGIMES = ["explicit_context", "ambiguous_conflict"]

# Limit pairs per regime for speed (set to None to run all)
MAX_PAIRS = 60
SEED = 42


def run_patching_for_model(model_name, model_id, base_dir):
    print(f"\n{'='*60}")
    print(f"  Phase 8: Residual Patching — {model_name}")
    print(f"{'='*60}")

    res_dir   = os.path.join(base_dir, "results", "latest", model_name)
    pairs_dir = os.path.join(base_dir, "data", "causal_tracing_pairs", model_name)
    os.makedirs(res_dir, exist_ok=True)

    # Load model config to get num_layers
    cfg_path = os.path.join(res_dir, "model_config.json")
    if not os.path.exists(cfg_path):
        print(f"  [ERROR] model_config.json not found. Run Phase 7 first.")
        return
    num_layers = json.load(open(cfg_path))["num_layers"]
    print(f"  Architecture: {num_layers} layers")

    try:
        model, tokenizer, device, dtype = load_model_and_tokenizer(model_id)
    except Exception as e:
        print(f"  [SKIP] {model_name}: {e}")
        return

    patcher = ResidualStreamPatcher(model)

    for regime in PAIR_REGIMES:
        # Skip if already completed
        out_path = os.path.join(res_dir, f"residual_patching_{regime}.json")
        if os.path.exists(out_path):
            print(f"  [SKIP] {regime} already completed, loading results...")
            d = json.load(open(out_path))
            print(f"    Peak layer: {d['peak_layer']}, ΔSPS: {d['peak_delta']:.4f}")
            continue

        pairs_file = os.path.join(pairs_dir, f"pairs_{regime}.jsonl")
        pairs = load_jsonl(pairs_file)
        if not pairs:
            print(f"  [SKIP] No pairs for {regime}")
            continue

        # Sample for speed
        if MAX_PAIRS and len(pairs) > MAX_PAIRS:
            import random; random.seed(SEED)
            pairs = random.sample(pairs, MAX_PAIRS)
        print(f"  [{regime}] Patching {len(pairs)} pairs × {num_layers} layers...")

        # Accumulators: layer -> list of ΔSPS values
        layer_delta_sps = {l: [] for l in range(num_layers)}
        pair_results = []

        for pair in tqdm(pairs, desc=f"  {regime}", leave=False):
            ctx_vars = pair.get("valid_context_single_token_variants") or \
                       generate_variants(pair["context_answer"])
            mem_vars = pair.get("valid_memory_single_token_variants") or \
                       generate_variants(pair["memory_answer"])

            # Tokenize both prompts
            clean_ids  = tokenizer(pair["clean_prompt"],
                                   return_tensors="pt").input_ids.to(device)
            corrupt_ids = tokenizer(pair["corrupt_prompt"],
                                    return_tensors="pt").input_ids.to(device)

            # Step 1: Capture clean activations at all layers
            clean_states = patcher.capture_clean(clean_ids)

            # Step 2: Get baseline corrupt score (re-use stored score if available)
            corrupt_score = pair.get("corrupt_score")
            if corrupt_score is None:
                with torch.no_grad():
                    out = model(corrupt_ids)
                corrupt_logits = out.logits[0, -1, :].float().cpu()
                info = compute_source_score(corrupt_logits, tokenizer,
                                            ctx_vars, mem_vars, single_token_only=True)
                corrupt_score = info["source_score"] or 0.0

            # Step 3: Patch each layer and measure ΔSPS
            per_layer = {}
            for l in range(num_layers):
                patched_logits = patcher.patch_at_layer(corrupt_ids, clean_states, l)
                patched_info = compute_source_score(
                    patched_logits, tokenizer, ctx_vars, mem_vars, single_token_only=True
                )
                patched_score = patched_info["source_score"]
                if patched_score is not None:
                    delta = patched_score - corrupt_score
                    layer_delta_sps[l].append(delta)
                    per_layer[l] = round(delta, 6)
                else:
                    per_layer[l] = None

            pair_results.append({
                "pair_id":       pair["pair_id"],
                "fact_id":       pair["fact_id"],
                "category":      pair.get("category",""),
                "corrupt_score": corrupt_score,
                "per_layer_delta_sps": per_layer,
            })

        # Average ΔSPS per layer
        avg_delta = {}
        for l in range(num_layers):
            vals = [v for v in layer_delta_sps[l] if v is not None]
            avg_delta[l] = round(sum(vals)/len(vals), 6) if vals else None

        # Find peak layer
        valid = {l: v for l, v in avg_delta.items() if v is not None}
        peak_layer = max(valid, key=lambda l: abs(valid[l])) if valid else None
        peak_delta = avg_delta.get(peak_layer)

        print(f"  [{regime}] Peak layer: {peak_layer}, Avg ΔSPS: {peak_delta:.4f}")
        print(f"  Top-5 layers by |ΔSPS|:")
        top5 = sorted(valid.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for l, v in top5:
            direction = "←memory" if v > 0 else "→context"
            print(f"    Layer {l:2d}: ΔSPS={v:+.4f} {direction}")

        save_json(os.path.join(res_dir, f"residual_patching_{regime}.json"), {
            "model":      model_name,
            "regime":     regime,
            "num_pairs":  len(pairs),
            "num_layers": num_layers,
            "max_pairs_sampled": MAX_PAIRS,
            "seed":       SEED,
            "timestamp":  datetime.datetime.now().isoformat(),
            "avg_delta_sps_per_layer": avg_delta,
            "peak_layer":  peak_layer,
            "peak_delta":  peak_delta,
            "top5_layers": [{"layer": l, "avg_delta_sps": v} for l, v in top5],
            "pair_results": pair_results,
        })
        print(f"  Saved -> results/latest/{model_name}/residual_patching_{regime}.json")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Phase 7 first
    print("Running Phase 7 (config lock)...")
    import subprocess, sys as _sys
    subprocess.run([_sys.executable, os.path.join(os.path.dirname(__file__),
                    "04_lock_model_configs.py")], check=True)

    for model_name, model_id in MODELS.items():
        run_patching_for_model(model_name, model_id, base_dir)

    print("\nResidual stream patching complete.")

if __name__ == "__main__":
    main()
