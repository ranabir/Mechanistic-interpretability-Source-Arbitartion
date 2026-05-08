"""
Phase 15: Activation Steering.

Compute a "context direction" vector from the identified heads and add/subtract
it during inference to steer the model toward context or memory.

Method:
  1. Collect head outputs on context-winning and memory-winning prompts
  2. Compute mean_ctx - mean_mem = "context direction" for each head
  3. During inference, add alpha * context_direction to the head output
     - alpha > 0: push toward context
     - alpha < 0: push toward memory
  4. Measure ctx_win% at different alpha values

If steering works, it proves the circuit is SUFFICIENT, not just necessary.

Output: results/latest/{model}/steering_{regime}.json
"""
import sys, os, json, datetime, random
import torch
import numpy as np
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
SEED = 42
N_DIRECTION = 30   # pairs to compute direction
N_EVAL = 40        # pairs to evaluate steering
# Large alphas needed — the direction is unit-normalized so we need to scale
# to match the magnitude of the attention output (~10-50 in practice)
ALPHAS = [-50.0, -30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0, 50.0]


def collect_head_outputs(model, tokenizer, prompts, target_layer, head_dim, device):
    """Collect head outputs at the target layer for all heads, last token position."""
    layers = model.model.layers
    all_outputs = []

    for prompt_text in prompts:
        captured = {}

        def hook(module, input, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            captured["attn"] = attn_out[0, -1, :].detach().float().cpu()

        handle = layers[target_layer].self_attn.register_forward_hook(hook)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model(input_ids)
        handle.remove()

        all_outputs.append(captured["attn"].numpy())

    return np.array(all_outputs)  # [n_prompts, hidden_dim]


def run_steering(model_name, model_id, base_dir):
    print(f"\n{'='*60}")
    print(f"  Phase 15: Steering — {model_name}")
    print(f"{'='*60}")

    res_dir = os.path.join(base_dir, "results", "latest", model_name)
    pairs_dir = os.path.join(base_dir, "data", "causal_tracing_pairs", model_name)
    cfg = json.load(open(os.path.join(res_dir, "model_config.json")))
    head_dim = cfg["head_dim"]
    num_heads = cfg["num_heads"]

    out_path = os.path.join(res_dir, f"steering_{REGIME}.json")
    # Always re-run (we're iterating on alpha scale)
    if os.path.exists(out_path):
        os.remove(out_path)

    # Get the top layer from residual patching (the layer with strongest effect)
    hp = json.load(open(os.path.join(res_dir, f"head_patching_{REGIME}.json")))
    target_layer = hp["top_heads"][0]["layer"]
    print(f"  Target layer: L{target_layer}")

    model, tokenizer, device, dtype = load_model_and_tokenizer(model_id)

    # Load pairs for direction computation
    pairs = load_jsonl(os.path.join(pairs_dir, f"pairs_{REGIME}.jsonl"))
    random.seed(SEED)
    random.shuffle(pairs)

    direction_pairs = pairs[:N_DIRECTION]
    eval_pairs = pairs[N_DIRECTION:N_DIRECTION + N_EVAL]

    if len(eval_pairs) < 10:
        # Not enough pairs, reuse with different seed
        eval_pairs = pairs[:N_EVAL]

    # Step 1: Compute context direction
    # Use corrupt prompts (context-winning) vs clean prompts (memory/no-doc)
    print(f"  Computing context direction from {len(direction_pairs)} pairs...")

    corrupt_prompts = [p["corrupt_prompt"] for p in direction_pairs]
    clean_prompts = [p["clean_prompt"] for p in direction_pairs]

    corrupt_outputs = collect_head_outputs(model, tokenizer, corrupt_prompts, target_layer, head_dim, device)
    clean_outputs = collect_head_outputs(model, tokenizer, clean_prompts, target_layer, head_dim, device)

    # Context direction = mean(corrupt) - mean(clean)
    # corrupt = model sees document and follows it
    # clean = model sees only question and uses memory
    context_direction = corrupt_outputs.mean(axis=0) - clean_outputs.mean(axis=0)
    context_direction = context_direction / (np.linalg.norm(context_direction) + 1e-8)  # normalize
    context_dir_tensor = torch.tensor(context_direction, dtype=torch.float32)

    print(f"  Context direction norm (pre-normalization): {np.linalg.norm(corrupt_outputs.mean(0) - clean_outputs.mean(0)):.4f}")

    # Step 2: Evaluate at different alpha values
    print(f"  Evaluating {len(ALPHAS)} alpha values on {len(eval_pairs)} pairs...")

    results = {}
    for alpha in ALPHAS:
        scores = []
        ctx_wins = 0
        layers = model.model.layers

        for pair in tqdm(eval_pairs, desc=f"  α={alpha:+.1f}", leave=False):
            ctx_vars = pair.get("valid_context_single_token_variants") or \
                       generate_variants(pair["context_answer"])
            mem_vars = pair.get("valid_memory_single_token_variants") or \
                       generate_variants(pair["memory_answer"])

            # Hook to add steering vector
            def make_steer_hook(alpha_val, direction):
                def hook(module, input, output):
                    attn_out = output[0] if isinstance(output, tuple) else output
                    steered = attn_out.clone()
                    steered[0, -1, :] += alpha_val * direction.to(attn_out.device).to(attn_out.dtype)
                    if isinstance(output, tuple):
                        return (steered,) + output[1:]
                    return steered
                return hook

            handle = layers[target_layer].self_attn.register_forward_hook(
                make_steer_hook(alpha, context_dir_tensor)
            )

            input_ids = tokenizer(pair["corrupt_prompt"], return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids)
            handle.remove()

            logits = out.logits[0, -1, :].float().cpu()
            info = compute_source_score(logits, tokenizer, ctx_vars, mem_vars, single_token_only=True)
            if info["source_score"] is not None:
                scores.append(info["source_score"])
                if info["source_score"] > 0:
                    ctx_wins += 1

        n = len(scores)
        results[str(alpha)] = {
            "alpha": alpha,
            "n": n,
            "ctx_win": ctx_wins,
            "ctx_win_pct": round(100 * ctx_wins / max(1, n), 1),
            "mean_sps": round(np.mean(scores), 4) if scores else 0,
            "std_sps": round(np.std(scores), 4) if scores else 0,
        }
        print(f"    α={alpha:+.1f}: ctx_win={results[str(alpha)]['ctx_win_pct']:.1f}%, "
              f"mean_sps={results[str(alpha)]['mean_sps']:+.2f}")

    save_json(out_path, {
        "model": model_name,
        "regime": REGIME,
        "target_layer": target_layer,
        "n_direction_pairs": len(direction_pairs),
        "n_eval_pairs": len(eval_pairs),
        "alphas": ALPHAS,
        "seed": SEED,
        "timestamp": datetime.datetime.now().isoformat(),
        "results": results,
    })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for model_name, model_id in MODELS.items():
        run_steering(model_name, model_id, base_dir)
    print("\nSteering analysis complete.")

if __name__ == "__main__":
    main()
