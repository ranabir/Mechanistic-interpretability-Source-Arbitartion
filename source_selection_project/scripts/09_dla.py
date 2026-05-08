"""
Phase 12: Direct Logit Attribution (DLA).

For each model's top context-override heads, compute the direct logit
contribution to context_answer vs memory_answer tokens.

Method:
  1. Run corrupt prompt → capture each head's output vector at the answer position
  2. Project head output through the unembedding matrix: logit_contrib = W_U @ head_out
  3. DLA score = logit_contrib[context_token] - logit_contrib[memory_token]

Positive DLA → head writes in favor of context answer
Negative DLA → head writes in favor of memory answer

Output: results/latest/{model}/dla_{regime}.json
"""
import sys, os, json, datetime, random
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from source_selection.data import load_jsonl, save_json
from source_selection.model_loading import load_model_and_tokenizer
from source_selection.token_variants import generate_variants

MODELS = {
    "qwen2_5_3b_instruct":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_5_1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma_2_2b_it":         "google/gemma-2-2b-it",
}

REGIMES = ["explicit_context", "ambiguous_conflict"]
MAX_PAIRS = 40
SEED = 42


def get_unembedding(model):
    """Get the unembedding matrix W_U [vocab_size, hidden_dim]."""
    # For most models: model.lm_head.weight
    return model.lm_head.weight.detach().float().cpu()  # [vocab, hidden]


def get_head_output(model, tokenizer, prompt, target_heads, device, num_heads, head_dim):
    """
    Run forward pass and capture each target head's output contribution
    at the last token position. Returns dict: (layer, head) -> vector [hidden_dim].
    """
    # Hook the self_attn module to capture the pre-o_proj attention output
    captured = {}
    handles = []

    layers = model.model.layers

    for (layer_idx, head_idx) in target_heads:
        def make_hook(l, h):
            def hook(module, input, output):
                # output[0] is the attention output after o_proj: [1, seq, hidden]
                attn_out = output[0] if isinstance(output, tuple) else output
                # Extract the head's slice at the last token
                start = h * head_dim
                end = start + head_dim
                head_vec = attn_out[0, -1, start:end].detach().float().cpu()
                captured[(l, h)] = head_vec
            return hook

        handles.append(layers[layer_idx].self_attn.register_forward_hook(
            make_hook(layer_idx, head_idx)
        ))

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        model(input_ids)

    for handle in handles:
        handle.remove()

    return captured


def compute_dla_for_head(head_vec, unembed, ctx_token_ids, mem_token_ids, head_dim, head_start):
    """
    Project a head's output through the unembedding to get logit contributions.
    
    head_vec: [head_dim] — the head's output slice
    unembed: [vocab, hidden] — full unembedding matrix
    
    We need the slice of W_U corresponding to this head's dimensions.
    DLA = W_U[ctx_tokens, head_slice] @ head_vec - W_U[mem_tokens, head_slice] @ head_vec
    """
    # Extract the relevant columns from the unembedding
    W_slice = unembed[:, head_start:head_start + head_dim]  # [vocab, head_dim]

    # Project
    logit_contrib = W_slice @ head_vec  # [vocab]

    # Get contributions for context and memory tokens
    ctx_logits = max(logit_contrib[tid].item() for tid in ctx_token_ids) if ctx_token_ids else 0
    mem_logits = max(logit_contrib[tid].item() for tid in mem_token_ids) if mem_token_ids else 0

    return ctx_logits - mem_logits, ctx_logits, mem_logits


def run_dla(model_name, model_id, base_dir):
    print(f"\n{'='*60}")
    print(f"  Phase 12: DLA — {model_name}")
    print(f"{'='*60}")

    res_dir = os.path.join(base_dir, "results", "latest", model_name)
    pairs_dir = os.path.join(base_dir, "data", "causal_tracing_pairs", model_name)
    cfg = json.load(open(os.path.join(res_dir, "model_config.json")))
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]

    # Check if all done
    all_done = all(
        os.path.exists(os.path.join(res_dir, f"dla_{r}.json"))
        for r in REGIMES
    )
    if all_done:
        print(f"  [SKIP] All regimes done.")
        return

    model, tokenizer, device, dtype = load_model_and_tokenizer(model_id)
    unembed = get_unembedding(model)

    for regime in REGIMES:
        out_path = os.path.join(res_dir, f"dla_{regime}.json")
        if os.path.exists(out_path):
            print(f"  [{regime}] SKIP")
            continue

        hp = json.load(open(os.path.join(res_dir, f"head_patching_{regime}.json")))
        top_heads = [(h["layer"], h["head"]) for h in hp["top_heads"][:10]]
        print(f"  [{regime}] Top heads: {top_heads[:5]}...")

        pairs = load_jsonl(os.path.join(pairs_dir, f"pairs_{regime}.jsonl"))
        random.seed(SEED)
        if MAX_PAIRS and len(pairs) > MAX_PAIRS:
            pairs = random.sample(pairs, MAX_PAIRS)

        # Accumulators
        head_dla = {str(h): [] for h in top_heads}
        head_ctx_logit = {str(h): [] for h in top_heads}
        head_mem_logit = {str(h): [] for h in top_heads}

        for pair in tqdm(pairs, desc=f"  {regime}", leave=False):
            ctx_vars = pair.get("valid_context_single_token_variants") or \
                       generate_variants(pair["context_answer"])
            mem_vars = pair.get("valid_memory_single_token_variants") or \
                       generate_variants(pair["memory_answer"])

            # Get token IDs for context and memory answers
            ctx_ids = [tokenizer.encode(v, add_special_tokens=False)[0]
                       for v in ctx_vars if len(tokenizer.encode(v, add_special_tokens=False)) == 1]
            mem_ids = [tokenizer.encode(v, add_special_tokens=False)[0]
                       for v in mem_vars if len(tokenizer.encode(v, add_special_tokens=False)) == 1]

            if not ctx_ids or not mem_ids:
                continue

            # Get head outputs on corrupt prompt
            head_outputs = get_head_output(
                model, tokenizer, pair["corrupt_prompt"],
                top_heads, device, num_heads, head_dim
            )

            for h_key, h_vec in head_outputs.items():
                h_str = str(h_key)
                head_start = h_key[1] * head_dim
                dla, ctx_l, mem_l = compute_dla_for_head(
                    h_vec, unembed, ctx_ids, mem_ids, head_dim, head_start
                )
                head_dla[h_str].append(dla)
                head_ctx_logit[h_str].append(ctx_l)
                head_mem_logit[h_str].append(mem_l)

        # Summarize
        summary = {}
        for h in top_heads:
            h_str = str(h)
            vals = head_dla[h_str]
            if vals:
                summary[h_str] = {
                    "layer": h[0], "head": h[1],
                    "avg_dla": round(np.mean(vals), 4),
                    "std_dla": round(np.std(vals), 4),
                    "avg_ctx_logit": round(np.mean(head_ctx_logit[h_str]), 4),
                    "avg_mem_logit": round(np.mean(head_mem_logit[h_str]), 4),
                    "n": len(vals),
                    "pct_positive": round(100 * sum(1 for v in vals if v > 0) / len(vals), 1),
                }

        ranked = sorted(summary.values(), key=lambda x: abs(x["avg_dla"]), reverse=True)
        print(f"  [{regime}] Top-5 by |DLA|:")
        for r in ranked[:5]:
            direction = "→ctx" if r["avg_dla"] > 0 else "←mem"
            print(f"    L{r['layer']:2d}.H{r['head']:2d}: DLA={r['avg_dla']:+.3f} "
                  f"(ctx={r['avg_ctx_logit']:+.3f}, mem={r['avg_mem_logit']:+.3f}) "
                  f"{direction} ({r['pct_positive']:.0f}% positive)")

        save_json(out_path, {
            "model": model_name,
            "regime": regime,
            "num_pairs": len(pairs),
            "seed": SEED,
            "timestamp": datetime.datetime.now().isoformat(),
            "heads": summary,
            "ranked": ranked,
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for model_name, model_id in MODELS.items():
        run_dla(model_name, model_id, base_dir)
    print("\nDLA analysis complete.")

if __name__ == "__main__":
    main()
