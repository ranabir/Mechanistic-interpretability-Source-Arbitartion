"""
Phase 9b: Mid-layer validation.

Quick sanity check: patch heads in 2-3 random mid-layers to confirm
they have ~0 ΔSPS, validating our decision to only test top-5 layers.
"""
import sys, os, json, random
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from source_selection.data import load_jsonl, save_json
from source_selection.model_loading import load_model_and_tokenizer
from source_selection.scoring import compute_source_score
from source_selection.token_variants import generate_variants
from source_selection.patching import HeadPatcher

MODELS = {
    "qwen2_5_3b_instruct":   ("Qwen/Qwen2.5-3B-Instruct",   [5, 15, 20]),  # early, mid, mid-late
    "qwen2_5_1_5b_instruct": ("Qwen/Qwen2.5-1.5B-Instruct", [5, 10, 15]),
    "gemma_2_2b_it":         ("google/gemma-2-2b-it",         [5, 10, 15]),
}

MAX_PAIRS = 20
SEED = 42


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results = {}

    for model_name, (model_id, check_layers) in MODELS.items():
        print(f"\n=== {model_name} ===")
        res_dir = os.path.join(base_dir, "results", "latest", model_name)
        cfg = json.load(open(os.path.join(res_dir, "model_config.json")))

        model, tokenizer, device, dtype = load_model_and_tokenizer(model_id)
        patcher = HeadPatcher(model, cfg["num_heads"], cfg["head_dim"])

        pairs = load_jsonl(os.path.join(base_dir, "data", "causal_tracing_pairs",
                                        model_name, "pairs_explicit_context.jsonl"))
        random.seed(SEED)
        pairs = random.sample(pairs, min(MAX_PAIRS, len(pairs)))

        model_results = {}
        for layer in check_layers:
            head_deltas = []
            for pair in tqdm(pairs, desc=f"  L{layer}", leave=False):
                ctx_vars = pair.get("valid_context_single_token_variants") or generate_variants(pair["context_answer"])
                mem_vars = pair.get("valid_memory_single_token_variants") or generate_variants(pair["memory_answer"])

                clean_ids = tokenizer(pair["clean_prompt"], return_tensors="pt").input_ids.to(device)
                corrupt_ids = tokenizer(pair["corrupt_prompt"], return_tensors="pt").input_ids.to(device)
                corrupt_score = pair.get("corrupt_score", 0.0)

                clean_attn = patcher.capture_attn_output(clean_ids, layer)
                for h in range(cfg["num_heads"]):
                    logits = patcher.patch_single_head(corrupt_ids, clean_attn, layer, h)
                    info = compute_source_score(logits, tokenizer, ctx_vars, mem_vars, single_token_only=True)
                    if info["source_score"] is not None:
                        head_deltas.append(info["source_score"] - corrupt_score)

            avg = sum(head_deltas) / len(head_deltas) if head_deltas else 0
            max_abs = max(abs(d) for d in head_deltas) if head_deltas else 0
            model_results[f"L{layer}"] = {"avg_delta": round(avg, 6), "max_abs_delta": round(max_abs, 6)}
            print(f"  L{layer}: avg ΔSPS = {avg:+.4f}, max |ΔSPS| = {max_abs:.4f}")

        results[model_name] = model_results
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(os.path.join(base_dir, "results", "latest", "midlayer_validation.json"), results)
    print("\nMid-layer validation complete.")

if __name__ == "__main__":
    main()
