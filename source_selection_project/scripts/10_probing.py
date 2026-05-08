"""
Phase 14: Linear Probing for Source Selection Decision.

Uses the full filtered prompt set (pre-matching), scores them on the fly, 
captures hidden states, and trains a probe at each layer.

Output: results/latest/{model}/probing_{regime}.json
"""
import sys, os, json, datetime, random
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

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

REGIMES = ["explicit_context", "ambiguous_conflict"]
MAX_SAMPLES = 120
SEED = 42


def run_probing(model_name, model_id, base_dir):
    print(f"\n{'='*60}")
    print(f"  Phase 14: Probing — {model_name}")
    print(f"{'='*60}")

    res_dir = os.path.join(base_dir, "results", "latest", model_name)
    cfg = json.load(open(os.path.join(res_dir, "model_config.json")))
    num_layers = cfg["num_layers"]

    all_done = all(
        os.path.exists(os.path.join(res_dir, f"probing_{r}.json"))
        for r in REGIMES
    )
    if all_done:
        print(f"  [SKIP] All regimes done.")
        return

    model, tokenizer, device, dtype = load_model_and_tokenizer(model_id)

    for regime in REGIMES:
        out_path = os.path.join(res_dir, f"probing_{regime}.json")
        if os.path.exists(out_path):
            print(f"  [{regime}] SKIP")
            continue

        # Load prompts from the filtered set
        prompt_file = os.path.join(
            base_dir, "data", "filtered", model_name,
            f"single_token_{regime}_prompts.jsonl"
        )
        if not os.path.exists(prompt_file):
            print(f"  [{regime}] No prompt file found. Skipping.")
            continue

        prompts = load_jsonl(prompt_file)
        random.seed(SEED)
        if len(prompts) > MAX_SAMPLES:
            prompts = random.sample(prompts, MAX_SAMPLES)

        print(f"  [{regime}] Processing {len(prompts)} prompts...")

        layer_activations = [[] for _ in range(num_layers + 1)]
        labels = []
        sps_scores = []

        for item in tqdm(prompts, desc=f"  {regime}", leave=False):
            prompt = item["prompt"]
            ctx_vars = item.get("valid_context_single_token_variants") or \
                       generate_variants(item["context_answer"])
            mem_vars = item.get("valid_memory_single_token_variants") or \
                       generate_variants(item["memory_answer"])

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)

            logits = out.logits[0, -1, :].float().cpu()
            info = compute_source_score(logits, tokenizer, ctx_vars, mem_vars, single_token_only=True)

            if info["source_score"] is None:
                continue

            sps = info["source_score"]
            label = 1 if sps > 0 else 0  # 1=context, 0=memory
            labels.append(label)
            sps_scores.append(sps)

            for layer_idx, hs in enumerate(out.hidden_states):
                vec = hs[0, -1, :].float().cpu().numpy()
                layer_activations[layer_idx].append(vec)

        labels = np.array(labels)
        n_ctx = int(labels.sum())
        n_mem = len(labels) - n_ctx
        print(f"  [{regime}] {len(labels)} valid: {n_ctx} ctx, {n_mem} mem")

        if min(n_ctx, n_mem) < 5:
            print(f"  [{regime}] Insufficient class diversity. Skipping.")
            continue

        # Train probes
        results_by_layer = []
        for layer_idx in range(num_layers + 1):
            X = np.array(layer_activations[layer_idx])
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0) + 1e-8
            X_norm = (X - X_mean) / X_std

            n_folds = min(5, min(n_ctx, n_mem))
            if n_folds < 2:
                n_folds = 2

            try:
                clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
                scores = cross_val_score(clf, X_norm, labels, cv=n_folds, scoring='accuracy')
                acc = scores.mean()
                std = scores.std()
            except Exception as e:
                acc, std = 0.5, 0.0

            layer_name = "embedding" if layer_idx == 0 else f"L{layer_idx - 1}"
            results_by_layer.append({
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "accuracy": round(acc, 4),
                "std": round(std, 4),
                "depth_pct": round(100 * layer_idx / num_layers, 1),
            })

            if layer_idx % 5 == 0 or layer_idx == num_layers:
                print(f"    {layer_name:12s}: acc={acc:.1%} ± {std:.1%}")

        save_json(out_path, {
            "model": model_name,
            "regime": regime,
            "num_samples": len(labels),
            "n_context": n_ctx,
            "n_memory": n_mem,
            "seed": SEED,
            "timestamp": datetime.datetime.now().isoformat(),
            "layers": results_by_layer,
        })

        accs = [r["accuracy"] for r in results_by_layer]
        transition = next((i for i, a in enumerate(accs) if a > 0.85), len(accs))
        if transition < len(accs):
            print(f"  [{regime}] Crystallization at {results_by_layer[transition]['layer_name']} "
                  f"({results_by_layer[transition]['depth_pct']:.0f}% depth, acc={accs[transition]:.1%})")
        else:
            print(f"  [{regime}] No clear crystallization (max acc: {max(accs):.1%})")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for model_name, model_id in MODELS.items():
        run_probing(model_name, model_id, base_dir)
    print("\nProbing analysis complete.")

if __name__ == "__main__":
    main()
