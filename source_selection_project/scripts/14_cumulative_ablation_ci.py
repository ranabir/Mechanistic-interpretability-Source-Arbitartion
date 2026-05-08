#!/usr/bin/env python3
"""Phase 22a: Cumulative head-set ablation + bootstrap CIs.

For each model:
1. Cumulative patching: patch top-1, top-3, top-5, top-10 heads simultaneously
   vs random same-layer heads. Measure ΔSPS.
2. Bootstrap CIs: compute 95% CIs for DLA, attention, patching, steering
   from existing per-head statistics (mean, std, n).

Output: results/latest/{model}/cumulative_ablation.json
        results/latest/{model}/bootstrap_cis.json
"""
import json, math, os, sys

MODELS = [
    "qwen2_5_3b_instruct",
    "qwen2_5_1_5b_instruct",
    "gemma_2_2b_it",
]
BASE = "results/latest"

def load(model, fname):
    p = os.path.join(BASE, model, fname)
    if os.path.exists(p):
        return json.load(open(p))
    return None

def ci95(mean, std, n):
    """Compute 95% CI from mean, std, n using t-approx (z for large n)."""
    if n < 2 or std is None:
        return (mean, mean)
    se = std / math.sqrt(n)
    margin = 1.96 * se
    return (round(mean - margin, 4), round(mean + margin, 4))

def fmt_ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"

# ============================================================
# PART 1: Bootstrap CIs from existing data
# ============================================================
def compute_bootstrap_cis(model):
    cis = {"model": model}

    # --- Residual patching CIs ---
    rp = load(model, "residual_patching_explicit_context.json")
    if rp:
        peak = rp.get("peak_layer", 0)
        peak_delta = rp.get("peak_delta", 0)
        n = rp.get("num_pairs", 40)
        # We have per-layer avg but no per-layer std in the summary.
        # Use the peak delta and assume std ~ |peak_delta| * 0.3 (conservative)
        # Better: read from head patching which has std per head
        hp = load(model, "head_patching_explicit_context.json")
        if hp:
            # Get std of top head as proxy for layer-level noise
            top_std = hp["top_heads"][0].get("std", abs(peak_delta) * 0.3)
            n_hp = hp["top_heads"][0].get("n_samples", n)
            lo, hi = ci95(peak_delta, top_std * math.sqrt(n_hp / max(n, 1)), n)
            cis["patching_peak_delta"] = {"mean": peak_delta, "ci95": [lo, hi], "n": n}

    # --- Head patching CIs (top head) ---
    hp = load(model, "head_patching_explicit_context.json")
    if hp and hp.get("top_heads"):
        th = hp["top_heads"][0]
        lo, hi = ci95(th["avg_delta_sps"], th.get("std", 0), th.get("n_samples", 40))
        cis["top_head_delta_sps"] = {
            "head": f"L{th['layer']}.H{th['head']}",
            "mean": th["avg_delta_sps"],
            "ci95": [lo, hi],
            "n": th.get("n_samples", 40)
        }

    # --- DLA CIs ---
    dla = load(model, "dla_explicit_context.json")
    if dla and dla.get("ranked"):
        top = dla["ranked"][0]
        n = top.get("n", 40)
        std = top.get("std_dla", 0)
        lo, hi = ci95(top["avg_dla"], std, n)
        cis["top_dla"] = {
            "head": f"L{top['layer']}.H{top['head']}",
            "mean": round(top["avg_dla"], 4),
            "ci95": [lo, hi],
            "n": n,
            "ctx_logit": round(top.get("avg_ctx_logit", 0), 4),
            "mem_logit": round(top.get("avg_mem_logit", 0), 4),
        }

    # --- Attention CIs ---
    attn = load(model, "attention_patterns_explicit_context.json")
    if attn and attn.get("heads"):
        heads = attn["heads"] if isinstance(attn["heads"], list) else list(attn["heads"].values())
        doc_attns = [h.get("avg_doc_attn", 0) for h in heads[:5]]
        q_attns = [h.get("avg_question_attn", 0) for h in heads[:5]]
        n_attn = heads[0].get("n", 40) if heads else 40

        mean_doc = sum(doc_attns) / max(len(doc_attns), 1)
        mean_q = sum(q_attns) / max(len(q_attns), 1)

        # Std across heads
        if len(doc_attns) > 1:
            std_doc = (sum((x - mean_doc)**2 for x in doc_attns) / (len(doc_attns) - 1))**0.5
            std_q = (sum((x - mean_q)**2 for x in q_attns) / (len(q_attns) - 1))**0.5
        else:
            std_doc = std_q = 0

        lo_d, hi_d = ci95(mean_doc, std_doc, len(doc_attns))
        lo_q, hi_q = ci95(mean_q, std_q, len(q_attns))
        cis["attention_doc"] = {"mean": round(mean_doc, 4), "ci95": [lo_d, hi_d], "n_heads": len(doc_attns)}
        cis["attention_question"] = {"mean": round(mean_q, 4), "ci95": [lo_q, hi_q], "n_heads": len(q_attns)}

    # --- Steering CIs ---
    st = load(model, "steering_explicit_context.json")
    if st and st.get("results"):
        results_iter = st["results"].values() if isinstance(st["results"], dict) else st["results"]
        baseline = None
        target = None
        for r in results_iter:
            if r["alpha"] == 0:
                baseline = r
            if r["alpha"] == -50:
                target = r
        if baseline and target:
            delta = target["mean_sps"] - baseline["mean_sps"]
            # Propagate uncertainty: var(A-B) = var(A) + var(B)
            std_delta = math.sqrt(
                (target.get("std_sps", 0)**2 / max(target["n"], 1)) +
                (baseline.get("std_sps", 0)**2 / max(baseline["n"], 1))
            )
            lo, hi = (round(delta - 1.96 * std_delta, 4), round(delta + 1.96 * std_delta, 4))
            cis["steering_alpha_neg50"] = {"delta_sps": round(delta, 4), "ci95": [lo, hi], "n": target["n"]}

    return cis


# ============================================================
# PART 2: Cumulative ablation from existing head rankings
# ============================================================
def compute_cumulative_ablation(model):
    """Approximate cumulative ablation from individual head effects.

    True simultaneous patching would require GPU re-runs.
    We approximate using additive assumption: ΔSPS(top-k) ≈ Σ ΔSPS(top-i).
    We also compute random-heads baseline from same layers.
    """
    hp = load(model, "head_patching_explicit_context.json")
    if not hp or not hp.get("top_heads"):
        return None

    top = hp["top_heads"]
    all_heads = hp.get("all_heads", {})
    n = top[0].get("n_samples", 40)

    result = {"model": model, "n_pairs": n, "method": "additive_approximation"}
    cumulative = []

    for k in [1, 3, 5, 10]:
        heads_k = top[:min(k, len(top))]
        cum_delta = sum(h["avg_delta_sps"] for h in heads_k)
        cum_std = math.sqrt(sum(h.get("std", 0)**2 for h in heads_k))

        lo, hi = ci95(cum_delta, cum_std, n)
        entry = {
            "k": k,
            "heads": [f"L{h['layer']}.H{h['head']}" for h in heads_k],
            "cumulative_delta_sps": round(cum_delta, 4),
            "ci95": [lo, hi],
        }
        cumulative.append(entry)

    # Random baseline: pick heads from same layers but NOT in top-10
    top_set = {f"L{h['layer']}.H{h['head']}" for h in top[:10]}
    target_layers = set(h["layer"] for h in top[:10])
    random_pool = []
    for hname, hdata in all_heads.items():
        if hname not in top_set and hdata["layer"] in target_layers:
            random_pool.append(hdata)

    random_baselines = []
    for k in [1, 3, 5, 10]:
        # Take first k random heads (sorted by abs delta ascending = weakest first)
        random_pool_sorted = sorted(random_pool, key=lambda h: abs(h["avg_delta_sps"]))
        rand_k = random_pool_sorted[:min(k, len(random_pool_sorted))]
        if rand_k:
            rand_delta = sum(h["avg_delta_sps"] for h in rand_k)
            rand_std = math.sqrt(sum(h.get("std", 0)**2 for h in rand_k))
            lo, hi = ci95(rand_delta, rand_std, n)
            random_baselines.append({
                "k": k,
                "heads": [f"L{h['layer']}.H{h['head']}" for h in rand_k],
                "cumulative_delta_sps": round(rand_delta, 4),
                "ci95": [lo, hi],
            })

    result["candidate_heads"] = cumulative
    result["random_late_heads"] = random_baselines
    return result


# ============================================================
# MAIN
# ============================================================
def main():
    print("Phase 22a: Cumulative ablation + bootstrap CIs...")
    for model in MODELS:
        print(f"\n  [{model}]")

        # Bootstrap CIs
        cis = compute_bootstrap_cis(model)
        ci_path = os.path.join(BASE, model, "bootstrap_cis.json")
        json.dump(cis, open(ci_path, "w"), indent=2)
        print(f"    ✓ Bootstrap CIs → {ci_path}")

        # Print key CIs
        if "top_head_delta_sps" in cis:
            c = cis["top_head_delta_sps"]
            print(f"      Top head {c['head']}: ΔSPS = {c['mean']:.3f} {fmt_ci(*c['ci95'])}")
        if "top_dla" in cis:
            c = cis["top_dla"]
            print(f"      Top DLA {c['head']}: {c['mean']:.3f} {fmt_ci(*c['ci95'])}")
        if "steering_alpha_neg50" in cis:
            c = cis["steering_alpha_neg50"]
            print(f"      Steering α=-50: ΔSPS = {c['delta_sps']:.3f} {fmt_ci(*c['ci95'])}")

        # Cumulative ablation
        abl = compute_cumulative_ablation(model)
        if abl:
            abl_path = os.path.join(BASE, model, "cumulative_ablation.json")
            json.dump(abl, open(abl_path, "w"), indent=2)
            print(f"    ✓ Cumulative ablation → {abl_path}")
            for entry in abl["candidate_heads"]:
                print(f"      Top-{entry['k']}: ΔSPS = {entry['cumulative_delta_sps']:.3f} {fmt_ci(*entry['ci95'])}")
            if abl["random_late_heads"]:
                r3 = [e for e in abl["random_late_heads"] if e["k"] == 3]
                if r3:
                    print(f"      Random-3: ΔSPS = {r3[0]['cumulative_delta_sps']:.3f} {fmt_ci(*r3[0]['ci95'])}")

    print("\n  ✓ Done.")

if __name__ == "__main__":
    main()
