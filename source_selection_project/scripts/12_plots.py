"""
Phase 19: Publication-quality plots from all experimental results.

Generates 6 figures for the paper:
  Fig 1: Residual Stream Patching — layer-wise SPS curves (3 models × 2 regimes)
  Fig 2: Head-level DLA heatmap — top heads' logit contributions
  Fig 3: Attention Pattern bars — doc vs question attention
  Fig 4: Steering dose-response — SPS vs alpha
  Fig 5: Probing crystallization — probe accuracy by layer
  Fig 6: Cross-model summary dashboard

Output: results/figures/
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(BASE, "results", "latest")
FIG_DIR = os.path.join(BASE, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = {
    "qwen2_5_3b_instruct":   ("Qwen2.5-3B", "#2563eb", 36),
    "qwen2_5_1_5b_instruct": ("Qwen2.5-1.5B", "#7c3aed", 28),
    "gemma_2_2b_it":         ("Gemma-2-2B", "#059669", 26),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(model, filename):
    path = os.path.join(RES, model, filename)
    if os.path.exists(path):
        return json.load(open(path))
    return None


# ═══════════════════════════════════════════════════════════
# Fig 1: Residual Stream Patching Curves
# ═══════════════════════════════════════════════════════════
def plot_residual_patching():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for regime_idx, regime in enumerate(["explicit_context", "ambiguous_conflict"]):
        ax = axes[regime_idx]
        for model_key, (label, color, n_layers) in MODELS.items():
            data = load(model_key, f"residual_patching_{regime}.json")
            if not data:
                continue
            layer_data = data["avg_delta_sps_per_layer"]
            xs = sorted(int(k) for k in layer_data.keys())
            ys = [layer_data[str(x)] for x in xs]
            # Normalize x to depth percentage
            xs_pct = [100 * x / (n_layers - 1) for x in xs]
            ax.plot(xs_pct, ys, "o-", label=label, color=color,
                    markersize=3, linewidth=1.8, alpha=0.85)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_xlabel("Depth (%)")
        ax.set_title(regime.replace("_", " ").title(), fontweight="bold")
        if regime_idx == 0:
            ax.set_ylabel("ΔSPS (patched − baseline)")
            ax.legend(frameon=True, fancybox=True, shadow=False, fontsize=10)

    fig.suptitle("Residual Stream Patching: Where Does Source Selection Happen?",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_residual_patching.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig1_residual_patching.pdf"))
    print("  ✓ Fig 1: Residual patching curves")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Fig 2: DLA Heatmap — Head-level logit attribution
# ═══════════════════════════════════════════════════════════
def plot_dla_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (model_key, (label, color, n_layers)) in enumerate(MODELS.items()):
        ax = axes[idx]
        data = load(model_key, "dla_explicit_context.json")
        if not data:
            continue

        heads = data["ranked"][:10]
        names = [f"L{h['layer']}.H{h['head']}" for h in heads]
        dlas = [h["avg_dla"] for h in heads]
        ctx_l = [h["avg_ctx_logit"] for h in heads]
        mem_l = [h["avg_mem_logit"] for h in heads]

        x = np.arange(len(names))
        width = 0.35
        bars_ctx = ax.bar(x - width/2, ctx_l, width, label="Context logit",
                          color="#3b82f6", alpha=0.8, edgecolor="white")
        bars_mem = ax.bar(x + width/2, mem_l, width, label="Memory logit",
                          color="#ef4444", alpha=0.8, edgecolor="white")

        # DLA as line
        ax2 = ax.twinx()
        ax2.plot(x, dlas, "k^-", markersize=7, linewidth=1.5, label="DLA", zorder=5)
        ax2.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax2.set_ylabel("DLA score", fontsize=10)
        ax2.spines["top"].set_visible(False)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_title(label, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Logit contribution")
            ax.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("Direct Logit Attribution: What Do Context-Override Heads Write?",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_dla_heatmap.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig2_dla_heatmap.pdf"))
    print("  ✓ Fig 2: DLA head-level")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Fig 3: Attention Patterns — doc vs question
# ═══════════════════════════════════════════════════════════
def plot_attention_patterns():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (model_key, (label, color, n_layers)) in enumerate(MODELS.items()):
        ax = axes[idx]
        data = load(model_key, "attention_patterns_explicit_context.json")
        if not data:
            continue

        heads = list(data["heads"].values())[:5]
        names = [f"L{h['layer']}.H{h['head']}" for h in heads]
        doc_attn = [h["corrupt_doc_attn_avg"] * 100 for h in heads]
        q_attn = [h["corrupt_question_attn_avg"] * 100 for h in heads]
        other = [100 - d - q for d, q in zip(doc_attn, q_attn)]

        x = np.arange(len(names))
        width = 0.6

        ax.barh(x, doc_attn, height=width, label="Document",
                color="#ef4444", alpha=0.85, edgecolor="white")
        ax.barh(x, q_attn, height=width, left=doc_attn, label="Question",
                color="#3b82f6", alpha=0.85, edgecolor="white")
        ax.barh(x, other, height=width, left=[d+q for d,q in zip(doc_attn, q_attn)],
                label="Other", color="#d1d5db", alpha=0.6, edgecolor="white")

        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Attention %")
        ax.set_xlim(0, 100)
        ax.set_title(label, fontweight="bold")
        if idx == 0:
            ax.legend(loc="lower right", fontsize=9, frameon=True)

        # Annotate 0% doc
        for i, d in enumerate(doc_attn):
            if d < 1:
                ax.annotate("0%", xy=(2, i), fontsize=8, color="#ef4444",
                           fontweight="bold", va="center")

    fig.suptitle("Attention Patterns: Context-Override Heads Read the Question, Not the Document",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_attention_patterns.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig3_attention_patterns.pdf"))
    print("  ✓ Fig 3: Attention patterns")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Fig 4: Steering Dose-Response
# ═══════════════════════════════════════════════════════════
def plot_steering():
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_key, (label, color, n_layers) in MODELS.items():
        data = load(model_key, "steering_explicit_context.json")
        if not data:
            continue

        alphas = []
        sps_vals = []
        baseline = data["results"]["0.0"]["mean_sps"]
        for a_str, r in data["results"].items():
            alphas.append(r["alpha"])
            sps_vals.append(r["mean_sps"] - baseline)

        ax.plot(alphas, sps_vals, "o-", label=f"{label} (L{data['target_layer']})",
                color=color, markersize=6, linewidth=2)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Steering strength (α)")
    ax.set_ylabel("ΔSPS from baseline")
    ax.set_title("Activation Steering: Directional Control of Source Preference",
                 fontweight="bold")
    ax.legend(frameon=True, fancybox=True)

    # Annotate the 1.5B inversion
    ax.annotate("1.5B inverted:\nmemory-defense heads",
               xy=(-50, 0.49), fontsize=8, color="#7c3aed",
               ha="center", va="bottom",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3e8ff", alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_steering.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig4_steering.pdf"))
    print("  ✓ Fig 4: Steering dose-response")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Fig 5: Probing Crystallization
# ═══════════════════════════════════════════════════════════
def plot_probing():
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_key, (label, color, n_layers) in MODELS.items():
        # Try ambiguous first (better class balance), fall back to explicit
        data = load(model_key, "probing_ambiguous_conflict.json")
        regime_label = "amb"
        if not data:
            data = load(model_key, "probing_explicit_context.json")
            regime_label = "exp"
        if not data:
            continue

        layers = data["layers"]
        xs = [l["depth_pct"] for l in layers]
        ys = [l["accuracy"] * 100 for l in layers]
        n_ctx = data["n_context"]
        n_mem = data["n_memory"]
        majority = max(n_ctx, n_mem) / (n_ctx + n_mem) * 100

        ax.plot(xs, ys, "o-", label=f"{label} ({regime_label}, {n_ctx}c/{n_mem}m)",
                color=color, markersize=3, linewidth=1.8, alpha=0.85)
        ax.axhline(majority, color=color, linestyle=":", alpha=0.3, linewidth=1)

    ax.axhline(50, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Depth (%)")
    ax.set_ylabel("Probe Accuracy (%)")
    ax.set_title("Linear Probing: Where Does the Source Decision Crystallize?",
                 fontweight="bold")
    ax.legend(frameon=True, fancybox=True, fontsize=9)
    ax.set_ylim(45, 100)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_probing.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig5_probing.pdf"))
    print("  ✓ Fig 5: Probing crystallization")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Fig 6: Cross-Model Summary Dashboard
# ═══════════════════════════════════════════════════════════
def plot_summary_dashboard():
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Patching peak layers
    ax_a = fig.add_subplot(gs[0, 0])
    model_names = []
    peak_depths = []
    colors_list = []
    for model_key, (label, color, n_layers) in MODELS.items():
        data = load(model_key, "residual_patching_explicit_context.json")
        if data:
            peak_layer = data["peak_layer"]
            peak_depth = 100 * peak_layer / (n_layers - 1)
            model_names.append(label)
            peak_depths.append(peak_depth)
            colors_list.append(color)
    ax_a.barh(model_names, peak_depths, color=colors_list, alpha=0.8, edgecolor="white")
    ax_a.set_xlabel("Peak Patching Depth (%)")
    ax_a.set_title("A. Circuit Location", fontweight="bold")
    ax_a.set_xlim(80, 100)

    # Panel B: Top DLA per model
    ax_b = fig.add_subplot(gs[0, 1])
    model_names = []
    top_dlas = []
    colors_list = []
    for model_key, (label, color, n_layers) in MODELS.items():
        data = load(model_key, "dla_explicit_context.json")
        if data:
            top_dla = data["ranked"][0]["avg_dla"]
            model_names.append(label)
            top_dlas.append(top_dla)
            colors_list.append(color)
    ax_b.barh(model_names, top_dlas, color=colors_list, alpha=0.8, edgecolor="white")
    ax_b.set_xlabel("Top Head DLA Score")
    ax_b.set_title("B. Max Logit Attribution", fontweight="bold")

    # Panel C: Attention pattern (doc vs question avg)
    ax_c = fig.add_subplot(gs[0, 2])
    model_names = []
    avg_doc = []
    avg_q = []
    colors_list = []
    for model_key, (label, color, n_layers) in MODELS.items():
        data = load(model_key, "attention_patterns_explicit_context.json")
        if data:
            heads = list(data["heads"].values())
            doc_mean = np.mean([h["corrupt_doc_attn_avg"] for h in heads]) * 100
            q_mean = np.mean([h["corrupt_question_attn_avg"] for h in heads]) * 100
            model_names.append(label)
            avg_doc.append(doc_mean)
            avg_q.append(q_mean)
            colors_list.append(color)
    x = np.arange(len(model_names))
    ax_c.bar(x - 0.15, avg_doc, 0.3, label="Document", color="#ef4444", alpha=0.8)
    ax_c.bar(x + 0.15, avg_q, 0.3, label="Question", color="#3b82f6", alpha=0.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(model_names, fontsize=9)
    ax_c.set_ylabel("Attention %")
    ax_c.set_title("C. What Heads Attend To", fontweight="bold")
    ax_c.legend(fontsize=8)

    # Panel D: Steering effect
    ax_d = fig.add_subplot(gs[1, 0])
    model_names = []
    steer_effects = []
    colors_list = []
    for model_key, (label, color, n_layers) in MODELS.items():
        data = load(model_key, "steering_explicit_context.json")
        if data:
            baseline = data["results"]["0.0"]["mean_sps"]
            worst = data["results"]["-50.0"]["mean_sps"]
            delta = worst - baseline
            model_names.append(label)
            steer_effects.append(delta)
            colors_list.append(color)
    ax_d.barh(model_names, steer_effects, color=colors_list, alpha=0.8, edgecolor="white")
    ax_d.axvline(0, color="gray", linestyle="--", alpha=0.4)
    ax_d.set_xlabel("ΔSPS at α=-50")
    ax_d.set_title("D. Steering Effect", fontweight="bold")

    # Panel E: Head count per model
    ax_e = fig.add_subplot(gs[1, 1])
    model_names_e = []
    head_counts = []
    colors_e = []
    for model_key, (label, color, n_layers) in MODELS.items():
        data = load(model_key, "head_patching_explicit_context.json")
        if data:
            sig_heads = sum(1 for h in data["top_heads"] if abs(h["avg_delta_sps"]) > 0.1)
            model_names_e.append(label)
            head_counts.append(sig_heads)
            colors_e.append(color)
    ax_e.barh(model_names_e, head_counts, color=colors_e, alpha=0.8, edgecolor="white")
    ax_e.set_xlabel("# Heads with |ΔSPS| > 0.1")
    ax_e.set_title("E. Circuit Sparsity", fontweight="bold")

    # Panel F: Summary text
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    summary = (
        "Key Findings:\n\n"
        "1. Context-override circuit at 92-96%\n"
        "   depth (last 2-3 layers)\n\n"
        "2. Heads read question (50-74%),\n"
        "   NOT document (0%)\n\n"
        "3. Qwen-1.5B has memory-defense\n"
        "   heads (inverted DLA & steering)\n\n"
        "4. Gemma uses distributed circuit\n"
        "   (many weak heads vs few strong)"
    )
    ax_f.text(0.05, 0.95, summary, transform=ax_f.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", alpha=0.8))
    ax_f.set_title("F. Summary", fontweight="bold")

    fig.suptitle("Source Arbitration: Cross-Model Mechanistic Analysis",
                 fontweight="bold", fontsize=15, y=1.01)
    fig.savefig(os.path.join(FIG_DIR, "fig6_summary_dashboard.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig6_summary_dashboard.pdf"))
    print("  ✓ Fig 6: Summary dashboard")
    plt.close(fig)


def main():
    print("Phase 19: Generating publication figures...")
    plot_residual_patching()
    plot_dla_heatmap()
    plot_attention_patterns()
    plot_steering()
    plot_probing()
    plot_summary_dashboard()
    print(f"\nAll figures saved to: {FIG_DIR}/")

if __name__ == "__main__":
    main()
