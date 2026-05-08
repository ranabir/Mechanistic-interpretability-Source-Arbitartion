"""
Phase 11: Attention Pattern Baselines.

For each model's top context-override heads, extract the attention weights
on both clean and corrupt prompts. Measure:
  1. What fraction of attention goes to DOCUMENT tokens vs QUESTION tokens
  2. How this changes between clean (no doc) and corrupt (with doc) prompts
  3. Whether the top heads preferentially attend to the conflict sentence

This tells us the "reading" side of the circuit: where the heads look.

Output: results/latest/{model}/attention_patterns_{regime}.json
"""
import sys, os, json, datetime, random, re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from source_selection.data import load_jsonl, save_json
from source_selection.model_loading import get_device, get_dtype

MODELS = {
    "qwen2_5_3b_instruct":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_5_1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma_2_2b_it":         "google/gemma-2-2b-it",
}

REGIMES = ["explicit_context", "ambiguous_conflict"]
MAX_PAIRS = 40
SEED = 42


def find_token_spans(tokenizer, prompt, conflict_sentence, subject):
    """
    Tokenize prompt and identify spans:
      - doc_span: tokens belonging to the document/conflict sentence
      - question_span: tokens belonging to the question
      - subject_span: tokens of the subject entity
    Returns dict of span names -> list of token indices.
    """
    tokens = tokenizer.encode(prompt)
    token_strs = [tokenizer.decode([t]) for t in tokens]

    # Find conflict sentence tokens
    conflict_toks = tokenizer.encode(conflict_sentence, add_special_tokens=False)
    doc_indices = []
    for start in range(len(tokens) - len(conflict_toks) + 1):
        if tokens[start:start+len(conflict_toks)] == conflict_toks:
            doc_indices = list(range(start, start + len(conflict_toks)))
            break

    # Find "Question:" or "Q:" marker to separate doc from question
    full_text = tokenizer.decode(tokens)
    q_markers = ["Question:", "Q:"]
    q_start = len(tokens)  # default: end
    for marker in q_markers:
        marker_toks = tokenizer.encode(marker, add_special_tokens=False)
        for start in range(len(tokens) - len(marker_toks) + 1):
            if tokens[start:start+len(marker_toks)] == marker_toks:
                q_start = start
                break
        if q_start < len(tokens):
            break

    question_indices = list(range(q_start, len(tokens)))

    # Find subject tokens
    subject_toks = tokenizer.encode(subject, add_special_tokens=False)
    subject_indices = []
    for start in range(len(tokens) - len(subject_toks) + 1):
        if tokens[start:start+len(subject_toks)] == subject_toks:
            subject_indices = list(range(start, start + len(subject_toks)))
            # Take first occurrence
            break

    return {
        "doc": doc_indices,
        "question": question_indices,
        "subject": subject_indices,
        "all": list(range(len(tokens))),
        "total_len": len(tokens),
    }


def extract_attention(model, tokenizer, prompt, target_heads, device):
    """
    Run forward pass with attention output and extract attention weights
    for specified heads at the last token position.
    Returns dict: (layer, head) -> attention weights [seq_len].
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    attn_weights = outputs.attentions  # tuple of [1, num_heads, seq, seq]
    result = {}
    for (layer, head) in target_heads:
        # Get attention from last token position to all positions
        w = attn_weights[layer][0, head, -1, :].float().cpu().numpy()
        result[(layer, head)] = w

    return result


def run_attention_analysis(model_name, model_id, base_dir):
    print(f"\n{'='*60}")
    print(f"  Phase 11: Attention Patterns — {model_name}")
    print(f"{'='*60}")

    res_dir = os.path.join(base_dir, "results", "latest", model_name)
    pairs_dir = os.path.join(base_dir, "data", "causal_tracing_pairs", model_name)

    # Check if all regimes already done
    all_done = all(
        os.path.exists(os.path.join(res_dir, f"attention_patterns_{r}.json"))
        for r in REGIMES
    )
    if all_done:
        print(f"  [SKIP] All regimes already done.")
        return

    # Load model once
    device = get_device()
    dtype = get_dtype(device)
    if "1.5B" in model_id or "1.5b" in model_id:
        dtype = torch.float32
        print(f"  [NOTE] Using float32 for {model_id} (float16 causes NaN in eager attn)")
    print(f"  Loading {model_id} with eager attention (dtype={dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    for regime in REGIMES:
        out_path = os.path.join(res_dir, f"attention_patterns_{regime}.json")
        if os.path.exists(out_path):
            print(f"  [{regime}] SKIP — already done.")
            continue

        # Get top heads from Phase 9
        hp_path = os.path.join(res_dir, f"head_patching_{regime}.json")
        if not os.path.exists(hp_path):
            print(f"  [{regime}] SKIP — no head patching results.")
            continue
        hp = json.load(open(hp_path))
        top_heads = [(h["layer"], h["head"]) for h in hp["top_heads"][:5]]
        print(f"  [{regime}] Target heads: {top_heads}")

        pairs = load_jsonl(os.path.join(pairs_dir, f"pairs_{regime}.jsonl"))
        random.seed(SEED)
        if MAX_PAIRS and len(pairs) > MAX_PAIRS:
            pairs = random.sample(pairs, MAX_PAIRS)

        # Accumulators per head
        head_stats = {str(h): {
            "layer": h[0], "head": h[1],
            "corrupt_doc_attn": [], "corrupt_question_attn": [], "corrupt_subject_attn": [],
            "clean_question_attn": [], "clean_subject_attn": [],
        } for h in top_heads}

        for pair in tqdm(pairs, desc=f"  {regime}", leave=False):
            spans_c = find_token_spans(
                tokenizer, pair["corrupt_prompt"],
                pair["conflict_sentence"], pair["subject"]
            )
            attn_c = extract_attention(model, tokenizer, pair["corrupt_prompt"], top_heads, device)
            for h_key, weights in attn_c.items():
                h_str = str(h_key)
                total = weights.sum()
                if total > 0:
                    doc_attn = weights[spans_c["doc"]].sum() / total if spans_c["doc"] else 0
                    q_attn = weights[spans_c["question"]].sum() / total if spans_c["question"] else 0
                    s_attn = weights[spans_c["subject"]].sum() / total if spans_c["subject"] else 0
                    head_stats[h_str]["corrupt_doc_attn"].append(float(doc_attn))
                    head_stats[h_str]["corrupt_question_attn"].append(float(q_attn))
                    head_stats[h_str]["corrupt_subject_attn"].append(float(s_attn))

            spans_cl = find_token_spans(
                tokenizer, pair["clean_prompt"],
                pair["conflict_sentence"], pair["subject"]
            )
            attn_cl = extract_attention(model, tokenizer, pair["clean_prompt"], top_heads, device)
            for h_key, weights in attn_cl.items():
                h_str = str(h_key)
                total = weights.sum()
                if total > 0:
                    q_attn = weights[spans_cl["question"]].sum() / total if spans_cl["question"] else 0
                    s_attn = weights[spans_cl["subject"]].sum() / total if spans_cl["subject"] else 0
                    head_stats[h_str]["clean_question_attn"].append(float(q_attn))
                    head_stats[h_str]["clean_subject_attn"].append(float(s_attn))

        summary = {}
        for h_str, stats in head_stats.items():
            avg = lambda lst: round(sum(lst)/len(lst), 4) if lst else 0
            summary[h_str] = {
                "layer": stats["layer"], "head": stats["head"],
                "corrupt_doc_attn_avg": avg(stats["corrupt_doc_attn"]),
                "corrupt_question_attn_avg": avg(stats["corrupt_question_attn"]),
                "corrupt_subject_attn_avg": avg(stats["corrupt_subject_attn"]),
                "clean_question_attn_avg": avg(stats["clean_question_attn"]),
                "clean_subject_attn_avg": avg(stats["clean_subject_attn"]),
                "n": len(stats["corrupt_doc_attn"]),
            }
            print(f"    {h_str}: doc={summary[h_str]['corrupt_doc_attn_avg']:.1%}, "
                  f"question={summary[h_str]['corrupt_question_attn_avg']:.1%}, "
                  f"subject={summary[h_str]['corrupt_subject_attn_avg']:.1%}")

        save_json(out_path, {
            "model": model_name,
            "regime": regime,
            "num_pairs": len(pairs),
            "seed": SEED,
            "timestamp": datetime.datetime.now().isoformat(),
            "heads": summary,
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for model_name, model_id in MODELS.items():
        run_attention_analysis(model_name, model_id, base_dir)
    print("\nAttention pattern analysis complete.")

if __name__ == "__main__":
    main()
