"""
Phase 7: Lock model configs and verify reproducibility.
Saves model architecture metadata (num_layers, num_heads, hidden_dim, etc.)
to results/latest/{model}/model_config.json for all three models.
This must be run before causal tracing so all patching scripts
know the architecture dimensions without loading the full model again.
"""
import sys, os, json, datetime
from transformers import AutoConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from source_selection.data import save_json

MODELS = {
    "qwen2_5_3b_instruct":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_5_1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma_2_2b_it":         "google/gemma-2-2b-it",
}

def extract_arch(cfg) -> dict:
    """Extract key architecture fields regardless of config class."""
    def _get(*keys, default=None):
        for k in keys:
            v = getattr(cfg, k, None)
            if v is not None:
                return v
        return default

    return {
        "num_layers":      _get("num_hidden_layers"),
        "num_heads":       _get("num_attention_heads"),
        "num_kv_heads":    _get("num_key_value_heads", default=_get("num_attention_heads")),
        "hidden_dim":      _get("hidden_size"),
        "intermediate":    _get("intermediate_size"),
        "head_dim":        _get("head_dim",
                               default=(_get("hidden_size",0) // max(1,_get("num_attention_heads",1)))),
        "vocab_size":      _get("vocab_size"),
        "max_position":    _get("max_position_embeddings"),
        "model_type":      _get("model_type"),
        "architectures":   _get("architectures"),
        "torch_dtype":     str(_get("torch_dtype", default="")),
    }

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    configs_out = {}

    for model_name, model_id in MODELS.items():
        print(f"Loading config: {model_id}")
        try:
            cfg = AutoConfig.from_pretrained(model_id)
            arch = extract_arch(cfg)
            arch["model_id"] = model_id
            arch["timestamp"] = datetime.datetime.now().isoformat()

            out_path = os.path.join(base_dir, "results", "latest", model_name, "model_config.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_json(out_path, arch)
            configs_out[model_name] = arch

            print(f"  layers={arch['num_layers']}, heads={arch['num_heads']}, "
                  f"kv_heads={arch['num_kv_heads']}, hidden={arch['hidden_dim']}")
        except Exception as e:
            print(f"  [ERROR] {model_name}: {e}")

    # Save combined summary
    save_json(
        os.path.join(base_dir, "results", "latest", "all_model_configs.json"),
        configs_out
    )
    print("\nModel config locking complete.")

if __name__ == "__main__":
    main()
