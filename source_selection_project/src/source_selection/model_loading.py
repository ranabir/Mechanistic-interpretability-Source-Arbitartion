"""Model loading utilities with device detection and reproducibility metadata."""
import os, sys, json, datetime, subprocess, platform
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_dtype(device):
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32

def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"

def load_model_and_tokenizer(model_id: str, device: str = None, dtype=None):
    """
    Load a causal LM and its tokenizer.
    Returns (model, tokenizer, device, dtype) or raises with a helpful message.
    """
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = get_dtype(device)

    print(f"  Loading tokenizer: {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        if "gated" in str(e).lower() or "token" in str(e).lower():
            print(f"\n[ERROR] {model_id} requires Hugging Face authentication.")
            print("  1. Run: huggingface-cli login")
            print(f"  2. Accept the license at: https://huggingface.co/{model_id}")
            raise
        raise

    print(f"  Loading model: {model_id} -> device={device}, dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer, device, dtype

def save_run_metadata(model_id: str, device: str, dtype, config_file: str,
                      seed: int, out_path: str):
    import transformers
    meta = {
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "platform": platform.platform(),
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config_file": config_file,
        "random_seed": seed,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta
