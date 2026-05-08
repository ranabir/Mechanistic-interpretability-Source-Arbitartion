"""
Source preference scoring.

Score(S) = max_{t in T_context} logit(t) - max_{t in T_memory} logit(t)

Positive => model prefers context answer.
Negative => model prefers memory answer.

Works for both:
  - single_token prompts: T_context/T_memory are pre-validated single-token ids.
  - multi_token prompts:  T_context/T_memory use ALL variant token ids (first subword),
    providing an approximate preference signal useful for behavioral analysis.
"""
import torch
from typing import List, Optional


@torch.no_grad()
def get_answer_position_logits(
    model, tokenizer, prompt: str, device: str
) -> torch.Tensor:
    """
    Tokenize prompt and return logits at the final (answer) token position.
    Returns shape [vocab_size].
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # logits: [1, seq_len, vocab_size]
    logits = outputs.logits[0, -1, :]  # last position
    return logits.float().cpu()


def compute_source_score(
    logits: torch.Tensor,
    tokenizer,
    context_variants: List[str],
    memory_variants: List[str],
    single_token_only: bool = True,
) -> dict:
    """
    Compute the source preference score given logits at the answer position.

    Args:
        logits: [vocab_size] float tensor
        tokenizer: HF tokenizer
        context_variants: list of string variants for the context answer
        memory_variants: list of string variants for the memory answer
        single_token_only: if True, only uses variants that encode to 1 token.
            If False, uses all variants (first subword token of each).

    Returns dict with:
        source_score, context_max_logit, memory_max_logit,
        context_token_ids, memory_token_ids,
        predicted_source, top_k_tokens
    """
    ctx_ids = _get_token_ids(tokenizer, context_variants, single_token_only)
    mem_ids = _get_token_ids(tokenizer, memory_variants, single_token_only)

    if not ctx_ids or not mem_ids:
        return {
            "source_score": None,
            "context_max_logit": None,
            "memory_max_logit": None,
            "context_token_ids": ctx_ids,
            "memory_token_ids": mem_ids,
            "predicted_source": "invalid",
            "top_5_tokens": _top_k_tokens(logits, tokenizer, k=5),
        }

    ctx_logits = logits[ctx_ids]
    mem_logits = logits[mem_ids]

    ctx_max = ctx_logits.max().item()
    mem_max = mem_logits.max().item()
    score = ctx_max - mem_max

    predicted = "context" if score > 0 else ("memory" if score < 0 else "tied")

    return {
        "source_score": round(score, 6),
        "context_max_logit": round(ctx_max, 6),
        "memory_max_logit": round(mem_max, 6),
        "context_token_ids": ctx_ids,
        "memory_token_ids": mem_ids,
        "predicted_source": predicted,
        "top_5_tokens": _top_k_tokens(logits, tokenizer, k=5),
    }


def _get_token_ids(tokenizer, variants: List[str], single_token_only: bool) -> List[int]:
    ids = []
    for v in variants:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if single_token_only:
            if len(toks) == 1:
                ids.append(toks[0])
        else:
            if toks:
                ids.append(toks[0])  # first subword approximation
    return list(set(ids))


def _top_k_tokens(logits: torch.Tensor, tokenizer, k: int = 5) -> List[dict]:
    topk = torch.topk(logits, k)
    return [
        {"token": tokenizer.decode([tid.item()]), "token_id": tid.item(),
         "logit": round(lv.item(), 4)}
        for tid, lv in zip(topk.indices, topk.values)
    ]
