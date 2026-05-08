"""
Residual stream patching utilities for causal tracing.

Algorithm (per matched pair):
  1. Run CLEAN prompt → capture residual stream at every layer: h_clean[l]
  2. Run CORRUPT prompt → get corrupt source preference score (baseline)
  3. For each layer l:
       - Run corrupt prompt but REPLACE residual stream at layer l with h_clean[l]
       - Measure patched source preference score
       - ΔSPS[l] = patched_score - corrupt_score
  4. Positive ΔSPS[l] → patching layer l from clean restored memory preference
     → that layer is causally responsible for context override

Implementation uses PyTorch forward hooks (no library dependencies).
"""
import torch
from typing import List, Callable, Optional


class ResidualStreamPatcher:
    """
    Captures and patches residual stream states using forward hooks.
    Works with any HuggingFace causal LM that uses standard layer norms
    (Qwen2, Gemma2, LLaMA-style architectures).
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self._captured = {}

    def _get_layers(self):
        """Return the list of transformer blocks."""
        # Qwen2: model.model.layers
        # Gemma2: model.model.layers
        # LLaMA: model.model.layers
        # Works for all standard decoder-only models
        return self.model.model.layers

    def capture_clean(self, input_ids: torch.Tensor) -> dict:
        """
        Run a forward pass and capture the residual stream
        (output of each transformer block) at every layer.
        Returns {layer_idx: tensor of shape [seq_len, hidden_dim]}.
        """
        captured = {}
        handles = []

        layers = self._get_layers()
        for i, layer in enumerate(layers):
            def make_hook(idx):
                def hook(module, input, output):
                    # output is a tuple; first element is the hidden state tensor
                    h = output[0] if isinstance(output, tuple) else output
                    captured[idx] = h.detach().clone()  # [1, seq_len, hidden]
                return hook
            handles.append(layer.register_forward_hook(make_hook(i)))

        with torch.no_grad():
            self.model(input_ids)

        for h in handles:
            h.remove()

        return captured

    def patch_at_layer(
        self,
        corrupt_input_ids: torch.Tensor,
        clean_states: dict,
        patch_layer: int,
        answer_position: int = -1
    ) -> torch.Tensor:
        """
        Run corrupt prompt but replace the residual stream at `patch_layer`
        with the corresponding clean states. Returns logits at answer_position.
        """
        handles = []
        layers = self._get_layers()

        clean_h = clean_states[patch_layer]  # [1, seq_len, hidden]

        def patch_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # Align sequence lengths (corrupt may be longer than clean due to doc)
            # We patch the LAST min(seq_len_clean, seq_len_corrupt) positions
            # but only the final token position is what we care about for logits.
            # Strategy: patch only the last token position of the residual stream.
            # This localises the intervention to the answer slot.
            seq_len = h.shape[1]
            clean_seq = clean_h.shape[1]

            if isinstance(output, tuple):
                h_new = h.clone()
                # Patch last token of clean into last token of corrupt
                h_new[0, -1, :] = clean_h[0, -1, :].to(h.device)
                return (h_new,) + output[1:]
            else:
                h_new = output.clone()
                h_new[0, -1, :] = clean_h[0, -1, :].to(h.device)
                return h_new

        handles.append(layers[patch_layer].register_forward_hook(patch_hook))

        with torch.no_grad():
            outputs = self.model(corrupt_input_ids)

        for h in handles:
            h.remove()

        logits = outputs.logits[0, answer_position, :].float().cpu()
        return logits

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class HeadPatcher:
    """
    Patches individual attention head outputs within a specific layer.

    Strategy:
      1. Run CLEAN → capture the attention layer's output (post o_proj)
         and decompose by head: head_out_clean[h] for each head h
      2. Run CORRUPT → but hook the attention layer and replace one head's
         contribution with the clean version
      3. Measure ΔSPS for that head

    Works by hooking the self_attn module and manipulating its output.
    Since we can't easily decompose the fused o_proj, we use a different
    approach: capture the full attention output for both clean and corrupt,
    decompose by head dimension, and swap individual head slices.
    """

    def __init__(self, model, num_heads: int, head_dim: int):
        self.model = model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim

    def _get_layers(self):
        return self.model.model.layers

    def _get_attn_module(self, layer_idx: int):
        return self._get_layers()[layer_idx].self_attn

    def capture_attn_output(self, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Run forward pass and capture the attention module output at layer_idx.
        Returns tensor [1, seq_len, hidden_dim].
        """
        captured = {}

        def hook(module, input, output):
            # output is typically a tuple: (attn_output, attn_weights, past_kv)
            # or (attn_output,) depending on config
            h = output[0] if isinstance(output, tuple) else output
            captured["out"] = h.detach().clone()

        attn = self._get_attn_module(layer_idx)
        handle = attn.register_forward_hook(hook)

        with torch.no_grad():
            self.model(input_ids)

        handle.remove()
        return captured["out"]  # [1, seq_len, hidden_dim]

    def patch_single_head(
        self,
        corrupt_input_ids: torch.Tensor,
        clean_attn_out: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        answer_position: int = -1,
    ) -> torch.Tensor:
        """
        Run corrupt forward pass but replace head_idx's slice of the attention
        output at layer_idx with the clean version.
        Returns logits at answer_position.
        """
        hd = self.head_dim
        start = head_idx * hd
        end = start + hd

        def patch_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            h_new = h.clone()
            # Patch only last token, only for the specific head's slice
            h_new[0, -1, start:end] = clean_attn_out[0, -1, start:end].to(h.device)
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        attn = self._get_attn_module(layer_idx)
        handle = attn.register_forward_hook(patch_hook)

        with torch.no_grad():
            outputs = self.model(corrupt_input_ids)

        handle.remove()
        logits = outputs.logits[0, answer_position, :].float().cpu()
        return logits
