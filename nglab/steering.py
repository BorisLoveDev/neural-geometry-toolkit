"""Lightweight activation steering helpers.

These helpers are intentionally conservative. They support common Hugging Face
CausalLM families (GPT-2, Llama/Mistral/Qwen-style, GPT-NeoX, OPT) and are meant
for quick causal sanity checks after you have found a promising layer/subspace.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from .geometry import group_centroids, pca_project, fit_circle_2d


def find_decoder_layers(model):
    """Return the transformer block list for common decoder-only architectures."""
    candidates = [
        ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("model.decoder.layers", lambda m: getattr(getattr(getattr(m, "model", None), "decoder", None), "layers", None)),
        ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
        ("gpt_neox.layers", lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
        ("transformer.blocks", lambda m: getattr(getattr(m, "transformer", None), "blocks", None)),
    ]
    for name, getter in candidates:
        layers = getter(model)
        if layers is not None:
            try:
                if len(layers) > 0:
                    return layers
            except TypeError:
                pass
    raise ValueError(
        "Could not find decoder layers for this model. Add a custom accessor in nglab/steering.py."
    )


def hidden_state_index_to_module_index(model, hidden_state_index: int) -> int:
    """Map outputs.hidden_states index to transformer block module index.

    hidden_states[0] is the embedding output, hidden_states[1] is after block 0.
    Therefore hidden_state_index=18 maps to module index 17. Negative indices
    follow Python convention; -1 maps to the last transformer block.
    """
    layers = find_decoder_layers(model)
    n = len(layers)
    idx = int(hidden_state_index)
    if idx < 0:
        return n + idx
    if idx == 0:
        raise ValueError("Steering hidden_states[0] / embeddings is not supported by this block hook.")
    module_idx = idx - 1
    if module_idx < 0 or module_idx >= n:
        raise ValueError(f"Layer index {hidden_state_index} maps to invalid module index {module_idx}; model has {n} blocks")
    return module_idx


@contextmanager
def activation_addition_hook(model, *, hidden_state_layer: int, delta: np.ndarray | torch.Tensor, token_position: int = -1):
    """Temporarily add delta to a residual stream position at a layer.

    hidden_state_layer uses the same convention as outputs.hidden_states.
    token_position=-1 means the last token in the forward pass.
    """
    layers = find_decoder_layers(model)
    module_idx = hidden_state_index_to_module_index(model, hidden_state_layer)
    module = layers[module_idx]

    delta_t = torch.as_tensor(delta)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None
        d = delta_t.to(device=h.device, dtype=h.dtype)
        while d.ndim < h.ndim:
            d = d.unsqueeze(0)
        h2 = h.clone()
        h2[:, token_position, :] = h2[:, token_position, :] + d.reshape(1, -1)
        if rest is None:
            return h2
        return (h2, *rest)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def next_token_topk(loaded, prompt: str, *, top_k: int = 10, hidden_state_layer: int | None = None, delta=None):
    """Return top-k next-token probabilities, optionally with activation steering."""
    tokenizer = loaded.tokenizer
    model = loaded.model
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)

    ctx = activation_addition_hook(model, hidden_state_layer=hidden_state_layer, delta=delta) if delta is not None else None
    with torch.inference_mode():
        if ctx is None:
            out = model(**encoded, return_dict=True)
        else:
            with ctx:
                out = model(**encoded, return_dict=True)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        vals, idxs = torch.topk(probs, k=top_k)
    rows = []
    for prob, idx in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
        rows.append({"token_id": int(idx), "token": tokenizer.decode([idx]), "prob": float(prob)})
    return pd.DataFrame(rows)


def centroid_delta(activations: np.ndarray, values: Sequence[int], *, from_value: int, to_value: int, period: int) -> np.ndarray:
    """Delta from one cyclic concept centroid to another in activation space."""
    centroids, group_values = group_centroids(activations, np.asarray(values) % period)
    lookup = {int(v) % period: i for i, v in enumerate(group_values)}
    return centroids[lookup[to_value % period]] - centroids[lookup[from_value % period]]


def pca_circle_delta(
    activations: np.ndarray,
    values: Sequence[int],
    *,
    from_value: int,
    to_angle_value: float,
    period: int,
) -> np.ndarray:
    """Continuous circle delta using a PCA plane fitted to cyclic centroids.

    This is a simple approximation of manifold steering: move from a source
    centroid to a point on the fitted circle at angle 2π*to_angle_value/period,
    then lift that 2D point back through PCA.
    """
    values_arr = np.asarray(values)
    centroids, group_values = group_centroids(activations, values_arr % period)
    z, pca = pca_project(centroids, 2)
    circle = fit_circle_2d(z)

    lookup = {int(v) % period: i for i, v in enumerate(group_values)}
    source_vec = centroids[lookup[from_value % period]]

    # Align true cyclic angle to fitted circle orientation/phase by using the observed source angle.
    source_z = z[lookup[from_value % period]]
    source_angle_observed = np.arctan2(source_z[1] - circle["center"][1], source_z[0] - circle["center"][0])
    source_angle_true = 2 * np.pi * (from_value % period) / period
    phase = source_angle_observed - source_angle_true
    target_angle = 2 * np.pi * (to_angle_value % period) / period + phase
    target_z = circle["center"] + circle["radius"] * np.array([np.cos(target_angle), np.sin(target_angle)])

    # PCA inverse transform returns centered-space plus pca.mean_; pca was fit on centered centroids in pca_project,
    # whose mean is close to zero. Add original centroid mean to lift into activation coordinates.
    centered_target = pca.inverse_transform(target_z.reshape(1, -1))[0]
    target_vec = centered_target + centroids.mean(axis=0)
    return target_vec - source_vec
