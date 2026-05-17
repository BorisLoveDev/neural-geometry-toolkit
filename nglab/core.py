"""Model loading and activation extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


@dataclass
class LoadedModel:
    model_name: str
    tokenizer: object
    model: object
    device: torch.device | str


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_lm(
    model_name: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    device_map: str | None = None,
    trust_remote_code: bool = False,
) -> LoadedModel:
    """Load a Hugging Face causal LM with a matching tokenizer.

    For large models, pass device_map="auto". For CPU experiments, omit it.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding keeps the last non-padding token at the final sequence index.
    tokenizer.padding_side = "left"

    torch_dtype = None
    if dtype == "auto":
        if resolved_device == "cuda":
            # bfloat16 is safer on modern GPUs, fp16 on older ones.
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif dtype in {"float16", "fp16"}:
        torch_dtype = torch.float16
    elif dtype in {"bfloat16", "bf16"}:
        torch_dtype = torch.bfloat16
    elif dtype in {"float32", "fp32"}:
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    kwargs = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    if device_map is None:
        model.to(resolved_device)
    return LoadedModel(model_name=model_name, tokenizer=tokenizer, model=model, device=resolved_device)


def _model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _token_spans_from_offsets(offset_mapping, start: int, end: int) -> list[int]:
    if start is None or end is None or start < 0 or end <= start:
        return []
    positions = []
    for i, (a, b) in enumerate(offset_mapping):
        # Ignore special/pad tokens with zero-width offsets.
        if a == b:
            continue
        if a < end and b > start:
            positions.append(i)
    return positions


def _last_nonpad_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    # Works for both left and right padding: last index where mask == 1.
    positions = []
    for row in attention_mask:
        nz = torch.nonzero(row, as_tuple=False).flatten()
        positions.append(nz[-1] if len(nz) else torch.tensor(0, device=row.device))
    return torch.stack(positions)


def extract_activations(
    loaded: LoadedModel,
    df: pd.DataFrame,
    *,
    layers: Sequence[int] = (-1,),
    batch_size: int = 8,
    extraction: str = "target_or_last",
    show_progress: bool = True,
) -> dict[int, np.ndarray]:
    """Extract hidden states for prompts in df.

    extraction:
      - "target_or_last": average hidden states over target_start/target_end when present, else final token
      - "target": require target span and average over it
      - "last": final non-padding token
    Returns {layer_index: activations [n_prompts, hidden_dim]}.
    """
    if "prompt" not in df.columns:
        raise ValueError("DataFrame must contain a 'prompt' column")
    extraction = extraction.lower()
    if extraction not in {"target_or_last", "target", "last"}:
        raise ValueError("extraction must be one of: target_or_last, target, last")

    tokenizer = loaded.tokenizer
    model = loaded.model
    device = _model_input_device(model)
    layers = tuple(int(l) for l in layers)
    out: dict[int, list[np.ndarray]] = {l: [] for l in layers}

    prompts = df["prompt"].tolist()
    starts = df["target_start"].tolist() if "target_start" in df.columns else [-1] * len(df)
    ends = df["target_end"].tolist() if "target_end" in df.columns else [-1] * len(df)

    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="extracting activations")

    for start_i in iterator:
        batch_prompts = prompts[start_i : start_i + batch_size]
        batch_starts = starts[start_i : start_i + batch_size]
        batch_ends = ends[start_i : start_i + batch_size]

        tok_kwargs = dict(return_tensors="pt", padding=True, truncation=False)
        # Fast tokenizers can return offsets; slow tokenizers cannot.
        want_offsets = extraction in {"target_or_last", "target"}
        offsets = None
        if want_offsets:
            try:
                encoded = tokenizer(batch_prompts, return_offsets_mapping=True, **tok_kwargs)
                offsets = encoded.pop("offset_mapping").tolist()
            except Exception:
                encoded = tokenizer(batch_prompts, **tok_kwargs)
        else:
            encoded = tokenizer(batch_prompts, **tok_kwargs)

        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        last_positions = _last_nonpad_positions(encoded["attention_mask"])

        span_positions: list[list[int]] = []
        if offsets is not None:
            for off, s, e in zip(offsets, batch_starts, batch_ends):
                span_positions.append(_token_spans_from_offsets(off, int(s), int(e)))
        else:
            span_positions = [[] for _ in batch_prompts]

        for layer in layers:
            hs = hidden_states[layer].detach().float().cpu()  # [batch, seq, dim]
            vectors = []
            for bi in range(hs.shape[0]):
                span = span_positions[bi]
                if extraction == "last" or (extraction == "target_or_last" and not span):
                    vectors.append(hs[bi, int(last_positions[bi].cpu().item())].numpy())
                elif extraction == "target":
                    if not span:
                        raise ValueError(
                            f"Could not locate target span for prompt index {start_i + bi}: {batch_prompts[bi]!r}"
                        )
                    vectors.append(hs[bi, span].mean(dim=0).numpy())
                else:
                    vectors.append(hs[bi, span].mean(dim=0).numpy())
            out[layer].append(np.vstack(vectors))

        del outputs, hidden_states
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {layer: np.vstack(chunks) for layer, chunks in out.items()}


def save_activation_bundle(path: str, df: pd.DataFrame, activations: dict[int, np.ndarray], *, extra: dict | None = None) -> None:
    """Save prompts metadata plus activations as a compressed npz."""
    arrays = {f"layer_{layer}": x for layer, x in activations.items()}
    meta_json = df.to_json(orient="records", force_ascii=False)
    import json

    payload = {"meta_json": np.array(meta_json), "layers": np.array(list(activations.keys()), dtype=int)}
    if extra is not None:
        payload["extra_json"] = np.array(json.dumps(extra, ensure_ascii=False, default=str))
    np.savez_compressed(path, **arrays, **payload)
