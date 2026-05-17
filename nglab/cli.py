"""Command-line interface for Neural Geometry Lab."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .core import extract_activations, load_lm, save_activation_bundle
from .datasets import (
    MONTHS,
    WEEKDAYS,
    make_addition_dataset,
    make_cyclic_addition_dataset,
    make_month_dataset,
    make_number_dataset,
    make_weekday_dataset,
)
from .geometry import circular_concept_metrics, fit_fourier_probes
from .plotting import (
    plot_concept_circle,
    plot_fourier_scores,
    plot_layer_heatmap,
    plot_linear_vs_arc,
    plot_predicted_mod_circle,
)


def parse_int_list(s: str) -> list[int]:
    s = str(s).strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def resolve_layers(layer_arg: str, loaded) -> list[int]:
    layer_arg = str(layer_arg).strip().lower()
    if layer_arg in {"all", "*"}:
        n = int(getattr(loaded.model.config, "num_hidden_layers", 0))
        # Include embedding layer 0? For mechanistic reports, transformer blocks are usually 1..n or -1.
        # Here we use hidden_states tuple indices: 0 is embedding output, 1..n are block outputs.
        return list(range(0, n + 1))
    return parse_int_list(layer_arg)


def limit_contexts(df: pd.DataFrame, n_contexts: int | None) -> pd.DataFrame:
    if n_contexts is None or n_contexts <= 0 or "context" not in df.columns:
        return df
    contexts = list(dict.fromkeys(df["context"].tolist()))[:n_contexts]
    return df[df["context"].isin(contexts)].reset_index(drop=True)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="gpt2", help="Hugging Face model id, e.g. gpt2 or meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--layers", default="-1", help="Comma-separated hidden_state indices, e.g. -1,18, or 'all'. 0 is embeddings.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto", help="auto/cuda/cpu/mps")
    p.add_argument("--dtype", default="auto", help="auto/fp16/bf16/fp32")
    p.add_argument("--device-map", default=None, help="Use 'auto' for large models with accelerate.")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--outdir", default="runs/nglab")
    p.add_argument("--no-progress", action="store_true")


def run_weekdays(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = limit_contexts(make_weekday_dataset(monday_is=0), args.contexts)
    loaded = load_lm(
        args.model,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    layers = resolve_layers(args.layers, loaded)
    acts = extract_activations(
        loaded, df, layers=layers, batch_size=args.batch_size,
        extraction="target_or_last", show_progress=not args.no_progress,
    )
    save_activation_bundle(str(outdir / "activations_weekdays.npz"), df, acts, extra=vars(args))
    df.to_csv(outdir / "prompts_weekdays.csv", index=False)

    metrics = []
    for layer, x in acts.items():
        m = circular_concept_metrics(x, df["value"].to_numpy(), period=7)
        m["layer"] = int(layer)
        metrics.append(m)
        plot_concept_circle(
            x,
            df["value"].to_numpy(),
            df["label"].tolist(),
            period=7,
            title=f"Weekday activation geometry — {args.model}, layer {layer}",
            outpath=outdir / f"weekday_circle_layer_{layer}.png",
        )
        plot_linear_vs_arc(
            x,
            df["value"].to_numpy(),
            WEEKDAYS,
            period=7,
            start_value=0,
            end_value=4,
            title=f"Linear chord vs weekday manifold arc — layer {layer}",
            outpath=outdir / f"weekday_linear_vs_arc_layer_{layer}.png",
        )
    pd.DataFrame(metrics).to_csv(outdir / "metrics_weekdays.csv", index=False)
    _write_json(outdir / "metrics_weekdays.json", metrics)
    print(f"Saved weekday report to {outdir}")


def run_months(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = limit_contexts(make_month_dataset(january_is=1), args.contexts)
    loaded = load_lm(
        args.model,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    layers = resolve_layers(args.layers, loaded)
    acts = extract_activations(
        loaded, df, layers=layers, batch_size=args.batch_size,
        extraction="target_or_last", show_progress=not args.no_progress,
    )
    save_activation_bundle(str(outdir / "activations_months.npz"), df, acts, extra=vars(args))
    df.to_csv(outdir / "prompts_months.csv", index=False)

    metrics = []
    for layer, x in acts.items():
        m = circular_concept_metrics(x, df["value"].to_numpy(), period=12)
        m["layer"] = int(layer)
        metrics.append(m)
        plot_concept_circle(
            x,
            df["value"].to_numpy(),
            df["label"].tolist(),
            period=12,
            title=f"Month activation geometry — {args.model}, layer {layer}",
            outpath=outdir / f"month_circle_layer_{layer}.png",
        )
    pd.DataFrame(metrics).to_csv(outdir / "metrics_months.csv", index=False)
    _write_json(outdir / "metrics_months.json", metrics)
    print(f"Saved month report to {outdir}")


def run_numbers(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    periods = parse_int_list(args.periods)
    df = limit_contexts(make_number_dataset(args.start, args.end), args.contexts)
    loaded = load_lm(
        args.model,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    layers = resolve_layers(args.layers, loaded)
    acts = extract_activations(
        loaded, df, layers=layers, batch_size=args.batch_size,
        extraction="target_or_last", show_progress=not args.no_progress,
    )
    save_activation_bundle(str(outdir / "activations_numbers.npz"), df, acts, extra=vars(args))
    df.to_csv(outdir / "prompts_numbers.csv", index=False)

    all_scores = []
    for layer, x in acts.items():
        scores, preds = fit_fourier_probes(
            x, df["value"].to_numpy(), periods=periods, alpha=args.alpha, folds=args.folds, random_state=args.seed
        )
        scores["layer"] = int(layer)
        all_scores.append(scores)
        scores.to_csv(outdir / f"fourier_scores_numbers_layer_{layer}.csv", index=False)
        plot_fourier_scores(
            scores,
            title=f"Number Fourier probes — {args.model}, layer {layer}",
            outpath=outdir / f"fourier_scores_numbers_layer_{layer}.png",
        )
        for p, pred in preds.items():
            # One point per prompt can be busy; annotate only for small datasets.
            plot_predicted_mod_circle(
                pred,
                df["value"].to_numpy(),
                period=p,
                title=f"Predicted number positions mod {p} — layer {layer}",
                outpath=outdir / f"number_mod_{p}_circle_layer_{layer}.png",
                annotate=len(df) <= 120,
            )
    combined = pd.concat(all_scores, ignore_index=True)
    combined.to_csv(outdir / "fourier_scores_numbers_all_layers.csv", index=False)
    if len(layers) > 1:
        plot_layer_heatmap(
            combined,
            metric="r2_mean",
            title=f"Number Fourier probe layer sweep — {args.model}",
            outpath=outdir / "fourier_scores_numbers_heatmap.png",
        )
    print(f"Saved number report to {outdir}")


def run_addition(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    periods = parse_int_list(args.periods)
    df = limit_contexts(make_addition_dataset(args.max_a, args.max_b, min_a=args.min_a, min_b=args.min_b), args.contexts)
    loaded = load_lm(
        args.model,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    layers = resolve_layers(args.layers, loaded)
    acts = extract_activations(
        loaded, df, layers=layers, batch_size=args.batch_size,
        extraction="last", show_progress=not args.no_progress,
    )
    save_activation_bundle(str(outdir / "activations_addition.npz"), df, acts, extra=vars(args))
    df.to_csv(outdir / "prompts_addition.csv", index=False)

    all_scores = []
    for layer, x in acts.items():
        scores, preds = fit_fourier_probes(
            x, df["sum"].to_numpy(), periods=periods, alpha=args.alpha, folds=args.folds, random_state=args.seed
        )
        scores["layer"] = int(layer)
        all_scores.append(scores)
        scores.to_csv(outdir / f"fourier_scores_addition_layer_{layer}.csv", index=False)
        plot_fourier_scores(
            scores,
            title=f"Addition Fourier probes for a+b — {args.model}, layer {layer}",
            outpath=outdir / f"fourier_scores_addition_layer_{layer}.png",
        )
        for p, pred in preds.items():
            plot_predicted_mod_circle(
                pred,
                df["sum"].to_numpy(),
                period=p,
                title=f"Predicted addition sum mod {p} — layer {layer}",
                outpath=outdir / f"addition_sum_mod_{p}_circle_layer_{layer}.png",
                annotate=False,
            )
    combined = pd.concat(all_scores, ignore_index=True)
    combined.to_csv(outdir / "fourier_scores_addition_all_layers.csv", index=False)
    if len(layers) > 1:
        plot_layer_heatmap(
            combined,
            metric="r2_mean",
            title=f"Addition Fourier probe layer sweep — {args.model}",
            outpath=outdir / "fourier_scores_addition_heatmap.png",
        )
    print(f"Saved addition report to {outdir}")


def run_cyclic(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    periods = parse_int_list(args.periods)
    df = limit_contexts(make_cyclic_addition_dataset(args.kind, max_offset=args.max_offset), args.contexts)
    loaded = load_lm(
        args.model,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    layers = resolve_layers(args.layers, loaded)
    acts = extract_activations(
        loaded, df, layers=layers, batch_size=args.batch_size,
        extraction="last", show_progress=not args.no_progress,
    )
    save_activation_bundle(str(outdir / f"activations_cyclic_{args.kind}.npz"), df, acts, extra=vars(args))
    df.to_csv(outdir / f"prompts_cyclic_{args.kind}.csv", index=False)

    all_scores = []
    for layer, x in acts.items():
        for target_col in ["premod_sum", "output_value"]:
            scores, preds = fit_fourier_probes(
                x, df[target_col].to_numpy(), periods=periods, alpha=args.alpha, folds=args.folds, random_state=args.seed
            )
            scores["layer"] = int(layer)
            scores["target"] = target_col
            all_scores.append(scores)
            scores.to_csv(outdir / f"fourier_scores_cyclic_{args.kind}_{target_col}_layer_{layer}.csv", index=False)
            plot_fourier_scores(
                scores,
                title=f"Cyclic {args.kind}: probe {target_col} — {args.model}, layer {layer}",
                outpath=outdir / f"fourier_scores_cyclic_{args.kind}_{target_col}_layer_{layer}.png",
            )
    combined = pd.concat(all_scores, ignore_index=True)
    combined.to_csv(outdir / f"fourier_scores_cyclic_{args.kind}_all_layers.csv", index=False)
    if len(layers) > 1:
        for target_col in ["premod_sum", "output_value"]:
            subset = combined[combined["target"] == target_col]
            plot_layer_heatmap(
                subset,
                metric="r2_mean",
                title=f"Cyclic {args.kind} {target_col} layer sweep — {args.model}",
                outpath=outdir / f"fourier_scores_cyclic_{args.kind}_{target_col}_heatmap.png",
            )
    print(f"Saved cyclic report to {outdir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neural Geometry Lab: inspect activation manifolds in language models")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("weekdays", help="Visualize weekday concept geometry")
    _common_model_args(p)
    p.add_argument("--contexts", type=int, default=None, help="Limit number of prompt templates")
    p.set_defaults(func=run_weekdays)

    p = sub.add_parser("months", help="Visualize month concept geometry")
    _common_model_args(p)
    p.add_argument("--contexts", type=int, default=None)
    p.set_defaults(func=run_months)

    p = sub.add_parser("numbers", help="Probe number activations for Fourier/modular circles")
    _common_model_args(p)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=99)
    p.add_argument("--periods", default="2,5,10,20,50,100")
    p.add_argument("--contexts", type=int, default=None)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=run_numbers)

    p = sub.add_parser("addition", help="Probe a+b prompts for Fourier features of the sum")
    _common_model_args(p)
    p.add_argument("--min-a", type=int, default=0)
    p.add_argument("--min-b", type=int, default=0)
    p.add_argument("--max-a", type=int, default=20)
    p.add_argument("--max-b", type=int, default=20)
    p.add_argument("--periods", default="2,5,10,20,50")
    p.add_argument("--contexts", type=int, default=None)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=run_addition)

    p = sub.add_parser("cyclic", help="Probe cyclic arithmetic prompts: weekdays/months/hours")
    _common_model_args(p)
    p.add_argument("--kind", default="weekday", choices=["weekday", "month", "hour"])
    p.add_argument("--max-offset", type=int, default=None)
    p.add_argument("--periods", default="2,5,7,10,12,24")
    p.add_argument("--contexts", type=int, default=None)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=run_cyclic)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
