"""Plotting helpers for neural geometry experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geometry import fit_circle_2d, group_centroids, pca_project


def _ensure_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_concept_circle(
    x: np.ndarray,
    values: Sequence[int],
    labels: Sequence[str] | None,
    *,
    period: int,
    title: str,
    outpath: str | Path,
) -> Path:
    """PCA plot of concept centroids with a fitted circle and cyclic path."""
    values_arr = np.asarray(values)
    centroids, group_values = group_centroids(x, values_arr % period)
    z, pca = pca_project(centroids, 2)
    circle = fit_circle_2d(z)
    p = _ensure_path(outpath)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(z[:, 0], z[:, 1], s=80)

    # Draw fitted circle.
    theta = np.linspace(0, 2 * np.pi, 300)
    cx, cy = circle["center"]
    r = circle["radius"]
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), linewidth=1.5)

    # Draw cyclic path in true order.
    order = np.argsort(group_values.astype(float) % period)
    z_ordered = z[order]
    ax.plot(
        np.r_[z_ordered[:, 0], z_ordered[0, 0]],
        np.r_[z_ordered[:, 1], z_ordered[0, 1]],
        linestyle="--",
        linewidth=1,
    )

    label_lookup = None
    if labels is not None:
        # Choose first label seen for each group value.
        label_lookup = {}
        for v, lab in zip(values_arr % period, labels):
            label_lookup.setdefault(int(v), str(lab))
    for i, gv in enumerate(group_values):
        text = label_lookup.get(int(gv), str(gv)) if label_lookup else str(gv)
        ax.annotate(text, (z[i, 0], z[i, 1]), xytext=(5, 5), textcoords="offset points")

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def plot_linear_vs_arc(
    x: np.ndarray,
    values: Sequence[int],
    labels: Sequence[str],
    *,
    period: int,
    start_value: int,
    end_value: int,
    title: str,
    outpath: str | Path,
) -> Path:
    values_arr = np.asarray(values)
    centroids, group_values = group_centroids(x, values_arr % period)
    z, _ = pca_project(centroids, 2)
    circle = fit_circle_2d(z)
    p = _ensure_path(outpath)

    group_int = (group_values.astype(int) % period).tolist()
    sidx = group_int.index(start_value % period)
    eidx = group_int.index(end_value % period)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(z[:, 0], z[:, 1], s=80)
    for i, gv in enumerate(group_values):
        text = labels[group_int.index(int(gv))] if len(labels) >= len(group_values) else str(gv)
        ax.annotate(text, (z[i, 0], z[i, 1]), xytext=(5, 5), textcoords="offset points")

    # Straight chord.
    ax.plot([z[sidx, 0], z[eidx, 0]], [z[sidx, 1], z[eidx, 1]], linewidth=2, label="linear chord")

    # Arc between fitted circle angles, shortest positive cyclic path by value order.
    cx, cy = circle["center"]
    r = circle["radius"]
    ordered_values = list(range(start_value % period, end_value % period + 1))
    if end_value % period < start_value % period:
        ordered_values = list(range(start_value % period, period)) + list(range(0, end_value % period + 1))
    path_idxs = [group_int.index(v) for v in ordered_values if v in group_int]
    ax.plot(z[path_idxs, 0], z[path_idxs, 1], linestyle="--", linewidth=2, label="manifold arc")

    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), linewidth=1, alpha=0.6)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def plot_fourier_scores(scores: pd.DataFrame, *, title: str, outpath: str | Path) -> Path:
    p = _ensure_path(outpath)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = scores.sort_values("period")
    ax.bar([str(x) for x in s["period"]], s["r2_mean"])
    ax.set_ylim(min(-0.1, float(s["r2_mean"].min()) - 0.05), 1.0)
    ax.set_xlabel("period / modulus")
    ax.set_ylabel("cross-validated mean R² for cos/sin")
    ax.set_title(title)
    for i, val in enumerate(s["r2_mean"]):
        ax.text(i, float(val), f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def plot_predicted_mod_circle(
    pred: np.ndarray,
    y: Sequence[int | float],
    *,
    period: int,
    title: str,
    outpath: str | Path,
    annotate: bool = False,
) -> Path:
    pred = np.asarray(pred)
    y_arr = np.asarray(y)
    p = _ensure_path(outpath)
    fig, ax = plt.subplots(figsize=(7, 7))
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=1)
    ax.scatter(pred[:, 0], pred[:, 1], s=35)
    if annotate:
        for xi, yi, label in zip(pred[:, 0], pred[:, 1], y_arr):
            ax.annotate(str(int(label)), (xi, yi), fontsize=7, xytext=(2, 2), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(f"cos(2πn/{period}) probe")
    ax.set_ylabel(f"sin(2πn/{period}) probe")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def plot_layer_heatmap(scores: pd.DataFrame, *, metric: str, title: str, outpath: str | Path) -> Path:
    p = _ensure_path(outpath)
    pivot = scores.pivot_table(index="period", columns="layer", values=metric, aggfunc="mean").sort_index()
    fig, ax = plt.subplots(figsize=(max(7, pivot.shape[1] * 0.45), max(4, pivot.shape[0] * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=90)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("layer")
    ax.set_ylabel("period / modulus")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p
