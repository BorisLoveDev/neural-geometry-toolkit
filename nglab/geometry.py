"""Geometry and probe utilities for neural geometry experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")
    return x


def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def center(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=0, keepdims=True)


def pca_project(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    x = as_2d(x)
    pca = PCA(n_components=n_components)
    z = pca.fit_transform(center(x))
    return z, pca


def group_centroids(x: np.ndarray, values: Sequence[int | float | str]) -> tuple[np.ndarray, np.ndarray]:
    """Return centroids sorted by group value."""
    x = as_2d(x)
    df = pd.DataFrame({"value": list(values)})
    groups = []
    labels = []
    for value in sorted(df["value"].unique(), key=lambda v: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v))):
        idx = df.index[df["value"] == value].to_numpy()
        groups.append(x[idx].mean(axis=0))
        labels.append(value)
    return np.vstack(groups), np.asarray(labels)


def fit_circle_2d(points: np.ndarray) -> dict:
    """Algebraic least-squares circle fit for 2D points."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape [n, 2]")
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
    radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    residual = radii - radius
    return {
        "center": np.array([cx, cy], dtype=float),
        "radius": radius,
        "radii": radii,
        "radius_cv": float(radii.std() / (radii.mean() + 1e-12)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
    }


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed circular difference a-b in radians, in [-pi, pi)."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def align_angles(pred: np.ndarray, true: np.ndarray) -> dict:
    """Align predicted angles to true angles by choosing orientation and phase."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    best = None
    for orientation in [1.0, -1.0]:
        oriented = orientation * pred
        # circular mean of offset true - oriented
        offset = np.angle(np.mean(np.exp(1j * (true - oriented))))
        aligned = oriented + offset
        err = angle_diff(aligned, true)
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        candidate = {
            "orientation": orientation,
            "offset": float(offset),
            "aligned": aligned,
            "errors": err,
            "mae_rad": mae,
            "mae_deg": float(np.degrees(mae)),
            "rmse_rad": rmse,
            "rmse_deg": float(np.degrees(rmse)),
        }
        if best is None or candidate["mae_rad"] < best["mae_rad"]:
            best = candidate
    assert best is not None
    return best


def circular_concept_metrics(
    x: np.ndarray,
    values: Sequence[int],
    *,
    period: int,
    use_centroids: bool = True,
) -> dict:
    """Quantify how much a concept looks like a circle in a PCA plane.

    This is a lightweight diagnostic, not a proof of causality.
    """
    x = as_2d(x)
    values_arr = np.asarray(values)
    if use_centroids:
        x_eval, labels = group_centroids(x, values_arr % period)
    else:
        x_eval, labels = x, values_arr % period
    z, pca = pca_project(x_eval, 2)
    circle = fit_circle_2d(z)
    angles = np.arctan2(z[:, 1] - circle["center"][1], z[:, 0] - circle["center"][0])
    true_angles = 2 * np.pi * (labels.astype(float) % period) / period
    alignment = align_angles(angles, true_angles)

    # Similarity gap: are adjacent cyclic labels closer than non-adjacent labels?
    centroids, labels2 = group_centroids(x, values_arr % period)
    cn = l2_normalize(center(centroids))
    sim = cn @ cn.T
    adj, non = [], []
    for i, vi in enumerate(labels2.astype(int) % period):
        for j, vj in enumerate(labels2.astype(int) % period):
            if i >= j:
                continue
            d = min((vi - vj) % period, (vj - vi) % period)
            if d == 1:
                adj.append(sim[i, j])
            elif d > 1:
                non.append(sim[i, j])
    adjacent_mean = float(np.mean(adj)) if adj else float("nan")
    nonadjacent_mean = float(np.mean(non)) if non else float("nan")

    return {
        "period": int(period),
        "n_points": int(len(values_arr)),
        "n_groups": int(len(np.unique(values_arr % period))),
        "pca_explained_variance_1": float(pca.explained_variance_ratio_[0]),
        "pca_explained_variance_2": float(pca.explained_variance_ratio_[1]),
        "pca_explained_variance_2d": float(pca.explained_variance_ratio_[:2].sum()),
        "circle_radius_cv": circle["radius_cv"],
        "circle_rmse": circle["rmse"],
        "angle_mae_deg": alignment["mae_deg"],
        "angle_rmse_deg": alignment["rmse_deg"],
        "adjacent_cosine_mean": adjacent_mean,
        "nonadjacent_cosine_mean": nonadjacent_mean,
        "adjacent_similarity_gap": adjacent_mean - nonadjacent_mean,
    }


def _cv_for_n(n: int, folds: int, random_state: int):
    if n < 3:
        return None
    if n <= folds:
        return LeaveOneOut()
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def fourier_targets(y: Sequence[int | float], period: int, harmonic: int = 1) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    theta = 2 * np.pi * harmonic * y / period
    return np.column_stack([np.cos(theta), np.sin(theta)])


def fit_fourier_probe(
    x: np.ndarray,
    y: Sequence[int | float],
    *,
    period: int,
    harmonic: int = 1,
    alpha: float = 10.0,
    folds: int = 5,
    random_state: int = 0,
) -> dict:
    """Cross-validated ridge probe for cos/sin(2*pi*y/period)."""
    x = as_2d(x)
    target = fourier_targets(y, period, harmonic=harmonic)
    model = make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=alpha))
    cv = _cv_for_n(len(x), folds, random_state)
    if cv is None:
        pred = model.fit(x, target).predict(x)
        cv_note = "fit_on_train_too_few_samples"
    else:
        pred = cross_val_predict(model, x, target, cv=cv)
        cv_note = type(cv).__name__
    # Normalize predictions to angles; keep vector length as confidence diagnostic.
    true_angle = np.arctan2(target[:, 1], target[:, 0])
    pred_angle = np.arctan2(pred[:, 1], pred[:, 0])
    angle_err = angle_diff(pred_angle, true_angle)
    pred_norm = np.linalg.norm(pred, axis=1)
    return {
        "period": int(period),
        "harmonic": int(harmonic),
        "r2_cos": float(r2_score(target[:, 0], pred[:, 0])),
        "r2_sin": float(r2_score(target[:, 1], pred[:, 1])),
        "r2_mean": float(0.5 * (r2_score(target[:, 0], pred[:, 0]) + r2_score(target[:, 1], pred[:, 1]))),
        "angle_mae_deg": float(np.degrees(np.mean(np.abs(angle_err)))),
        "angle_rmse_deg": float(np.degrees(np.sqrt(np.mean(angle_err ** 2)))),
        "pred_norm_mean": float(pred_norm.mean()),
        "pred_norm_std": float(pred_norm.std()),
        "cv": cv_note,
        "pred": pred,
    }


def fit_fourier_probes(
    x: np.ndarray,
    y: Sequence[int | float],
    periods: Sequence[int],
    *,
    harmonic: int = 1,
    alpha: float = 10.0,
    folds: int = 5,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows = []
    preds: dict[int, np.ndarray] = {}
    for p in periods:
        res = fit_fourier_probe(
            x, y, period=int(p), harmonic=harmonic, alpha=alpha, folds=folds, random_state=random_state
        )
        preds[int(p)] = res.pop("pred")
        rows.append(res)
    return pd.DataFrame(rows), preds
