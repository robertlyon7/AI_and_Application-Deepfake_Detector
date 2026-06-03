"""Fusion strategies for combining per-model deepfake scores.

Each function takes per-model arrays of per-video fake-probability scores and
returns a single fused score array. All inputs are assumed aligned (same order
of videos across models).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


def _stack(scores_per_model: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a list of per-model score lists into a (N, M) matrix.

    N = number of videos, M = number of models.
    """
    arr = np.asarray(scores_per_model, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected (M, N) input; got shape {arr.shape}")
    return arr.T  # -> (N, M)


def fuse_mean(scores_per_model: Sequence[Sequence[float]]) -> np.ndarray:
    """Simple arithmetic mean across models (baseline ensemble)."""
    X = _stack(scores_per_model)
    return X.mean(axis=1)


def fuse_weighted(
    scores_per_model: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> np.ndarray:
    """Convex combination with caller-supplied non-negative weights.

    Weights are renormalized so they sum to 1, so callers can pass any
    positive numbers (e.g. AUC-derived weights).
    """
    X = _stack(scores_per_model)
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != X.shape[1]:
        raise ValueError(f"Got {len(weights)} weights for {X.shape[1]} models")
    w = w / w.sum()
    return X @ w


def fuse_max(scores_per_model: Sequence[Sequence[float]]) -> np.ndarray:
    """Per-video maximum across models.

    Favours recall: any single model claiming "fake" with high confidence wins.
    """
    return _stack(scores_per_model).max(axis=1)


def fuse_logistic(
    val_scores_per_model: Sequence[Sequence[float]],
    val_labels: Sequence[int],
    test_scores_per_model: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Logistic-regression stacking: train a meta-classifier on validation data.

    Returns (test_scores, learned_coefficients_including_bias). Fitting on the
    validation split avoids test-set leakage, which is the standard protocol
    when you tune any combination rule.
    """
    Xv = _stack(val_scores_per_model)
    yv = np.asarray(val_labels, dtype=int)
    Xt = _stack(test_scores_per_model)

    meta = LogisticRegression(max_iter=1000)
    meta.fit(Xv, yv)
    test_scores = meta.predict_proba(Xt)[:, 1]
    coefs = np.concatenate([meta.coef_.ravel(), meta.intercept_])
    return test_scores, coefs


def align_predictions(*pred_lists: list[dict]) -> tuple[list[str], list[int], list[np.ndarray]]:
    """Align multiple per-model prediction lists by video_name.

    Returns:
      - video_names: list of length N
      - true_labels: list of length N (ground truth, taken from the first list)
      - score_lists: list of M numpy arrays, each of length N, in the same
        order as the input pred_lists.

    Skips any video that is not present in *all* prediction lists.
    """
    if not pred_lists:
        raise ValueError("At least one prediction list required.")

    # Build index per model: video_name -> (true_label, pred_score)
    indexes = [
        {p["video_name"]: (int(p["true_label"]), float(p["pred_score"])) for p in pl}
        for pl in pred_lists
    ]

    # Use the first model's videos as the canonical order, filtered to the
    # intersection of all model coverages.
    common = [name for name in indexes[0] if all(name in idx for idx in indexes[1:])]

    video_names = common
    true_labels = [indexes[0][name][0] for name in common]
    score_lists = [
        np.array([idx[name][1] for name in common], dtype=float)
        for idx in indexes
    ]
    return video_names, true_labels, score_lists
