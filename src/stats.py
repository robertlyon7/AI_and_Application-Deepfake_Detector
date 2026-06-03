"""Bootstrap confidence intervals and paired comparison tests for AUC/F1/etc.

These helpers let the notebook turn point estimates ("AUC = 0.95") into
defensible statistical claims ("AUC = 0.95 [0.93, 0.97], significantly higher
than the single-model baseline at p = 0.003").
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_FNS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "AUC":       lambda y, s: roc_auc_score(y, s),
    "Accuracy":  lambda y, s: accuracy_score(y, (s >= 0.5).astype(int)),
    "F1":        lambda y, s: f1_score(y, (s >= 0.5).astype(int), zero_division=0),
    "Precision": lambda y, s: precision_score(y, (s >= 0.5).astype(int), zero_division=0),
    "Recall":    lambda y, s: recall_score(y, (s >= 0.5).astype(int), zero_division=0),
}


def bootstrap_ci(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    metric: str = "AUC",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute a point estimate and bootstrap CI for one metric.

    Stratified resampling: we draw real and fake samples with replacement
    separately so the bootstrap dataset always has both classes (otherwise AUC
    is undefined on any resample that ends up single-class).
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_scores, dtype=float)
    fn = METRIC_FNS[metric]

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)

    point = float(fn(y, s))
    samples = []
    for _ in range(n_bootstrap):
        # Stratified bootstrap: separate resample within each class.
        pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pos, neg])
        try:
            samples.append(float(fn(y[idx], s[idx])))
        except ValueError:
            # Rare degenerate case (e.g. all-same class after resample): skip.
            continue

    samples = np.asarray(samples)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return {
        "metric": metric,
        "point": point,
        "lo": lo,
        "hi": hi,
        "n_bootstrap": int(len(samples)),
        "confidence": confidence,
    }


def paired_bootstrap_test(
    y_true: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    metric: str = "AUC",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test: is metric(A) > metric(B) on the same labels?

    On each bootstrap resample we recompute metric(A) and metric(B) using the
    SAME resampled indices (that's the "paired" part — controls for sample
    variation) and record their difference. The two-sided p-value is twice the
    proportion of bootstrap samples whose sign opposes the observed difference.

    Returns the observed difference, its CI, and a two-sided p-value.
    """
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    fn = METRIC_FNS[metric]

    observed = float(fn(y, a) - fn(y, b))

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)

    diffs = []
    for _ in range(n_bootstrap):
        pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pos, neg])
        try:
            d = float(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
            diffs.append(d)
        except ValueError:
            continue
    diffs = np.asarray(diffs)

    # Two-sided p-value via the "achieved significance level" reflection:
    # under the null (no real difference) the bootstrap distribution is roughly
    # symmetric around 0, so the p-value is twice the proportion of the tail
    # opposite the observed sign.
    if observed >= 0:
        p_value = float(2.0 * np.mean(diffs <= 0.0))
    else:
        p_value = float(2.0 * np.mean(diffs >= 0.0))
    p_value = min(max(p_value, 0.0), 1.0)

    lo = float(np.quantile(diffs, 0.025))
    hi = float(np.quantile(diffs, 0.975))
    return {
        "metric": metric,
        "observed_diff": observed,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_value,
        "n_bootstrap": int(len(diffs)),
    }
