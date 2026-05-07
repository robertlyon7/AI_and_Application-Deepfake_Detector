"""Metrics computation and dataset-level evaluation."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _compute_eer(y_true, y_scores) -> float:
    """Compute the Equal Error Rate from the ROC curve.

    EER is the threshold point where FPR == FNR (i.e. 1 - TPR).
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find the index where |FPR - FNR| is minimized.
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer


def compute_metrics(y_true, y_scores) -> dict:
    """Compute standard deepfake-detection metrics at threshold 0.5."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    y_pred = (y_scores >= 0.5).astype(int)

    # zero_division=0 prevents warnings when a class has no predictions.
    return {
        "AUC": float(roc_auc_score(y_true, y_scores)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "EER": _compute_eer(y_true, y_scores),
    }


def _print_metrics_table(metrics: dict, title: str) -> None:
    """Pretty-print a metrics dict as an aligned table."""
    print()
    print(f"=== {title} ===")
    print(f"{'Metric':<12} {'Value':>10}")
    print("-" * 24)
    for k in ("AUC", "Accuracy", "F1", "Precision", "Recall", "EER"):
        print(f"{k:<12} {metrics[k]:>10.4f}")
    print()


def evaluate_dataset(detector, manifest_path: str, dataset_name: str):
    """Run the detector on the cached dataset and compute + persist metrics.

    Returns (metrics_dict, predictions_list).
    """
    predictions = detector.predict_dataset(manifest_path)

    y_true = [p["true_label"] for p in predictions]
    y_scores = [p["pred_score"] for p in predictions]
    metrics = compute_metrics(y_true, y_scores)

    _print_metrics_table(metrics, f"Results on {dataset_name}")

    # Save metrics for later reuse / archiving.
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset_name}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Wrote metrics -> {out_path}")

    return metrics, predictions
