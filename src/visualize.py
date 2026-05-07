"""Plotting and Grad-CAM utilities."""

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import confusion_matrix, roc_curve

from .dataset import IMAGENET_MEAN, IMAGENET_STD, default_transform


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# ROC, score distribution, confusion matrix
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true, y_scores, dataset_name: str) -> Path:
    """Plot the ROC curve with shaded AUC region and a diagonal baseline."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = float(np.trapz(tpr, fpr))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC = {auc:.3f}")
    # Shade the area under the ROC curve for visual emphasis.
    ax.fill_between(fpr, tpr, alpha=0.2, color="#1f77b4")
    # Random classifier reference line.
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {dataset_name}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)

    out_path = RESULTS_DIR / f"{dataset_name}_roc_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_score_distribution(y_true, y_scores, dataset_name: str) -> Path:
    """Overlapping histogram of real vs fake predicted scores."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Separate real (0) and fake (1) scores for overlapping histograms.
    ax.hist(y_scores[y_true == 0], bins=30, color="#2ca02c",
            alpha=0.55, label="Real", edgecolor="black")
    ax.hist(y_scores[y_true == 1], bins=30, color="#d62728",
            alpha=0.55, label="Fake", edgecolor="black")
    # Decision threshold at 0.5.
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold = 0.5")
    ax.set_xlabel("Predicted fake probability")
    ax.set_ylabel("Number of videos")
    ax.set_title(f"Score Distribution — {dataset_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = RESULTS_DIR / f"{dataset_name}_score_dist.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(y_true, y_pred, dataset_name: str) -> Path:
    """Normalized confusion matrix rendered as a seaborn heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Row-normalize so each class sums to 1 -> easier to read when classes are imbalanced.
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (normalized) — {dataset_name}")

    out_path = RESULTS_DIR / f"{dataset_name}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_comparison_bar(metrics_ff: dict, metrics_celeb: dict) -> Path:
    """Grouped bar chart comparing FF++ and Celeb-DF v2 across metrics."""
    metric_names = ["AUC", "Accuracy", "F1", "Precision", "Recall"]
    ff_vals = [metrics_ff[m] for m in metric_names]
    celeb_vals = [metrics_celeb[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - width / 2, ff_vals, width, label="FF++", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, celeb_vals, width, label="Celeb-DF v2", color="#ff7f0e")

    # Annotate each bar with its numeric value for readability.
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("FF++ vs Celeb-DF v2 — Metric Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path = RESULTS_DIR / "comparison_bar.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Invert ImageNet normalization and return an HxWx3 float array in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.detach().cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def generate_gradcam(model, face_tensor: torch.Tensor, original_face_np: np.ndarray) -> np.ndarray:
    """Produce a Grad-CAM heatmap overlaid on the original face crop.

    `original_face_np` must be an HxWx3 RGB image in uint8 or float[0,1].
    """
    if face_tensor.dim() == 3:
        face_tensor = face_tensor.unsqueeze(0)
    device = next(model.parameters()).device
    face_tensor = face_tensor.to(device)

    # efficientnet_pytorch exposes the last conv as `_conv_head`.
    target_layer = model._conv_head
    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=face_tensor, targets=None)[0]

    # Ensure the base image is float in [0, 1] for overlay.
    if original_face_np.dtype != np.float32 and original_face_np.dtype != np.float64:
        base = original_face_np.astype(np.float32) / 255.0
    else:
        base = np.clip(original_face_np, 0, 1).astype(np.float32)

    # Resize base to match CAM if needed (CAM is 224x224).
    if base.shape[:2] != grayscale_cam.shape:
        base = cv2.resize(base, (grayscale_cam.shape[1], grayscale_cam.shape[0]))

    overlay = show_cam_on_image(base, grayscale_cam, use_rgb=True)
    return overlay


# ---------------------------------------------------------------------------
# Sample visualization (correct + wrong predictions with Grad-CAM)
# ---------------------------------------------------------------------------

def _find_example_frame_from_predictions(predictions, video_name: str) -> str | None:
    """Look up a representative cached frame path for a video_name."""
    for p in predictions:
        if p["video_name"] == video_name and p.get("frames"):
            return p["frames"][0]
    return None


def visualize_samples(detector, manifest_path: str, dataset_name: str,
                      n_correct: int = 8, n_wrong: int = 4,
                      predictions: list | None = None) -> Path:
    """Visualize a grid of correct + misclassified samples with Grad-CAM overlays.

    If `predictions` is not supplied, this function will run inference on the
    full manifest via `detector.predict_dataset(manifest_path)`.
    """
    # Reuse prior predictions when available to avoid expensive re-inference.
    if predictions is None:
        predictions = detector.predict_dataset(manifest_path)

    correct = [p for p in predictions if p["pred_label"] == p["true_label"] and p.get("frames")]
    wrong = [p for p in predictions if p["pred_label"] != p["true_label"] and p.get("frames")]

    random.seed(42)
    random.shuffle(correct)
    random.shuffle(wrong)

    picks = correct[:n_correct] + wrong[:n_wrong]
    if len(picks) == 0:
        print(f"[visualize] No samples available for {dataset_name}.")
        return RESULTS_DIR / f"{dataset_name}_gradcam_samples.png"

    transform = default_transform()

    # Grid: one row per sample, two columns (face | grad-cam).
    n_rows = len(picks)
    fig, axes = plt.subplots(n_rows, 2, figsize=(6, 2.8 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for i, pred in enumerate(picks):
        frame_path = pred["frames"][0]
        face_rgb = np.array(Image.open(frame_path).convert("RGB"))
        tensor = transform(Image.fromarray(face_rgb))

        # Generate Grad-CAM overlay.
        cam_img = generate_gradcam(detector.model, tensor, face_rgb)

        true_name = "FAKE" if pred["true_label"] == 1 else "REAL"
        pred_name = "FAKE" if pred["pred_label"] == 1 else "REAL"
        status = "correct" if pred["pred_label"] == pred["true_label"] else "wrong"
        title = (f"[{status}] True: {true_name} | Pred: {pred_name} | "
                 f"Score: {pred['pred_score']:.2f}")

        axes[i, 0].imshow(face_rgb)
        axes[i, 0].set_title("Original", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cam_img)
        axes[i, 1].set_title("Grad-CAM", fontsize=9)
        axes[i, 1].axis("off")

        # Row-level caption as a suptitle-like label on the left column.
        axes[i, 0].set_ylabel(title, fontsize=8)

    fig.suptitle(f"Grad-CAM samples — {dataset_name}", fontsize=12)

    out_path = RESULTS_DIR / f"{dataset_name}_gradcam_samples.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
