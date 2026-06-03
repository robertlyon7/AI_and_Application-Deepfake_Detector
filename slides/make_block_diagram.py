"""Render the DUET-DF project block diagram to a JPG."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PATH = Path(__file__).parent / "block_diagram.jpg"

# Colour palette.
C_INPUT = "#d9d9d9"
C_PREP = "#bdbdbd"
C_SBI = "#1f77b4"
C_XC = "#ff7f0e"
C_FUSION = "#2ca02c"
C_EVAL = "#9467bd"
C_INTERP = "#8c564b"
C_TEXT_DARK = "#1a1a1a"


def box(ax, x, y, w, h, text, facecolor, textcolor="white", fontsize=10, bold=True):
    """Draw a rounded box centred at (x, y) with wrapped text."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor="#404040", facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x, y, text, ha="center", va="center",
        fontsize=fontsize, color=textcolor,
        fontweight="bold" if bold else "normal",
    )


def arrow(ax, x1, y1, x2, y2, color="#404040"):
    """Draw an arrow from (x1, y1) to (x2, y2)."""
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=1.6, color=color,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 13))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis("off")

    # --- Title ---
    ax.text(6, 15.5, "DUET-DF — Project Pipeline", ha="center",
            fontsize=15, fontweight="bold", color=C_TEXT_DARK)

    # --- Input ---
    box(ax, 6, 14.4, 5.4, 0.9,
        "INPUT VIDEOS\nFaceForensics++ c23  ·  Celeb-DF v2",
        C_INPUT, textcolor=C_TEXT_DARK, fontsize=10)
    arrow(ax, 6, 13.95, 6, 13.35)

    # --- Preprocessing ---
    box(ax, 6, 12.7, 7.2, 1.25,
        "PREPROCESSING  (shared)\n"
        "sample every 10th frame  →  MTCNN face detection\n"
        "→  crop (+12.5% margin)  →  cache 380x380 JPG",
        C_PREP, textcolor=C_TEXT_DARK, fontsize=9.5)

    # Fork down to the two base models.
    arrow(ax, 6, 12.05, 3.2, 11.05)
    arrow(ax, 6, 12.05, 8.8, 11.05)

    # --- Base model A: SBI ---
    box(ax, 3.2, 10.3, 4.4, 1.5,
        "BASE MODEL A — SBI\nEfficientNet-B4\n"
        "input 380x380, /255\nblending-boundary cue",
        C_SBI, fontsize=9.5)
    # --- Base model B: Xception ---
    box(ax, 8.8, 10.3, 4.4, 1.5,
        "BASE MODEL B — Xception\nDeepfakeBench (FF++)\n"
        "input 299x299, [-1,1]\ntexture cue",
        C_XC, fontsize=9.5)

    arrow(ax, 3.2, 9.55, 3.2, 8.95)
    arrow(ax, 8.8, 9.55, 8.8, 8.95)

    # --- Per-model aggregation ---
    box(ax, 3.2, 8.35, 4.4, 1.0,
        "per-frame softmax\n→  mean over frames  →  s_SBI",
        "#aec7e8", textcolor=C_TEXT_DARK, fontsize=9.5, bold=False)
    box(ax, 8.8, 8.35, 4.4, 1.0,
        "per-frame softmax\n→  mean over frames  →  s_XC",
        "#ffbb78", textcolor=C_TEXT_DARK, fontsize=9.5, bold=False)

    # Merge into fusion.
    arrow(ax, 3.2, 7.85, 6, 6.95)
    arrow(ax, 8.8, 7.85, 6, 6.95)

    # --- Late fusion ---
    box(ax, 6, 6.2, 8.0, 1.45,
        "LATE FUSION  —  4 strategies\n"
        "mean   ·   weighted   ·   max   ·   logistic stacking\n"
        "(tuned on held-out 20% Celeb-DF — no leakage)",
        C_FUSION, fontsize=10)

    arrow(ax, 6, 5.475, 6, 4.95)
    ax.text(6.25, 5.2, "ensemble score  s_E", ha="left", va="center",
            fontsize=9, style="italic", color=C_TEXT_DARK)

    # Fork to evaluation + interpretation.
    arrow(ax, 6, 4.55, 3.2, 3.65)
    arrow(ax, 6, 4.55, 8.8, 3.65)

    # --- Evaluation ---
    box(ax, 3.2, 2.85, 4.6, 1.6,
        "EVALUATION\nAUC · F1 · EER\nAccuracy · Precision · Recall\n"
        "bootstrap 95% CIs\npaired significance test",
        C_EVAL, fontsize=9.5)
    # --- Interpretation ---
    box(ax, 8.8, 2.85, 4.6, 1.6,
        "INTERPRETATION\nGrad-CAM  (pixel level)\n"
        "SHAP  (ensemble level)\n"
        "where each model looks\n+ how it contributes",
        C_INTERP, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", format="jpg")
    print(f"Saved block diagram -> {OUT_PATH}")


if __name__ == "__main__":
    main()
