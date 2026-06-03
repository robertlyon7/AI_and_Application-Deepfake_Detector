"""DeepfakeDetector wrapping a pretrained SBI EfficientNet-B4 model."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from PIL import Image
from tqdm import tqdm

from .dataset import default_transform, xception_transform
from .face_utils import detect_and_crop_face, iter_video_frames


def _load_sbi_state_dict(weights_path: str) -> dict:
    """Load an SBI checkpoint and strip the `net.` wrapper prefix.

    The official SBI release wraps the EfficientNet-PyTorch model under a
    `net.` attribute (the "Detector" class), so checkpoint keys look like
    `net._conv_stem.weight`. We strip that single prefix so the keys map
    directly onto an `EfficientNet` instance.
    """
    checkpoint = torch.load(weights_path, map_location="cpu")

    # Unwrap nested containers commonly used in research code.
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    state_dict = checkpoint if isinstance(checkpoint, dict) else {}
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        # Only strip wrapper prefixes — keep the `_conv_stem`, `_fc` etc.
        for prefix in ("module.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


class DeepfakeDetector:
    """Pretrained SBI deepfake detector built on EfficientNet-B4."""

    # Human-readable model id; used to tag saved predictions per detector.
    name = "sbi"

    def __init__(self, weights_path: str):
        # Auto-detect device with CUDA preference.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SBI was trained with the `efficientnet_pytorch` library, NOT timm.
        # The classifier head has 2 outputs (real vs fake), so we use num_classes=2
        # and apply softmax to recover the fake probability at inference time.
        self.model = EfficientNet.from_name("efficientnet-b4", num_classes=2)

        weights_path = str(weights_path)
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Weights not found at {weights_path}. "
                f"Run scripts/download_weights.py first."
            )

        state_dict = _load_sbi_state_dict(weights_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            print(f"[detector:{self.name}] Warning: {len(missing)} missing keys during load.")
            print(f"[detector:{self.name}] Example missing: {missing[:3]}")
        if len(unexpected) > 0:
            print(f"[detector:{self.name}] Warning: {len(unexpected)} unexpected keys during load.")
            print(f"[detector:{self.name}] Example unexpected: {unexpected[:3]}")

        self.model.to(self.device)
        self.model.eval()

        self.transform = default_transform()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[detector:{self.name}] Backbone: efficientnet-b4 (efficientnet_pytorch, num_classes=2)")
        print(f"[detector:{self.name}] Parameters: {n_params/1e6:.2f}M")
        print(f"[detector:{self.name}] Device: {self.device}")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_image(self, face_tensor: torch.Tensor) -> float:
        """Predict the fake probability of a single normalized face tensor.

        `face_tensor` may be shaped (3, 224, 224) or (1, 3, 224, 224).
        Returns a Python float in [0, 1].
        """
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        logits = self.model(face_tensor)  # (1, 2)
        # 2-class softmax; index 1 is the FAKE class probability.
        probs = F.softmax(logits, dim=1)
        score = probs[0, 1].item()
        return float(score)

    @torch.inference_mode()
    def predict_video(self, video_path: str, sample_every: int = 10) -> dict:
        """Run inference on every `sample_every`-th frame of the given video."""
        video_name = Path(video_path).stem
        frame_scores = []

        for frame_idx, frame_bgr in iter_video_frames(video_path, sample_every=sample_every):
            face_rgb = detect_and_crop_face(frame_bgr)
            if face_rgb is None:
                continue
            # Convert RGB numpy crop -> PIL -> normalized tensor.
            pil = Image.fromarray(face_rgb)
            tensor = self.transform(pil)
            score = self.predict_image(tensor)
            frame_scores.append({"frame_idx": int(frame_idx), "score": float(score)})

        # Free GPU memory between videos to stay inside 4 GB VRAM budgets.
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Aggregate per-video decision from mean of frame scores.
        mean_score = float(np.mean([f["score"] for f in frame_scores])) if frame_scores else 0.0
        label = "FAKE" if mean_score >= 0.5 else "REAL"
        return {
            "video_name": video_name,
            "frame_scores": frame_scores,
            "mean_score": mean_score,
            "label": label,
            "confidence": mean_score * 100.0,
        }

    @torch.inference_mode()
    def predict_dataset(self, manifest_path: str) -> list[dict]:
        """Run inference on every cached frame in the manifest, grouped by video.

        Returns a list of per-video prediction dicts and writes the raw predictions
        to `results/{dataset_name}_predictions.json` (dataset name inferred from the
        manifest path).
        """
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Derive a clean dataset identifier from the manifest path layout
        # (…/cache/<dataset_name>/manifest.json).
        dataset_name = manifest_path.parent.name

        predictions = []
        for entry in tqdm(manifest, desc=f"Inference on {dataset_name}"):
            video_name = entry["video"]
            true_label = int(entry["label"])
            frames = entry["frames"]

            scores = []
            for frame_path in frames:
                img = Image.open(frame_path).convert("RGB")
                tensor = self.transform(img)
                scores.append(self.predict_image(tensor))

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            mean_score = float(np.mean(scores)) if scores else 0.0
            predictions.append({
                "video_name": video_name,
                "true_label": true_label,
                "pred_score": mean_score,
                "pred_label": 1 if mean_score >= 0.5 else 0,
                "frames": frames,  # keep for downstream visualization / error analysis
            })

        # Persist raw predictions for downstream analysis.
        # Filename includes the model name so multiple detectors don't clobber
        # each other (e.g. ff++_sbi_predictions.json vs ff++_xception_predictions.json).
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{dataset_name}_{self.name}_predictions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            serializable = [
                {k: v for k, v in p.items() if k != "frames"} for p in predictions
            ]
            json.dump(serializable, f, indent=2)
        print(f"[detector:{self.name}] Wrote predictions -> {out_path}")

        return predictions


# ---------------------------------------------------------------------------
# Xception (DeepfakeBench checkpoint)
# ---------------------------------------------------------------------------

def _load_xception_state_dict(weights_path: str) -> dict:
    """Load a DeepfakeBench Xception checkpoint and remap keys for timm.

    The checkpoint stores parameters under `backbone.*` and uses
    `last_linear.{weight,bias}` for the 2-class classifier head. The timm
    `legacy_xception` model expects bare keys without the `backbone.` prefix
    and names the classifier `fc.{weight,bias}`. DeepfakeBench's training
    setup also adds an `adjust_channel` head that we don't need at inference;
    those keys are intentionally dropped.
    """
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    cleaned = {}
    for k, v in checkpoint.items():
        nk = k
        # Strip wrapper prefixes used by DeepfakeBench.
        for prefix in ("module.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        # Skip the auxiliary projection head that's only used during training.
        if nk.startswith("adjust_channel"):
            continue
        # DeepfakeBench: last_linear.* ; timm legacy_xception: fc.*
        if nk.startswith("last_linear"):
            nk = "fc" + nk[len("last_linear"):]
        cleaned[nk] = v
    return cleaned


class XceptionDetector:
    """Deepfake detector using Xception trained on FaceForensics++ (DeepfakeBench).

    Acts as a second model for ensembling with `DeepfakeDetector` (SBI). Uses
    the timm `legacy_xception` backbone (Cadene's classic Xception) with the
    classifier head replaced for 2-class output.
    """

    name = "xception"

    def __init__(self, weights_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # legacy_xception matches the DeepfakeBench checkpoint's architecture.
        self.model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)

        weights_path = str(weights_path)
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Xception weights not found at {weights_path}. "
                f"Run scripts/download_weights.py to fetch them."
            )

        state_dict = _load_xception_state_dict(weights_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            print(f"[detector:{self.name}] Warning: {len(missing)} missing keys.")
            print(f"[detector:{self.name}] Example missing: {missing[:3]}")
        if len(unexpected) > 0:
            print(f"[detector:{self.name}] Warning: {len(unexpected)} unexpected keys.")
            print(f"[detector:{self.name}] Example unexpected: {unexpected[:3]}")

        self.model.to(self.device)
        self.model.eval()

        self.transform = xception_transform()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[detector:{self.name}] Backbone: legacy_xception (timm, num_classes=2)")
        print(f"[detector:{self.name}] Parameters: {n_params/1e6:.2f}M")
        print(f"[detector:{self.name}] Device: {self.device}")

    # The Xception inference/aggregation logic is identical to SBI; we just
    # reuse `DeepfakeDetector`'s methods by binding them to this class.
    predict_image = DeepfakeDetector.predict_image
    predict_video = DeepfakeDetector.predict_video
    predict_dataset = DeepfakeDetector.predict_dataset
