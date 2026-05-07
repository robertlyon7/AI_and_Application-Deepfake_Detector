"""Face detection and cropping utilities using MTCNN."""

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN


# Global MTCNN instance to avoid reinitializing on every call.
_MTCNN = None


def get_mtcnn(device: str = None) -> MTCNN:
    """Return a lazily-constructed MTCNN detector pinned to the given device.

    We use MTCNN purely as a bounding-box detector and apply our own crop
    margin afterwards (SBI expects significant surrounding context).
    """
    global _MTCNN
    if _MTCNN is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _MTCNN = MTCNN(
            image_size=380,
            margin=0,
            keep_all=False,
            post_process=False,
            device=device,
        )
    return _MTCNN


# SBI was trained on face crops with substantial padding around the face so the
# model can see the blending boundary at the edge of the swap. Expand the
# detected face bounding box by this fraction of its longer side on every side
# before cropping. Matches the convention used by the official SBI repo.
SBI_FACE_MARGIN = 0.125


def detect_and_crop_face(frame_bgr: np.ndarray, size: int = 380) -> np.ndarray | None:
    """Detect the primary face in an OpenCV BGR frame and return a resized RGB crop.

    The crop is expanded by `SBI_FACE_MARGIN` of the face's longer side on every
    side before resizing, so the model sees enough context around the face.
    Returns None if no face is detected.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detector = get_mtcnn()

    boxes, _ = detector.detect(frame_rgb)
    if boxes is None or len(boxes) == 0:
        return None

    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    best_idx = int(np.argmax(areas))
    x1, y1, x2, y2 = boxes[best_idx]

    # Expand the box by SBI_FACE_MARGIN of the face's longer side on each side.
    fw = x2 - x1
    fh = y2 - y1
    pad = int(max(fw, fh) * SBI_FACE_MARGIN)
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad

    h, w = frame_rgb.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_rgb[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop_resized


def iter_video_frames(video_path: str, sample_every: int = 10):
    """Yield (frame_idx, frame_bgr) for every `sample_every`-th frame of the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()
