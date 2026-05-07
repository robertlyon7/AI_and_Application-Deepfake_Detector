"""Extract and cache face crops from FF++ or Celeb-DF v2 videos.

Run:
    python scripts/preprocess.py --dataset ff++
    python scripts/preprocess.py --dataset celebdf
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

# Make the project root importable when running as a standalone script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.face_utils import detect_and_crop_face, iter_video_frames  # noqa: E402


DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"


# ---------------------------------------------------------------------------
# FF++
# ---------------------------------------------------------------------------

FF_TEST_MIN = 860
FF_TEST_MAX = 999  # inclusive


def _ff_video_index(path: Path) -> int | None:
    """Return the leading numeric index of an FF++ video filename, or None."""
    # FF++ real videos are named like "860.mp4"; fakes are "860_861.mp4".
    # We key off the *first* integer in the stem, which is consistent across both.
    stem = path.stem
    first = stem.split("_")[0]
    if first.isdigit():
        return int(first)
    return None


def _collect_ff_videos() -> list[tuple[Path, int]]:
    """Collect FF++ test-split videos as (path, label) pairs. Label 0=real, 1=fake."""
    root = DATA_DIR / "FaceForensics++"
    real_dir = root / "original_sequences" / "youtube" / "c23" / "videos"
    fake_dirs = [
        root / "manipulated_sequences" / "Deepfakes" / "c23" / "videos",
        root / "manipulated_sequences" / "FaceSwap" / "c23" / "videos",
    ]

    pairs: list[tuple[Path, int]] = []

    for vid in sorted(real_dir.glob("*.mp4")) if real_dir.exists() else []:
        idx = _ff_video_index(vid)
        if idx is not None and FF_TEST_MIN <= idx <= FF_TEST_MAX:
            pairs.append((vid, 0))

    for fd in fake_dirs:
        if not fd.exists():
            continue
        for vid in sorted(fd.glob("*.mp4")):
            idx = _ff_video_index(vid)
            if idx is not None and FF_TEST_MIN <= idx <= FF_TEST_MAX:
                pairs.append((vid, 1))

    return pairs


# ---------------------------------------------------------------------------
# Celeb-DF v2
# ---------------------------------------------------------------------------

def _collect_celebdf_videos() -> list[tuple[Path, int]]:
    """Collect Celeb-DF v2 test videos from List_of_testing_videos.txt."""
    root = DATA_DIR / "Celeb-DF-v2"
    list_path = root / "List_of_testing_videos.txt"
    if not list_path.exists():
        raise FileNotFoundError(
            f"Expected {list_path} — did you download Celeb-DF v2 and place it under data/?"
        )

    pairs: list[tuple[Path, int]] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # Celeb-DF labels: 1 = REAL, 0 = FAKE in the official list file.
            # We standardize internally to 0 = REAL, 1 = FAKE.
            raw_label = int(parts[0])
            internal_label = 0 if raw_label == 1 else 1
            rel_path = parts[1]
            vid_path = root / rel_path
            # Some distributions of Celeb-DF store videos directly under Celeb-real/ and
            # Celeb-synthesis/ rather than inside a nested videos/ folder.
            if not vid_path.exists():
                alt = root / rel_path.replace("/videos/", "/")
                if alt.exists():
                    vid_path = alt
            if vid_path.exists():
                pairs.append((vid_path, internal_label))
            else:
                print(f"[preprocess] Warning: video listed but missing on disk: {vid_path}")

    return pairs


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def _process_video(video_path: Path, label: int, cache_root: Path,
                   sample_every: int = 10) -> list[str]:
    """Extract faces from `video_path` and write JPGs to the cache. Return frame paths."""
    video_name = video_path.stem
    subdir = "real" if label == 0 else "fake"
    out_dir = cache_root / subdir / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip re-processing if we already have cached frames for this video.
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if existing:
        return [str(p) for p in existing]

    saved_paths: list[str] = []
    for frame_idx, frame_bgr in iter_video_frames(str(video_path), sample_every=sample_every):
        face_rgb = detect_and_crop_face(frame_bgr)
        if face_rgb is None:
            continue
        # cv2 writes BGR; convert RGB crop back to BGR for encoding as a JPEG.
        out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
        saved_paths.append(str(out_path))

    return saved_paths


def run(dataset: str) -> None:
    dataset = dataset.lower()
    if dataset == "ff++":
        cache_root = CACHE_DIR / "ff++"
        pairs = _collect_ff_videos()
    elif dataset == "celebdf":
        cache_root = CACHE_DIR / "celebdf"
        pairs = _collect_celebdf_videos()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    cache_root.mkdir(parents=True, exist_ok=True)
    skipped_path = cache_root / "skipped.txt"
    manifest_path = cache_root / "manifest.json"

    if not pairs:
        print(f"[preprocess] No videos found for dataset={dataset}. "
              f"Check that data/ is populated correctly.")
        return

    manifest: list[dict] = []
    skipped: list[str] = []
    total_frames = 0

    for video_path, label in tqdm(pairs, desc=f"Preprocessing {dataset}"):
        frames = _process_video(video_path, label, cache_root)
        if not frames:
            skipped.append(str(video_path))
            continue
        manifest.append({
            "video": video_path.stem,
            "label": int(label),
            "frames": frames,
        })
        total_frames += len(frames)

    # Persist the manifest and skipped list.
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(skipped_path, "w", encoding="utf-8") as f:
        for s in skipped:
            f.write(s + "\n")

    print()
    print("=== Preprocessing summary ===")
    print(f"Dataset:         {dataset}")
    print(f"Total videos:    {len(pairs)}")
    print(f"Videos cached:   {len(manifest)}")
    print(f"Videos skipped:  {len(skipped)}")
    print(f"Total frames:    {total_frames}")
    print(f"Manifest:        {manifest_path}")
    print(f"Skipped log:     {skipped_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract face crops from deepfake datasets.")
    parser.add_argument("--dataset", required=True, choices=["ff++", "celebdf"],
                        help="Which dataset to preprocess.")
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
