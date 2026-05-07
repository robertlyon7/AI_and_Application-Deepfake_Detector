"""Download the pretrained SBI EfficientNet-B4 checkpoint into weights/."""

import os
import sys
from pathlib import Path

import gdown


# Official SBI release hosts the EfficientNet-B4 weights on Google Drive.
# File id from the public release at https://github.com/mapooon/SelfBlendedImages.
SBI_EFFNETB4_GDRIVE_ID = "1X0-NYT8KPursLZZdxduRQju6E52hauV0"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
OUTPUT_PATH = WEIGHTS_DIR / "sbi_efficientb4.pth"


def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"[download] Weights already exist at {OUTPUT_PATH} ({size_mb:.2f} MB). Skipping.")
        return

    url = f"https://drive.google.com/uc?id={SBI_EFFNETB4_GDRIVE_ID}"
    print(f"[download] Fetching SBI EfficientNet-B4 weights from {url}")
    try:
        gdown.download(url, str(OUTPUT_PATH), quiet=False)
    except Exception as exc:
        print(f"[download] ERROR while downloading: {exc}")
        print("[download] If the file is flagged by Google Drive quota, retry later or")
        print("           manually download it from the SBI GitHub release and place it at:")
        print(f"           {OUTPUT_PATH}")
        sys.exit(1)

    if not OUTPUT_PATH.exists():
        print("[download] ERROR: download did not produce the expected file.")
        sys.exit(1)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"[download] Saved {OUTPUT_PATH} ({size_mb:.2f} MB)")
    print("[download] Success.")


if __name__ == "__main__":
    main()
