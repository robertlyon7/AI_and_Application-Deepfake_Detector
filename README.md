# Deepfake Detection Analysis — SBI on FF++ and Celeb-DF v2

A research analysis project that evaluates the pretrained **Self-Blended Images (SBI)** deepfake detector on two major benchmarks — **FaceForensics++** and **Celeb-DF v2** — to measure both in-distribution performance and cross-dataset generalization. The project loads a pretrained EfficientNet-B4 SBI model, runs inference frame-by-frame on detected face crops, aggregates scores per video, and reports metrics (AUC, Accuracy, F1, Precision, Recall, EER) alongside ROC curves, score distributions, confusion matrices, and Grad-CAM saliency maps. All output is consolidated in a single Jupyter notebook (`notebooks/analysis.ipynb`).

---

## Research Foundation

- **SBI — Self-Blended Images**: Shiohara, K. & Yamasaki, T. *"Detecting Deepfakes with Self-Blended Images."* CVPR 2022.
- **DeepfakeBench**: Yan, Z. et al. *"DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection."* NeurIPS 2023.
- **FaceForensics++**: Rossler, A. et al. *"FaceForensics++: Learning to Detect Manipulated Facial Images."* ICCV 2019.
- **Celeb-DF v2**: Li, Y. et al. *"Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics."* CVPR 2020.

---

## Requirements

- Python 3.10+
- NVIDIA GPU (4GB+ VRAM recommended; CPU fallback available but very slow)
- ~25 GB free disk space for datasets and cached face crops

---

## Setup Instructions

### Step 1 — Clone repo and create virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2 — Download pretrained SBI weights

```bash
python scripts/download_weights.py
```

### Step 3 — Place datasets in the `data/` folder

Required layout:

```
data/
├── FaceForensics++/
│   ├── original_sequences/
│   │   └── youtube/
│   │       └── c23/
│   │           └── videos/
│   └── manipulated_sequences/
│       ├── Deepfakes/
│       │   └── c23/
│       │       └── videos/
│       └── FaceSwap/
│           └── c23/
│               └── videos/
└── Celeb-DF-v2/
    ├── Celeb-real/
    │   └── videos/
    ├── Celeb-synthesis/
    │   └── videos/
    └── List_of_testing_videos.txt
```

### Step 4 — Run preprocessing (extracts and caches face crops)

```bash
python scripts/preprocess.py --dataset ff++
python scripts/preprocess.py --dataset celebdf
```

### Step 5 — Open and run the notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

Run all cells top-to-bottom. Charts and metrics are saved to `results/`.

---

## Expected Results

| Dataset        | Setting              | AUC (expected) |
|----------------|----------------------|----------------|
| FF++           | in-distribution      | ~0.90 – 0.95   |
| Celeb-DF v2    | cross-dataset (zero-shot) | ~0.75 – 0.85 |

The drop between FF++ and Celeb-DF v2 illustrates the **generalization gap** that motivates ongoing deepfake detection research.

---

## Dataset Citations and Download Links

- **FaceForensics++**: Access request form and download script at https://github.com/ondyari/FaceForensics
- **Celeb-DF v2**: Access request form at https://github.com/yuezunli/celeb-deepfakeforensics

Please cite the original papers if you use this work in research.
