# COVID-19 Detection from Chest X-Rays using ResNet-50

**MA305 — Team 9** | Multimodal Deep Learning for COVID-19 Classification

A multimodal deep learning model that detects COVID-19 from **chest X-ray images** fused with **patient metadata** (age, sex, days since symptom onset) using a ResNet-50 backbone + Tabular MLP. Achieves a **patient-level AUC of 0.8673** on the held-out test set.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Results](#2-results)
3. [Model Architecture](#3-model-architecture)
4. [Dataset](#4-dataset)
5. [Requirements](#5-requirements)
6. [How to Run](#6-how-to-run)
7. [Training Details](#7-training-details)
8. [Data Processing](#8-data-processing)
9. [Repository Structure](#9-repository-structure)
10. [References](#10-references)

---

## 1. Project Overview

This project implements a **multimodal deep learning pipeline** for binary COVID-19 classification:

- **Image branch** — ResNet-50 pretrained on ImageNet extracts visual features from chest X-rays
- **Tabular branch** — 2-layer MLP encodes patient metadata (age, sex, symptom offset days)
- **Fusion** — features are concatenated before a final classification head
- **Auxiliary head** — predicts X-ray view type (AP / PA / Axial) as a regularisation signal
- **Patient-level stratified split** — prevents data leakage (same patient never appears in both train & test)

---

## 2. Results

### Test Set Performance (best checkpoint, epoch 18)

| Metric | Image-level | Patient-level |
|--------|------------|---------------|
| **AUC** | **0.8602** | **0.8673** |

### Classification Report (Patient-level)

| Metric | Value |
|--------|-------|
| Precision | **0.854** |
| Recall | **0.891** |
| F1-Score | **0.872** |

### Confusion Matrix (Patient-level, 65 patients)

```
                  Predicted Non-COVID    Predicted COVID
Actual Non-COVID        12                    7
Actual COVID             5                   41
```

- **True Positives (COVID correctly caught):** 41
- **True Negatives:** 12
- **False Positives (healthy wrongly flagged):** 7
- **False Negatives (COVID missed):** 5

### Training Curve Summary

| Epoch | Train Loss | Train AUC | Val AUC |
|-------|-----------|-----------|---------|
| 1 | 0.9790 | 0.6923 | 0.7364 |
| 5 | 0.6219 | 0.8632 | 0.8038 |
| 10 | 0.5047 | 0.9321 | 0.8404 |
| 15 | 0.3170 | 0.9816 | 0.8827 |
| **18** ⭐ | **0.2752** | **0.9928** | **0.8831** ← best |
| 25 | 0.2429 | 0.9914 | 0.8784 |

> The best model was saved at **epoch 18** (Val AUC = 0.8831). The model converges steadily — training AUC exceeds 0.99 by epoch 18, with validation AUC plateauing around 0.88.

---

## 3. Model Architecture

```
Chest X-ray (224×224) ──► ResNet-50 backbone ──► 2048-d feature vector ──┐
                          (ImageNet pretrained)                            │
                                                                           ├──► Concat ──► COVID head (BCEWithLogits)
Patient Metadata ──► Sex Embedding (8-d)  ──┐                             │
  (age, sex, offset)   Age (1-d)            ├──► MLP (10→64→64) ──────────┘
                        Offset days (1-d) ──┘         │
                                                       └──► View head (CrossEntropy: AP/PA/Axial)
```

| Component | Details |
|-----------|---------|
| Image backbone | ResNet-50 (pretrained on ImageNet) → 2048-d |
| Sex encoding | Learnable embedding: `{M:0, F:1, Unknown:2}` → 8-d |
| Tabular MLP | Input(10) → Linear(64) → BN → Dropout(0.3) → Linear(64) → ReLU |
| Fusion | Concat image + tabular features → FC → sigmoid |
| Auxiliary head | View classification: AP / PA / Axial (CrossEntropyLoss) |

---

## 4. Dataset

This project uses the **[COVID-19 Image Data Collection](https://github.com/ieee8023/covid-chestxray-dataset)** by Joseph Paul Cohen et al.

| Split | Images | Patients |
|-------|--------|---------|
| Train | 510 | 312 |
| Val | 116 | 67 |
| Test | 98 | 67 |
| **Total** | **745** | **446** |

**Label distribution (patient-level):**
- COVID-positive: 321 patients
- Non-COVID: 125 patients

**X-ray views:** AP, PA, Axial

> ⚠️ The `images/` folder (~528 MB, 930 files) is **not included** in this repo due to GitHub size limits.

### Setup Instructions

1. Download `metadata.csv` and the `images/` folder from:
   👉 https://github.com/ieee8023/covid-chestxray-dataset

2. For **Google Colab** (recommended):
   - Create a folder `covidproject/` in your Google Drive
   - Upload `metadata.csv` → `covidproject/metadata.csv`
   - Upload `images/` → `covidproject/images/`

---

## 5. Requirements

- **Python:** 3.8+
- **Hardware:** NVIDIA GPU strongly recommended (code supports CUDA)

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework |
| `timm` | Model zoo utilities |
| `scikit-learn` | Stratified splitting, metrics |
| `pandas` / `numpy` | Data processing |
| `Pillow` | Image loading |
| `torchmetrics` | AUC and other metrics |
| `matplotlib` | Plotting results |

---

## 6. How to Run

> **Recommended:** Run on [Google Colab](https://colab.research.google.com/) with GPU runtime.

1. **Set runtime to GPU:** `Runtime → Change runtime type → T4 GPU`
2. **Mount Google Drive** (first cell handles this)
3. **Ensure dataset is in place** — see [Dataset Setup](#4-dataset)
4. **Open** `COVID19_XRAY_CODE.ipynb` in Google Colab
5. **Run All Cells** — the notebook will:
   - Load and clean `metadata.csv`
   - Perform patient-level stratified train/val/test split
   - Train the multimodal ResNet-50 model for 25 epochs
   - Save the best checkpoint (`best_multimodal_resnet.pth`)
   - Evaluate on the test set and print AUC, confusion matrix, Precision/Recall/F1

---

## 7. Training Details

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 25 |
| Optimizer | SGD |
| Learning Rate | 0.01 |
| LR Scheduler | StepLR (×0.1 every 10 epochs) |
| Momentum | 0.9 |
| Weight Decay | 1e-3 |
| Batch Size | 32 |
| Random Seed | 42 |
| Primary Loss | `BCEWithLogitsLoss` (class-weighted for imbalance) |
| Auxiliary Loss | `CrossEntropyLoss` (view type: AP/PA/Axial) |
| Best Epoch | 18 (Val AUC = 0.8831) |

---

## 8. Data Processing

| Setting | Value |
|---------|-------|
| Input resolution | 224 × 224 pixels |
| Normalization mean | `[0.485, 0.456, 0.406]` (ImageNet) |
| Normalization std | `[0.229, 0.224, 0.225]` (ImageNet) |
| Train augmentations | RandomResizedCrop(224, scale=0.7–1.0), RandomHorizontalFlip, RandomRotation(±15°) |
| Val/Test transforms | Resize(224×224), Normalize |
| Leakage prevention | Patient-level stratified split (same patient never in 2 splits) |

---

## 9. Repository Structure

```
MA305_TEAM9/
├── COVID19_XRAY_CODE.ipynb    # Main notebook — training + evaluation (with saved outputs)
├── metadata.csv               # Patient metadata from the dataset
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excludes images/, .DS_Store, model weights
├── README.md                  # This file
└── images/                    # ⚠️ NOT included — download from dataset source
```

> The `.ipynb` file contains all training outputs, loss curves, and final evaluation results as saved cell outputs — viewable directly on GitHub.

---

## 10. References

- Cohen, J. P., et al. *COVID-19 Image Data Collection.* arXiv:2003.11597 (2020).
  https://github.com/ieee8023/covid-chestxray-dataset

- He, K., Zhang, X., Ren, S., & Sun, J. *Deep Residual Learning for Image Recognition.* CVPR 2016.

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html