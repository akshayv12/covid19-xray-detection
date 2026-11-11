# 🫁 COVID-19 Detection from Chest X-Rays using ResNet-50

> **MA305 Team 9** — Multimodal Deep Learning for COVID-19 Classification

A multimodal deep learning model that detects COVID-19 from **Chest X-ray images** combined with **patient metadata** (age, sex, days since symptom onset) using a ResNet-50 backbone fused with a Tabular MLP.

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Model Architecture](#2-model-architecture)
3. [Dataset Setup](#3-dataset-setup)
4. [Requirements & Environment](#4-requirements--environment)
5. [How to Run](#5-how-to-run)
6. [Training Configuration](#6-training-configuration)
7. [Data Processing](#7-data-processing)
8. [Output](#8-output)

---

## 1. Project Overview

This project implements a **multimodal deep learning pipeline** to classify COVID-19 from chest X-rays:

- 🖼️ **Image branch:** ResNet-50 pretrained on ImageNet
- 📊 **Tabular branch:** 2-layer MLP for patient metadata (age, sex, days since onset)
- 🔀 **Fusion:** Feature concatenation before final classification head
- ✅ **Patient-level stratified split** to prevent data leakage across train/val/test

---

## 2. Model Architecture

```
Chest X-ray ──► ResNet-50 (ImageNet pretrained) ──► Feature Vector (2048-d)
                                                             │
Patient Metadata ──► MLP (input → 64 → 64) ──► Feature Vec  │
                                                             ▼
                                              Concat → Classifier Head
                                                             │
                                               ┌────────────┴────────────┐
                                          COVID-19                 View Type
                                        (BCEWithLogits)         (CrossEntropy)
```

| Component | Details |
|-----------|---------|
| Backbone | ResNet-50 (pretrained on ImageNet) |
| Tabular Embedding | 2-layer MLP (input → 64 → output) |
| Auxiliary Head | View classification (PA / AP / etc.) |
| Final Activation | Sigmoid (binary COVID-19 prediction) |

---

## 3. Dataset Setup

This project uses the **[COVID-19 Image Data Collection](https://github.com/ieee8023/covid-chestxray-dataset)** by Joseph Paul Cohen et al.

> ⚠️ The `images/` folder (~528 MB) is **not included** in this repo due to GitHub size limits. Download it directly from the source.

### Steps:
1. Download `metadata.csv` and the `images/` folder from:  
   👉 [https://github.com/ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)

2. For **Google Colab** (recommended):
   - Create a folder named `covidproject` in your Google Drive
   - Upload `metadata.csv` → `covidproject/metadata.csv`
   - Upload `images/` → `covidproject/images/`

---

## 4. Requirements & Environment

- **Python:** 3.8+
- **Hardware:** NVIDIA GPU strongly recommended (code supports CUDA)

Install dependencies:
```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework |
| `timm` | Model zoo (ResNet-50) |
| `scikit-learn` | Stratified splitting, metrics |
| `pandas` / `numpy` | Data processing |
| `Pillow` | Image loading |
| `torchmetrics` | Evaluation metrics |
| `matplotlib` | Plotting results |

---

## 5. How to Run

> 💡 **Recommended:** Run on [Google Colab](https://colab.research.google.com/) with GPU runtime.

1. Change Colab runtime to **GPU**: `Runtime → Change runtime type → GPU`
2. Open `COVID19_XRAY_CODE.ipynb` in Google Colab
3. Mount your Google Drive and ensure the dataset is placed correctly (see [Dataset Setup](#3-dataset-setup))
4. Run all cells sequentially

---

## 6. Training Configuration

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
| Loss (Primary) | `BCEWithLogitsLoss` (class-weighted) |
| Loss (Auxiliary) | `CrossEntropyLoss` (view type) |

---

## 7. Data Processing

| Setting | Value |
|---------|-------|
| Input Size | 224 × 224 pixels |
| Normalization Mean | `[0.485, 0.456, 0.406]` |
| Normalization Std | `[0.229, 0.224, 0.225]` |
| Augmentation (Train) | Random Horizontal Flip, Random Rotation (±15°) |

---

## 8. Output

After training completes, the following files are generated:

| File | Description |
|------|-------------|
| `train.csv` | Training split |
| `val.csv` | Validation split |
| `test.csv` | Test split |
| `best_model.pth` | Best model checkpoint (by val loss) |
| `test_predictions.csv` | Final predictions on test set |

> 📌 All intermediate files are saved automatically in the Colab session. Early training results are saved inside the `.ipynb` notebook itself.

---

## 📂 Repository Structure

```
MA305_TEAM9/
├── COVID19_XRAY_CODE.ipynb   # Main notebook (training + evaluation)
├── metadata.csv               # Patient metadata (from dataset)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── images/                    # ⚠️ NOT included — download separately
```

---

## 📚 References

- Cohen, J. P., et al. *COVID-19 Image Data Collection.* arXiv:2003.11597 (2020).  
  [https://github.com/ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
- He, K., et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.