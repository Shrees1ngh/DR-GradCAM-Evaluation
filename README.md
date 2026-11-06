# Evaluating Grad-CAM Explanations for Diabetic Retinopathy Grading

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic%20Research-green.svg)]()

> **B.Tech CSE Minor Research Project**  
> **Institution:** HMR Institute of Technology and Management  
> **Team:** Shree Bhagwan, Shivam Tiwari, Sushant Kumar  
> **Supervisor:** Dr. Pradeep  
> **Base Reference:** *Biswas et al., "Interpreting Deep Neural Networks in Diabetic Retinopathy Grading: A Comparison with Human Decision Criteria", Life 2025.*

---

## 1. Research Overview

Deep learning models achieve high diagnostic accuracy in Diabetic Retinopathy (DR) grading from digital fundus photography. However, clinical adoption requires interpretable visual evidence. While Gradient-weighted Class Activation Mapping (**Grad-CAM**) is widely deployed to provide visual saliency maps, literature frequently conflates **localization** with **faithfulness**:

1. **Localization (Plausibility / Ground-Truth Overlap):** Does the heatmap highlight pathological retinal lesions (microaneurysms, haemorrhages, hard exudates, soft exudates) marked by expert ophthalmologists? Measured via **Intersection over Union (IoU)** and the **Dice Similarity Coefficient**.
2. **Faithfulness (Decision Fidelity):** Did the highlighted pixels actually drive the CNN's classification score? Evaluated via the functional **Deletion Test** (measuring the drop in prediction confidence $\Delta p$ when salient regions are ablated, bench-marked against an area-matched **random-region control**).

### Core Research Question
> **Does good localization imply good faithfulness in Grad-CAM explanations across all five ICDR diabetic retinopathy grades?**

---

## 2. Nine-Phase Research Roadmap

* **Phase 1: Project Setup & Dataset Audit** *(Current Phase)* — Environment scaffolding, configuration, and structural audit of the IDRiD dataset.
* **Phase 2: Data Preparation & Preprocessing** — Circular FOV masking, aspect-ratio preserving resizing ($512 \times 512$), normalization, and leakage-free dataset splitting.
* **Phase 3: CNN Model Development** — Transfer learning and fine-tuning with ResNet-50 / EfficientNet backbones for 5-class ICDR grading (Grades 0–4).
* **Phase 4: Model Evaluation** — Evaluation reporting Accuracy, Macro F1, Confusion Matrix, and Quadratic Weighted Kappa (QWK).
* **Phase 5: Grad-CAM Explainability** — Hooking target convolutional feature maps (`layer4`) to generate normalized activation maps for predicted classes.
* **Phase 6: Localization Evaluation** — Multi-threshold binarization (top 5%, 10%, 20%) and IoU/Dice overlap computation against expert lesion masks (81 segmentation images). Grade 0 handled as N/A.
* **Phase 7: Faithfulness Evaluation** — Deletion test with progressive salient region ablation ($10\%, 20\%, 30\%$) and area-matched random-region control.
* **Phase 8: Statistical Analysis & Research Findings** — Spearman rank correlation ($\rho$), bootstrap confidence intervals, and grade-by-grade synthesis.
* **Phase 9: Finalization & Presentation** — Research paper manuscript, minor project report, viva defence deck, and optional web demo.

---

## 3. Repository Architecture

```text
DR-GradCAM-Evaluation/
├── configs/
│   └── config.yaml                 # Centralized configuration (paths, hyperparameters, thresholds)
├── data/
│   ├── raw/                        # Original IDRiD archive (git-ignored)
│   │   └── README.md               # Dataset download and placement instructions
│   └── processed/                  # Processed splits, caches, aligned masks (git-ignored)
├── experiments/
│   ├── checkpoints/                # Trained model weights (git-ignored)
│   └── visualizations/             # Generated Grad-CAM overlays & figures (git-ignored)
├── notebooks/                      # Colab/Jupyter research exploration notebooks
├── reports/
│   ├── dataset_audit_report.md     # Multi-dataset audit findings (IDRiD + APTOS)
│   ├── dataset_audit_report.json   # Machine-readable audit metrics
│   ├── idrid_seg_grading_matches.json # Empirical IDRiD segmentation-to-grading alignment
│   └── preprocessing_report.md     # Phase 2 preprocessing and split stratification report
├── scripts/
│   ├── audit_dataset.py            # CLI entrypoint for dataset audit
│   └── preprocess_data.py          # Phase 2 preprocessing and split generator
├── src/
│   ├── __init__.py                 # Core package init
│   ├── audit.py                    # Multi-dataset verification & leakage detection logic
│   ├── dataset.py                  # PyTorch Datasets (DRGradingDataset, SegmentationLesionDataset)
│   └── preprocessing.py            # Aspect-ratio preserving resizing, padding, mask alignment
├── tests/
│   ├── __init__.py
│   ├── test_audit.py               # Dataset audit test suite (IDRiD & APTOS)
│   └── test_preprocessing.py       # Preprocessing & spatial mask alignment test suite
├── .gitignore                      # Git exclusion rules
├── requirements.txt                # Pinned dependencies
└── README.md                       # Research specification & documentation
```

---

## 4. Environment Setup

### Prerequisites
* Python 3.10+ (Recommended: Python 3.11 or 3.12)
* Recommended: Virtual environment (`venv` or `conda`)

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/<your-username>/DR-GradCAM-Evaluation.git
cd DR-GradCAM-Evaluation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Dataset Setup (IDRiD)

The **Indian Diabetic Retinopathy Image Dataset (IDRiD)** consists of:
1. **Disease Grading Set (516 images):** 413 training images and 103 testing images with ICDR grade labels (0 to 4).
2. **Segmentation Set (81 images):** Pixel-level binary masks for Microaneurysms (`MA`), Haemorrhages (`HE`), Hard Exudates (`EX`), Soft Exudates (`SE`), and Optic Disc (`OD`).

### Placement Guide
Place raw uncompressed IDRiD folders into `data/raw/IDRiD/` as specified in [`data/raw/README.md`](data/raw/README.md):

```text
data/raw/IDRiD/
├── 1. Original Images/
│   ├── a. Training Set/                  # 413 fundus images (JPG)
│   └── b. Testing Set/                   # 103 fundus images (JPG)
├── 2. Groundtruths/
│   ├── a. DR AC Groundtruths/
│   │   ├── train.csv                     # Training labels (0-4)
│   │   └── test.csv                      # Testing labels (0-4)
└── Segmentation/
    ├── Original Images/                  # 81 images
    └── Groundtruths/
        ├── 1. Microaneurysms/            # MA masks (.tif)
        ├── 2. Haemorrhages/              # HE masks (.tif)
        ├── 3. Hard Exudates/             # EX masks (.tif)
        ├── 4. Soft Exudates/             # SE masks (.tif)
        └── 5. Optic Disc/                # OD masks (.tif)
```

---

## 6. Running Phase 1 Dataset Audit

Execute the automated dataset audit CLI to inspect image integrity, class distributions, segmentation mask availability, and cross-subset leakage:

```bash
python scripts/audit_dataset.py
```

### Optional Arguments
```bash
# Custom config or raw data path override:
python scripts/audit_dataset.py --config configs/config.yaml --raw-dir /path/to/custom/IDRiD

# Verbose logging:
python scripts/audit_dataset.py -v
```

Audit results are written to:
* **Human-readable Report:** [`reports/dataset_audit_report.md`](reports/dataset_audit_report.md)
* **JSON Structured Summary:** [`reports/dataset_audit_report.json`](reports/dataset_audit_report.json)

---

## 7. Running Tests

Run the test suite to verify the audit engine and mock dataset validations:

```bash
pytest tests/ -v
```

---

## 8. Academic & Medical Disclaimer

This software and research pipeline is developed strictly for academic investigation and empirical evaluation of Explainable Artificial Intelligence (XAI) methods. It is **not** certified as a medical device and must **never** be used for clinical diagnosis, patient management, or real-world treatment planning.

