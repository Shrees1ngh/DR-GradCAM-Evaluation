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

* **Phase 1: Project Setup & Dataset Audit** *(Complete ✅)* — Multi-dataset auditing (APTOS 3,662 images + IDRiD 516 grading + 81 segmentation images), integrity verification, and cross-dataset duplicate auditing.
* **Phase 2: Data Preparation & Preprocessing** *(Complete ✅)* — Aspect-ratio preserving resizing ($512 \times 512$), black border removal, circular FOV masking, synchronized multi-lesion mask alignment (MA, HE, EX, SE), and stratified 80/20 APTOS splitting.
* **Phase 3: CNN Model Development** *(Current Phase — Pipeline Implemented & Verified ✅)* — Reusable ResNet-50 backbone with ImageNet pretraining, dropout (0.3), inverse class-weighted CrossEntropyLoss, AdamW optimizer, Cosine Annealing schedule, validation QWK early stopping with `min_delta`, target layer (`layer4`), and opt-in external evaluation.
* **Phase 4: Model Evaluation** *(Diagnostic Framework Verified ✅)* — Comprehensive diagnostic assessment (Accuracy, Macro F1, Sensitivity, Specificity, QWK, Confusion Matrix) on validation and external test sets, with strict dataset isolation and checkpoint immutability.
* **Phase 5: Grad-CAM Explainability** *(Pipeline Implemented & Verified ✅)* — Hooking target convolutional feature maps (`layer4`) to generate normalized activation maps for predicted classes, multi-threshold binarization (top 5%, 10%, 20%), and alpha-blended fundus overlays.
* **Phase 6: Localization Evaluation** *(Pipeline Implemented & Verified ✅)* — Multi-threshold binarization (top 5%, 10%, 20%) and IoU/Dice overlap computation against expert lesion masks (81 segmentation images: 78 primary cohort, 3 supplementary test overlap cohort). Grade 0 handled as N/A.
* **Phase 7: Faithfulness Evaluation** *(Pipeline Implemented & Verified ✅)* — Deletion test with progressive salient region ablation ($10\%, 20\%, 30\%$), area-matched random-region controls, and AUDC metrics.
* **Phase 8: Statistical Analysis & Research Findings** — Spearman rank correlation ($\rho$), bootstrap confidence intervals, and grade-by-grade synthesis.
* **Phase 9: Finalization & Presentation** — Research paper manuscript, minor project report, viva defence deck, and optional web demo.

---

## 3. Repository Architecture

```text
DR-GradCAM-Evaluation/
├── configs/
│   └── config.yaml                 # Centralized configuration (paths, hyperparameters, thresholds)
├── data/
│   ├── raw/                        # Original raw datasets (git-ignored)
│   │   └── README.md               # Dataset download and placement instructions
│   └── processed/                  # Standardized 512x512 images, masks, and split manifests (git-ignored)
├── experiments/
│   ├── checkpoints/                # Trained model weights (*.pth, git-ignored)
│   └── visualizations/             # Generated Grad-CAM overlays & figures (git-ignored)
├── docs/
│   └── GENUINE_INFERENCE_GUIDE.md  # Instructions for executing genuine inference on GPU/Colab
├── notebooks/                      # Colab/Jupyter research exploration notebooks
├── reports/
│   ├── SIMULATED_DATA_NOTICE.md    # Formal audit notice regarding simulated vs genuine artifacts
│   ├── dataset_audit_report.md     # Phase 1 dataset audit report
│   ├── idrid_seg_grading_matches.json # Empirical IDRiD segmentation-to-grading alignment
│   ├── preprocessing_report.md     # Phase 2 preprocessing and split stratification report
│   └── phase_3_training_report.md  # Phase 3 model training & evaluation report
├── scripts/
│   ├── audit_dataset.py            # CLI entrypoint for dataset audit
│   ├── diagnose_idrid_domain.py    # Phase 4 IDRiD domain shift diagnostic CLI
│   ├── evaluate_localization.py    # Phase 6 Grad-CAM lesion localization CLI
│   ├── preprocess_data.py          # Phase 2 preprocessing and split generator
│   └── train.py                    # Phase 3 CNN training and evaluation CLI
├── src/
│   ├── __init__.py                 # Core package init
│   ├── audit.py                    # Multi-dataset verification & leakage detection logic
│   ├── dataset.py                  # PyTorch Datasets with configurable augmentations
│   ├── explainability.py           # Phase 5: Native Grad-CAM hooks, binarization, & overlay blending
│   ├── faithfulness.py             # Phase 7: Deletion tests, salient region ablation, & AUDC
│   ├── localization.py             # Phase 6: Multi-threshold IoU/Dice lesion localization
│   ├── model.py                    # ResNet-50 model builder & Grad-CAM target layer extractor
│   ├── preprocessing.py            # Aspect-ratio preserving resizing, padding, mask alignment
│   └── similarity.py               # Visual hash duplicate verification (pHash, dHash)
├── tests/
│   ├── __init__.py
│   ├── test_audit.py               # Dataset audit test suite
│   ├── test_diagnose_domain.py     # Domain diagnostic and dataset isolation tests
│   ├── test_explainability.py      # Grad-CAM hooks, binarization, & overlay tests
│   ├── test_localization.py        # Lesion localization, cohort split, and immutability tests
│   ├── test_model.py               # Model architecture & Grad-CAM layer tests
│   ├── test_preprocessing.py       # Preprocessing & spatial mask alignment test suite
│   ├── test_similarity.py          # Perceptual hash similarity tests
│   └── test_train.py               # Training loop, metric, early stopping, and AMP tests
├── .gitignore                      # Git exclusion rules
├── project.md                      # Complete Project Understanding Document (SRS format)
├── pytest.ini                      # Test runner configuration
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

## 7. Phase 3: CNN Model Development & Training

Phase 3 implements a research-quality transfer learning and fine-tuning pipeline based on **ResNet-50** (`torchvision.models.resnet50`) pre-trained on ImageNet.

### Architectural & Training Specifications
- **Input Dimensions:** $512 \times 512 \times 3$ with aspect-ratio preserving padding.
- **Classification Head:** Linear layer ($2048 \to 5$) with dropout probability ($p=0.3$) mapping to ICDR Grades 0–4.
- **Grad-CAM Target Layer:** `model.layer4` (2048-dim feature maps at $16 \times 16$ spatial resolution).
- **Optimization:** AdamW ($\text{LR}=10^{-4}$, weight decay $=10^{-4}$) with Cosine Annealing learning rate schedule (`CosineAnnealingLR`).
- **Class Imbalance:** Inverse class weighting ($w_c = \frac{N}{C \cdot N_c}$) applied to `nn.CrossEntropyLoss`.
- **Early Stopping & Checkpointing:** Monitored strictly on **Validation Quadratic Weighted Kappa (QWK)** with configurable `min_delta` (default `0.001`) and patience (default 5 epochs). The training script saves the best checkpoint whenever validation QWK strictly improves ($QWK > QWK_{\text{best}}$). `min_delta` is used only for resetting the early-stopping patience benchmark ($QWK \ge QWK_{\text{benchmark}} + \text{min\_delta}$), not for deciding whether to save a checkpoint.
- **Modern Mixed Precision (AMP):** Utilizes `torch.amp.autocast(device_type="cuda")` and `torch.amp.GradScaler("cuda")` when CUDA is available, while supporting deterministic CPU execution.
- **Strict Dataset Isolation:** The model is trained exclusively on the primary APTOS 2019 dataset ($N=2,929$ train, $N=733$ val). External test evaluation on IDRiD ($N=103$) is **opt-in** and strictly evaluated post-training using the selected best checkpoint.

### CLI Execution Commands

#### 1. Full Model Training (Primary APTOS Train + Val)
```bash
python scripts/train.py
```

#### 2. Quick Dry-Run Smoke Test (Pipeline Verification)
Runs 1 quick epoch on a truncated 16-sample subset to verify tensor shapes, gradient computation, and reporting without running full training:
```bash
python scripts/train.py --dry-run
```

#### 3. Training with Post-Training External Test Evaluation (IDRiD)
Trains on APTOS, selects the best checkpoint via validation QWK, and subsequently evaluates the held-out IDRiD test cohort:
```bash
python scripts/train.py --eval-external
```

#### 4. Training with Output Persistence to Google Drive (Google Colab)
Safely saves checkpoints, training history, and evaluation reports directly to Google Drive while also keeping local runtime copies:
```bash
python scripts/train.py --output-dir /content/drive/MyDrive/DR-GradCAM-Evaluation/outputs
```

#### 5. Evaluate an Existing Checkpoint
```bash
# Evaluate on APTOS validation split:
python scripts/train.py --eval-only

# Evaluate on both APTOS validation and IDRiD external test:
python scripts/train.py --eval-only --eval-external

# Evaluate with custom checkpoint:
python scripts/train.py --eval-only --eval-external --checkpoint-path experiments/checkpoints/best_model_qwk.pth
```

#### 6. Cross-Dataset Domain Shift Diagnostic (N=413 Training Images)
Executes non-destructive evaluation verifying label mapping, preprocessing geometry, and per-grade recall:
```bash
python scripts/diagnose_idrid_domain.py \
    --checkpoint experiments/checkpoints/best_model_qwk.pth \
    --idrid-manifest data/processed/idrid/external_train.csv
```

#### 7. Phase 6: Grad-CAM Lesion Grounding & Localization Evaluation
Evaluates IoU, Dice, EPG, AUC-ROC, and Hit Rate against expert lesion masks (78 primary, 3 supplementary):
```bash
python scripts/evaluate_localization.py \
    --checkpoint experiments/checkpoints/best_model_qwk.pth \
    --manifest data/processed/idrid/segmentation_manifest.csv
```

#### 8. Phase 7: Grad-CAM Faithfulness Evaluation (Deletion Test)
Evaluates decision fidelity via progressive salient region deletion and area-matched random controls:
```bash
python scripts/evaluate_faithfulness.py \
    --checkpoint experiments/checkpoints/best_model_qwk.pth \
    --manifest data/processed/idrid/segmentation_manifest.csv
```

#### 9. Launch RetinaScope Streamlit Web Application
Interactive web interface for DR grading and Grad-CAM visualization:
```bash
streamlit run app.py
```

### Artifact & Output Locations
Output directories are fully configurable via the `--output-dir` CLI flag or `outputs.output_dir` in `configs/config.yaml`. The target directory structure is automatically created if it does not exist:
- **Default Local Output Locations:**
  - **Best Model Checkpoint:** `experiments/checkpoints/best_model_qwk.pth`
  - **Training Epoch History:** `experiments/training_history.csv`
  - **Evaluation Metrics JSON:** `experiments/evaluation_metrics.json`
  - **Phase 3 Markdown Report:** `reports/phase_3_training_report.md`
- **Google Drive Output Structure (when `--output-dir` is configured):**
  - **Checkpoints:** `<output-dir>/checkpoints/best_model_qwk.pth`
  - **Training History:** `<output-dir>/training_history.csv`
  - **Evaluation Metrics JSON:** `<output-dir>/evaluation_metrics.json`
  - **Phase 3 Report:** `<output-dir>/reports/phase_3_training_report.md`
  *(Local runtime copies are also simultaneously preserved.)*

> [!NOTE]
> **Current Project Status:** Phase 3 model architecture, data loaders, training engine, early stopping with `min_delta`, AMP acceleration, and unit tests are fully implemented and verified. Real model convergence weights and evaluation metrics will be produced when full training is launched on GPU hardware (e.g., Google Colab with Tesla T4).

---

## 8. Running Tests

Run the complete test suite across audit, preprocessing, similarity, model, and training components:

```bash
pytest tests/ -v
```

---

## 9. Academic & Medical Disclaimer

This software and research pipeline is developed strictly for academic investigation and empirical evaluation of Explainable Artificial Intelligence (XAI) methods. It is **not** certified as a medical device and must **never** be used for clinical diagnosis, patient management, or real-world treatment planning.

