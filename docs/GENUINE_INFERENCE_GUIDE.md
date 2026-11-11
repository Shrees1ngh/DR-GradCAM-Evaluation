# Guide for Executing Genuine Diagnostic & Localization Inference

> **Target Environment:** GPU Workstation or Google Colab with access to trained weights and clinical images.  
> **Backbone Checkpoint:** `experiments/checkpoints/best_model_qwk.pth` (Trained exclusively on APTOS 2019, $N=2,929$)  
> **Integrity Guarantee:** Zero weight modification (Dual verified: Checkpoint file SHA-256 and bit-exact in-memory `state_dict` matching under `torch.no_grad()`).

---

## 1. Prerequisites Checklist

Before executing genuine inference, ensure that:
1. **Checkpoint:** `experiments/checkpoints/best_model_qwk.pth` exists and is read-only.
2. **IDRiD Dataset:** The official IDRiD dataset directory is mounted or extracted under:
   ```text
   data/raw/IDRiD/
   ├── A. Segmentation/
   │   └── A. Segmentation/
   │       ├── 1. Original Images/                  (81 images: IDRiD_01.jpg to IDRiD_81.jpg)
   │       └── 2. All Segmentation Groundtruths/    (MA, HE, EX, SE, OD mask tif files)
   └── B. Disease Grading/
       └── B. Disease Grading/
           ├── 1. Original Images/
           │   ├── a. Training Set/                 (413 images: IDRiD_001.jpg to IDRiD_413.jpg)
           │   └── b. Testing Set/                  (103 images: IDRiD_414.jpg to IDRiD_516.jpg)
           └── 2. Groundtruths/
               ├── a. IDRiD_Disease Grading_Training Labels.csv
               └── b. IDRiD_Disease Grading_Testing Labels.csv
   ```
3. **Python Environment:** Python 3.11 with PyTorch, Torchvision, OpenCV, Scikit-learn, and Pandas installed.

---

## 2. Step 1: Preprocess Data & Generate Split Manifests

If `data/processed/` has not yet been populated, run the deterministic preprocessor using the `--idrid-only` flag (recommended for Colab and inference environments where IDRiD images and the pre-trained APTOS checkpoint are provided):

```bash
python scripts/preprocess_data.py --size 512 --idrid-only --validation-mode strict
```

To explicitly force re-generation and overwriting of all processed images, masks, and FOV masks:
```bash
python scripts/preprocess_data.py --size 512 --idrid-only --force-reprocess --validation-mode strict
```

> [!NOTE]
> The `--idrid-only` flag allows IDRiD grading cohorts (413 train, 103 test) and segmentation masks (81 images across 5 lesion types) to be fully preprocessed and aligned without requiring raw APTOS 2019 data. The existing APTOS-trained checkpoint (`best_model_qwk.pth`) remains frozen and untouched.
> Sample count validation strictly verifies: 413 train, 103 test, 81 segmentation images, and cohort uniqueness.

### Preprocessing Geometry & Quality Control Guarantees:
- **FOV Cropping:** `crop_to_retinal_fov` detects circular retinal boundaries in the green channel and excises empty horizontal sensor margins ($1500+$ px on IDRiD's $4288 \times 2848$ frames).
- **Aspect Ratio:** The cropped retina is approximately square ($1.01:1$).
- **Actual Model Letterbox Padding:** Minimal ($\le 3\%$), unlike the uncropped native sensor padding ($33.59\%$).
- **Active Retinal Tissue:** Covers $\sim 75\%$ of the $512 \times 512$ canvas (with $\sim 25\%$ representing the four natural corners outside the circular optical field).
- **Persistent FOV Masks:** Generated directly from each processed fundus image, verified binary {0, 1}, and zeroed out on padding margins.
- **FOV Quality Control Report:** `reports/fov_qc_report.csv` and `reports/fov_qc_report.md` validate active area ratio, circularity, and flag anomalies (`EMPTY_FOV`, `UNUSUALLY_SMALL`, `UNUSUALLY_LARGE`, `PADDING_LEAK`).
- **Manifests Generated:**
  - `data/processed/idrid/external_train.csv` (413 records)
  - `data/processed/idrid/external_test.csv` (103 records — strictly held out)
  - `data/processed/idrid/segmentation_manifest.csv` (81 records: 57 train-matched, 3 test-matched, 21 unique)

---

## 3. Step 2: Run Genuine IDRiD Domain Shift Diagnostic (N=413)

Execute non-destructive inference on all 413 training images using the original frozen APTOS checkpoint:

```bash
python scripts/diagnose_idrid_domain.py \
    --checkpoint experiments/checkpoints/best_model_qwk.pth \
    --idrid-manifest data/processed/idrid/external_train.csv
```

### Safety and Integrity Guarantees Enforced:
1. **Dual Immutability Verification:**
   - File SHA-256 hash computed before and after inference.
   - In-memory `state_dict` tensors cloned before inference and verified 100% bit-exact (`torch.equal`) after inference.
2. **Optimizer Inactive:** Model weights are frozen (`param.requires_grad = False`), and zero optimizer steps or loss functions are executed (`optimizer = None`).
3. **Dataset Isolation:** The candidate cohort is verified against `external_test.csv` and official test labels. Zero test-set images are permitted.
4. **Hard Failure on Missing Inputs:** If any of the 413 images or weights are missing, the script aborts immediately with returncode 1 (no silent fallback).
5. **Outputs Generated:**
   - Predictions CSV: `reports/idrid_training_predictions.csv`
   - Detailed Markdown: `reports/idrid_training_domain_diagnostic_report.md`
   - Machine-Readable JSON: `reports/idrid_training_domain_diagnostic_report.json`

---

## 4. Step 3: Verify and Recalculate Diagnostic Metrics

To independently verify the numbers from the generated predictions CSV without running model forward passes:

```python
from pathlib import Path
from scripts.diagnose_idrid_domain import recalculate_metrics_from_csv

metrics = recalculate_metrics_from_csv(Path("reports/idrid_training_predictions.csv"))
print(f"Verified Accuracy: {metrics['accuracy'] * 100:.2f}%")
print(f"Verified QWK:      {metrics['qwk']:.4f}")
print(f"Off-by-<=1 Rate:   {metrics['off_by_one_pct']:.2f}%")
print(f"Referable DR Sens: {metrics['referable_dr']['sensitivity'] * 100:.2f}%")
print(f"Referable DR Spec: {metrics['referable_dr']['specificity'] * 100:.2f}%")
```

---

## 5. Step 4: Run Genuine Phase 6 Grad-CAM Localization (N=81)

Evaluate spatial overlap against expert-annotated pixel masks:

```bash
python scripts/evaluate_localization.py \
    --checkpoint experiments/checkpoints/best_model_qwk.pth \
    --manifest data/processed/idrid/segmentation_manifest.csv
```

### Key Guarantees & Cohort Separation Enforced:
1. **Primary Localization Cohort ($N=78$ Non-Overlapping):**
   - Strictly excludes the 3 images that overlap with the IDRiD disease grading testing set.
   - Evaluates 57 train-matched and 21 unique segmentation images as the primary benchmark.
2. **Supplementary Test Overlap Cohort ($N=3$):**
   - Evaluates the 3 test-matched images (`IDRiD_09` -> `IDRiD_478`, `IDRiD_22` -> `IDRiD_470`, `IDRiD_31` -> `IDRiD_471`) separately.
   - These 3 images are **never** used for model training, fine-tuning, or hyperparameter selection.
3. **Full Dataset Preservation:** The complete 81-image segmentation dataset is preserved and reported.
4. **Grad-CAM Gradients vs Training Updates:**
   - Grad-CAM activation gradients ($\partial y^c / \partial A^k$) compute feature map channel weights.
   - Model parameters have `param.requires_grad = False`. No gradients with respect to weights are computed.
   - In-memory `state_dict` verified 100% bit-exact identical before and after Grad-CAM generation.
5. **Multi-Threshold Analysis:** Evaluates top 5%, 10% (baseline), and 20% heatmap cutoffs.
6. **Grade 0 Exclusion:** Images without retinal lesions (Grade 0) are treated as N/A (not penalized as 0.0).
7. **Negative Control:** Computes overlap against the Optic Disc (OD) to ensure Grad-CAM does not fixate on healthy anatomy.
8. **Outputs Generated:**
   - Detailed Markdown: `reports/phase_6_localization_report.md`
   - Structured JSON: `reports/phase_6_localization_report.json`

---

## 6. Step 5: Run Genuine Phase 7 Grad-CAM Faithfulness Evaluation (Deletion Test)

Evaluate decision fidelity via progressive salient region deletion and area-matched random controls:

```bash
python scripts/evaluate_faithfulness.py \
    --checkpoint experiments/checkpoints/best_model_qwk.pth \
    --manifest data/processed/idrid/segmentation_manifest.csv \
    --deletion-steps 0.10,0.20,0.30 \
    --fill-strategies black,gaussian_blur \
    --num-random-trials 5
```

### Key Guarantees & Faithfulness Metrics:
1. **Progressive Salient Deletion:** Progressively removes top 10%, 20%, and 30% most salient retinal regions identified by Grad-CAM.
2. **Area-Matched Random Controls & Preprocessed FOV Parity:** Generates 5 independent area-matched random binary masks per step. Reuses the exact preprocessed retinal FOV mask generated during preprocessing (`mask_fov`), ensuring exact pixel-count parity ($K = \text{round}(f \times N_{\text{fov}})$) and 0.0% border bleed outside the circular retinal boundary.
3. **Dual Fill Strategies:**
   - `black`: Canonical zero-fill baseline `(0, 0, 0)`.
   - `gaussian_blur`: Neutral blurred fill ($k=21$) to control for high-frequency edge/OOD artifacts.
4. **Primary vs Supplementary Separation:** Evaluates the 78 non-overlapping primary cohort and 3 test-matched supplementary cohort separately.
5. **Metrics Computed:**
   - Confidence drop: $\Delta p = p_{\text{base}} - p_{\text{ablated}}$
   - Faithfulness Margin: $\mathcal{M} = \Delta p_{\text{salient}} - \Delta p_{\text{random}}$
   - Area Under Deletion Curve (AUDC) and AUDC margin.
   - Grade-wise breakdowns across ICDR grades 0 to 4.
6. **Immutability Guarantee:** Model weights snapshot verified 100% bit-exact identical before and after execution.
7. **Zero-Download Offline Inference:** Models are constructed with `pretrained=False`, loading the saved APTOS checkpoint weights directly without downloading ImageNet weights from the network.
8. **Outputs Generated:**
   - Detailed Markdown: `reports/phase_7_faithfulness_report.md`
   - Structured JSON: `reports/phase_7_faithfulness_report.json`
   - Predictions CSV: `reports/faithfulness_predictions.csv`

---

## 7. Complete Google Colab Execution Notebook

Copy-paste these exact code blocks into your Google Colab notebook with GPU runtime enabled (**Runtime > Change runtime type > T4 GPU**):

### Cell 1: Mount Google Drive & Clone Repository
```python
from google.colab import drive
import os

drive.mount('/content/drive')

# Clone repo and enter directory
!git clone https://github.com/Shrees1ngh/DR-GradCAM-Evaluation.git
%cd DR-GradCAM-Evaluation
```

### Cell 2: Install Dependencies
```python
!pip install -r requirements.txt
```

### Cell 3: Link Trained Checkpoint & Physical IDRiD Dataset
```python
import os
import shutil

# Ensure checkpoints directory exists
os.makedirs("experiments/checkpoints", exist_ok=True)

# Copy trained APTOS checkpoint from Google Drive
# (Adjust source path to match your Google Drive folder structure)
DRIVE_CKPT = "/content/drive/MyDrive/DR-GradCAM-Evaluation/best_model_qwk.pth"
LOCAL_CKPT = "experiments/checkpoints/best_model_qwk.pth"

if os.path.exists(DRIVE_CKPT):
    shutil.copy(DRIVE_CKPT, LOCAL_CKPT)
    print(f"✅ Checkpoint copied to {LOCAL_CKPT}")
else:
    print(f"⚠️ Checkpoint not found at {DRIVE_CKPT}. Please verify path!")

# Extract or link raw IDRiD dataset into data/raw/IDRiD
# Example: if stored as IDRiD.zip on Drive:
DRIVE_IDRID_ZIP = "/content/drive/MyDrive/DR-GradCAM-Evaluation/IDRiD.zip"
if os.path.exists(DRIVE_IDRID_ZIP):
    !unzip -q {DRIVE_IDRID_ZIP} -d data/raw/
    print("✅ IDRiD raw dataset unzipped to data/raw/")
```

### Cell 4: Preprocess IDRiD Images & Generate Aligned Manifests
```python
!python scripts/preprocess_data.py --size 512 --idrid-only
```

### Cell 5: Run Genuine Domain Shift Diagnostic (N=413 IDRiD Train Images)
```python
!python scripts/diagnose_idrid_domain.py \
    --checkpoint experiments/checkpoints/best_model_qwk.pth \
    --idrid-manifest data/processed/idrid/external_train.csv
```

### Cell 6: Run Genuine Phase 6 Localization Evaluation (N=81 Images)
```python
!python scripts/evaluate_localization.py \
    --checkpoint experiments/checkpoints/best_model_qwk.pth \
    --manifest data/processed/idrid/segmentation_manifest.csv
```

### Cell 7: Run Genuine Phase 7 Faithfulness Evaluation (Deletion Test)
```python
!python scripts/evaluate_faithfulness.py \
    --checkpoint experiments/checkpoints/best_model_qwk.pth \
    --manifest data/processed/idrid/segmentation_manifest.csv \
    --deletion-steps 0.10,0.20,0.30 \
    --fill-strategies black,gaussian_blur \
    --num-random-trials 5
```

### Cell 8: Sync Reports and Predictions Back to Google Drive
```python
DRIVE_REPORTS = "/content/drive/MyDrive/DR-GradCAM-Evaluation/reports"
os.makedirs(DRIVE_REPORTS, exist_ok=True)

!cp -r reports/* {DRIVE_REPORTS}/
print(f"✅ All reports and predictions successfully backed up to {DRIVE_REPORTS}!")
```


