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

If `data/processed/` has not yet been populated, run the deterministic preprocessor:

```bash
python scripts/preprocess_data.py --size 512
```

### Preprocessing Geometry Guarantees:
- **FOV Cropping:** `crop_to_retinal_fov` detects circular retinal boundaries in the green channel and excises empty horizontal sensor margins ($1500+$ px on IDRiD's $4288 \times 2848$ frames).
- **Aspect Ratio:** The cropped retina is approximately square ($1.01:1$).
- **Actual Model Letterbox Padding:** Minimal ($\le 3\%$), unlike the uncropped native sensor padding ($33.59\%$).
- **Active Retinal Tissue:** Covers $\sim 75\%$ of the $512 \times 512$ canvas (with $\sim 25\%$ representing the four natural corners outside the circular optical field).
- **Manifests Generated:**
  - `data/processed/idrid/external_train.csv` (413 records)
  - `data/processed/idrid/external_test.csv` (103 records — strictly held out)
  - `data/processed/idrid/segmentation_manifest.csv` (81 records)

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
2. **Area-Matched Random Controls:** Generates 5 independent area-matched random binary masks per step, strictly constrained within the retinal Field of View (FOV) to prevent padding bias.
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
7. **Outputs Generated:**
   - Detailed Markdown: `reports/phase_7_faithfulness_report.md`
   - Structured JSON: `reports/phase_7_faithfulness_report.json`
   - Predictions CSV: `reports/faithfulness_predictions.csv`

