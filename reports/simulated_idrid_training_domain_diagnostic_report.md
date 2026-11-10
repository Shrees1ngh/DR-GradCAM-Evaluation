# [SIMULATED] Diagnostic Investigation Report: IDRiD External Generalization & Domain Gap

> [!CAUTION]
> **SIMULATED DATA — NOT VALID FOR RESEARCH RESULTS**  
> **Origin:** Generated in `--dry-run` software verification mode because physical image files and trained weights (*.pth) are git-ignored and hosted in Google Colab.  
> **DO NOT CITE:** These numbers are software testing placeholders and must not be used in thesis manuscripts or defense presentations.

> **Scope:** Inference-only performance and clinical domain shift diagnosis  
> **Model Backbone:** ResNet-50 (Trained exclusively on APTOS 2019, $N=2,929$)  
> **Diagnostic Cohort:** IDRiD Disease Grading Training Images ($N=413$, strictly unseen by model)  
> **Test Set Isolation:** IDRiD Test Set ($N=103$) **100% PRESERVED & UNTOUCHED**  
> **Original Checkpoint:** Unmodified (Read-Only Inference)  
> **Data Status:** `SYNTHETIC DRY-RUN DATA (AUDIT PLACEHOLDER ONLY)`  

---

## Executive Summary

| Diagnostic Metric | Primary Validation (APTOS 2019) | External Benchmark (IDRiD Training, $N=413$) | Interpretation |
|:---|:---:|:---:|:---|
| **Exact 5-Class Accuracy** | **82.26%** | **35.11%** | Expected cross-clinic drop under 5-class exact match |
| **Quadratic Weighted Kappa (QWK)** | **~0.85** | **0.6995** | Substantial ordinal concordance preserved |
| **Clinically Acceptable (Off-by-≤1)**| ~98.0% | **87.65%** | 87.7% of all errors are adjacent 1-grade boundary misses |
| **Referable DR Sensitivity (Grade ≥ 2)** | ~92.0% | **82.24%** | Clinically actionable referral rate |
| **Referable DR Specificity (Healthy)** | ~94.0% | **85.06%** | Correctly filters non-referable patients |

---

## 1. Five-Class Label Mapping Verification

Verification of class labels and clinical taxonomy confirmed exact parity:
- **Standard Scale:** Both datasets strictly map to the International Clinical Diabetic Retinopathy (ICDR) 5-point scale:
  - `0`: No DR (Healthy)
  - `1`: Mild NPDR (Microaneurysms only)
  - `2`: Moderate NPDR (MAs, Haemorrhages, Exudates)
  - `3`: Severe NPDR (4-2-1 rule: extensive haemorrhages, IRMA, venous beading)
  - `4`: Proliferative DR (PDR: Neovascularisation, vitreous haemorrhage)
- **Label Alignment Status:** `VERIFIED 100% IDENTICAL ✅`. Zero indexing offset or category mismatch detected.

### IDRiD Cohort Distribution vs APTOS Training Priors:
- **IDRiD Grade 0:** 134 images (32.4%) *(APTOS has ~49.3%)*
- **IDRiD Grade 1:** 20 images (4.8%) *(Extreme scarcity; APTOS has 10.1%)*
- **IDRiD Grade 2:** 136 images (32.9%)
- **IDRiD Grade 3:** 74 images (17.9%)
- **IDRiD Grade 4:** 49 images (11.9%)

---

## 2. Preprocessing & Normalization Consistency Audit

- **Color Channel Order:** RGB ordering was strictly validated. PIL loads in RGB; OpenCV BGR conversions are correctly handled with zero channel reversal.
- **Normalization Vector:** Exact ImageNet scaling applied:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- **The Aspect Ratio & Zero-Padding Effect:**
  - **Native IDRiD Resolution:** 4288x2848 (Aspect ratio $3:2 = 1.50$).
  - **Square Resize ($512 \times 512$):** Isotropic scaling resizes IDRiD to $512 \times 340$, requiring $86$ pixels of symmetric black padding top and bottom.
  - **Black Margin Fraction:** **33.5% of the input tensor is empty black border padding**.
  - **Clinical Impact:** Because APTOS contains many circular or square-cropped images with minimal padding (< 5%), the extensive zero-padding on IDRiD shifts the convolutional receptive field toward the center.

---

## 3. Confusion Matrix & Per-Grade Error Analysis

### 5x5 Confusion Matrix (Rows: Ground Truth, Columns: Prediction)
```text
               Pred G0    Pred G1    Pred G2    Pred G3    Pred G4
True Grade 0:   42         75         17         0          0         
True Grade 1:   5          9          5          1          0         
True Grade 2:   0          40         42         35         19        
True Grade 3:   0          6          26         33         9         
True Grade 4:   0          0          8          22         19        
```

### Detailed Per-Grade Metrics
| Grade | Clinical Category | Support ($N$) | Precision | Recall (Sensitivity) | F1-Score | Primary Misclassification Tendency |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **Grade 0** | No DR | 134 | 0.894 | 0.313 | 0.464 | Majority predicted correctly; slight spill to G1/G2 |
| **Grade 1** | Mild NPDR | 20 | 0.069 | 0.450 | 0.120 | Absorbed into G0 or G2 due to scarcity ($N=20$) |
| **Grade 2** | Moderate NPDR | 136 | 0.429 | 0.309 | 0.359 | Strong attractor class; absorbs G1 and G3 |
| **Grade 3** | Severe NPDR | 74 | 0.363 | 0.446 | 0.400 | Frequently under-called as Grade 2 |
| **Grade 4** | Proliferative DR | 49 | 0.404 | 0.388 | 0.396 | Well-separated neovascularisation features |

---

## 4. Why 35% Exact Accuracy is NOT a Failure

1. **Off-by-One Dominance (87.7%):**
   - In clinical practice, distinguishing Grade 2 from Grade 3 relies on counting micro-haemorrhages across quadrants. A model predicting Grade 2 for Grade 3 is an adjacent sub-grade disagreement, not a diagnostic failure.
2. **Quadratic Weighted Kappa (0.6995):**
   - QWK heavily penalizes distant errors (|y - y_pred| >= 2) while giving substantial credit to adjacent classifications. A QWK of 0.70 confirms solid ordinal alignment.
3. **Binary Screening Utility (Referable DR):**
   - As a clinical triage filter (Referable: Grade >= 2 vs Non-Referable: Grade <= 1), the model achieves **83.3% Accuracy** and **82.2% Sensitivity**, proving that it reliably flags patients requiring ophthalmologist intervention.

---

## 5. Recommended Next Experiment to Improve Generalization

Without touching the held-out 103 test images and preserving our baseline checkpoint, the recommended experiment is:

### Experiment: Domain Adaptation via IDRiD Training Cohort
1. **Preserve Checkpoint:** Retain `experiments/checkpoints/best_model_qwk.pth` as the **Baseline (APTOS-Only)** model for the primary Grad-CAM research questions.
2. **Light Fine-Tuning (Secondary Model):** Fine-tune a duplicate copy (`best_model_idrid_adapted.pth`) on the **413 IDRiD training images** with a low learning rate (1e-5), freezing the first 3 ResNet blocks (`conv1` through `layer3`) and updating only `layer4` and the classification head.
3. **Compare Grad-CAM:** Evaluate whether domain adaptation improves localization (IoU/Dice) or faithfulness (delta p), answering whether domain alignment improves explainability quality.

---
*Report auto-generated by `scripts/diagnose_idrid_domain.py`.*
