# Phase 2 Completion Report: Data Preparation & Preprocessing

**Project:** DR-GradCAM-Evaluation  
**Phase:** Phase 2 — Dataset Strategy, Image Preprocessing & Multi-Lesion Alignment  
**Date:** November 7, 2026  
**Status:** COMPLETE & VERIFIED ✅  

---

## Executive Summary

Phase 2 implementation and empirical verification are complete. All approved datasets (APTOS 2019 primary backbone and IDRiD external benchmarks) have been physically preprocessed to standardized $512 \times 512$ resolution, strictly preserving native retinal aspect ratios via circular FOV detection, isotropic scaling, and symmetric zero-padding.

All methodological corrections mandated for Phase 2 have been strictly enforced and empirically validated on disk:
1. **IDRiD Segmentation Independence:** All 81 IDRiD segmentation images are documented as **100% unseen** by the primary classification model. The 57 images overlapping IDRiD's own grading set are classified strictly as `idrid_grading_train_cohort` and never as model training-distribution data.
2. **Perceptual Hash Audit:** Comprehensive 64-bit DCT pHash and gradient dHash verification across all 2,186,214 cross-dataset pairs confirmed zero exact visual duplicates, zero frequency-domain near-duplicates, and zero joint near-duplicates between APTOS and IDRiD.
3. **Mask Spatial Alignment & Binary Integrity:** All 486 generated lesion masks have been verified as strictly binary ($\{0, 255\}$) at $512 \times 512$, with zero positive pixels falling into padded margins (100% spatial FOV confinement).
4. **Missing Annotation Distinctions:** Unannotated lesion files (41 SE masks, 1 HE mask) are strictly flagged as `annotated=False` with `pixel_count=-1` and `lesion_present=None`, preventing artificial zero-IoU penalties in Phase 6.

---

## 1. Cross-Dataset Perceptual Similarity Audit

Pairwise similarity was evaluated between all 3,662 APTOS fundus images and all 597 unique IDRiD images (413 grading train + 103 grading test + 81 segmentation) using both frequency-domain DCT pHash and spatial-gradient dHash:

$$\text{Total Pairwise Comparisons} = 3,662 \times 597 = 2,186,214 \text{ pairs}$$

| Metric | pHash (2D DCT Frequency) | dHash (Horizontal Gradient) | Joint (pHash $\le 5$ AND dHash $\le 5$) |
|:---|:---:|:---:|:---:|
| **Minimum Distance** | **6 bits** | **0 bits** | **None (0 pairs)** |
| **1st Percentile ($P_1$)** | 15.0 bits | 4.0 bits | — |
| **Median ($P_{50}$)** | 28.0 bits | 16.0 bits | — |
| **Mean $\pm$ Std** | $27.41 \pm 4.81$ bits | $15.71 \pm 5.88$ bits | — |
| **Exact Duplicates ($D = 0$)** | 0 | 1 | **0** |
| **Near Duplicates ($D \le 5$)** | **0** | 7,651 | **0** |

### Methodological Analysis of Hash Discrepancy
- **pHash Analysis:** Evaluates the lowest $8 \times 8$ 2D DCT frequency coefficients. pHash is robust against circular boundary artifacts. The empirical minimum across all 2.18M pairs was **6 bits** (mean 27.4 bits). Zero pairs exhibited pHash $\le 5$ bits.
- **dHash Analysis:** Evaluates monotonic horizontal gradients across an $8 \times 8$ grid. Because retinal fundus photography shares an invariant geometric structure (black background $\rightarrow$ bright circular retina $\rightarrow$ black background), distinct fundus images with similar disc placement can produce identical horizontal gradient signs. 
- **Joint Metric Verification:** Cross-referencing all candidate pairs with low dHash confirmed that every single pair exhibited high pHash distance (13 to 22 bits), proving that low dHash values are geometric boundary artifacts rather than true visual duplicates.
- **Verdict:** **Visual independence confirmed.** Primary classification training on APTOS can proceed without cross-dataset contamination.

---

## 2. Processed Image & Mask Inventory

All physical image files and binary masks were generated on disk under `data/processed/` and verified:

```
data/processed/
├── aptos/
│   ├── images/                        # 3,662 PNG files (512x512x3)
│   ├── train_split.csv                # 2,929 rows (Stratified 80%)
│   └── val_split.csv                  # 733 rows (Stratified 20%)
└── idrid/
    ├── images/                        # 516 JPG files (512x512x3)
    ├── external_train.csv             # 413 rows (Kept separate)
    ├── external_test.csv              # 103 rows (Held-out test set)
    └── segmentation/
        ├── images/                    # 81 JPG files (512x512x3)
        ├── masks/                     # 486 PNG files (512x512, binary {0, 255})
        │   ├── MA/                    # 81 files
        │   ├── HE/                    # 81 files
        │   ├── EX/                    # 81 files
        │   ├── SE/                    # 81 files
        │   ├── OD/                    # 81 files
        │   └── composite/             # 81 files (MA U HE U EX U SE)
        └── segmentation_manifest.csv  # 81 rows (Enriched metadata)
```

### Actual Count Verification

| Dataset / Cohort | Target Role | On-Disk Files | Manifest Records | Verification Status |
|:---|:---|:---:|:---:|:---:|
| **APTOS Train Split** | Primary CNN Training (80%) | 2,929 | 2,929 | VERIFIED ✅ |
| **APTOS Val Split** | Primary Model Selection (20%) | 733 | 733 | VERIFIED ✅ |
| **IDRiD Grading External Train** | Secondary / Transfer (Not in primary training) | 413 | 413 | VERIFIED ✅ |
| **IDRiD Grading External Test** | Primary Model Held-Out Test Benchmark | 103 | 103 | VERIFIED ✅ |
| **IDRiD Segmentation Images** | XAI Localization Benchmark (100% Unseen) | 81 | 81 | VERIFIED ✅ |
| **IDRiD MA Masks** | Microaneurysm Ground Truth | 81 | 81 | VERIFIED ✅ |
| **IDRiD HE Masks** | Haemorrhage Ground Truth | 81 | 81 | VERIFIED ✅ |
| **IDRiD EX Masks** | Hard Exudates Ground Truth | 81 | 81 | VERIFIED ✅ |
| **IDRiD SE Masks** | Soft Exudates Ground Truth | 81 | 81 | VERIFIED ✅ |
| **IDRiD OD Masks** | Optic Disc (Landmark Control) | 81 | 81 | VERIFIED ✅ |
| **IDRiD Composite Masks** | $\text{MA} \cup \text{HE} \cup \text{EX} \cup \text{SE}$ | 81 | 81 | VERIFIED ✅ |
| **Total Processed Files** | — | **4,745 files** | — | **100% DISK VERIFIED** |

---

## 3. APTOS Stratified Split Balance

Stratified 80/20 train/validation split with fixed seed 42 preserved exact disease grade proportions:

| ICDR Grade | Clinical Classification | Train Split ($N=2,929$) | Val Split ($N=733$) | Total ($N=3,662$) | Train % | Val % |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Grade 0** | No DR | 1,444 | 361 | 1,805 | 49.30% | 49.25% |
| **Grade 1** | Mild NPDR | 296 | 74 | 370 | 10.11% | 10.10% |
| **Grade 2** | Moderate NPDR | 799 | 200 | 999 | 27.28% | 27.29% |
| **Grade 3** | Severe NPDR | 154 | 39 | 193 | 5.26% | 5.32% |
| **Grade 4** | Proliferative DR (PDR) | 236 | 59 | 295 | 8.06% | 8.05% |
| **Total** | — | **2,929** | **733** | **3,662** | **100.0%** | **100.0%** |

---

## 4. IDRiD Segmentation Cohort & Lesion Masks

### Cohort Origin Breakdown
All 81 segmentation images are marked as `model_seen_status = "unseen"`. Downstream localization in Phase 6 will be reported both overall and stratified by cohort origin:
- `idrid_grading_train_cohort`: **57 images** (originating from IDRiD's grading training set).
- `idrid_seg_unique`: **21 images** (unique to IDRiD's segmentation set).
- `idrid_grading_test_cohort`: **3 images** (originating from IDRiD's grading test set).

### Annotation Availability vs. Absence

| Lesion Code | Clinical Description | Annotated Files Present | Missing Annotation Files | Non-Zero Lesion Masks | All-Zero Lesion Masks |
|:---|:---|:---:|:---:|:---:|:---:|
| **MA** | Microaneurysms | 81 | 0 | 81 | 0 |
| **HE** | Haemorrhages | 80 | 1 (`IDRiD_43`) | 80 | 0 |
| **EX** | Hard Exudates | 81 | 0 | 81 | 0 |
| **SE** | Soft Exudates | 40 | 41 | 40 | 0 |
| **OD** | Optic Disc (Control) | 81 | 0 | 81 | 0 |
| **Composite** | $\text{MA} \cup \text{HE} \cup \text{EX} \cup \text{SE}$ | 81 | 0 | 81 | 0 |

> [!NOTE]
> Every existing annotation file contains non-zero positive lesion pixels. The 41 unannotated SE masks and 1 unannotated HE mask are strictly flagged in `segmentation_manifest.csv` as `annotated=False` with `pixel_count=-1` and `lesion_present=None`. They will be dynamically excluded from lesion-specific IoU/Dice denominators in Phase 6.

---

## 5. Spatial Alignment & Margin Confinement

Image-mask spatial alignment was verified across all 81 segmentation images:
- **FOV Margin Confinement:** Evaluated whether any lesion pixel falls into the black padding area $(pad_{top}, pad_{bottom}, pad_{left}, pad_{right})$. **Zero boundary leakage detected across all 81 images.**
- **Nearest-Neighbor Interpolation:** Verified that all 486 masks retain strictly binary pixel values in $\{0, 255\}$ (zero interpolation blur or anti-aliasing artifacts).
- **Optic Disc Exclusion:** Confirmed that Optic Disc (`OD`) pixels are strictly excluded from the composite lesion mask.

---

## 6. Test Suite Execution Results

Running the complete test suite across the repository (`pytest tests/ -v`):

```
============================= test session starts =============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Projects\DR-GradCAM-Evaluation
collected 12 items

tests/test_audit.py::test_audit_missing_dataset PASSED                   [  8%]
tests/test_audit.py::test_audit_synthetic_dataset PASSED                 [ 16%]
tests/test_audit.py::test_audit_aptos_synthetic PASSED                   [ 25%]
tests/test_preprocessing.py::test_crop_to_retinal_fov PASSED             [ 33%]
tests/test_preprocessing.py::test_resize_and_pad_aspect_ratio PASSED     [ 41%]
tests/test_preprocessing.py::test_mask_spatial_alignment_and_binarization PASSED [ 50%]
tests/test_preprocessing.py::test_build_composite_lesion_mask PASSED     [ 58%]
tests/test_preprocessing.py::test_dr_grading_dataset SKIPPED             [ 66%]
tests/test_preprocessing.py::test_segmentation_lesion_dataset SKIPPED    [ 75%]
tests/test_similarity.py::test_phash_and_dhash_exact_match PASSED        [ 83%]
tests/test_similarity.py::test_phash_near_duplicate_robustness PASSED    [ 91%]
tests/test_similarity.py::test_pairwise_hamming_distance PASSED          [100%]

======================== 10 passed, 2 skipped in 1.91s ========================
```

- **Pass Rate:** 100% of runnable tests passed (10 passed, 0 failed).
- **Skipped Tests:** 2 PyTorch tests gracefully skipped pending PyTorch installation in the global environment (`test_dr_grading_dataset` and `test_segmentation_lesion_dataset`).

---

## 7. Data Leakage & Git Tracking Audit

- **Raw Datasets (`data/raw/`):** 100% preserved in read-only state. Zero files modified, moved, or deleted.
- **Git Status:** `.gitignore` strictly excludes all raw medical images, processed images (`*.png`, `*.jpg`), masks, and cache files (`.similarity_cache.npz`).
- **Working Tree:** No dataset files are tracked or staged in Git. Only code, tests, and markdown reports are tracked.

---

## 8. Summary of Failures & Unresolved Issues

- **Hardware/Server Restart During Run:** Task-650 was interrupted midway by a server restart after processing 1,716 APTOS images. The preprocessing script was upgraded with safe caching/resume logic and multi-threading, successfully resuming and completing all 4,745 files to disk without data loss.
- **Host PyTorch Dependency:** The global Python environment currently lacks `torch` and `torchvision`. The dataset module `src/dataset.py` and test suite dynamically detect this and skip PyTorch-dependent tests gracefully. PyTorch with CUDA support will be configured for Phase 3 model training.

---

## 9. Deliverables Ready for Review
1. Multi-dataset audit reports: [`reports/dataset_audit_report.md`](file:///c:/Projects/DR-GradCAM-Evaluation/reports/dataset_audit_report.md) & [`reports/dataset_audit_report.json`](file:///c:/Projects/DR-GradCAM-Evaluation/reports/dataset_audit_report.json)
2. Cross-dataset perceptual similarity report: [`reports/cross_dataset_similarity_report.md`](file:///c:/Projects/DR-GradCAM-Evaluation/reports/cross_dataset_similarity_report.md) & [`reports/cross_dataset_similarity_report.json`](file:///c:/Projects/DR-GradCAM-Evaluation/reports/cross_dataset_similarity_report.json)
3. Preprocessing report: [`reports/preprocessing_report.md`](file:///c:/Projects/DR-GradCAM-Evaluation/reports/preprocessing_report.md)
4. Five CSV manifests under `data/processed/`:
   - `aptos/train_split.csv` ($N=2,929$)
   - `aptos/val_split.csv` ($N=733$)
   - `idrid/external_train.csv` ($N=413$)
   - `idrid/external_test.csv` ($N=103$)
   - `idrid/segmentation_manifest.csv` ($N=81$)
5. Preprocessing codebase & test suites:
   - [`src/preprocessing.py`](file:///c:/Projects/DR-GradCAM-Evaluation/src/preprocessing.py)
   - [`src/dataset.py`](file:///c:/Projects/DR-GradCAM-Evaluation/src/dataset.py)
   - [`src/similarity.py`](file:///c:/Projects/DR-GradCAM-Evaluation/src/similarity.py)
   - [`scripts/preprocess_data.py`](file:///c:/Projects/DR-GradCAM-Evaluation/scripts/preprocess_data.py)
   - [`scripts/verify_cross_dataset_similarity.py`](file:///c:/Projects/DR-GradCAM-Evaluation/scripts/verify_cross_dataset_similarity.py)
   - Unit tests: [`tests/test_audit.py`](file:///c:/Projects/DR-GradCAM-Evaluation/tests/test_audit.py), [`tests/test_preprocessing.py`](file:///c:/Projects/DR-GradCAM-Evaluation/tests/test_preprocessing.py), [`tests/test_similarity.py`](file:///c:/Projects/DR-GradCAM-Evaluation/tests/test_similarity.py)

