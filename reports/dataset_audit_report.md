# Multi-Dataset Audit Report — Phase 1

> **Project:** DR-GradCAM-Evaluation  
> **Scope:** Audit of IDRiD (Benchmark & Ground-Truth Lesion Masks) and APTOS 2019 (Large-Scale Screening)  

---

## 1. Executive Summary

- **IDRiD Dataset Status:** `VERIFIED ✅`
- **APTOS 2019 Dataset Status:** `VERIFIED ✅`

---

## 2. IDRiD Dataset Audit (Primary Benchmark & Lesion Ground Truth)

- **Dataset Root:** `C:\Projects\DR-GradCAM-Evaluation\data\raw\IDRiD`
- **Disease Grading Images:** 516 (Train: 413, Test: 103)
- **Segmentation Images:** 81 (Train: 54, Test: 27)
- **Total Segmentation Masks:** 363 masks across 5 structures
- **Corrupted / Unreadable Files:** 0 (Zero corruption across all 597 images and 363 masks)

### 2.1 IDRiD Image Files & Integrity

| Subset | Found | Valid | Corrupted | Resolution | Color Mode | Aspect Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Grading (Training Set) | 413 | 413 | 0 | 4288x2848 | RGB | 3:2 |
| Grading (Testing Set) | 103 | 103 | 0 | 4288x2848 | RGB | 3:2 |
| Segmentation (Combined) | 81 | 81 | 0 | 4288x2848 | RGB | 3:2 |

### 2.2 IDRiD Disease Grading Class Distribution (ICDR Scale)

| ICDR Grade | Clinical Diagnosis | Training Set | Testing Set | Combined Total | Combined % | Clinical Severity |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Grade 0** | No DR | 134 | 34 | 168 | **32.6%** | Healthy / Non-referable |
| **Grade 1** | Mild NPDR | 20 | 5 | 25 | **4.8%** | Microaneurysms only (Non-referable) |
| **Grade 2** | Moderate NPDR | 136 | 32 | 168 | **32.6%** | Referable DR |
| **Grade 3** | Severe NPDR | 74 | 19 | 93 | **18.0%** | Referable DR (High Risk) |
| **Grade 4** | Proliferative DR (PDR) | 49 | 13 | 62 | **12.0%** | Sight-Threatening (Immediate Care) |
| **Total** | — | 413 | 103 | 516 | **100.0%** | — |

> **Key IDRiD Imbalance Observation:** Grade 1 is scarce (25 total images, 4.8%). Grade 0 (32.6%) and Grade 2 (32.6%) comprise ~65.2% of the dataset.

### 2.3 IDRiD Lesion Masks Availability & Integrity

- **Total Segmentation Images:** 81 (54 Training + 27 Testing)
- **Mask Dimension Alignment:** 100% verified (All masks match image size 4288x2848)
- **Corrupted Masks:** 0

| Lesion Type | Code | Total Masks | Train Split | Test Split | Missing / Absent | Clinical Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Microaneurysms | `MA` | 81 / 81 | 54 | 27 | 0 | Present in all 81 segmentation images |
| Haemorrhages | `HE` | 80 / 81 | 53 | 27 | 1 (`IDRiD_43`) | Single case without haemorrhages |
| Hard Exudates | `EX` | 81 / 81 | 54 | 27 | 0 | Present in all 81 segmentation images |
| Soft Exudates | `SE` | 40 / 81 | 26 | 14 | 41 images | Absence indicates zero soft exudate lesions |
| Optic Disc | `OD` | 81 / 81 | 54 | 27 | 0 | Normal anatomical structure (negative control) |

### 2.4 IDRiD Cross-Subset Overlap & Data Leakage Analysis

- **Filename Sequence Collision:** Note: IDRiD test set reuses numbering IDRiD_001 to IDRiD_103. However, file hashes and labels confirm they are completely distinct images stored in separate directories. They must be indexed with split prefixes (e.g. 'train/IDRiD_001' vs 'test/IDRiD_001') to avoid collisions.
- **Patient Identifiers Available:** `False` (Anonymized sequential IDs; no clinical patient IDs provided).
- **Empirical Segmentation-to-Grading Alignment (MAE < 2.0):**
  - **57 images** match images in the **Disease Grading Training Set** (`a. Training Set`).
  - **3 images** match images in the **Disease Grading Testing Set** (`b. Testing Set`).
  - **21 images** are unique to the segmentation set (not in the grading split).
  - Full per-image match details stored in: `reports/idrid_seg_grading_matches.json`.

> [!IMPORTANT]
> In Phase 6 (Localization), we must explicitly evaluate and report localization scores separately for training-distribution images (57 images) versus held-out test images (3 images) to avoid conflating train-set memorization with true generalization.

---

## 3. APTOS 2019 Dataset Audit (Large-Scale Screening)

- **Dataset Root:** `C:\Projects\DR-GradCAM-Evaluation\data\raw\APTOS`
- **Total CSV Rows:** 3662
- **Total Images on Disk:** 3662
- **Missing Images (CSV vs Disk):** 0 (Exact 1:1 match)
- **Extra Images on Disk:** 0
- **Duplicate CSV Identifiers:** 0
- **Duplicate Disk Filenames:** 0
- **Unreadable / Corrupt Images:** 0 (Zero corruption)

### 3.1 APTOS Disease Grading Class Distribution

| ICDR Grade | Clinical Diagnosis | Image Count | Percentage | Clinical Note |
| :---: | :--- | :---: | :---: | :--- |
| **Grade 0** | No DR | 1805 | **49.3%** | Healthy / Non-referable |
| **Grade 1** | Mild NPDR | 370 | **10.1%** | Microaneurysms only (Non-referable) |
| **Grade 2** | Moderate NPDR | 999 | **27.3%** | Referable DR |
| **Grade 3** | Severe NPDR | 193 | **5.3%** | Referable DR (High Risk) |
| **Grade 4** | Proliferative DR (PDR) | 295 | **8.1%** | Sight-Threatening (Immediate Care) |
| **Total** | — | **3662** | **100.0%** | — |

### 3.2 APTOS Image Specifications & Resolutions

- **Primary Format:** PNG (100% of images)
- **Color Modes:** 100% 3-channel RGB (3662 images)
- **Total Unique Resolutions:** 17 distinct resolutions

| Rank | Resolution | Count | Percentage | Aspect Ratio |
| :---: | :---: | :---: | :---: | :---: |
| 1 | `1050x1050` | 974 | 26.6% | 1:1 (Square) |
| 2 | `2416x1736` | 638 | 17.4% | 1.39:1 |
| 3 | `2588x1958` | 533 | 14.6% | 1.32:1 |
| 4 | `3216x2136` | 410 | 11.2% | 1.51:1 |
| 5 | `2048x1536` | 351 | 9.6% | 1.33:1 |

> **Key APTOS Characteristics:**
> 1. **Substantial Scale:** With 3,662 images, APTOS provides ~7.1x more training data than IDRiD (516 images).
> 2. **Varied Resolutions:** Unlike IDRiD where 100% of images are 4288x2848, APTOS contains images from different clinics with varying resolutions (square 1050x1050, rectangular 2416x1736, etc.). Unified preprocessing and circular FOV cropping are essential.
> 3. **Absence of Lesion Masks:** APTOS contains classification labels only. It serves as an ideal external cohort for classification and faithfulness evaluation, while IDRiD remains the exclusive dataset for localization.

---

## 4. Cross-Dataset Comparison: IDRiD vs. APTOS 2019

| Research Dimension | IDRiD Dataset | APTOS 2019 Dataset |
| :--- | :--- | :--- |
| **Total Images** | 516 grading + 81 segmentation (597 total) | 3662 grading images |
| **Expert Lesion Masks** | **Available (81 images, 363 masks)** | **None** (Image-level labels only) |
| **Native Resolution** | Uniform $4288 \times 2848$ (100%) | Multi-resolution (17 formats, top: $1050 \times 1050$) |
| **Native Image Format** | JPEG (Grading) / TIF (Masks) | PNG |
| **Grade 0 (No DR)** | 168 (32.6%) | 1,805 (49.3%) |
| **Grade 1 (Mild NPDR)** | 25 (4.8%) | 370 (10.1%) |
| **Grade 2 (Moderate NPDR)** | 168 (32.6%) | 999 (27.3%) |
| **Grade 3 (Severe NPDR)** | 93 (18.0%) | 193 (5.3%) |
| **Grade 4 (PDR)** | 62 (12.0%) | 295 (8.1%) |
| **Role in Project** | **Core Benchmark (Localization & Faithfulness)** | **Large-Scale Pretraining & External Evaluation** |

---
*Report auto-generated by `scripts/audit_dataset.py` as part of Phase 1.*