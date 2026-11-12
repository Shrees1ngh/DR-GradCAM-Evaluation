# Data Preparation & Preprocessing Report — Phase 2

> **Project:** DR-GradCAM-Evaluation  
> **Scope:** Preprocessing for Primary Backbone (APTOS 2019) and External Benchmarks (IDRiD)  

---

## 1. Executive Summary

- **Pipeline Status:** `FAILED` ❌
- **Data Leakage Check:** `FAILED ⚠️`
- **Target Resolution:** `512 x 512` (Preserving native aspect ratio with symmetric zero-padding)
- **Primary Training Set (APTOS 80%):** 2929 images
- **Primary Validation Set (APTOS 20%):** 733 images
- **External Test Benchmark (IDRiD):** 103 images
- **External Training Cohort (IDRiD):** 413 images (Kept separate; not used for primary training)
- **Localization Benchmark (IDRiD):** 81 images (100% UNSEEN by primary model)

## 2. Primary Classification Backbone: APTOS 2019 Split Balance

| ICDR Grade | Clinical Diagnosis | Train Split (80%) | Val Split (20%) | Total Images | Train % | Val % |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Grade 0** | No DR | 1444 | 361 | 1805 | 49.3% | 49.2% |
| **Grade 1** | Mild NPDR | 296 | 74 | 370 | 10.1% | 10.1% |
| **Grade 2** | Moderate NPDR | 799 | 200 | 999 | 27.3% | 27.3% |
| **Grade 3** | Severe NPDR | 154 | 39 | 193 | 5.3% | 5.3% |
| **Grade 4** | Proliferative DR (PDR) | 236 | 59 | 295 | 8.1% | 8.0% |
| **Total** | — | **2929** | **733** | **3662** | **100.0%** | **100.0%** |

## 3. IDRiD Segmentation Cohort & Lesion Masks

> [!IMPORTANT]
> **Model Independence Guarantee:** Our primary CNN is trained SOLELY on APTOS 2019.
> Therefore, **ALL 81 IDRiD segmentation images are 100% unseen by the model**.
> The 57 images originating from IDRiD's grading training set must NOT be described as
> 'model training-distribution images'. They are documented as `idrid_grading_train_cohort`.

### Cohort Origin Breakdown
- `idrid_grading_train_cohort`: **57** images (Overlap with IDRiD's own grading train set)
- `idrid_grading_test_cohort`: **3** images (Overlap with IDRiD's own grading test set)
- `idrid_seg_unique`: **21** images (Unique to IDRiD segmentation)
- **Model Seen Status:** 100% marked as `unseen`.

### Lesion Mask Annotation Availability vs. Absence

| Lesion Type | Annotated Files Present | Missing Annotation Files | All-Zero Annotated Masks | Non-Zero Lesion Masks |
| :--- | :---: | :---: | :---: | :---: |
| **MA** (Microaneurysms) | 81 | 0 | 0 | 81 |
| **HE** (Haemorrhages) | 80 | 1 | 0 | 80 |
| **EX** (Hard Exudates) | 81 | 0 | 0 | 81 |
| **SE** (Soft Exudates) | 40 | 41 | 0 | 40 |
| **OD** (Optic Disc (landmark control)) | 81 | 0 | 0 | 81 |

- **Composite Lesion Mask:** Union of `MA ∪ HE ∪ EX ∪ SE` (Optic Disc strictly excluded).

## 4. Zero-Leakage & Integrity Verification

| Verification Check | Target Cohorts | Status | Audit Findings |
| :--- | :--- | :---: | :--- |
| **APTOS Train Uniqueness** | APTOS Train Cohort | `PASSED ✅` | All 2929 image identities in APTOS Train are unique. |
| **APTOS Val Uniqueness** | APTOS Val Cohort | `PASSED ✅` | All 733 image identities in APTOS Val are unique. |
| **IDRiD Train Uniqueness** | IDRiD Train Cohort | `PASSED ✅` | All 413 image identities in IDRiD Train are unique. |
| **IDRiD Test Uniqueness** | IDRiD Test Cohort | `PASSED ✅` | All 103 image identities in IDRiD Test are unique. |
| **IDRiD Segmentation Uniqueness** | IDRiD Segmentation Cohort | `PASSED ✅` | All 81 image identities in IDRiD Segmentation are unique. |
| **APTOS Split Disjointness** | APTOS Train (80%) vs Val (20%) | `PASSED ✅` | APTOS Train (2929) and Val (733) splits are 100% disjoint. |
| **APTOS Sample Conservation** | Total APTOS Cohort (N=3,662) | `PASSED ✅` | Sample count conserved: 2929 train + 733 val = 3,662. |
| **IDRiD Cohort Isolation** | IDRiD Train vs Test | `FAILED ❌` | Critical Leakage: 103 image identities overlap between IDRiD Train and Test: ['IDRiD_001', 'IDRiD_002', 'IDRiD_003', 'IDRiD_004', 'IDRiD_005', 'IDRiD_006', 'IDRiD_007', 'IDRiD_008', 'IDRiD_009', 'IDRiD_010', 'IDRiD_011', 'IDRiD_012', 'IDRiD_013', 'IDRiD_014', 'IDRiD_015', 'IDRiD_016', 'IDRiD_017', 'IDRiD_018', 'IDRiD_019', 'IDRiD_020', 'IDRiD_021', 'IDRiD_022', 'IDRiD_023', 'IDRiD_024', 'IDRiD_025', 'IDRiD_026', 'IDRiD_027', 'IDRiD_028', 'IDRiD_029', 'IDRiD_030', 'IDRiD_031', 'IDRiD_032', 'IDRiD_033', 'IDRiD_034', 'IDRiD_035', 'IDRiD_036', 'IDRiD_037', 'IDRiD_038', 'IDRiD_039', 'IDRiD_040', 'IDRiD_041', 'IDRiD_042', 'IDRiD_043', 'IDRiD_044', 'IDRiD_045', 'IDRiD_046', 'IDRiD_047', 'IDRiD_048', 'IDRiD_049', 'IDRiD_050', 'IDRiD_051', 'IDRiD_052', 'IDRiD_053', 'IDRiD_054', 'IDRiD_055', 'IDRiD_056', 'IDRiD_057', 'IDRiD_058', 'IDRiD_059', 'IDRiD_060', 'IDRiD_061', 'IDRiD_062', 'IDRiD_063', 'IDRiD_064', 'IDRiD_065', 'IDRiD_066', 'IDRiD_067', 'IDRiD_068', 'IDRiD_069', 'IDRiD_070', 'IDRiD_071', 'IDRiD_072', 'IDRiD_073', 'IDRiD_074', 'IDRiD_075', 'IDRiD_076', 'IDRiD_077', 'IDRiD_078', 'IDRiD_079', 'IDRiD_080', 'IDRiD_081', 'IDRiD_082', 'IDRiD_083', 'IDRiD_084', 'IDRiD_085', 'IDRiD_086', 'IDRiD_087', 'IDRiD_088', 'IDRiD_089', 'IDRiD_090', 'IDRiD_091', 'IDRiD_092', 'IDRiD_093', 'IDRiD_094', 'IDRiD_095', 'IDRiD_096', 'IDRiD_097', 'IDRiD_098', 'IDRiD_099', 'IDRiD_100', 'IDRiD_101', 'IDRiD_102', 'IDRiD_103']. |
| **IDRiD Training Set Integrity** | IDRiD Grading Training Set (N=413) | `PASSED ✅` | IDRiD grading training set preserved intact: exactly 413 images. |
| **IDRiD Test Set Integrity** | IDRiD External Test Benchmark (N=103) | `PASSED ✅` | IDRiD external test set preserved intact: exactly 103 images. |
| **IDRiD Segmentation Set Integrity** | IDRiD Segmentation Dataset (N=81) | `PASSED ✅` | IDRiD segmentation set preserved intact: exactly 81 images. |

## 5. Processed Manifests

- `data/processed/aptos/train_split.csv` ($N=2,929$)
- `data/processed/aptos/val_split.csv` ($N=733$)
- `data/processed/idrid/external_test.csv` ($N=103$)
- `data/processed/idrid/external_train.csv` ($N=413$)
- `data/processed/idrid/segmentation_manifest.csv` ($N=81$)

---
*Report auto-generated by `scripts/preprocess_data.py` as part of Phase 2.*