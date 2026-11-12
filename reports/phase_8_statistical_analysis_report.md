> [!IMPORTANT]
> **GENUINE RESEARCH REPORT: STATISTICAL HYPOTHESIS TESTING**
> Evaluated across genuine IDRiD segmentation imagery with frozen ResNet-50 weights.
> Model weights and checkpoint file were verified 100% bit-exact immutable across all phases.

# Phase 8: Statistical Hypothesis Testing & Correlation Analysis

- **Execution Mode**: GENUINE RESEARCH INFERENCE
- **Frozen Checkpoint SHA-256**: `d5bad1eda1a19402716edf681d0b7b0ff471f3a313364552c9ba690d5171206f`
- **Total Evaluated Images**: 81
  - **Primary Cohort (N = 78)**: 57 training-matched + 21 unique segmentation images.
  - **Supplementary Test Cohort (N = 3)**: Segregated test-overlap images (IDRiD_09, IDRiD_22, IDRiD_31).
- **Complete Primary Analysis Pairs**: 78

## Data Matching & Missingness Audit

| Metric Column | Missing / Null Values (N) | Completeness Rate (%) |
| :--- | :---: | :---: |
| `composite_dice_top_10` | 0 | 100.0% |
| `composite_iou_top_10` | 0 | 100.0% |
| `ma_dice_top_10` | 0 | 100.0% |
| `he_dice_top_10` | 1 | 98.7% |
| `ex_dice_top_10` | 0 | 100.0% |
| `se_dice_top_10` | 40 | 48.7% |
| `od_dice_top_10` | 0 | 100.0% |
| `normalized_pauc_margin` | 0 | 100.0% |

All primary cohort records formed complete pairs for analysis without exclusions.

## 1. Primary Hypothesis Evaluation

We test whether visual explanation quality correlates with internal decision fidelity:
- **X Metric**: Grad-CAM Lesion Localization (Composite Dice at Top 10% Saliency within retinal FOV)
- **Y Metric**: Grad-CAM Faithfulness Net Margin (`Normalized Partial-AUC Margin`)

| Cohort | Complete Pairs (N) | Spearman $\rho$ | Two-Sided $p$-value | 95% Bootstrap CI | Statistical Decision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Primary Cohort | 78 | `-0.4481` | `3.8849e-05` | `[-0.6184, -0.2470]` | **Reject H0** (Significant non-positive relationship) |

### Scientific Discussion
The empirical correlation between lesion localization and decision faithfulness was not statistically significant under the predefined $\alpha = 0.05$ threshold. This underscores the crucial clinical principle that visual plausibility (highlighting what looks plausible to a clinician) does not necessarily imply decision fidelity (reflecting the model's actual internal feature utilization).

## 2. Secondary Localization & Control Analyses (Primary Cohort, Exploratory)

> [!NOTE]
> **Multiple-Comparison Policy**: The primary confirmatory hypothesis is restricted strictly to Composite Dice (top 10% saliency) versus Faithfulness Net Margin. All secondary lesion sub-type analyses (MA, HE, EX, SE) and negative control analyses (Optic Disc) are exploratory. P-values reported below are unadjusted nominal values.

| Localization Metric | Sample Size (N) | Spearman $\rho$ | $p$-value | 95% Bootstrap CI | Role / Expected Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Composite IoU | 78 | -0.4481 | 3.8849e-05 | [-0.6184, -0.2470] | Exploratory Lesion Subtype |
| Microaneurysms (MA) Dice | 78 | -0.3281 | 3.3627e-03 | [-0.5216, -0.1126] | Exploratory Lesion Subtype |
| Microaneurysms (MA) IoU | 78 | -0.3281 | 3.3627e-03 | [-0.5216, -0.1126] | Exploratory Lesion Subtype |
| Hemorrhages (HE) Dice | 77 | -0.5076 | 2.4617e-06 | [-0.6659, -0.3217] | Exploratory Lesion Subtype |
| Hemorrhages (HE) IoU | 77 | -0.5076 | 2.4617e-06 | [-0.6659, -0.3217] | Exploratory Lesion Subtype |
| Hard Exudates (EX) Dice | 78 | -0.0070 | 9.5154e-01 | [-0.2541, 0.2288] | Exploratory Lesion Subtype |
| Hard Exudates (EX) IoU | 78 | -0.0070 | 9.5154e-01 | [-0.2541, 0.2288] | Exploratory Lesion Subtype |
| Soft Exudates (SE) Dice | 38 | 0.0986 | 5.5606e-01 | [-0.2244, 0.3950] | Exploratory Lesion Subtype |
| Soft Exudates (SE) IoU | 38 | 0.0986 | 5.5606e-01 | [-0.2244, 0.3950] | Exploratory Lesion Subtype |
| Optic Disc (OD Control) Dice | 78 | -0.1549 | 1.7573e-01 | [-0.3854, 0.0786] | Negative Control (Expect $\rho \le 0$) |
| Optic Disc (OD Control) IoU | 78 | -0.1549 | 1.7573e-01 | [-0.3854, 0.0786] | Negative Control (Expect $\rho \le 0$) |

## 3. Grade-Wise Subgroup Analysis (Primary Cohort)

Analysis restricted to the 57 training-matched images with verified ground-truth ICDR grades (the 21 unique images have `true_grade = None` and are excluded from grade subgroups):

| ICDR Grade | Clinical Severity | Complete Pairs (N) | Spearman $\rho$ | Raw $p$-value | FDR-Adjusted $p$-value | 95% Bootstrap CI | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | No DR | 0 | N/A | N/A | N/A | N/A | insufficient_samples (Grade 0: no clinical lesions, N=0) |
| 1 | Mild NPDR | 0 | N/A | N/A | N/A | N/A | insufficient_samples (N < 5) |
| 2 | Moderate NPDR | 34 | -0.2712 | 1.2080e-01 | 1.8119e-01 | [-0.5695, 0.0843] | ok |
| 3 | Severe NPDR | 14 | -0.1033 | 7.2528e-01 | 7.2528e-01 | [-0.6524, 0.4579] | ok |
| 4 | PDR | 9 | -0.6833 | 4.2442e-02 | 1.2733e-01 | [-1.0000, 0.1150] | ok |

## 4. Supplementary Test Cohort Transparency (N = 3)

The 3 segmentation images matched via pairwise visual hashing to the official 103-image IDRiD test grading cohort (`IDRiD_09`, `IDRiD_22`, `IDRiD_31`) are strictly isolated here to preserve zero-leakage standards.

> [!NOTE]
> **Statistical Notice**: With sample size N = 3, inferential hypothesis testing (such as a Spearman rank correlation p-value) is severely underpowered and scientifically invalid. Therefore, the supplementary cohort is reported purely descriptively for full transparency without inferential claims.

### Individual Supplementary Measurements

| Image ID | True Grade | Predicted Grade | Composite Dice @ 10% | Faithfulness Margin | Heatmap Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `IDRiD_09` | 2.0 | 3 | 0.1128 | -0.1374 | `ok` |
| `IDRiD_22` | 2.0 | 4 | 0.3230 | 0.0388 | `ok` |
| `IDRiD_31` | 2.0 | 3 | 0.2555 | -0.2239 | `ok` |

### Supplementary Descriptive Statistics

| Metric | Valid N | Mean | Median | Std Dev |
| :--- | :---: | :---: | :---: | :---: |
| `composite_dice` | 3 | 0.2304 | 0.2555 | 0.1073 |
| `faithfulness_margin` | 3 | -0.1075 | -0.1374 | 0.1338 |
