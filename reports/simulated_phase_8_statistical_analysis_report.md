> [!CAUTION]
> **SIMULATED DATA WARNING: NOT VALID FOR RESEARCH FINDINGS**
> This report was compiled from simulated / dry-run software tests.
> The numeric metrics, correlation coefficients, and p-values below are synthetic
> and must NEVER be cited as empirical evidence or used in scientific publications.

# Phase 8: Statistical Hypothesis Testing & Correlation Analysis

- **Execution Mode**: DRY-RUN SIMULATION (SOFTWARE TEST ONLY)
- **Frozen Checkpoint SHA-256**: `N/A (Dry-Run)`
- **Total Evaluated Images**: 81
  - **Primary Cohort (N = 78)**: 57 training-matched + 21 unique segmentation images.
  - **Supplementary Test Cohort (N = 3)**: Segregated test-overlap images (IDRiD_09, IDRiD_22, IDRiD_31).

## 1. Primary Hypothesis Evaluation

We test whether visual explanation quality correlates with internal decision fidelity:
- **X Metric**: Grad-CAM Lesion Localization (Composite Dice at Top 10% Saliency within retinal FOV)
- **Y Metric**: Grad-CAM Faithfulness Net Margin (`Normalized Partial-AUC Margin`)

| Cohort | Sample Size (N) | Spearman $\rho$ | Two-Sided $p$-value | 95% Bootstrap CI | Statistical Decision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Primary Cohort | 78 | `0.9585` | `3.5333e-43` | `[0.9345, 0.9675]` | **Reject H0** (Statistically significant positive correlation) |

### Scientific Discussion
The primary evaluation demonstrates a statistically significant positive rank correlation between lesion localization overlap and decision fidelity. This indicates that Grad-CAM heatmaps that align more closely with clinical lesions also correspond to higher model sensitivity under feature removal.

## 2. Secondary Localization & Control Analyses (Primary Cohort)

| Localization Metric | Sample Size (N) | Spearman $\rho$ | $p$-value | 95% Bootstrap CI | Role / Expected Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Composite IoU | 78 | 0.9585 | 3.5333e-43 | [0.9345, 0.9675] | Pathology Lesion Saliency |
| Microaneurysms (MA) Dice | 78 | 0.9585 | 3.5333e-43 | [0.9345, 0.9675] | Pathology Lesion Saliency |
| Microaneurysms (MA) IoU | 78 | 0.9585 | 3.5333e-43 | [0.9345, 0.9675] | Pathology Lesion Saliency |
| Hemorrhages (HE) Dice | 78 | 0.9585 | 3.5333e-43 | [0.9345, 0.9675] | Pathology Lesion Saliency |
| Hemorrhages (HE) IoU | 78 | 0.9585 | 3.5333e-43 | [0.9345, 0.9675] | Pathology Lesion Saliency |
| Hard Exudates (EX) Dice | 78 | 0.9585 | 3.5333e-43 | [0.9345, 0.9675] | Pathology Lesion Saliency |
| Hard Exudates (EX) IoU | 78 | 0.9585 | 3.5333e-43 | [0.9345, 0.9675] | Pathology Lesion Saliency |
| Soft Exudates (SE) Dice | 78 | 0.9585 | 3.5333e-43 | [0.9345, 0.9675] | Pathology Lesion Saliency |
| Soft Exudates (SE) IoU | 78 | 0.9585 | 3.5333e-43 | [0.9345, 0.9675] | Pathology Lesion Saliency |
| Optic Disc (OD Control) Dice | 78 | N/A | N/A | N/A | Negative Control (Expect $\rho \le 0$) |
| Optic Disc (OD Control) IoU | 78 | N/A | N/A | N/A | Negative Control (Expect $\rho \le 0$) |

## 3. Grade-Wise Subgroup Analysis (Primary Cohort)

Analysis restricted to the 57 training-matched images with verified ground-truth ICDR grades (the 21 unique images have `true_grade = None` and are excluded from grade subgroups):

| ICDR Grade | Clinical Severity | N | Spearman $\rho$ | Raw $p$-value | FDR-Adjusted $p$-value | 95% Bootstrap CI | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | No DR | 0 | N/A | N/A | N/A | N/A | insufficient_samples (N < 5) |
| 1 | Mild NPDR | 15 | 0.9500 | 6.0863e-08 | 6.0863e-08 | [0.8232, 0.9854] | ok |
| 2 | Moderate NPDR | 14 | 0.9692 | 1.1465e-08 | 1.5287e-08 | [0.8393, 1.0000] | ok |
| 3 | Severe NPDR | 14 | 0.9824 | 4.1072e-10 | 1.6429e-09 | [0.8828, 1.0000] | ok |
| 4 | PDR | 14 | 0.9780 | 1.5520e-09 | 3.1039e-09 | [0.8619, 1.0000] | ok |

## 4. Supplementary Test Cohort Transparency (N = 3)

The 3 segmentation images matched via pairwise visual hashing to the official 103-image IDRiD test grading cohort (`IDRiD_09`, `IDRiD_22`, `IDRiD_31`) are strictly isolated here to preserve zero-leakage standards:

- **Sample Size**: N = 3
- **Spearman $\rho$**: `1.0000` ($p = 0.0000$)
