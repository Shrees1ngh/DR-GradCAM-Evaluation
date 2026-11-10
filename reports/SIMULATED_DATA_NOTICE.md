# AUDIT NOTICE: Identification of Synthetic & Dry-Run Artifacts

> [!CAUTION]
> **SIMULATED DATA — NOT VALID FOR RESEARCH FINDINGS OR MANUSCRIPT RESULTS**
>
> The files listed below were generated during local software unit testing and pipeline verification in `--dry-run` mode. Because raw clinical images and trained model weights (*.pth) are git-ignored and hosted in the research GPU environment (Google Colab), the local execution engine used synthetic distributions to verify that tensor shapes, gradient hooks, metric functions, and report formatters execute without runtime exceptions.
>
> **THESE NUMBERS DO NOT REPRESENT REAL MODEL PERFORMANCE OR EMPIRICAL OBSERVATIONS.**
> Under no circumstances should these files be cited in thesis submissions, presentations, or research publications.

---

## 1. List of Preserved Simulated Artifacts (For Pipeline Audit Only)

| Preserved File | Origin | Status | Intended Purpose |
|---|---|:---:|---|
| [`reports/simulated_idrid_training_predictions.csv`](file:///e:/diabeticretinopathy/DR-GradCAM-Evaluation/reports/simulated_idrid_training_predictions.csv) | `--dry-run` software test | **SIMULATED ⚠️** | Unit test verification of CSV schema and QWK calculation functions |
| [`reports/simulated_idrid_training_domain_diagnostic_report.md`](file:///e:/diabeticretinopathy/DR-GradCAM-Evaluation/reports/simulated_idrid_training_domain_diagnostic_report.md) | `--dry-run` software test | **SIMULATED ⚠️** | Formatting and layout verification of diagnostic report |
| [`reports/simulated_idrid_training_domain_diagnostic_report.json`](file:///e:/diabeticretinopathy/DR-GradCAM-Evaluation/reports/simulated_idrid_training_domain_diagnostic_report.json) | `--dry-run` software test | **SIMULATED ⚠️** | Schema verification of serialized JSON diagnostic outputs |
| [`reports/simulated_phase_6_localization_report.md`](file:///e:/diabeticretinopathy/DR-GradCAM-Evaluation/reports/simulated_phase_6_localization_report.md) | `--dry-run` software test | **SIMULATED ⚠️** | Layout verification of Phase 6 localization report (78 primary, 3 supplementary) |
| [`reports/simulated_phase_6_localization_report.json`](file:///e:/diabeticretinopathy/DR-GradCAM-Evaluation/reports/simulated_phase_6_localization_report.json) | `--dry-run` software test | **SIMULATED ⚠️** | Schema verification of serialized JSON localization metrics |

---

## 2. Research Protocol & Scientific Isolation Guarantees

1. **Dataset Isolation & 3-Image Overlap Resolution:**
   * **100 of the 103** IDRiD disease grading test images are completely untouched across all project phases.
   * **3 images** (`IDRiD_478`, `IDRiD_470`, `IDRiD_471`) overlap with the 81-image segmentation dataset as `IDRiD_09`, `IDRiD_22`, and `IDRiD_31`.
   * **Primary Localization Cohort:** Defined strictly as the **78 non-overlapping images** (57 train-matched + 21 unique segmentation images).
   * **Supplementary Analysis:** The 3 test-overlapping images are reported separately as supplementary results. They are **never** used for model training, fine-tuning, or hyperparameter selection.
   * **Cohort Preservation:** The full 81-image segmentation cohort is preserved intact.

2. **Checkpoint Immutability & Model Weight Protection:**
   * **Dual Verification:** Checkpoint preservation is verified by both SHA-256 file hashing before/after execution and 100% bit-exact in-memory `state_dict` tensor comparison (`torch.equal`).
   * **Optimizer Inactive:** Model weights are frozen (`param.requires_grad = False`), and zero optimizer steps or loss functions are executed.
   * **Grad-CAM Gradients vs Training Updates:** Grad-CAM activation gradients ($\partial y^c / \partial A^k$) compute sensitivity with respect to feature maps, completely isolated from model parameter updates ($\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$).

3. **Preprocessing Geometry Audit:**
   * **Native Sensor Geometry:** $4288 \times 2848$ sensor resolution has a $3:2$ aspect ratio ($1.5056$), mathematically creating $33.59\%$ black margin if directly isotropically padded into $512 \times 512$ without cropping.
   * **Pipeline Preprocessed Geometry:** The preprocessing pipeline applies `crop_to_retinal_fov`, excising the side-pillars and producing an approximately square cropped retina ($1.01:1$). The letterbox padding added by `resize_and_pad` is negligible ($\le 3\%$), with active retinal tissue covering $\sim 75\%$ of the $512 \times 512$ tensor.

4. **Genuine Output Filenames:**
   * Genuine diagnostic outputs must strictly be written to:
     - `reports/idrid_training_predictions.csv`
     - `reports/idrid_training_domain_diagnostic_report.md`
     - `reports/idrid_training_domain_diagnostic_report.json`
     - `reports/phase_6_localization_report.md`
     - `reports/phase_6_localization_report.json`

5. **Missing Input Policy:**
   * If any required physical image file or checkpoint is missing on disk, scripts must **exit immediately with returncode 1**.
   * Simulated data generation in normal runtime is strictly forbidden.

6. **Execution Instructions:**
   * See [`docs/GENUINE_INFERENCE_GUIDE.md`](file:///e:/diabeticretinopathy/DR-GradCAM-Evaluation/docs/GENUINE_INFERENCE_GUIDE.md) for full execution commands in GPU environments.
