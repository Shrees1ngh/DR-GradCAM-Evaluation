# Model B Provenance & Evaluation Protocol Comparison

## 1. Model B Training Source

- **Checkpoint:** `experiments/checkpoints/model_b_best.pth` (`epoch: 10`, `val_qwk: 0.8915`)
- **Training Source:** Trained in Google Colab (`DR_GradCAM_Training.ipynb`), not via a standalone fine-tuning script in `scripts/`.
- **Initialization & Split:** Initialized from APTOS-trained Model A (`experiments/checkpoints/best_model_qwk.pth`) and fine-tuned on IDRiD `external_train.csv` after strictly excluding all 57 images that overlap with IDRiD Segmentation (`pool = 356`: `284` train / `72` val).
- **Unseen Evaluation Sets:** All 103 images in IDRiD `external_test.csv` (`acc = 0.612`, `QWK = 0.746`) and all 81 IDRiD segmentation images remain strictly unseen during Model B training.

## 2. Preprocessing & Metric Naming Clarifications

- **`circular_fov_mask` in `configs/config.yaml`:** Set to `false`. Both `best_model_qwk.pth` and `model_b_best.pth` were trained using `crop_to_retinal_fov` + `resize_and_pad` without circular corner zeroing. Retinal FOV boundary masking is applied strictly at evaluation time in Phases 6 and 7 via `mask_fov`.
- **AUDC vs. Normalized Partial-AUC (`normalized_pauc_*`):** In `src/faithfulness.py` and `scripts/evaluate_faithfulness.py`, `audc_salient`, `audc_random`, and `audc_margin` are backward-compatible aliases for **discrete mean deletion drops** across steps, not trapezoidal area under the curve. True trapezoidal area under the deletion curve normalized to $[0, x_{\max}]$ is computed separately as `normalized_pauc_salient`, `normalized_pauc_random`, and `normalized_pauc_margin`, and `normalized_pauc_margin` is the metric used in Phase 8 statistical testing.

## 3. Colab vs. Repository Protocol Differences

Colab notebook experiments and the repository Phase 6–8 pipeline use **different evaluation protocols** and must be reported separately without mixing numbers in the same comparison table:

| Setting | Colab (`DR_GradCAM_Training.ipynb`) | Repository Pipeline (`scripts/evaluate_*.py`, `analyze_results.py`) |
| :--- | :--- | :--- |
| **Evaluation Cohort** | 81 images combined | 78 primary (non-test-overlap) + 3 supplementary (`IDRiD_09`, `IDRiD_22`, `IDRiD_31`) |
| **Localization Binarization** | Fixed intensity thresholds | Top-percentile cutoffs (`top_5_pct`, `top_10_pct`, `top_20_pct`) |
| **FOV Restriction in Localization** | Not applied | Strictly restricted to retinal FOV (`mask_fov`) |
| **Deletion Fill Strategy** | ImageNet mean color | RGB black `(0, 0, 0)` primary (`gaussian_blur` secondary control) |
| **Random Control** | Rotated / translated heatmap | Area-matched uniform random pixels inside retinal FOV |
| **Deletion Levels** | 5%, 10%, 20%, 30%, 50% | 10%, 20%, 30% |
| **Primary Faithfulness Metric** | Remaining-confidence AUC | Normalized partial-AUC (`normalized_pauc_margin`) |

### Genuine Protocol-Dependent Direction Difference

- Under the **Colab ImageNet mean-color deletion protocol**, Grad-CAM remaining-confidence AUC was lower than the rotated/translated control, suggesting positive faithfulness.
- Under the **Repository RGB black-fill deletion protocol**, salient deletion drop (`0.4859`) was lower than area-matched random FOV deletion (`0.5850`; normalized pAUC margin `-0.0972`), and Phase 8 found a **statistically significant negative correlation** between localization overlap (`composite_dice_top_10`) and faithfulness margin ($\rho = -0.4481$, $p = 3.8849 \times 10^{-5}$, 95% CI $[-0.6184, -0.2470]$).
- Both results are genuine artifacts of their respective perturbation and control definitions and should be reported separately unless evaluated under a single unified protocol.

