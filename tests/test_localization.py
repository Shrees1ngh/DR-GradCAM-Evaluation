"""Unit tests for Phase 6 Localization evaluation metrics and aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from src.localization import (
    aggregate_localization_scores,
    compute_dice,
    compute_iou,
    evaluate_single_image_localization,
)


def test_compute_iou_and_dice_perfect_overlap():
    """Identical non-empty masks must yield IoU=1.0 and Dice=1.0."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    iou = compute_iou(mask, mask)
    dice = compute_dice(mask, mask)

    assert iou == 1.0
    assert dice == 1.0


def test_compute_iou_and_dice_disjoint():
    """Disjoint masks must yield IoU=0.0 and Dice=0.0."""
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[0:2, 0:2] = 1

    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[5:8, 5:8] = 1

    assert compute_iou(pred, gt) == 0.0
    assert compute_dice(pred, gt) == 0.0


def test_compute_iou_and_dice_partial_overlap():
    """Verify exact formula on known partial overlap."""
    # Pred: 4 pixels. GT: 4 pixels. Intersection: 2 pixels. Union: 6 pixels.
    pred = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ], dtype=np.uint8)

    gt = np.array([
        [0, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
    ], dtype=np.uint8)

    # Intersection = 2 (at (0, 1) and (1, 1))
    # Union = 6
    # IoU = 2 / 6 = 1/3
    # Dice = (2 * 2) / (4 + 4) = 4 / 8 = 0.5
    iou = compute_iou(pred, gt)
    dice = compute_dice(pred, gt)

    assert pytest.approx(iou) == 1.0 / 3.0
    assert pytest.approx(dice) == 0.5


def test_empty_ground_truth_returns_none_or_na():
    """Ground truth with 0 lesions (e.g. Grade 0) must return None/N/A."""
    pred = np.ones((10, 10), dtype=np.uint8)
    empty_gt = np.zeros((10, 10), dtype=np.uint8)

    assert compute_iou(pred, empty_gt) is None
    assert compute_dice(pred, empty_gt) is None

    # Custom empty_value override
    assert compute_iou(pred, empty_gt, empty_value=-1.0) == -1.0


def test_shape_mismatch_raises_error():
    """Dimension mismatch must raise ValueError."""
    m1 = np.zeros((10, 10), dtype=np.uint8)
    m2 = np.zeros((12, 12), dtype=np.uint8)

    with pytest.raises(ValueError, match="mismatch"):
        compute_iou(m1, m2)

    with pytest.raises(ValueError, match="mismatch"):
        compute_dice(m1, m2)


def test_evaluate_single_image_localization_grade_0():
    """Grade 0 images must return None for lesion IoU and Dice (designated N/A)."""
    heatmap = np.random.rand(50, 50).astype(np.float32)
    masks = {
        "MA": np.zeros((50, 50), dtype=np.uint8),
        "composite": np.zeros((50, 50), dtype=np.uint8),
        "OD": np.ones((50, 50), dtype=np.uint8),
    }

    res = evaluate_single_image_localization(
        heatmap=heatmap,
        ground_truth_masks=masks,
        image_id="img_grade_0",
        grade=0,
        thresholds=[0.10],
    )

    assert res["is_grade_0"] is True
    t_metrics = res["metrics_by_threshold"]["top_10_pct"]
    assert t_metrics["lesions"]["MA"]["iou"] is None
    assert t_metrics["lesions"]["composite"]["iou"] is None
    # Optic disc control is still evaluated
    assert t_metrics["control"]["OD"]["iou"] is not None


def test_evaluate_single_image_localization_diseased():
    """Diseased image (Grade 2) must calculate valid IoU/Dice across thresholds."""
    # Synthetic heatmap with high activation in upper-left
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[0:20, 0:20] = 1.0

    ma_mask = np.zeros((100, 100), dtype=np.uint8)
    ma_mask[0:10, 0:10] = 1

    masks = {
        "MA": ma_mask,
        "HE": np.zeros((100, 100), dtype=np.uint8),
        "EX": np.zeros((100, 100), dtype=np.uint8),
        "SE": np.zeros((100, 100), dtype=np.uint8),
        "composite": ma_mask,
        "OD": np.zeros((100, 100), dtype=np.uint8),
    }

    res = evaluate_single_image_localization(
        heatmap=heatmap,
        ground_truth_masks=masks,
        image_id="img_grade_2",
        grade=2,
        thresholds=[0.05, 0.10, 0.20],
        annotation_status={"SE": False},  # SE unannotated
    )

    assert res["is_grade_0"] is False
    assert len(res["metrics_by_threshold"]) == 3

    top10 = res["metrics_by_threshold"]["top_10_pct"]
    assert top10["lesions"]["MA"]["iou"] > 0.0
    assert top10["lesions"]["composite"]["iou"] > 0.0
    # Unannotated SE must be None
    assert top10["lesions"]["SE"]["iou"] is None


def test_aggregate_localization_scores():
    """Verify aggregation excludes Grade 0 N/A from averages and computes statistics."""
    results = []

    # Image 1: Grade 2, IoU = 0.20, Dice = 0.30
    results.append({
        "image_id": "img1",
        "grade": 2,
        "metrics_by_threshold": {
            "top_10_pct": {
                "lesions": {
                    "composite": {"iou": 0.20, "dice": 0.30},
                    "MA": {"iou": 0.10, "dice": 0.15},
                },
                "control": {"OD": {"iou": 0.05, "dice": 0.08}},
            }
        }
    })

    # Image 2: Grade 2, IoU = 0.40, Dice = 0.50
    results.append({
        "image_id": "img2",
        "grade": 2,
        "metrics_by_threshold": {
            "top_10_pct": {
                "lesions": {
                    "composite": {"iou": 0.40, "dice": 0.50},
                    "MA": {"iou": 0.30, "dice": 0.35},
                },
                "control": {"OD": {"iou": 0.01, "dice": 0.02}},
            }
        }
    })

    # Image 3: Grade 0 (N/A)
    results.append({
        "image_id": "img3",
        "grade": 0,
        "metrics_by_threshold": {
            "top_10_pct": {
                "lesions": {
                    "composite": {"iou": None, "dice": None},
                    "MA": {"iou": None, "dice": None},
                },
                "control": {"OD": {"iou": 0.00, "dice": 0.00}},
            }
        }
    })

    summary = aggregate_localization_scores(results, baseline_threshold_key="top_10_pct")

    # Total 3 images, but only 2 diseased images evaluated for composite
    assert summary["total_images"] == 3
    assert summary["overall_composite"]["iou"]["count"] == 2
    # Mean of 0.20 and 0.40 = 0.30
    assert pytest.approx(summary["overall_composite"]["iou"]["mean"]) == 0.30
    # Mean of 0.30 and 0.50 = 0.40
    assert pytest.approx(summary["overall_composite"]["dice"]["mean"]) == 0.40

    # Grade 2 stats
    assert pytest.approx(summary["grade_wise_composite"]["grade_2"]["iou"]["mean"]) == 0.30


def test_aggregate_cohort_origin_stratification():
    """Verify that cohort origin (train-matched, test-matched, unique) is stratified."""
    results = [
        {
            "image_id": "seg_01",
            "grade": 2,
            "cohort_origin": "idrid_grading_train_cohort",
            "metrics_by_threshold": {
                "top_10_pct": {
                    "lesions": {"composite": {"iou": 0.10, "dice": 0.18}},
                    "control": {"OD": {"iou": 0.01, "dice": 0.02}},
                }
            },
        },
        {
            "image_id": "seg_09",
            "grade": 2,
            "cohort_origin": "idrid_grading_test_cohort",  # Overlap with test set
            "metrics_by_threshold": {
                "top_10_pct": {
                    "lesions": {"composite": {"iou": 0.05, "dice": 0.09}},
                    "control": {"OD": {"iou": 0.00, "dice": 0.00}},
                }
            },
        },
        {
            "image_id": "seg_04",
            "grade": 3,
            "cohort_origin": "idrid_seg_unique",
            "metrics_by_threshold": {
                "top_10_pct": {
                    "lesions": {"composite": {"iou": 0.15, "dice": 0.25}},
                    "control": {"OD": {"iou": 0.02, "dice": 0.03}},
                }
            },
        },
    ]

    summary = aggregate_localization_scores(results, baseline_threshold_key="top_10_pct")
    cohort_stats = summary["cohort_origin_composite"]

    assert "idrid_grading_train_cohort" in cohort_stats
    assert "idrid_grading_test_cohort" in cohort_stats
    assert "idrid_seg_unique" in cohort_stats

    assert cohort_stats["idrid_grading_train_cohort"]["iou"]["count"] == 1
    assert pytest.approx(cohort_stats["idrid_grading_train_cohort"]["iou"]["mean"]) == 0.10

    assert cohort_stats["idrid_grading_test_cohort"]["iou"]["count"] == 1
    assert pytest.approx(cohort_stats["idrid_grading_test_cohort"]["iou"]["mean"]) == 0.05

    assert cohort_stats["idrid_seg_unique"]["iou"]["count"] == 1
    assert pytest.approx(cohort_stats["idrid_seg_unique"]["iou"]["mean"]) == 0.15


def test_primary_and_supplementary_cohort_separation():
    """Verify that aggregate_localization_scores strictly isolates the 3 test-matched images into supplementary results."""
    # Create 4 synthetic results: 3 primary (2 train-matched, 1 unique), 1 supplementary test-matched
    results = [
        {
            "image_id": "IDRiD_01",
            "grade": 2,
            "cohort_origin": "idrid_grading_train_cohort",
            "metrics_by_threshold": {
                "top_10_pct": {
                    "lesions": {"composite": {"iou": 0.20, "dice": 0.30}, "MA": {"iou": 0.15, "dice": 0.20}},
                    "control": {"OD": {"iou": 0.01, "dice": 0.02}},
                }
            },
        },
        {
            "image_id": "IDRiD_02",
            "grade": 3,
            "cohort_origin": "idrid_grading_train_cohort",
            "metrics_by_threshold": {
                "top_10_pct": {
                    "lesions": {"composite": {"iou": 0.30, "dice": 0.40}, "MA": {"iou": 0.25, "dice": 0.30}},
                    "control": {"OD": {"iou": 0.02, "dice": 0.03}},
                }
            },
        },
        {
            "image_id": "IDRiD_04",
            "grade": 2,
            "cohort_origin": "idrid_seg_unique",
            "metrics_by_threshold": {
                "top_10_pct": {
                    "lesions": {"composite": {"iou": 0.40, "dice": 0.50}, "MA": {"iou": 0.35, "dice": 0.40}},
                    "control": {"OD": {"iou": 0.03, "dice": 0.04}},
                }
            },
        },
        {
            "image_id": "IDRiD_09",  # Overlap with test set
            "grade": 2,
            "cohort_origin": "idrid_grading_test_cohort",
            "metrics_by_threshold": {
                "top_10_pct": {
                    "lesions": {"composite": {"iou": 0.05, "dice": 0.09}, "MA": {"iou": 0.03, "dice": 0.05}},
                    "control": {"OD": {"iou": 0.00, "dice": 0.00}},
                }
            },
        },
    ]

    summary = aggregate_localization_scores(results, baseline_threshold_key="top_10_pct")

    # Primary cohort must contain exactly the 3 non-overlapping images
    primary = summary["primary_cohort"]
    assert primary["composite_iou"]["count"] == 3
    # Mean of (0.20, 0.30, 0.40) = 0.30
    assert pytest.approx(primary["composite_iou"]["mean"]) == 0.30
    # Mean of (0.30, 0.40, 0.50) = 0.40
    assert pytest.approx(primary["composite_dice"]["mean"]) == 0.40

    # Supplementary test cohort must contain exactly the 1 test-matched image
    supp = summary["supplementary_test_cohort"]
    assert supp["composite_iou"]["count"] == 1
    assert pytest.approx(supp["composite_iou"]["mean"]) == 0.05
    assert len(supp["images"]) == 1
    assert supp["images"][0]["image_id"] == "IDRiD_09"

    # Overall full dataset combines all 4
    assert summary["overall_composite"]["iou"]["count"] == 4


def test_verify_state_dict_unchanged():
    """Verify that in-memory model weights immutability checker detects unchanged and mutated states."""
    import torch
    import torch.nn as nn
    from scripts.evaluate_localization import verify_state_dict_unchanged

    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )

    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Check 1: unchanged model must return True
    assert verify_state_dict_unchanged(initial_state, model) is True

    # Check 2: mutate a single weight parameter
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.01)
            break

    assert verify_state_dict_unchanged(initial_state, model) is False


def test_gradcam_with_frozen_model_parameters():
    """Verify Grad-CAM operates normally when all model parameters are frozen (requires_grad=False)."""
    import torch
    import torch.nn as nn
    from src.explainability import GradCAM

    class ToyConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(8, 5)

        def forward(self, x):
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    model = ToyConvNet()
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    with GradCAM(model, target_layer=model.layer4) as cam:
        x = torch.randn(1, 3, 32, 32)
        heatmap, pred_class = cam.generate_cam(x)

        assert heatmap.shape == (32, 32)
        assert np.min(heatmap) >= 0.0
        assert np.max(heatmap) <= 1.0
        assert 0 <= pred_class < 5

    # Verify model weights remained untouched
    for param in model.parameters():
        assert param.grad is None or torch.all(param.grad == 0)


def test_grade_unavailable_single_image_localization():
    """Verify evaluate_single_image_localization gracefully handles grade=None (unavailable label)."""
    h = np.zeros((50, 50), dtype=np.float32)
    h[10:20, 10:20] = 1.0
    m = np.zeros((50, 50), dtype=np.uint8)
    m[10:20, 10:20] = 1
    masks = {"MA": m, "HE": m, "EX": m, "SE": m, "composite": m, "OD": m}

    res = evaluate_single_image_localization(
        heatmap=h,
        ground_truth_masks=masks,
        image_id="unique_seg_01",
        grade=None,  # Grade unavailable
        cohort_origin="idrid_seg_unique",
    )

    assert res["image_id"] == "unique_seg_01"
    assert res["grade"] is None
    assert res["is_grade_0"] is False
    assert res["cohort_origin"] == "idrid_seg_unique"

    # Metrics should still be computed for valid masks
    t10 = res["metrics_by_threshold"]["top_10_pct"]
    assert t10["lesions"]["composite"]["iou"] is not None


def test_evaluate_single_image_localization_non_informative_heatmap():
    """Verify non-informative (flat) heatmap sets status and None metrics."""
    h = np.ones((50, 50), dtype=np.float32) * 0.5
    m = np.zeros((50, 50), dtype=np.uint8)
    m[10:20, 10:20] = 1
    masks = {"MA": m, "HE": m, "EX": m, "SE": m, "composite": m, "OD": m}

    res = evaluate_single_image_localization(
        heatmap=h,
        ground_truth_masks=masks,
        image_id="flat_img_01",
        grade=2,
        cohort_origin="idrid_grading_train_cohort",
    )

    assert res["heatmap_status"] == "non_informative_heatmap"
    assert res["is_informative"] is False
    t10 = res["metrics_by_threshold"]["top_10_pct"]
    assert t10["lesions"]["composite"]["iou"] is None
    assert t10["lesions"]["composite"]["dice"] is None


def test_cohort_and_grade_stratification_exact_counts():
    """Verify exact cohort and grade stratification across all 81 segmentation images:
    - 57 training-matched images with verified grades.
    - 21 unique images with grade=None (grade unavailable).
    - 3 supplementary test-matched images strictly separate.
    - Primary cohort total = 78 (57 + 21).
    - Grade-wise analysis sums strictly to 57 images.
    """
    results = []
    rng = np.random.default_rng(42)

    for i in range(81):
        h = rng.random((64, 64), dtype=np.float32)
        m = np.zeros((64, 64), dtype=np.uint8)
        m[15:35, 15:35] = 1
        masks = {"MA": m, "HE": m, "EX": m, "SE": m, "composite": m, "OD": m}

        if i < 57:
            co = "idrid_grading_train_cohort"
            g = (i % 4) + 1  # verified grade 1 to 4
        elif i < 60:
            co = "idrid_grading_test_cohort"
            g = (i % 4) + 1  # supplementary test grade
        else:
            co = "idrid_seg_unique"
            g = None  # grade unavailable

        res = evaluate_single_image_localization(
            heatmap=h,
            ground_truth_masks=masks,
            image_id=f"seg_{i+1:02d}",
            grade=g,
            cohort_origin=co,
        )
        results.append(res)

    summary = aggregate_localization_scores(results, baseline_threshold_key="top_10_pct")

    # 1. Total dataset count
    assert summary["total_images"] == 81
    assert summary["overall_composite"]["total_images"] == 81

    # 2. Primary cohort count (57 matched + 21 unique = 78)
    assert summary["primary_cohort"]["total_images"] == 78
    assert summary["primary_cohort"]["composite_iou"]["count"] == 78
    assert summary["primary_cohort"]["composite_dice"]["count"] == 78

    # 3. Supplementary test cohort count (exactly 3)
    assert summary["supplementary_test_cohort"]["total_images"] == 3
    assert summary["supplementary_test_cohort"]["composite_iou"]["count"] == 3
    assert len(summary["supplementary_test_cohort"]["images"]) == 3

    # 4. Grade-wise stratification in primary cohort (sums strictly to 57 verified images)
    grade_wise_sum = sum(
        summary["primary_cohort"]["grade_wise"][f"grade_{g}"]["iou"]["count"]
        for g in range(1, 5)
    )
    assert grade_wise_sum == 57

    # 5. Grade unavailable count in distribution is exactly 21
    assert summary["grade_distribution"]["unavailable"] == 21


def test_localization_load_cohort_and_grade_mapping_strict(tmp_path: pytest.TempPathFactory):
    """Verify scripts.evaluate_localization.load_cohort_and_grade_mapping strictly validates inputs."""
    import json
    from pathlib import Path
    import pandas as pd
    from scripts.evaluate_localization import load_cohort_and_grade_mapping

    matches_file = Path("data/idrid_seg_grading_matches.json")
    if not matches_file.exists():
        matches_file = Path("reports/idrid_seg_grading_matches.json")
    if not matches_file.exists():
        pytest.skip("idrid_seg_grading_matches.json not found")

    with open(matches_file, "r", encoding="utf-8") as f:
        matches_data = json.load(f)

    # 1. Missing labels raises ValueError
    nonexistent = tmp_path / "nonexistent.csv"
    with pytest.raises(ValueError, match="Official IDRiD training grading labels are missing or empty"):
        load_cohort_and_grade_mapping(
            matches_file,
            train_manifest_path=nonexistent,
            test_manifest_path=nonexistent,
        )

    # 2. Valid mock labels load all 81 images correctly
    train_records = [
        {"image_id": Path(matches_data["all_details"][fn]["target_image"]).stem, "grade": 1}
        for fn in matches_data.get("matched_train_ids", [])
    ]
    test_records = [
        {"image_id": Path(matches_data["all_details"][fn]["target_image"]).stem, "grade": 2}
        for fn in matches_data.get("matched_test_ids", [])
    ]

    mock_tr = tmp_path / "tr_loc.csv"
    mock_te = tmp_path / "te_loc.csv"
    pd.DataFrame(train_records).to_csv(mock_tr, index=False)
    pd.DataFrame(test_records).to_csv(mock_te, index=False)

    cohort_map, grade_map = load_cohort_and_grade_mapping(
        matches_file,
        train_manifest_path=mock_tr,
        test_manifest_path=mock_te,
    )
    assert len(cohort_map) == 81
    assert len(grade_map) == 81

    # 57 train matched have grade 1
    for fn in matches_data["matched_train_ids"]:
        assert grade_map[Path(fn).stem] == 1

    # 3 test matched have grade 2
    for fn in matches_data["matched_test_ids"]:
        assert grade_map[Path(fn).stem] == 2

    # 21 unique have None
    for fn in matches_data["unmatched_ids"]:
        assert grade_map[Path(fn).stem] is None


def test_export_localization_predictions_csv(tmp_path: Path):
    """Verify export_localization_predictions_csv writes structured per-image records."""
    import pandas as pd
    from scripts.evaluate_localization import export_localization_predictions_csv

    mock_results = [
        {
            "image_id": "seg_01",
            "grade": 2,
            "predicted_class": 2,
            "cohort_origin": "idrid_grading_train_cohort",
            "metrics_by_threshold": {
                "top_10_pct": {
                    "lesions": {
                        "composite": {"iou": 0.45, "dice": 0.62},
                        "MA": {"iou": 0.30, "dice": 0.46},
                        "HE": {"iou": 0.25, "dice": 0.40},
                        "EX": {"iou": None, "dice": None},
                        "SE": {"iou": 0.15, "dice": 0.26},
                    },
                    "control": {
                        "OD": {"iou": 0.05, "dice": 0.10},
                    },
                }
            },
        },
        {
            "image_id": "seg_02",
            "grade": None,
            "predicted_class": 1,
            "cohort_origin": "idrid_seg_unique",
            "metrics_by_threshold": {
                "top_10_pct": {
                    "lesions": {
                        "composite": {"iou": 0.10, "dice": 0.18},
                        "MA": {"iou": None, "dice": None},
                        "HE": {"iou": None, "dice": None},
                        "EX": {"iou": None, "dice": None},
                        "SE": {"iou": None, "dice": None},
                    },
                    "control": {
                        "OD": {"iou": 0.0, "dice": 0.0},
                    },
                }
            },
        },
    ]

    out_csv = tmp_path / "test_localization_predictions.csv"
    df = export_localization_predictions_csv(mock_results, out_csv)

    assert out_csv.exists()
    assert len(df) == 2
    assert list(df["image_id"]) == ["seg_01", "seg_02"]
    assert df.loc[0, "true_grade"] == 2
    assert pd.isna(df.loc[1, "true_grade"])
    assert df.loc[0, "composite_iou_top_10"] == 0.45
    assert df.loc[0, "composite_dice_top_10"] == 0.62
    assert df.loc[0, "ma_iou_top_10"] == 0.30
    assert df.loc[0, "ma_dice_top_10"] == 0.46
    assert df.loc[0, "he_iou_top_10"] == 0.25
    assert df.loc[0, "he_dice_top_10"] == 0.40
    assert pd.isna(df.loc[0, "ex_iou_top_10"])
    assert pd.isna(df.loc[0, "ex_dice_top_10"])
    assert df.loc[0, "se_iou_top_10"] == 0.15
    assert df.loc[0, "se_dice_top_10"] == 0.26
    assert df.loc[0, "od_iou_top_10"] == 0.05
    assert df.loc[0, "od_dice_top_10"] == 0.10

    # Ensure unannotated lesions in seg_02 preserve None/NaN, not converted to 0.0
    assert pd.isna(df.loc[1, "ma_iou_top_10"])
    assert pd.isna(df.loc[1, "ma_dice_top_10"])
    assert pd.isna(df.loc[1, "he_iou_top_10"])
    assert pd.isna(df.loc[1, "he_dice_top_10"])
    assert df.loc[1, "cohort_origin"] == "idrid_seg_unique"


def test_fov_aware_localization_rejects_invalid_masks():
    """Verify evaluate_single_image_localization validates FOV mask integrity."""
    from src.localization import evaluate_single_image_localization

    h = np.zeros((100, 100), dtype=np.float32)
    masks = {"composite": np.zeros((100, 100), dtype=np.uint8)}

    # Shape mismatch
    with pytest.raises(ValueError, match="does not match heatmap shape"):
        evaluate_single_image_localization(
            heatmap=h,
            ground_truth_masks=masks,
            image_id="img_err1",
            fov_mask=np.zeros((50, 50), dtype=np.uint8),
        )

    # Non-binary mask
    with pytest.raises(ValueError, match="strictly binary"):
        evaluate_single_image_localization(
            heatmap=h,
            ground_truth_masks=masks,
            image_id="img_err2",
            fov_mask=np.ones((100, 100), dtype=np.uint8) * 128,
        )

    # Empty FOV mask
    with pytest.raises(ValueError, match="FOV mask is empty"):
        evaluate_single_image_localization(
            heatmap=h,
            ground_truth_masks=masks,
            image_id="img_err3",
            fov_mask=np.zeros((100, 100), dtype=np.uint8),
        )


def test_fov_aware_localization_masks_out_external_activations():
    """Verify strong artificial heatmap activations outside retinal FOV cannot influence localization."""
    from src.localization import evaluate_single_image_localization

    # Canvas 100x100. Retinal circular FOV centered with radius 35 (covers ~3849 pixels)
    y, x = np.ogrid[:100, :100]
    fov = ((y - 50)**2 + (x - 50)**2 <= 35**2).astype(np.uint8)
    n_fov = int(np.sum(fov))

    # Construct lesion ground truth inside FOV: 50 pixels near center
    lesion_mask = np.zeros((100, 100), dtype=np.uint8)
    lesion_mask[48:53, 45:55] = 1
    masks = {"composite": lesion_mask, "MA": lesion_mask}

    # Construct heatmap with massive false activation OUTSIDE FOV (in black border padding)
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[0:15, 0:15] = 1.0  # 225 pixels with peak 1.0 outside FOV!

    # Inside FOV, modest activation (0.8) exactly on the lesion
    heatmap[48:53, 45:55] = 0.8

    # 1. Without FOV awareness: top 2% over entire canvas (200 pixels) selects only the 225 border pixels
    res_no_fov = evaluate_single_image_localization(
        heatmap=heatmap,
        ground_truth_masks=masks,
        image_id="no_fov",
        grade=2,
        thresholds=[0.02],
        fov_mask=None,
    )
    # The lesion has 0 overlap because outside peak dominates
    assert res_no_fov["metrics_by_threshold"]["top_2_pct"]["lesions"]["composite"]["iou"] == 0.0

    # 2. With FOV awareness: border pixels are completely excluded
    res_fov = evaluate_single_image_localization(
        heatmap=heatmap,
        ground_truth_masks=masks,
        image_id="with_fov",
        grade=2,
        thresholds=[0.02],
        fov_mask=fov,
    )
    m_fov = res_fov["metrics_by_threshold"]["top_2_pct"]
    # Exactly round(0.02 * n_fov) pixels selected inside FOV
    exp_k = int(round(0.02 * n_fov))
    assert m_fov["salient_pixel_count"] == exp_k
    # Saliency successfully selected the lesion inside FOV
    assert m_fov["lesions"]["composite"]["iou"] > 0.0
    assert m_fov["lesions"]["composite"]["dice"] > 0.0
    assert res_fov["fov_aware"] is True
    assert res_fov["fov_pixel_count"] == n_fov





