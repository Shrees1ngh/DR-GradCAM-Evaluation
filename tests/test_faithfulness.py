"""Unit tests for Phase 7 Faithfulness evaluation (Progressive Deletion & Random Controls)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.faithfulness import (
    aggregate_faithfulness_scores,
    apply_ablation_mask,
    evaluate_progressive_deletion,
    generate_fov_restricted_salient_mask,
    generate_random_ablation_mask,
    run_single_deletion_step,
)


class DummyModel(nn.Module):
    """Simple deterministic mock model for testing ablation and faithfulness metrics."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

        # Initialize fixed weights
        nn.init.constant_(self.fc.weight, 0.1)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.relu(self.conv(x))
        pooled = self.pool(feat).flatten(1)
        return self.fc(pooled)


# ---------------------------------------------------------------------------
# 1. apply_ablation_mask Tests
# ---------------------------------------------------------------------------

def test_apply_ablation_mask_black():
    """Verify that 'black' strategy zeroes out masked pixels and preserves others."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 200
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    ablated = apply_ablation_mask(img, mask, fill_strategy="black")

    assert np.all(ablated[10:20, 10:20] == 0)
    assert np.all(ablated[0:10, :] == 200)
    assert np.all(ablated[20:, :] == 200)


def test_apply_ablation_mask_gaussian_blur():
    """Verify that 'gaussian_blur' blurs masked pixels without zeroes."""
    # Create checkerboard pattern
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[::2, ::2] = 255
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[15:35, 15:35] = 1

    ablated = apply_ablation_mask(img, mask, fill_strategy="gaussian_blur", blur_kernel_size=15)

    # Pixels in masked region should be smoothed (not strictly zero, not original extremes)
    masked_region = ablated[15:35, 15:35]
    assert np.any(masked_region > 0)
    # Outside mask remains strictly identical
    assert np.array_equal(ablated[:10, :10], img[:10, :10])


def test_apply_ablation_mask_mean():
    """Verify that 'mean' strategy replaces masked pixels with global mean color."""
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:, :, 0] = 100
    img[:, :, 1] = 150
    img[:, :, 2] = 200
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:15, 5:15] = 1

    ablated = apply_ablation_mask(img, mask, fill_strategy="mean")

    assert np.all(ablated[5:15, 5:15, 0] == 100)
    assert np.all(ablated[5:15, 5:15, 1] == 150)
    assert np.all(ablated[5:15, 5:15, 2] == 200)


def test_apply_ablation_mask_invalid_strategy():
    """Verify exception on unrecognized fill strategy."""
    img = np.ones((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unrecognized fill_strategy"):
        apply_ablation_mask(img, mask, fill_strategy="invalid_mode")


def test_apply_ablation_mask_shape_mismatch():
    """Verify exception on dimension mismatch."""
    img = np.ones((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((30, 30), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match"):
        apply_ablation_mask(img, mask)


# ---------------------------------------------------------------------------
# 2. generate_random_ablation_mask Tests
# ---------------------------------------------------------------------------

def test_generate_random_ablation_mask_count_and_bounds():
    """Verify random mask activates exact count and respects retinal FOV boundary."""
    h, w = 100, 100
    target_count = 500

    # Retinal FOV in center only
    fov_mask = np.zeros((h, w), dtype=np.uint8)
    fov_mask[20:80, 20:80] = 1

    rng = np.random.default_rng(123)
    rand_mask = generate_random_ablation_mask(
        target_pixel_count=target_count,
        shape=(h, w),
        valid_fov_mask=fov_mask,
        rng=rng,
    )

    assert rand_mask.shape == (h, w)
    assert np.sum(rand_mask) == target_count
    # All activated pixels must lie strictly inside valid_fov_mask
    assert np.all(fov_mask[rand_mask > 0] == 1)
    # Background border must have zero activations
    assert np.sum(rand_mask[:20, :]) == 0
    assert np.sum(rand_mask[80:, :]) == 0


def test_generate_random_ablation_mask_zero_count():
    """Zero target pixel count must return an empty mask."""
    mask = generate_random_ablation_mask(0, (50, 50))
    assert np.sum(mask) == 0


# ---------------------------------------------------------------------------
# 3. Progressive Deletion & Faithfulness Pipeline Tests
# ---------------------------------------------------------------------------

def test_run_single_deletion_step():
    """Verify forward inference on ablated image returns valid float probability."""
    model = DummyModel()
    device = torch.device("cpu")
    img = np.ones((64, 64, 3), dtype=np.uint8) * 150
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 1

    prob, ablated = run_single_deletion_step(
        model=model,
        image_rgb=img,
        ablation_mask=mask,
        target_class=2,
        device=device,
        fill_strategy="black",
    )

    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    assert ablated.shape == (64, 64, 3)


def test_evaluate_progressive_deletion_structure():
    """Verify evaluate_progressive_deletion output structure across steps and strategies."""
    model = DummyModel()
    device = torch.device("cpu")

    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    # Center salient hotspot
    heatmap = np.zeros((64, 64), dtype=np.float32)
    heatmap[24:40, 24:40] = 1.0

    res = evaluate_progressive_deletion(
        model=model,
        image_rgb=img,
        heatmap=heatmap,
        target_class=1,
        device=device,
        image_id="test_image_01",
        true_grade=2,
        deletion_steps=[0.10, 0.20],
        fill_strategies=["black", "gaussian_blur"],
        num_random_trials=3,
        seed=42,
    )

    assert res["image_id"] == "test_image_01"
    assert res["true_grade"] == 2
    assert "baseline_confidence" in res
    assert "black" in res["evaluations_by_fill"]
    assert "gaussian_blur" in res["evaluations_by_fill"]

    black_eval = res["evaluations_by_fill"]["black"]
    assert len(black_eval["steps"]) == 2
    assert "audc_salient" in black_eval
    assert "audc_random" in black_eval
    assert "audc_margin" in black_eval
    assert "overall_is_faithful" in black_eval


# ---------------------------------------------------------------------------
# 4. Aggregation Tests
# ---------------------------------------------------------------------------

def test_aggregate_faithfulness_scores():
    """Verify calculation of overall and grade-wise statistics."""
    records = []
    # Create 4 synthetic records: 2 faithful, 2 unfaithful
    for i in range(4):
        g = i % 2
        is_f = (i % 2 == 0)
        records.append({
            "image_id": f"img_{i}",
            "true_grade": g,
            "predicted_grade": g,
            "target_class": g,
            "baseline_confidence": 0.8,
            "cohort_origin": "idrid_seg_unique",
            "evaluations_by_fill": {
                "black": {
                    "audc_salient": 0.4 if is_f else 0.1,
                    "audc_random": 0.1 if is_f else 0.3,
                    "audc_margin": 0.3 if is_f else -0.2,
                    "overall_is_faithful": is_f,
                }
            },
        })

    summary = aggregate_faithfulness_scores(records, primary_fill="black")

    assert summary["total_images"] == 4
    assert summary["overall"]["faithful_cases_count"] == 2
    assert summary["overall"]["faithful_rate_pct"] == 50.0
    assert summary["overall"]["audc_margin"]["mean"] is not None
    assert summary["grade_wise"]["grade_0"]["audc_margin"]["count"] == 2


# ---------------------------------------------------------------------------
# 5. CLI & Immutability Verification Tests
# ---------------------------------------------------------------------------

def test_evaluate_faithfulness_cli_dry_run(tmp_path):
    """Test CLI runs in dry-run mode and writes simulated markdown and JSON reports."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/evaluate_faithfulness.py",
        "--dry-run",
        "--dry-run-samples", "10",
        "--output-dir", str(tmp_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI dry run failed: {res.stderr}"

    report_md = tmp_path / "simulated_phase_7_faithfulness_report.md"
    report_json = tmp_path / "simulated_phase_7_faithfulness_report.json"
    pred_csv = tmp_path / "simulated_faithfulness_predictions.csv"

    assert report_md.exists()
    assert report_json.exists()
    assert pred_csv.exists()

    content = report_md.read_text(encoding="utf-8")
    assert "SIMULATED DATA WARNING" in content
    assert "Faithfulness" in content


def test_evaluate_faithfulness_cli_missing_checkpoint_fails():
    """Verify that omitting --dry-run when checkpoint is missing exits with code 1."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/evaluate_faithfulness.py",
        "--checkpoint", "non_existent_model.pth",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "Missing required model checkpoint" in res.stderr or "Missing required model checkpoint" in res.stdout


def test_state_dict_immutability_guarantee():
    """Verify in-memory state_dict comparison detects any model modification."""
    from scripts.evaluate_faithfulness import verify_state_dict_unchanged

    model = DummyModel()
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 1. Unmodified model should verify as bit-exact unchanged
    assert verify_state_dict_unchanged(initial_state, model) is True

    # 2. Mutating a weight in memory should immediately fail verification
    with torch.no_grad():
        model.fc.weight[0, 0] += 1.0

    assert verify_state_dict_unchanged(initial_state, model) is False


def test_fov_restricted_salient_and_random_mask_exact_parity():
    """Verify salient and random ablation masks have exact pixel parity and zero border bleed.
    
    Guarantees:
    1. Both masks have the exact same number of deleted pixels.
    2. Both masks contain zero activations outside the valid retinal FOV.
    3. Target count is defined relative to valid retinal surface area (not total canvas).
    """
    h, w = 200, 200
    # Create circular retinal FOV
    y, x = np.ogrid[:h, :w]
    center_y, center_x = 100, 100
    radius = 80
    valid_fov = ((y - center_y)**2 + (x - center_x)**2 <= radius**2).astype(np.uint8)
    n_fov_pixels = int(np.sum(valid_fov))
    assert n_fov_pixels > 0

    # Create synthetic Grad-CAM heatmap with values both inside and outside the FOV
    heatmap = np.random.rand(h, w).astype(np.float32)

    rng = np.random.default_rng(999)

    for step in [0.10, 0.20, 0.30]:
        salient_mask, pixel_count = generate_fov_restricted_salient_mask(
            heatmap=heatmap,
            valid_fov_mask=valid_fov,
            fraction=step,
        )

        rand_mask = generate_random_ablation_mask(
            target_pixel_count=pixel_count,
            shape=(h, w),
            valid_fov_mask=valid_fov,
            rng=rng,
        )

        expected_count = int(round(step * n_fov_pixels))

        # 1. Exact count parity
        assert pixel_count == expected_count
        assert np.sum(salient_mask) == expected_count
        assert np.sum(rand_mask) == expected_count
        assert np.sum(salient_mask) == np.sum(rand_mask)

        # 2. Zero border bleed: all 1s must reside strictly inside valid_fov
        assert np.all(valid_fov[salient_mask > 0] == 1)
        assert np.all(valid_fov[rand_mask > 0] == 1)

        # 3. Canvas background outside FOV must be completely zero
        outside_fov = (valid_fov == 0)
        assert np.all(salient_mask[outside_fov] == 0)
        assert np.all(rand_mask[outside_fov] == 0)


def test_load_cohort_and_grade_mapping_idrid_matches(tmp_path: Path):
    """Verify load_cohort_and_grade_mapping correctly categorizes 57 train, 3 test, 21 unique."""
    from scripts.evaluate_faithfulness import load_cohort_and_grade_mapping
    from pathlib import Path
    import json
    import pandas as pd

    matches_file = Path("data/idrid_seg_grading_matches.json")
    if not matches_file.exists():
        matches_file = Path("reports/idrid_seg_grading_matches.json")
    if not matches_file.exists():
        pytest.skip("idrid_seg_grading_matches.json not found")

    with open(matches_file, "r", encoding="utf-8") as f:
        matches_data = json.load(f)

    # Build complete mock labels
    train_records = [
        {"image_id": Path(matches_data["all_details"][fn]["target_image"]).stem, "grade": 2}
        for fn in matches_data.get("matched_train_ids", [])
    ]
    test_records = [
        {"image_id": Path(matches_data["all_details"][fn]["target_image"]).stem, "grade": 3}
        for fn in matches_data.get("matched_test_ids", [])
    ]

    mock_tr = tmp_path / "tr.csv"
    mock_te = tmp_path / "te.csv"
    pd.DataFrame(train_records).to_csv(mock_tr, index=False)
    pd.DataFrame(test_records).to_csv(mock_te, index=False)

    cohort_map, grade_map = load_cohort_and_grade_mapping(
        matches_file,
        train_manifest_path=mock_tr,
        test_manifest_path=mock_te,
    )

    train_cohort = [k for k, v in cohort_map.items() if v == "idrid_grading_train_cohort"]
    test_cohort = [k for k, v in cohort_map.items() if v == "idrid_grading_test_cohort"]
    unique_cohort = [k for k, v in cohort_map.items() if v == "idrid_seg_unique"]

    assert len(train_cohort) == 57
    assert len(test_cohort) == 3
    assert len(unique_cohort) == 21
    assert len(cohort_map) == 81

    # The 21 unique images must strictly have grade None (unavailable, never invented)
    for k in unique_cohort:
        assert grade_map[k] is None


def test_load_cohort_and_grade_mapping_missing_labels_raises_error(tmp_path: Path):
    """Verify that missing or empty label files raise ValueError immediately."""
    from scripts.evaluate_faithfulness import load_cohort_and_grade_mapping
    from pathlib import Path

    matches_file = Path("data/idrid_seg_grading_matches.json")
    if not matches_file.exists():
        matches_file = Path("reports/idrid_seg_grading_matches.json")
    if not matches_file.exists():
        pytest.skip("idrid_seg_grading_matches.json not found")

    # 1. Missing label files completely
    nonexistent = tmp_path / "nonexistent.csv"
    with pytest.raises(ValueError, match="Official IDRiD training grading labels are missing or empty"):
        load_cohort_and_grade_mapping(
            matches_file,
            train_manifest_path=nonexistent,
            test_manifest_path=nonexistent,
        )

    # 2. Empty label DataFrame
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("image_id,grade\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Official IDRiD training grading labels are missing or empty"):
        load_cohort_and_grade_mapping(
            matches_file,
            train_manifest_path=empty_csv,
            test_manifest_path=empty_csv,
        )


def test_load_cohort_and_grade_mapping_invalid_grade_range_raises_error(tmp_path: Path):
    """Verify that grades outside range 0-4 raise ValueError."""
    import json
    import pandas as pd
    from scripts.evaluate_faithfulness import load_cohort_and_grade_mapping
    from pathlib import Path

    matches_file = Path("data/idrid_seg_grading_matches.json")
    if not matches_file.exists():
        matches_file = Path("reports/idrid_seg_grading_matches.json")
    if not matches_file.exists():
        pytest.skip("idrid_seg_grading_matches.json not found")

    with open(matches_file, "r", encoding="utf-8") as f:
        matches_data = json.load(f)

    # Invalid grade 5 (outside 0-4)
    train_records = [
        {"image_id": Path(matches_data["all_details"][fn]["target_image"]).stem, "grade": 5 if i == 0 else 2}
        for i, fn in enumerate(matches_data.get("matched_train_ids", []))
    ]
    test_records = [
        {"image_id": Path(matches_data["all_details"][fn]["target_image"]).stem, "grade": 3}
        for fn in matches_data.get("matched_test_ids", [])
    ]

    mock_tr = tmp_path / "tr_invalid.csv"
    mock_te = tmp_path / "te.csv"
    pd.DataFrame(train_records).to_csv(mock_tr, index=False)
    pd.DataFrame(test_records).to_csv(mock_te, index=False)

    with pytest.raises(ValueError, match="Invalid grade 5"):
        load_cohort_and_grade_mapping(
            matches_file,
            train_manifest_path=mock_tr,
            test_manifest_path=mock_te,
        )


def test_grade_unavailable_never_substituted_by_prediction():
    """Verify that when ground truth is unavailable, true_grade remains None and is never set to predicted_grade."""
    model = DummyModel()
    device = torch.device("cpu")
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    heatmap = np.zeros((64, 64), dtype=np.float32)
    heatmap[20:40, 20:40] = 1.0

    # Pass true_grade=None
    res = evaluate_progressive_deletion(
        model=model,
        image_rgb=img,
        heatmap=heatmap,
        target_class=1,
        device=device,
        image_id="unique_01",
        true_grade=None,  # Unavailable!
        deletion_steps=[0.10],
        fill_strategies=["black"],
        num_random_trials=2,
    )

    # true_grade must remain None, NOT predicted_grade
    assert res["true_grade"] is None
    assert res["predicted_grade"] is not None


def test_matched_grade_completeness_and_cohort_assignment(tmp_path: Path):
    """Verify that all 57 train and 3 test images receive valid grades, 21 unique receive None, and missing grades raise ValueError."""
    import json
    import pandas as pd
    from scripts.evaluate_faithfulness import load_cohort_and_grade_mapping

    matches_file = Path("reports/idrid_seg_grading_matches.json")
    if not matches_file.exists():
        pytest.skip("reports/idrid_seg_grading_matches.json not found")

    with open(matches_file, "r", encoding="utf-8") as f:
        matches_data = json.load(f)

    # 1. Build complete mock training and testing labels CSVs
    train_records = []
    for filename in matches_data.get("matched_train_ids", []):
        target = matches_data["all_details"][filename]["target_image"]
        train_records.append({"image_id": Path(target).stem, "grade": 2})

    test_records = []
    for filename in matches_data.get("matched_test_ids", []):
        target = matches_data["all_details"][filename]["target_image"]
        test_records.append({"image_id": Path(target).stem, "grade": 3})

    mock_train_csv = tmp_path / "mock_train.csv"
    mock_test_csv = tmp_path / "mock_test.csv"
    pd.DataFrame(train_records).to_csv(mock_train_csv, index=False)
    pd.DataFrame(test_records).to_csv(mock_test_csv, index=False)

    cohort_map, grade_map = load_cohort_and_grade_mapping(
        matches_file,
        train_manifest_path=mock_train_csv,
        test_manifest_path=mock_test_csv,
    )

    assert len(cohort_map) == 81
    assert len(grade_map) == 81

    # Verify 57 train images have valid grades
    train_stems = [Path(f).stem for f in matches_data["matched_train_ids"]]
    assert len(train_stems) == 57
    for s in train_stems:
        assert isinstance(grade_map[s], int)
        assert grade_map[s] == 2

    # Verify 3 test images have valid grades
    test_stems = [Path(f).stem for f in matches_data["matched_test_ids"]]
    assert len(test_stems) == 3
    for s in test_stems:
        assert isinstance(grade_map[s], int)
        assert grade_map[s] == 3

    # Verify 21 unique images are strictly None
    unique_stems = [Path(f).stem for f in matches_data["unmatched_ids"]]
    assert len(unique_stems) == 21
    for s in unique_stems:
        assert grade_map[s] is None

    # 2. Test missing training grade raises ValueError
    incomplete_train_csv = tmp_path / "incomplete_train.csv"
    pd.DataFrame(train_records[:-1]).to_csv(incomplete_train_csv, index=False)  # Drop 1 record
    with pytest.raises(ValueError, match="Expected ground-truth training grade missing"):
        load_cohort_and_grade_mapping(
            matches_file,
            train_manifest_path=incomplete_train_csv,
            test_manifest_path=mock_test_csv,
        )

    # 3. Test missing testing grade raises ValueError
    incomplete_test_csv = tmp_path / "incomplete_test.csv"
    pd.DataFrame(test_records[:-1]).to_csv(incomplete_test_csv, index=False)  # Drop 1 record
    with pytest.raises(ValueError, match="Expected ground-truth test grade missing"):
        load_cohort_and_grade_mapping(
            matches_file,
            train_manifest_path=mock_train_csv,
            test_manifest_path=incomplete_test_csv,
        )


