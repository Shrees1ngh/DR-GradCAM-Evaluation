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

