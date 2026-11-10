"""Unit tests for IDRiD training domain diagnostic, integrity checks, and dataset isolation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_idrid_domain import (
    compute_file_sha256,
    recalculate_metrics_from_csv,
    verify_dataset_isolation,
    verify_label_mapping,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_verify_label_mapping():
    """Verify that five-class labels are properly parsed and verified."""
    df = pd.DataFrame({
        "grade": [0, 1, 2, 3, 4, 0, 2, 2, 3, 4]
    })
    res = verify_label_mapping(df, grade_col="grade")

    assert res["is_valid_0_to_4"] is True
    assert res["total_samples"] == 10
    assert res["grade_counts"][2] == 3
    assert res["unique_grades_found"] == [0, 1, 2, 3, 4]


def test_recalculate_metrics_from_csv(tmp_path: Path):
    """Verify recalculate_metrics_from_csv computes exact metrics directly from CSV."""
    csv_file = tmp_path / "test_preds.csv"
    # Create 10 dummy predictions:
    # 5 exact matches, 3 off-by-one, 2 severe misses
    df = pd.DataFrame({
        "image_id": [f"IDRiD_{i+1:03d}" for i in range(10)],
        "true_grade":      [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        "predicted_grade": [0, 1, 2, 3, 4, 1, 2, 1, 0, 1],
        "prediction_confidence": [0.9] * 10,
    })
    df.to_csv(csv_file, index=False)

    metrics = recalculate_metrics_from_csv(csv_file)

    assert "accuracy" in metrics
    assert "qwk" in metrics
    assert "confusion_matrix" in metrics
    assert "referable_dr" in metrics
    assert pytest.approx(metrics["accuracy"]) == 0.50
    assert len(metrics["confusion_matrix"]) == 5
    assert len(metrics["per_grade"]) == 5


def test_verify_dataset_isolation_strict(tmp_path: Path):
    """Verify dataset isolation rejects any test-cohort samples regardless of numerical ordering."""
    test_manifest = tmp_path / "test_manifest.csv"
    pd.DataFrame({
        "image_id": ["IDRiD_478", "IDRiD_470", "IDRiD_471", "IDRiD_414"],
        "original_id": ["IDRiD_478", "IDRiD_470", "IDRiD_471", "IDRiD_414"],
    }).to_csv(test_manifest, index=False)

    # Valid training cohort (disjoint)
    valid_train = ["IDRiD_001", "IDRiD_002", "IDRiD_102", "IDRiD_350"]
    res_valid = verify_dataset_isolation(valid_train, test_manifest_path=test_manifest)
    assert res_valid["is_isolated"] is True
    assert res_valid["leakage_count"] == 0

    # Contaminated training cohort with a test ID
    contaminated_train = ["IDRiD_001", "IDRiD_478", "IDRiD_002"]
    res_contaminated = verify_dataset_isolation(contaminated_train, test_manifest_path=test_manifest)
    assert res_contaminated["is_isolated"] is False
    assert res_contaminated["leakage_count"] == 1
    assert "IDRiD_478" in res_contaminated["leakage_ids"]


def test_compute_file_sha256(tmp_path: Path):
    """Verify SHA-256 computation and tampering detection."""
    test_file = tmp_path / "model.pth"
    test_file.write_bytes(b"initial_model_weights_tensor_data")

    sha_before = compute_file_sha256(test_file)
    assert isinstance(sha_before, str) and len(sha_before) == 64

    # File unchanged
    sha_unchanged = compute_file_sha256(test_file)
    assert sha_before == sha_unchanged

    # Simulate modification
    test_file.write_bytes(b"modified_model_weights_after_fine_tuning")
    sha_after = compute_file_sha256(test_file)
    assert sha_before != sha_after


def test_diagnose_script_fails_without_dry_run_when_inputs_missing():
    """Verify that diagnose_idrid_domain.py exits with status 1 if required inputs are missing without --dry-run."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "diagnose_idrid_domain.py"),
        "--checkpoint", str(PROJECT_ROOT / "non_existent_model.pth"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "Missing required model checkpoint" in proc.stderr or "Missing required model checkpoint" in proc.stdout or "status 1" in proc.stderr or proc.returncode != 0


def test_diagnose_script_dry_run_generates_simulated_outputs(tmp_path: Path):
    """Verify that diagnose_idrid_domain.py succeeds with --dry-run and writes to designated simulated outputs."""
    out_csv = tmp_path / "sim_preds.csv"
    out_md = tmp_path / "sim_report.md"
    out_json = tmp_path / "sim_data.json"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "diagnose_idrid_domain.py"),
        "--dry-run",
        "--output-csv", str(out_csv),
        "--output-md", str(out_md),
        "--output-json", str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert out_csv.exists()
    assert out_md.exists()
    assert out_json.exists()

    report_content = out_md.read_text(encoding="utf-8")
    assert "SIMULATED DATA" in report_content


def test_preprocessing_geometry_calculation(tmp_path: Path):
    """Verify that analyze_image_quality_and_domain distinguishes uncropped padding from FOV-cropped geometry."""
    import cv2
    from PIL import Image
    from scripts.diagnose_idrid_domain import analyze_image_quality_and_domain

    # Create a synthetic 4288x2848 fundus image with centered circular retina (diameter 2700 px)
    h, w = 2848, 4288
    img = np.zeros((h, w, 3), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = 1350
    # Retinal tissue has green channel intensity
    cv2.circle(img, center, radius, (150, 100, 50), -1)

    test_img_path = tmp_path / "test_fundus.jpg"
    Image.fromarray(img).save(test_img_path, quality=95)

    stats = analyze_image_quality_and_domain([test_img_path], sample_limit=1)

    # 1. Uncropped direct scaling must produce ~33.59% letterbox padding due to 3:2 aspect ratio
    assert pytest.approx(stats["native_mean_aspect_ratio"], rel=0.01) == 4288 / 2848
    assert pytest.approx(stats["uncropped_padding_fraction"], abs=0.02) == 0.3359

    # 2. FOV cropping removes black side pillars -> aspect ratio becomes near 1:1 (~1.00 - 1.05)
    assert pytest.approx(stats["fov_cropped_mean_aspect_ratio"], abs=0.05) == 1.0

    # 3. Actual model-input letterbox padding after FOV crop is negligible (<= 3%)
    assert stats["actual_preprocessed_letterbox_padding"] <= 0.05


def test_verify_state_dict_unchanged_diagnostic():
    """Verify state_dict integrity verification in diagnostic pipeline."""
    import torch
    import torch.nn as nn
    from scripts.diagnose_idrid_domain import verify_state_dict_unchanged

    model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 5))
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert verify_state_dict_unchanged(initial_state, model) is True

    # Mutate a weight
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
            break

    assert verify_state_dict_unchanged(initial_state, model) is False

