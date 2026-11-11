"""Unit tests for Phase 8 Statistical Analysis & Hypothesis Testing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_results import (
    benjamini_hochberg_fdr,
    compute_spearman_with_bootstrap_ci,
    main,
    perform_statistical_analysis,
    verify_inputs_and_load_data,
)


def test_compute_spearman_with_bootstrap_ci_monotonic():
    """Verify Spearman correlation and bootstrap CI on monotonic positive relationship."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    y = [1.1, 1.9, 3.2, 3.8, 5.1, 6.2, 6.9, 8.1, 9.2, 9.8]

    res = compute_spearman_with_bootstrap_ci(x, y, n_bootstraps=500, seed=42)

    assert res["status"] == "ok"
    assert res["n"] == 10
    assert res["rho"] > 0.95
    assert res["p_value"] < 0.001
    assert res["ci_lower"] is not None
    assert res["ci_upper"] is not None
    assert res["ci_lower"] <= res["rho"] <= res["ci_upper"]


def test_compute_spearman_with_bootstrap_ci_edge_cases():
    """Verify handling of small sample sizes and zero variance degenerate data."""
    # N < 3
    res_small = compute_spearman_with_bootstrap_ci([1.0, 2.0], [2.0, 3.0])
    assert res_small["status"] == "insufficient_samples"
    assert res_small["rho"] is None

    # Degenerate (constant x)
    res_const = compute_spearman_with_bootstrap_ci([5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0])
    assert res_const["status"] == "degenerate_distribution"
    assert res_const["rho"] is None


def test_benjamini_hochberg_fdr_calculation():
    """Verify Benjamini-Hochberg FDR calculation against hand-computed benchmark."""
    # Input p-values
    p_vals = [0.01, 0.04, 0.03, 0.20]
    # Hand calculation:
    # m = 4
    # Sorted: (0.01, k=1), (0.03, k=2), (0.04, k=3), (0.20, k=4)
    # k=4: 4/4 * 0.20 = 0.20 -> cum_min = 0.20
    # k=3: 4/3 * 0.04 = 0.05333... -> cum_min = 0.05333...
    # k=2: 4/2 * 0.03 = 0.06 -> min(0.05333..., 0.06) = 0.05333...
    # k=1: 4/1 * 0.01 = 0.04 -> min(0.05333..., 0.04) = 0.04
    # Expected order [0.04, 0.05333..., 0.05333..., 0.20]
    adj = benjamini_hochberg_fdr(p_vals)

    assert len(adj) == 4
    assert pytest.approx(adj[0], abs=1e-4) == 0.04
    assert pytest.approx(adj[1], abs=1e-4) == 0.05333
    assert pytest.approx(adj[2], abs=1e-4) == 0.05333
    assert pytest.approx(adj[3], abs=1e-4) == 0.20


def test_verify_inputs_and_load_data_integrity_gate(tmp_path: Path):
    """Verify strict integrity gates: reject simulated data without flag and reject hash mismatches."""
    loc_csv = tmp_path / "loc.csv"
    faith_csv = tmp_path / "faith.csv"
    loc_json = tmp_path / "loc.json"
    faith_json = tmp_path / "faith.json"

    df = pd.DataFrame([{"image_id": "img1", "composite_dice_top_10": 0.5, "normalized_pauc_margin": 0.2}])
    df.to_csv(loc_csv, index=False)
    df.to_csv(faith_csv, index=False)

    # 1. Simulated status rejected by default
    with open(loc_json, "w") as f:
        json.dump({"status": "SIMULATED_DATA_NOT_VALID_FOR_RESEARCH"}, f)
    with open(faith_json, "w") as f:
        json.dump({"status": "SIMULATED_FAITHFULNESS_DATA"}, f)

    with pytest.raises(ValueError, match="Integrity Gate Failed: Simulated / Dry-Run"):
        verify_inputs_and_load_data(loc_csv, faith_csv, loc_json, faith_json, allow_simulated=False)

    # 2. Simulated allowed with allow_simulated=True
    merged, _, _, is_sim, _ = verify_inputs_and_load_data(loc_csv, faith_csv, loc_json, faith_json, allow_simulated=True)
    assert is_sim is True
    assert len(merged) == 1

    # 3. Checkpoint SHA-256 mismatch in genuine mode
    with open(loc_json, "w") as f:
        json.dump({"status": "GENUINE_LOCALIZATION_EVALUATION_COMPLETE", "checkpoint_sha256": "hash_aaa"}, f)
    with open(faith_json, "w") as f:
        json.dump({"status": "GENUINE_FAITHFULNESS_EVALUATION_COMPLETE", "checkpoint_sha256": "hash_bbb"}, f)

    with pytest.raises(ValueError, match="Checkpoint SHA-256 hash mismatch"):
        verify_inputs_and_load_data(loc_csv, faith_csv, loc_json, faith_json, allow_simulated=False)

    # 4. Matching SHA-256 passes in genuine mode
    with open(faith_json, "w") as f:
        json.dump({"status": "GENUINE_FAITHFULNESS_EVALUATION_COMPLETE", "checkpoint_sha256": "hash_aaa"}, f)

    merged, _, _, is_sim, sha = verify_inputs_and_load_data(loc_csv, faith_csv, loc_json, faith_json, allow_simulated=False)
    assert is_sim is False
    assert sha == "hash_aaa"


def test_verify_inputs_detects_duplicate_image_ids(tmp_path: Path):
    """Verify that duplicate image IDs in either CSV trigger ValueError."""
    loc_csv = tmp_path / "loc.csv"
    faith_csv = tmp_path / "faith.csv"
    loc_json = tmp_path / "loc.json"
    faith_json = tmp_path / "faith.json"

    with open(loc_json, "w") as f:
        json.dump({"status": "GENUINE_LOCALIZATION_COMPLETE", "checkpoint_sha256": "h1"}, f)
    with open(faith_json, "w") as f:
        json.dump({"status": "GENUINE_FAITHFULNESS_COMPLETE", "checkpoint_sha256": "h1"}, f)

    # Duplicate in loc_csv
    pd.DataFrame([
        {"image_id": "img1", "composite_dice_top_10": 0.5},
        {"image_id": "img1", "composite_dice_top_10": 0.6},
    ]).to_csv(loc_csv, index=False)

    pd.DataFrame([
        {"image_id": "img1", "normalized_pauc_margin": 0.2},
    ]).to_csv(faith_csv, index=False)

    with pytest.raises(ValueError, match="Duplicate image IDs detected in localization CSV"):
        verify_inputs_and_load_data(loc_csv, faith_csv, loc_json, faith_json, allow_simulated=False)


def test_perform_statistical_analysis_cohorts_and_grades():
    """Verify 78 primary vs 3 supplementary cohort separation and grade-wise breakdown."""
    rows = []
    rng = np.random.default_rng(42)

    # 81 records:
    # 57 train-matched (grades 1-4)
    # 21 unique (grade None)
    # 3 supplementary (IDRiD_09, IDRiD_22, IDRiD_31)
    for i in range(81):
        dice = float(0.10 + 0.05 * (i % 5) + 0.01 * rng.random())
        margin = float(0.15 + 0.04 * (i % 5) + 0.01 * rng.random())

        if i < 57:
            img_id = f"IDRiD_train_{i+1:02d}"
            cohort = "idrid_grading_train_cohort"
            grade = (i % 4) + 1  # grades 1 to 4 (~14 per grade)
        elif i < 60:
            supp_names = ["IDRiD_09", "IDRiD_22", "IDRiD_31"]
            img_id = supp_names[i - 57]
            cohort = "idrid_grading_test_cohort"
            grade = (i % 4) + 1
        else:
            img_id = f"IDRiD_unique_{i-59:02d}"
            cohort = "idrid_seg_unique"
            grade = None  # Grade unavailable for the 21 unique images

        rows.append({
            "clean_id": img_id,
            "cohort_origin": cohort,
            "true_grade": grade,
            "composite_dice_top_10": dice,
            "composite_iou_top_10": dice * 0.5,
            "ma_dice_top_10": dice * 0.8,
            "ma_iou_top_10": dice * 0.4,
            "od_dice_top_10": 0.01,
            "od_iou_top_10": 0.005,
            "normalized_pauc_margin": margin,
        })

    df = pd.DataFrame(rows)

    res = perform_statistical_analysis(df, seed=42, n_bootstraps=200)

    # 1. Check cohort counts
    assert res["primary_cohort_count"] == 78
    assert res["supplementary_cohort_count"] == 3
    assert res["total_merged_count"] == 81

    # 2. Check primary analysis
    p_an = res["primary_analysis"]
    assert p_an["n"] == 78
    assert p_an["status"] == "ok"
    assert p_an["rho"] is not None

    # 3. Check supplementary cohort
    s_an = res["supplementary_analysis"]
    assert s_an is not None
    assert s_an["n"] == 3

    # 4. Check grade-wise analyses:
    # Grades 1 to 4 each have ~14 images (>= 5), so they are evaluated
    gw = res["grade_wise_analyses"]
    total_graded = 0
    for g in range(1, 5):
        g_info = gw[f"grade_{g}"]
        assert g_info["n"] >= 5
        assert g_info["rho"] is not None
        assert g_info["p_value_fdr"] is not None
        total_graded += g_info["n"]
    assert total_graded == 57  # Exactly the 57 train-matched images

    # Grade 0 has 0 images (< 5) -> insufficient samples
    assert gw["grade_0"]["n"] == 0
    assert gw["grade_0"]["status"] == "insufficient_samples (N < 5)"

    # 5. Check secondary metrics
    assert "od_dice_top_10" in res["secondary_analyses"]
    assert "ma_dice_top_10" in res["secondary_analyses"]


def test_analyze_results_cli_end_to_end(tmp_path: Path):
    """Verify CLI end-to-end execution generates Markdown, JSON, and PNG figure."""
    loc_csv = tmp_path / "loc.csv"
    faith_csv = tmp_path / "faith.csv"
    loc_json = tmp_path / "loc.json"
    faith_json = tmp_path / "faith.json"

    out_md = tmp_path / "report.md"
    out_json = tmp_path / "report.json"
    out_png = tmp_path / "figures" / "plot.png"

    with open(loc_json, "w") as f:
        json.dump({"status": "SIMULATED_LOCALIZATION", "checkpoint_sha256": "fake_sha"}, f)
    with open(faith_json, "w") as f:
        json.dump({"status": "SIMULATED_FAITHFULNESS", "checkpoint_sha256": "fake_sha"}, f)

    rows_loc = []
    rows_faith = []
    for i in range(10):
        img_id = f"IDRiD_{i+1:02d}"
        rows_loc.append({
            "image_id": img_id,
            "cohort_origin": "idrid_grading_train_cohort",
            "true_grade": 2,
            "composite_dice_top_10": 0.1 * (i + 1),
            "composite_iou_top_10": 0.05 * (i + 1),
            "od_dice_top_10": 0.01,
        })
        rows_faith.append({
            "image_id": img_id,
            "cohort_origin": "idrid_grading_train_cohort",
            "true_grade": 2,
            "normalized_pauc_margin": 0.15 * (i + 1),
        })

    pd.DataFrame(rows_loc).to_csv(loc_csv, index=False)
    pd.DataFrame(rows_faith).to_csv(faith_csv, index=False)

    test_args = [
        "analyze_results.py",
        "--localization-csv", str(loc_csv),
        "--faithfulness-csv", str(faith_csv),
        "--localization-json", str(loc_json),
        "--faithfulness-json", str(faith_json),
        "--output-md", str(out_md),
        "--output-json", str(out_json),
        "--output-plot", str(out_png),
        "--allow-simulated",
        "--bootstrap-samples", "100",
    ]

    with patch("sys.argv", test_args):
        ret = main()

    assert ret == 0
    assert out_md.exists()
    assert out_json.exists()
    assert out_png.exists()

    with open(out_json, "r") as f:
        report_data = json.load(f)
    assert report_data["is_simulated"] is True
    assert "SIMULATED" in report_data["status"]
    assert report_data["analysis_results"]["primary_analysis"]["rho"] is not None
