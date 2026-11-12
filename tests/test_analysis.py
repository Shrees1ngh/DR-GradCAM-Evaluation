"""Unit tests for Phase 8 Statistical Analysis & Hypothesis Testing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_results import (
    audit_and_merge_predictions,
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

    df = pd.DataFrame([{
        "image_id": "img1",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "composite_dice_top_10": 0.5,
        "normalized_pauc_margin": 0.2,
    }])
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
    valid_sha1 = "a" * 64
    valid_sha2 = "b" * 64
    with open(loc_json, "w") as f:
        json.dump({
            "status": "GENUINE_LOCALIZATION_EVALUATION_COMPLETE",
            "checkpoint_sha256": valid_sha1,
            "state_dict_verified": True,
        }, f)
    with open(faith_json, "w") as f:
        json.dump({
            "status": "GENUINE_FAITHFULNESS_EVALUATION_COMPLETE",
            "checkpoint_sha256": valid_sha2,
            "state_dict_verified": True,
        }, f)

    with pytest.raises(ValueError, match="Checkpoint SHA-256 hash mismatch"):
        verify_inputs_and_load_data(loc_csv, faith_csv, loc_json, faith_json, allow_simulated=False)

    # 4. Matching SHA-256 passes in genuine mode with all 81 expected images
    df_81_rows = []
    for i in range(1, 82):
        cid = f"IDRiD_{i:02d}"
        if i <= 57:
            cohort = "idrid_grading_train_cohort"
            grade = 2
        elif i <= 60:
            cohort = "idrid_grading_test_cohort"
            grade = 3
        else:
            cohort = "idrid_seg_unique"
            grade = None
        df_81_rows.append({
            "image_id": cid,
            "cohort_origin": cohort,
            "true_grade": grade,
            "predicted_grade": 2,
            "target_class": 2,
            "is_informative": True,
            "composite_dice_top_10": 0.5,
            "normalized_pauc_margin": 0.2,
        })
    df_81 = pd.DataFrame(df_81_rows)
    df_81.to_csv(loc_csv, index=False)
    df_81.to_csv(faith_csv, index=False)

    with open(faith_json, "w") as f:
        json.dump({
            "status": "GENUINE_FAITHFULNESS_EVALUATION_COMPLETE",
            "checkpoint_sha256": valid_sha1,
            "state_dict_verified": True,
        }, f)

    merged, _, _, is_sim, sha = verify_inputs_and_load_data(loc_csv, faith_csv, loc_json, faith_json, allow_simulated=False)
    assert is_sim is False
    assert sha == valid_sha1
    assert len(merged) == 81


def test_verify_inputs_exact_genuine_statuses_and_state_dict(tmp_path: Path):
    """Verify that unknown/pending statuses and unverified state_dict are rejected."""
    loc_csv = tmp_path / "loc.csv"
    faith_csv = tmp_path / "faith.csv"
    loc_json = tmp_path / "loc.json"
    faith_json = tmp_path / "faith.json"

    df = pd.DataFrame([{"image_id": "img1", "composite_dice_top_10": 0.5, "normalized_pauc_margin": 0.2}])
    df.to_csv(loc_csv, index=False)
    df.to_csv(faith_csv, index=False)

    valid_sha = "c" * 64

    # 1. Pending or unknown Phase 6 status rejected
    with open(loc_json, "w") as f:
        json.dump({"status": "PENDING_GENUINE_EVALUATION", "checkpoint_sha256": valid_sha, "state_dict_verified": True}, f)
    with open(faith_json, "w") as f:
        json.dump({"status": "GENUINE_FAITHFULNESS_EVALUATION_COMPLETE", "checkpoint_sha256": valid_sha, "state_dict_verified": True}, f)

    with pytest.raises(ValueError, match="Phase 6 report status must be exactly"):
        verify_inputs_and_load_data(loc_csv, faith_csv, loc_json, faith_json, allow_simulated=False)

    # 2. Pending or unknown Phase 7 status rejected
    with open(loc_json, "w") as f:
        json.dump({"status": "GENUINE_LOCALIZATION_EVALUATION_COMPLETE", "checkpoint_sha256": valid_sha, "state_dict_verified": True}, f)
    with open(faith_json, "w") as f:
        json.dump({"status": "UNKNOWN_STATUS", "checkpoint_sha256": valid_sha, "state_dict_verified": True}, f)

    with pytest.raises(ValueError, match="Phase 7 report status must be exactly"):
        verify_inputs_and_load_data(loc_csv, faith_csv, loc_json, faith_json, allow_simulated=False)

    # 3. state_dict_verified is False or missing
    with open(faith_json, "w") as f:
        json.dump({"status": "GENUINE_FAITHFULNESS_EVALUATION_COMPLETE", "checkpoint_sha256": valid_sha, "state_dict_verified": False}, f)

    with pytest.raises(ValueError, match="state_dict_verified is not True"):
        verify_inputs_and_load_data(loc_csv, faith_csv, loc_json, faith_json, allow_simulated=False)

    # 4. Invalid non-hex or invalid-length SHA-256
    with open(faith_json, "w") as f:
        json.dump({"status": "GENUINE_FAITHFULNESS_EVALUATION_COMPLETE", "checkpoint_sha256": "not_a_valid_sha", "state_dict_verified": True}, f)

    with pytest.raises(ValueError, match="not a valid 64-character hexadecimal SHA-256 hash"):
        verify_inputs_and_load_data(loc_csv, faith_csv, loc_json, faith_json, allow_simulated=False)


def test_audit_and_merge_predictions_unmatched_records():
    """Verify outer merge audit catches records present in only one phase."""
    loc_df = pd.DataFrame([
        {"image_id": "IDRiD_01", "composite_dice_top_10": 0.5},
        {"image_id": "IDRiD_02", "composite_dice_top_10": 0.6},
    ])
    faith_df = pd.DataFrame([
        {"image_id": "IDRiD_01", "normalized_pauc_margin": 0.2},
        {"image_id": "IDRiD_03", "normalized_pauc_margin": 0.3},
    ])

    with pytest.raises(ValueError, match="Outer merge audit failed: Unmatched image records"):
        audit_and_merge_predictions(loc_df, faith_df, is_simulated=True)


def test_audit_and_merge_predictions_field_mismatches():
    """Verify detection and rejection of true_grade, cohort_origin, or predicted_grade mismatches."""
    # 1. true_grade mismatch
    loc_df = pd.DataFrame([
        {"image_id": "IDRiD_01", "true_grade": 2, "cohort_origin": "idrid_grading_train_cohort"},
    ])
    faith_df = pd.DataFrame([
        {"image_id": "IDRiD_01", "true_grade": 3, "cohort_origin": "idrid_grading_train_cohort"},
    ])
    with pytest.raises(ValueError, match="true_grade mismatch"):
        audit_and_merge_predictions(loc_df, faith_df, is_simulated=True)

    # 2. cohort_origin mismatch
    loc_df2 = pd.DataFrame([
        {"image_id": "IDRiD_01", "true_grade": 2, "cohort_origin": "idrid_grading_train_cohort"},
    ])
    faith_df2 = pd.DataFrame([
        {"image_id": "IDRiD_01", "true_grade": 2, "cohort_origin": "idrid_seg_unique"},
    ])
    with pytest.raises(ValueError, match="cohort_origin mismatch"):
        audit_and_merge_predictions(loc_df2, faith_df2, is_simulated=True)

    # 3. predicted_grade mismatch
    loc_df3 = pd.DataFrame([
        {
            "image_id": "IDRiD_01",
            "cohort_origin": "idrid_grading_train_cohort",
            "true_grade": 2,
            "predicted_grade": 2,
        },
    ])
    faith_df3 = pd.DataFrame([
        {
            "image_id": "IDRiD_01",
            "cohort_origin": "idrid_grading_train_cohort",
            "true_grade": 2,
            "predicted_grade": 4,
        },
    ])
    with pytest.raises(ValueError, match="predicted_grade mismatch"):
        audit_and_merge_predictions(loc_df3, faith_df3, is_simulated=True)


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
        verify_inputs_and_load_data(loc_csv, faith_csv, loc_json, faith_json, allow_simulated=True)


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
    assert s_an["rho"] is None
    assert s_an["p_value"] is None
    assert "descriptive_only" in s_an["status"]
    assert len(s_an["individual_measurements"]) == 3
    assert "composite_dice" in s_an["descriptive_summary"]

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
    assert "Grade 0: no clinical lesions" in gw["grade_0"]["status"]

    # 5. Check secondary metrics
    assert "od_dice_top_10" in res["secondary_analyses"]
    assert "ma_dice_top_10" in res["secondary_analyses"]
    assert res["secondary_analyses"]["od_dice_top_10"]["exploratory"] is True


def test_perform_statistical_analysis_non_informative_and_grade_0_exclusions():
    """Verify that non-informative heatmaps and Grade 0 N/A records are excluded from complete pairs."""
    rows = [
        # Informative train-matched
        {
            "clean_id": "img1",
            "cohort_origin": "idrid_grading_train_cohort",
            "true_grade": 2,
            "composite_dice_top_10": 0.4,
            "normalized_pauc_margin": 0.2,
            "heatmap_status": "ok",
            "is_informative": True,
        },
        # Non-informative heatmap record
        {
            "clean_id": "img2",
            "cohort_origin": "idrid_grading_train_cohort",
            "true_grade": 3,
            "composite_dice_top_10": None,
            "normalized_pauc_margin": None,
            "heatmap_status": "non_informative_heatmap",
            "is_informative": False,
        },
        # Grade 0 record (no lesions, composite dice is NaN/None)
        {
            "clean_id": "img3",
            "cohort_origin": "idrid_grading_train_cohort",
            "true_grade": 0,
            "composite_dice_top_10": None,
            "normalized_pauc_margin": 0.15,
            "heatmap_status": "ok",
            "is_informative": True,
        },
        # Another informative
        {
            "clean_id": "img4",
            "cohort_origin": "idrid_seg_unique",
            "true_grade": None,
            "composite_dice_top_10": 0.35,
            "normalized_pauc_margin": 0.18,
            "heatmap_status": "ok",
            "is_informative": True,
        },
    ]
    df = pd.DataFrame(rows)
    res = perform_statistical_analysis(df, seed=42, n_bootstraps=100)

    assert res["primary_cohort_count"] == 4
    assert res["complete_primary_pairs_count"] == 2
    excluded = res["excluded_primary_records"]
    assert len(excluded) == 2
    ex_reasons = {rec["clean_id"]: rec["primary_exclusion_reason"] for rec in excluded}
    assert ex_reasons["img2"] == "non_informative_heatmap"
    assert ex_reasons["img3"] == "grade_0_no_lesions_na"


def test_audit_and_merge_predictions_genuine_81_images_validation():
    """Verify that genuine evaluation strictly requires all 81 expected IDRiD segmentation images."""
    rows_loc = [{"image_id": f"IDRiD_{i:02d}", "true_grade": 2} for i in range(1, 81)]
    rows_faith = [{"image_id": f"IDRiD_{i:02d}", "true_grade": 2} for i in range(1, 81)]

    with pytest.raises(ValueError, match="Dataset count validation failed for IDRiD segmentation cohort"):
        audit_and_merge_predictions(pd.DataFrame(rows_loc), pd.DataFrame(rows_faith), is_simulated=False)


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


# ==============================================================================
# Phase 8 Strict Metadata & Validation Regression Tests
# ==============================================================================

def test_audit_and_merge_predictions_fractional_ground_truth_grades():
    """Verify rejection of fractional, out-of-range, and boolean ground-truth grades."""
    # 1. Fractional float grade
    loc_df = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": 2.5}])
    faith_df = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": 2.5}])
    with pytest.raises(ValueError, match="fractional value"):
        audit_and_merge_predictions(loc_df, faith_df, is_simulated=True)

    # 2. Fractional string grade
    loc_df_str = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": "2.5"}])
    faith_df_str = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": "2.5"}])
    with pytest.raises(ValueError, match="fractional value"):
        audit_and_merge_predictions(loc_df_str, faith_df_str, is_simulated=True)

    # 3. Out-of-bounds grade (> 4)
    loc_df_high = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": 5}])
    faith_df_high = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": 5}])
    with pytest.raises(ValueError, match="greater than maximum"):
        audit_and_merge_predictions(loc_df_high, faith_df_high, is_simulated=True)

    # 4. Negative grade (< 0)
    loc_df_neg = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": -1}])
    faith_df_neg = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": -1}])
    with pytest.raises(ValueError, match="less than minimum"):
        audit_and_merge_predictions(loc_df_neg, faith_df_neg, is_simulated=True)

    # 5. Boolean grade (e.g. True)
    loc_df_bool = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": True}])
    faith_df_bool = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": True}])
    with pytest.raises(ValueError, match="got boolean"):
        audit_and_merge_predictions(loc_df_bool, faith_df_bool, is_simulated=True)


def test_audit_and_merge_predictions_fractional_predicted_classes():
    """Verify rejection of fractional, out-of-range, and boolean predicted classes and grades."""
    # 1. Fractional predicted grade in Phase 6
    loc_df = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "predicted_grade": 1.5,
    }])
    faith_df = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "predicted_grade": 1,
    }])
    with pytest.raises(ValueError, match="fractional value"):
        audit_and_merge_predictions(loc_df, faith_df, is_simulated=True)

    # 2. Fractional target class in Phase 7
    loc_df2 = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "predicted_grade": 1,
    }])
    faith_df2 = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "predicted_grade": 1,
        "target_class": "1.5",
    }])
    with pytest.raises(ValueError, match="fractional value"):
        audit_and_merge_predictions(loc_df2, faith_df2, is_simulated=True)

    # 3. Out-of-bounds predicted class
    loc_df3 = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "predicted_grade": 5,
    }])
    faith_df3 = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "predicted_grade": 5,
    }])
    with pytest.raises(ValueError, match="greater than maximum"):
        audit_and_merge_predictions(loc_df3, faith_df3, is_simulated=True)


def test_audit_and_merge_predictions_cohort_metadata_strict():
    """Verify strict validation and rejection of missing, empty, or unrecognized cohorts."""
    # 1. Missing cohort in Phase 6
    loc_df = pd.DataFrame([{"image_id": "IDRiD_01", "true_grade": 2}])
    faith_df = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": 2}])
    with pytest.raises(ValueError, match="Missing cohort_origin in Phase 6"):
        audit_and_merge_predictions(loc_df, faith_df, is_simulated=True)

    # 2. Missing cohort in Phase 7
    loc_df2 = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": 2}])
    faith_df2 = pd.DataFrame([{"image_id": "IDRiD_01", "true_grade": 2}])
    with pytest.raises(ValueError, match="Missing cohort_origin in Phase 7"):
        audit_and_merge_predictions(loc_df2, faith_df2, is_simulated=True)

    # 3. Empty string cohort
    loc_df3 = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "", "true_grade": 2}])
    faith_df3 = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "", "true_grade": 2}])
    with pytest.raises(ValueError, match="Missing cohort_origin in Phase 6"):
        audit_and_merge_predictions(loc_df3, faith_df3, is_simulated=True)

    # 4. Unrecognized cohort label
    loc_df4 = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "unrecognized_cohort", "true_grade": 2}])
    faith_df4 = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "unrecognized_cohort", "true_grade": 2}])
    with pytest.raises(ValueError, match="Unrecognized cohort_origin 'unrecognized_cohort'"):
        audit_and_merge_predictions(loc_df4, faith_df4, is_simulated=True)


def test_audit_and_merge_predictions_matched_and_unique_cohort_null_grades():
    """Verify ground-truth grade rules: required for train/test matched cohorts, null for unique."""
    # 1. Null grade in idrid_grading_train_cohort (must raise)
    loc_df1 = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": None}])
    faith_df1 = pd.DataFrame([{"image_id": "IDRiD_01", "cohort_origin": "idrid_grading_train_cohort", "true_grade": None}])
    with pytest.raises(ValueError, match="Missing true_grade in Phase 6 for matched cohort"):
        audit_and_merge_predictions(loc_df1, faith_df1, is_simulated=True)

    # 2. Null grade in idrid_grading_test_cohort (must raise)
    loc_df2 = pd.DataFrame([{"image_id": "IDRiD_09", "cohort_origin": "idrid_grading_test_cohort", "true_grade": None}])
    faith_df2 = pd.DataFrame([{"image_id": "IDRiD_09", "cohort_origin": "idrid_grading_test_cohort", "true_grade": None}])
    with pytest.raises(ValueError, match="Missing true_grade in Phase 6 for matched cohort"):
        audit_and_merge_predictions(loc_df2, faith_df2, is_simulated=True)

    # 3. Non-null grade in idrid_seg_unique (must raise)
    loc_df3 = pd.DataFrame([{"image_id": "IDRiD_60", "cohort_origin": "idrid_seg_unique", "true_grade": 2}])
    faith_df3 = pd.DataFrame([{"image_id": "IDRiD_60", "cohort_origin": "idrid_seg_unique", "true_grade": 2}])
    with pytest.raises(ValueError, match="Expected null true_grade for unique segmentation record"):
        audit_and_merge_predictions(loc_df3, faith_df3, is_simulated=True)

    # 4. Valid null grade in idrid_seg_unique (must pass with unified_grade=None)
    loc_df4 = pd.DataFrame([{"image_id": "IDRiD_60", "cohort_origin": "idrid_seg_unique", "true_grade": None}])
    faith_df4 = pd.DataFrame([{"image_id": "IDRiD_60", "cohort_origin": "idrid_seg_unique", "true_grade": None}])
    merged, _ = audit_and_merge_predictions(loc_df4, faith_df4, is_simulated=True)
    assert merged["true_grade"].iloc[0] is None


def test_audit_and_merge_predictions_genuine_prediction_metadata_required():
    """Verify that genuine evaluation strictly requires predicted grade and target class metadata."""
    rows_loc = []
    rows_faith = []
    for i in range(1, 82):
        cid = f"IDRiD_{i:02d}"
        cohort = "idrid_grading_train_cohort" if i <= 57 else ("idrid_grading_test_cohort" if i <= 60 else "idrid_seg_unique")
        grade = 2 if cohort != "idrid_seg_unique" else None
        rows_loc.append({
            "image_id": cid,
            "cohort_origin": cohort,
            "true_grade": grade,
            "is_informative": True,
            # Missing predicted_grade / predicted_class on image 1
            "predicted_grade": 2 if i > 1 else None,
        })
        rows_faith.append({
            "image_id": cid,
            "cohort_origin": cohort,
            "true_grade": grade,
            "is_informative": True,
            "predicted_grade": 2,
            "target_class": 2,
        })

    with pytest.raises(ValueError, match="Missing Phase 6 predicted class / grade for genuine image 'IDRiD_01'"):
        audit_and_merge_predictions(pd.DataFrame(rows_loc), pd.DataFrame(rows_faith), is_simulated=False)


def test_audit_and_merge_predictions_strict_boolean_parsing():
    """Verify explicit boolean parsing: accepts 'True'/'False'/'1'/'0', rejects ambiguous/invalid."""
    # 1. String "False" is correctly parsed as False (not True like bool("False"))
    loc_df_false = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "is_informative": "False",
    }])
    faith_df_false = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "is_informative": "False",
    }])
    merged_false, _ = audit_and_merge_predictions(loc_df_false, faith_df_false, is_simulated=True)
    assert bool(merged_false["is_informative"].iloc[0]) is False
    assert merged_false["heatmap_status"].iloc[0] == "non_informative_heatmap"

    # 2. String "True" is parsed as True
    loc_df_true = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "is_informative": "True",
    }])
    faith_df_true = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "is_informative": "True",
    }])
    merged_true, _ = audit_and_merge_predictions(loc_df_true, faith_df_true, is_simulated=True)
    assert bool(merged_true["is_informative"].iloc[0]) is True
    assert merged_true["heatmap_status"].iloc[0] == "ok"

    # 3. Numeric 1 and 0 (and floats 1.0, 0.0)
    loc_df_num = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "is_informative": 0,
    }])
    faith_df_num = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "is_informative": 0.0,
    }])
    merged_num, _ = audit_and_merge_predictions(loc_df_num, faith_df_num, is_simulated=True)
    assert bool(merged_num["is_informative"].iloc[0]) is False

    # 4. Ambiguous string "yes" rejected
    loc_df_yes = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "is_informative": "yes",
    }])
    faith_df_yes = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "is_informative": True,
    }])
    with pytest.raises(ValueError, match="Invalid boolean string.*'yes'"):
        audit_and_merge_predictions(loc_df_yes, faith_df_yes, is_simulated=True)

    # 5. Invalid integer 2 rejected
    loc_df_two = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "is_informative": 2,
    }])
    faith_df_two = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "is_informative": True,
    }])
    with pytest.raises(ValueError, match="Invalid boolean integer.*2"):
        audit_and_merge_predictions(loc_df_two, faith_df_two, is_simulated=True)

    # 6. Null boolean value rejected when column present
    loc_df_null = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "is_informative": None,
    }])
    faith_df_null = pd.DataFrame([{
        "image_id": "IDRiD_01",
        "cohort_origin": "idrid_grading_train_cohort",
        "true_grade": 2,
        "is_informative": True,
    }])
    with pytest.raises(ValueError, match="Invalid null/empty boolean"):
        audit_and_merge_predictions(loc_df_null, faith_df_null, is_simulated=True)
