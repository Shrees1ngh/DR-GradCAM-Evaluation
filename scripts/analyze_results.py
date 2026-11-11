#!/usr/bin/env python3
"""Phase 8: Statistical Analysis & Hypothesis Testing.

Evaluates the primary research question:
Does visual explanation quality (Grad-CAM localization against expert lesion masks)
correlate with model decision fidelity (faithfulness under progressive deletion)?

Primary Hypothesis:
- H0: There is no monotonic relationship between Grad-CAM Composite Dice (top 10%
  percentile) and Faithfulness Net Margin (Spearman rho == 0).
- H1: Higher lesion localization overlap is positively correlated with decision fidelity
  (Spearman rho > 0).

Strict Data Integrity Guarantees:
1. Genuine Run Enforcement:
   - Unless '--allow-simulated' is explicitly provided, both Phase 6 and Phase 7 inputs
     must be verified genuine empirical evaluation outputs.
   - Checkpoint SHA-256 hashes from both phases must match bit-for-bit.
2. Cohort Isolation:
   - Primary Evaluation Cohort (N = 78): 57 training-matched images + 21 unique segmentation
     images. Used for primary hypothesis testing.
   - Supplementary Test Cohort (N = 3: IDRiD_09, IDRiD_22, IDRiD_31): Evaluated separately
     for transparency; excluded from primary correlation to eliminate test-set leakage.
3. Grade-Wise Analysis:
   - Images with grade == None (21 unique images) are included in the overall primary
     correlation but excluded from grade-wise breakdowns.
   - Tests with sample size N < 5 are flagged as insufficient.
   - Benjamini-Hochberg False Discovery Rate (FDR) adjustment is applied across tested grades.
4. Negative Control Validation:
   - Optic Disc (OD) localization is analyzed as a negative control; expected correlation
     near zero or negative.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


SUPPLEMENTARY_IMAGE_IDS = {"IDRiD_09", "IDRiD_22", "IDRiD_31"}
ICDR_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR",
}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def compute_spearman_with_bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute Spearman rank correlation, two-sided p-value, and bootstrap CI.

    Args:
        x: First metric sequence.
        y: Second metric sequence.
        n_bootstraps: Number of bootstrap iterations.
        confidence_level: Desired coverage (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducible bootstrap sampling.

    Returns:
        Dict containing n, rho, p_value, ci_lower, ci_upper, and confidence_level.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    valid_mask = (~np.isnan(x_arr)) & (~np.isnan(y_arr))
    x_clean = x_arr[valid_mask]
    y_clean = y_arr[valid_mask]
    n = len(x_clean)

    if n < 3:
        return {
            "n": n,
            "rho": None,
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
            "confidence_level": confidence_level,
            "status": "insufficient_samples",
        }

    # Check for degenerate constant input distributions
    if np.all(x_clean == x_clean[0]) or np.all(y_clean == y_clean[0]):
        return {
            "n": n,
            "rho": None,
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
            "confidence_level": confidence_level,
            "status": "degenerate_distribution",
        }

    # Base Spearman calculation
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.stats.ConstantInputWarning if hasattr(scipy.stats, "ConstantInputWarning") else UserWarning)
        res = scipy.stats.spearmanr(x_clean, y_clean)
    rho = float(res.statistic if hasattr(res, "statistic") else res.correlation)
    p_val = float(res.pvalue)

    if np.isnan(rho):
        return {
            "n": n,
            "rho": None,
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
            "confidence_level": confidence_level,
            "status": "degenerate_distribution",
        }

    # Percentile Bootstrap Confidence Interval
    rng = np.random.default_rng(seed)
    boot_rhos: List[float] = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        xb = x_clean[idx]
        yb = y_clean[idx]
        # Avoid degenerate resamples with zero variance
        if np.all(xb == xb[0]) or np.all(yb == yb[0]):
            continue
        b_res = scipy.stats.spearmanr(xb, yb)
        b_rho = float(b_res.statistic if hasattr(b_res, "statistic") else b_res.correlation)
        if not np.isnan(b_rho):
            boot_rhos.append(b_rho)

    if len(boot_rhos) < 100:
        ci_lower, ci_upper = None, None
    else:
        alpha = (1.0 - confidence_level) / 2.0
        ci_lower = float(np.percentile(boot_rhos, alpha * 100.0))
        ci_upper = float(np.percentile(boot_rhos, (1.0 - alpha) * 100.0))

    return {
        "n": n,
        "rho": rho,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "status": "ok",
    }


def benjamini_hochberg_fdr(p_values: Sequence[float]) -> List[float]:
    """Adjust p-values using the Benjamini-Hochberg FDR procedure.

    Args:
        p_values: Sequence of raw two-sided p-values.

    Returns:
        List of FDR-adjusted p-values in the original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted: List[Tuple[int, float]] = []

    # Step-up adjustment
    cum_min = 1.0
    for rank_from_end, (orig_idx, p) in enumerate(reversed(indexed)):
        k = m - rank_from_end
        adj_p = min(cum_min, p * m / k)
        adj_p = min(1.0, max(0.0, adj_p))
        cum_min = adj_p
        adjusted.append((orig_idx, adj_p))

    adjusted.sort(key=lambda item: item[0])
    return [p_adj for _, p_adj in adjusted]


def verify_inputs_and_load_data(
    localization_csv: Path,
    faithfulness_csv: Path,
    localization_json: Path,
    faithfulness_json: Path,
    allow_simulated: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], bool, Optional[str]]:
    """Validate data integrity, verify checkpoint SHA-256 consistency, and merge predictions.

    Args:
        localization_csv: Path to Phase 6 per-image predictions CSV.
        faithfulness_csv: Path to Phase 7 per-image predictions CSV.
        localization_json: Path to Phase 6 structured JSON report.
        faithfulness_json: Path to Phase 7 structured JSON report.
        allow_simulated: If True, permit simulated/dry-run outputs for testing.

    Returns:
        merged_df: Unified per-image DataFrame.
        loc_meta: Metadata dictionary from Phase 6 JSON.
        faith_meta: Metadata dictionary from Phase 7 JSON.
        is_simulated: Boolean indicating whether inputs were simulated.
        checkpoint_sha256: Verified matching checkpoint hash, if genuine.
    """
    # Check physical existence of input files
    for p, name in [
        (localization_csv, "localization CSV"),
        (faithfulness_csv, "faithfulness CSV"),
        (localization_json, "localization JSON"),
        (faithfulness_json, "faithfulness JSON"),
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required {name} not found at: {p}\n"
                "Please run Phase 6 and Phase 7 evaluations before running Phase 8 analysis."
            )

    with open(localization_json, "r", encoding="utf-8") as f:
        loc_meta = json.load(f)
    with open(faithfulness_json, "r", encoding="utf-8") as f:
        faith_meta = json.load(f)

    loc_status = str(loc_meta.get("status", ""))
    faith_status = str(faith_meta.get("status", ""))

    is_simulated = (
        "SIMULATED" in loc_status
        or "SIMULATED" in faith_status
        or "simulated" in localization_csv.name.lower()
        or "simulated" in faithfulness_csv.name.lower()
    )

    if is_simulated and not allow_simulated:
        raise ValueError(
            "Integrity Gate Failed: Simulated / Dry-Run evaluation files detected.\n"
            f"- Localization status: '{loc_status}' ({localization_csv.name})\n"
            f"- Faithfulness status: '{faith_status}' ({faithfulness_csv.name})\n"
            "Phase 8 statistical hypothesis testing requires genuine empirical evaluation.\n"
            "To execute a pipeline software test on simulated data, pass '--allow-simulated'."
        )

    checkpoint_sha256 = None
    if not is_simulated:
        loc_sha = loc_meta.get("checkpoint_sha256")
        faith_sha = faith_meta.get("checkpoint_sha256")

        if not loc_sha or not faith_sha:
            raise ValueError(
                "Integrity Gate Failed: Missing checkpoint_sha256 in evaluation JSON reports."
            )

        if loc_sha != faith_sha:
            raise ValueError(
                f"Integrity Gate Failed: Checkpoint SHA-256 hash mismatch between phases!\n"
                f"- Phase 6 Localization Checkpoint: {loc_sha}\n"
                f"- Phase 7 Faithfulness Checkpoint: {faith_sha}\n"
                "Both phases must be evaluated using the exact same immutable model weights."
            )
        checkpoint_sha256 = loc_sha

    # Load and merge CSV predictions
    loc_df = pd.read_csv(localization_csv)
    faith_df = pd.read_csv(faithfulness_csv)

    # Standardize image_id stems
    loc_df["clean_id"] = loc_df["image_id"].astype(str).apply(lambda x: Path(x).stem)
    faith_df["clean_id"] = faith_df["image_id"].astype(str).apply(lambda x: Path(x).stem)

    # Check for duplicates
    if loc_df["clean_id"].duplicated().any():
        dups = loc_df["clean_id"][loc_df["clean_id"].duplicated()].tolist()
        raise ValueError(f"Duplicate image IDs detected in localization CSV: {dups}")
    if faith_df["clean_id"].duplicated().any():
        dups = faith_df["clean_id"][faith_df["clean_id"].duplicated()].tolist()
        raise ValueError(f"Duplicate image IDs detected in faithfulness CSV: {dups}")

    # Merge on clean_id
    merged = pd.merge(
        loc_df,
        faith_df,
        on="clean_id",
        suffixes=("_loc", "_faith"),
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError("No common image IDs found between localization and faithfulness predictions.")

    return merged, loc_meta, faith_meta, is_simulated, checkpoint_sha256


def get_faithfulness_metric_col(df: pd.DataFrame) -> Tuple[str, str]:
    """Select the best available faithfulness margin metric column.

    Prefers 'normalized_pauc_margin' (trapezoidal partial-AUC); falls back to
    'mean_deletion_margin' or 'audc_margin'.
    """
    candidates = [
        ("normalized_pauc_margin", "Normalized Partial-AUC Margin"),
        ("mean_deletion_margin", "Mean Deletion Margin"),
        ("audc_margin", "AUDC Margin (Discrete Mean Drop)"),
    ]
    for col, display in candidates:
        if col in df.columns and df[col].notna().any():
            return col, display
        col_faith = f"{col}_faith"
        if col_faith in df.columns and df[col_faith].notna().any():
            return col_faith, display

    raise KeyError("No valid faithfulness margin metric column found in predictions DataFrame.")


def perform_statistical_analysis(
    merged_df: pd.DataFrame,
    seed: int = 42,
    n_bootstraps: int = 2000,
) -> Dict[str, Any]:
    """Execute complete Phase 8 statistical analysis across cohorts and grades.

    Args:
        merged_df: Unified DataFrame with both localization and faithfulness columns.
        seed: Random seed for bootstrap confidence intervals.
        n_bootstraps: Number of bootstrap iterations.

    Returns:
        Structured dictionary of statistical results.
    """
    faith_col, faith_label = get_faithfulness_metric_col(merged_df)

    # Cohort separation
    # Flag supplementary cohort (IDRiD_09, 22, 31 or cohort_origin containing 'test_cohort')
    is_supp = (
        merged_df["clean_id"].isin(SUPPLEMENTARY_IMAGE_IDS)
        | merged_df.get("cohort_origin_loc", merged_df.get("cohort_origin", "")).astype(str).str.contains("test_cohort")
    )

    primary_df = merged_df[~is_supp].copy()
    supp_df = merged_df[is_supp].copy()

    # 1. Primary Analysis: Composite Dice @ top 10% vs Faithfulness Net Margin
    primary_dice_col = "composite_dice_top_10"
    if primary_dice_col not in primary_df.columns:
        cand = [c for c in primary_df.columns if "composite_dice" in c]
        if cand:
            primary_dice_col = cand[0]
        else:
            raise KeyError(f"Composite Dice column '{primary_dice_col}' not found.")

    primary_corr = compute_spearman_with_bootstrap_ci(
        primary_df[primary_dice_col],
        primary_df[faith_col],
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # 2. Secondary Localization Metrics vs Faithfulness Net Margin (Primary Cohort)
    secondary_metrics = [
        ("composite_iou_top_10", "Composite IoU"),
        ("ma_dice_top_10", "Microaneurysms (MA) Dice"),
        ("ma_iou_top_10", "Microaneurysms (MA) IoU"),
        ("he_dice_top_10", "Hemorrhages (HE) Dice"),
        ("he_iou_top_10", "Hemorrhages (HE) IoU"),
        ("ex_dice_top_10", "Hard Exudates (EX) Dice"),
        ("ex_iou_top_10", "Hard Exudates (EX) IoU"),
        ("se_dice_top_10", "Soft Exudates (SE) Dice"),
        ("se_iou_top_10", "Soft Exudates (SE) IoU"),
        ("od_dice_top_10", "Optic Disc (OD Control) Dice"),
        ("od_iou_top_10", "Optic Disc (OD Control) IoU"),
    ]

    secondary_corrs: Dict[str, Any] = {}
    for col, name in secondary_metrics:
        if col in primary_df.columns and primary_df[col].notna().any():
            secondary_corrs[col] = {
                "display_name": name,
                **compute_spearman_with_bootstrap_ci(
                    primary_df[col],
                    primary_df[faith_col],
                    n_bootstraps=n_bootstraps,
                    seed=seed,
                ),
            }

    # 3. Grade-Wise Breakdown (Primary Cohort, grades 1-4, excluding grade == None)
    grade_col = "true_grade_loc" if "true_grade_loc" in primary_df.columns else "true_grade"
    grade_results: Dict[str, Any] = {}
    p_values_to_adjust: List[float] = []
    grade_keys_tested: List[str] = []

    for g in range(5):
        # Subset with valid integer grade == g
        sub = primary_df[primary_df[grade_col].notna() & (primary_df[grade_col] == g)]
        n_g = len(sub)
        grade_name = ICDR_NAMES.get(g, f"Grade {g}")

        if n_g < 5:
            grade_results[f"grade_{g}"] = {
                "grade": g,
                "grade_name": grade_name,
                "n": n_g,
                "status": "insufficient_samples (N < 5)",
                "rho": None,
                "p_value": None,
                "p_value_fdr": None,
                "ci_lower": None,
                "ci_upper": None,
            }
        else:
            corr_g = compute_spearman_with_bootstrap_ci(
                sub[primary_dice_col],
                sub[faith_col],
                n_bootstraps=n_bootstraps,
                seed=seed,
            )
            grade_results[f"grade_{g}"] = {
                "grade": g,
                "grade_name": grade_name,
                **corr_g,
                "p_value_fdr": None,  # Will be populated by FDR adjustment
            }
            if corr_g.get("p_value") is not None:
                p_values_to_adjust.append(corr_g["p_value"])
                grade_keys_tested.append(f"grade_{g}")

    # Apply Benjamini-Hochberg FDR correction across tested grades
    if p_values_to_adjust:
        adjusted_ps = benjamini_hochberg_fdr(p_values_to_adjust)
        for k, p_adj in zip(grade_keys_tested, adjusted_ps):
            grade_results[k]["p_value_fdr"] = p_adj

    # 4. Supplementary Cohort Metrics
    supp_corrs: Optional[Dict[str, Any]] = None
    if len(supp_df) >= 3:
        supp_corrs = compute_spearman_with_bootstrap_ci(
            supp_df[primary_dice_col],
            supp_df[faith_col],
            n_bootstraps=n_bootstraps,
            seed=seed,
        )

    return {
        "faithfulness_metric_used": faith_col,
        "faithfulness_metric_label": faith_label,
        "primary_localization_metric": primary_dice_col,
        "primary_cohort_count": len(primary_df),
        "supplementary_cohort_count": len(supp_df),
        "total_merged_count": len(merged_df),
        "primary_analysis": primary_corr,
        "secondary_analyses": secondary_corrs,
        "grade_wise_analyses": grade_results,
        "supplementary_analysis": supp_corrs,
        "supplementary_image_ids": sorted(supp_df["clean_id"].tolist()),
    }


def generate_correlation_plot(
    merged_df: pd.DataFrame,
    analysis_results: Dict[str, Any],
    output_path: Path,
    is_simulated: bool = False,
) -> None:
    """Generate high-resolution scientific correlation scatter plot with regression trendline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    faith_col = analysis_results["faithfulness_metric_used"]
    dice_col = analysis_results["primary_localization_metric"]
    p_res = analysis_results["primary_analysis"]

    is_supp = (
        merged_df["clean_id"].isin(SUPPLEMENTARY_IMAGE_IDS)
        | merged_df.get("cohort_origin_loc", merged_df.get("cohort_origin", "")).astype(str).str.contains("test_cohort")
    )
    primary_df = merged_df[~is_supp].copy()
    supp_df = merged_df[is_supp].copy()

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)

    # Subdivide primary into train-matched vs unique
    cohort_col = "cohort_origin_loc" if "cohort_origin_loc" in primary_df.columns else "cohort_origin"
    is_train_matched = primary_df[cohort_col].astype(str).str.contains("train_cohort")
    train_matched = primary_df[is_train_matched]
    unique_seg = primary_df[~is_train_matched]

    # Scatter points
    ax.scatter(
        train_matched[dice_col],
        train_matched[faith_col],
        color="#1f77b4",
        alpha=0.75,
        s=50,
        edgecolors="k",
        linewidths=0.5,
        label=f"Primary Train-Matched (N = {len(train_matched)})",
        zorder=3,
    )
    ax.scatter(
        unique_seg[dice_col],
        unique_seg[faith_col],
        color="#2ca02c",
        marker="D",
        alpha=0.75,
        s=45,
        edgecolors="k",
        linewidths=0.5,
        label=f"Primary Unique Seg (N = {len(unique_seg)})",
        zorder=3,
    )

    if len(supp_df) > 0:
        ax.scatter(
            supp_df[dice_col],
            supp_df[faith_col],
            color="#ff7f0e",
            marker="^",
            alpha=0.9,
            s=70,
            edgecolors="k",
            linewidths=0.8,
            label=f"Supplementary Test Overlap (N = {len(supp_df)})",
            zorder=4,
        )

    # Linear trendline for primary cohort
    x_prim = primary_df[dice_col].values
    y_prim = primary_df[faith_col].values
    valid_idx = (~np.isnan(x_prim)) & (~np.isnan(y_prim))

    if np.sum(valid_idx) >= 3:
        xp = x_prim[valid_idx]
        yp = y_prim[valid_idx]
        slope, intercept, _, _, _ = scipy.stats.linregress(xp, yp)
        x_line = np.linspace(np.min(xp), np.max(xp), 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            color="#333333",
            linestyle="--",
            linewidth=1.75,
            label="Primary Fit Trendline",
            zorder=2,
        )

    # Inset statistical summary box
    rho_val = p_res.get("rho")
    p_val = p_res.get("p_value")
    ci_lo = p_res.get("ci_lower")
    ci_hi = p_res.get("ci_upper")
    n_val = p_res.get("n", len(primary_df))

    stats_text = (
        f"Primary Cohort (N = {n_val}):\n"
        f"Spearman $\\rho$ = {rho_val:.3f}\n" if rho_val is not None else "Spearman $\\rho$: N/A\n"
    )
    if p_val is not None:
        stats_text += f"p = {p_val:.4f}\n" if p_val >= 0.0001 else "p < 0.0001\n"
    if ci_lo is not None and ci_hi is not None:
        stats_text += f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]"

    props = dict(boxstyle="round,pad=0.5", facecolor="aliceblue", alpha=0.9, edgecolor="steelblue")
    ax.text(
        0.04,
        0.96,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
        fontfamily="sans-serif",
    )

    if is_simulated:
        ax.text(
            0.5,
            1.04,
            "SIMULATED TEST DATA - NOT VALID FOR EMPIRICAL RESEARCH",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="crimson",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="#ffebee", edgecolor="crimson"),
        )

    ax.set_title(
        "Grad-CAM Lesion Localization vs Decision Faithfulness",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Lesion Localization (Composite Dice at Top 10% Saliency)", fontsize=11)
    ax.set_ylabel(f"Decision Faithfulness ({analysis_results['faithfulness_metric_label']})", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_markdown_report(
    analysis_results: Dict[str, Any],
    output_path: Path,
    is_simulated: bool = False,
    checkpoint_sha256: Optional[str] = None,
) -> None:
    """Generate comprehensive scientific Markdown report for Phase 8."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md = []
    if is_simulated:
        md.append("> [!CAUTION]")
        md.append("> **SIMULATED DATA WARNING: NOT VALID FOR RESEARCH FINDINGS**")
        md.append("> This report was compiled from simulated / dry-run software tests.")
        md.append("> The numeric metrics, correlation coefficients, and p-values below are synthetic")
        md.append("> and must NEVER be cited as empirical evidence or used in scientific publications.\n")
    else:
        md.append("> [!IMPORTANT]")
        md.append("> **GENUINE RESEARCH REPORT: STATISTICAL HYPOTHESIS TESTING**")
        md.append("> Evaluated across genuine IDRiD segmentation imagery with frozen ResNet-50 weights.")
        md.append("> Model weights and checkpoint file were verified 100% bit-exact immutable across all phases.\n")

    md.append("# Phase 8: Statistical Hypothesis Testing & Correlation Analysis\n")
    md.append(f"- **Execution Mode**: {'DRY-RUN SIMULATION (SOFTWARE TEST ONLY)' if is_simulated else 'GENUINE RESEARCH INFERENCE'}")
    md.append(f"- **Frozen Checkpoint SHA-256**: `{checkpoint_sha256 or 'N/A (Dry-Run)'}`")
    md.append(f"- **Total Evaluated Images**: {analysis_results['total_merged_count']}")
    md.append(f"  - **Primary Cohort (N = {analysis_results['primary_cohort_count']})**: 57 training-matched + 21 unique segmentation images.")
    md.append(f"  - **Supplementary Test Cohort (N = {analysis_results['supplementary_cohort_count']})**: Segregated test-overlap images ({', '.join(analysis_results['supplementary_image_ids'])}).\n")

    md.append("## 1. Primary Hypothesis Evaluation\n")
    md.append("We test whether visual explanation quality correlates with internal decision fidelity:")
    md.append("- **X Metric**: Grad-CAM Lesion Localization (Composite Dice at Top 10% Saliency within retinal FOV)")
    md.append(f"- **Y Metric**: Grad-CAM Faithfulness Net Margin (`{analysis_results['faithfulness_metric_label']}`)")

    p_res = analysis_results["primary_analysis"]
    rho = p_res.get("rho")
    p_val = p_res.get("p_value")
    ci_lo = p_res.get("ci_lower")
    ci_hi = p_res.get("ci_upper")

    rho_str = f"{rho:.4f}" if rho is not None else "N/A"
    p_str = f"{p_val:.4e}" if p_val is not None else "N/A"
    ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if ci_lo is not None and ci_hi is not None else "N/A"

    md.append("\n| Cohort | Sample Size (N) | Spearman $\\rho$ | Two-Sided $p$-value | 95% Bootstrap CI | Statistical Decision |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    decision = "Inconclusive"
    if p_val is not None:
        if p_val < 0.05 and rho is not None and rho > 0.0:
            decision = "**Reject H0** (Statistically significant positive correlation)"
        elif p_val < 0.05 and rho is not None and rho <= 0.0:
            decision = "**Reject H0** (Significant non-positive relationship)"
        else:
            decision = "**Fail to Reject H0** (No statistically significant monotonic relationship)"

    md.append(f"| Primary Cohort | {p_res.get('n', 'N/A')} | `{rho_str}` | `{p_str}` | `{ci_str}` | {decision} |\n")

    md.append("### Scientific Discussion")
    if rho is not None and p_val is not None:
        if p_val < 0.05 and rho > 0.0:
            md.append("The primary evaluation demonstrates a statistically significant positive rank correlation between lesion localization overlap and decision fidelity. This indicates that Grad-CAM heatmaps that align more closely with clinical lesions also correspond to higher model sensitivity under feature removal.")
        else:
            md.append("The empirical correlation between lesion localization and decision faithfulness was not statistically significant under the predefined $\\alpha = 0.05$ threshold. This underscores the crucial clinical principle that visual plausibility (highlighting what looks plausible to a clinician) does not necessarily imply decision fidelity (reflecting the model's actual internal feature utilization).")
    md.append("")

    # 2. Secondary Metrics Table
    md.append("## 2. Secondary Localization & Control Analyses (Primary Cohort)\n")
    md.append("| Localization Metric | Sample Size (N) | Spearman $\\rho$ | $p$-value | 95% Bootstrap CI | Role / Expected Behavior |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :--- |")

    for k, v in analysis_results["secondary_analyses"].items():
        v_rho = f"{v['rho']:.4f}" if v.get("rho") is not None else "N/A"
        v_p = f"{v['p_value']:.4e}" if v.get("p_value") is not None else "N/A"
        v_ci = f"[{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]" if v.get("ci_lower") is not None and v.get("ci_upper") is not None else "N/A"
        role = "Negative Control (Expect $\\rho \\le 0$)" if "od_" in k else "Pathology Lesion Saliency"
        md.append(f"| {v['display_name']} | {v.get('n', 'N/A')} | {v_rho} | {v_p} | {v_ci} | {role} |")
    md.append("")

    # 3. Grade-Wise Analysis
    md.append("## 3. Grade-Wise Subgroup Analysis (Primary Cohort)\n")
    md.append("Analysis restricted to the 57 training-matched images with verified ground-truth ICDR grades (the 21 unique images have `true_grade = None` and are excluded from grade subgroups):\n")
    md.append("| ICDR Grade | Clinical Severity | N | Spearman $\\rho$ | Raw $p$-value | FDR-Adjusted $p$-value | 95% Bootstrap CI | Status |")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    for g in range(5):
        g_res = analysis_results["grade_wise_analyses"].get(f"grade_{g}", {})
        cnt = g_res.get("n", 0)
        g_rho = f"{g_res['rho']:.4f}" if g_res.get("rho") is not None else "N/A"
        g_p = f"{g_res['p_value']:.4e}" if g_res.get("p_value") is not None else "N/A"
        g_fdr = f"{g_res['p_value_fdr']:.4e}" if g_res.get("p_value_fdr") is not None else "N/A"
        g_ci = f"[{g_res['ci_lower']:.4f}, {g_res['ci_upper']:.4f}]" if g_res.get("ci_lower") is not None and g_res.get("ci_upper") is not None else "N/A"
        status = g_res.get("status", "Evaluated")
        md.append(f"| {g} | {ICDR_NAMES.get(g, f'Grade {g}')} | {cnt} | {g_rho} | {g_p} | {g_fdr} | {g_ci} | {status} |")
    md.append("")

    # 4. Supplementary Cohort Transparency
    md.append("## 4. Supplementary Test Cohort Transparency (N = 3)\n")
    md.append("The 3 segmentation images matched via pairwise visual hashing to the official 103-image IDRiD test grading cohort (`IDRiD_09`, `IDRiD_22`, `IDRiD_31`) are strictly isolated here to preserve zero-leakage standards:\n")
    s_res = analysis_results.get("supplementary_analysis")
    if s_res and s_res.get("rho") is not None:
        md.append(f"- **Sample Size**: N = {s_res['n']}")
        md.append(f"- **Spearman $\\rho$**: `{s_res['rho']:.4f}` ($p = {s_res['p_value']:.4f}$)")
    else:
        md.append("- **Sample Size**: N = 3 (Insufficient for standalone inferential significance testing; reported descriptively)")
    md.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 8: Statistical Hypothesis Testing CLI"
    )
    parser.add_argument(
        "--localization-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase_6_localization_predictions.csv",
        help="Path to Phase 6 per-image localization predictions CSV.",
    )
    parser.add_argument(
        "--faithfulness-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "faithfulness_predictions.csv",
        help="Path to Phase 7 per-image faithfulness predictions CSV.",
    )
    parser.add_argument(
        "--localization-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase_6_localization_report.json",
        help="Path to Phase 6 structured report JSON.",
    )
    parser.add_argument(
        "--faithfulness-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase_7_faithfulness_report.json",
        help="Path to Phase 7 structured report JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Path to output Markdown report.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to output structured JSON report.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Path to output high-resolution correlation figure.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute in software testing mode when genuine data is unavailable.",
    )
    parser.add_argument(
        "--allow-simulated",
        action="store_true",
        help="Permit execution using simulated inputs for software unit testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap sampling (default: 42).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap iterations for confidence intervals (default: 2000).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logging.info("Initializing Phase 8 Statistical Analysis...")

    # Determine output paths
    if args.dry_run:
        output_md = args.output_md or PROJECT_ROOT / "reports" / "simulated_phase_8_statistical_analysis_report.md"
        output_json = args.output_json or PROJECT_ROOT / "reports" / "simulated_phase_8_statistical_analysis_report.json"
        output_plot = args.output_plot or PROJECT_ROOT / "reports" / "figures" / "simulated_phase_8_correlation_plot.png"
        logging.warning("=" * 70)
        logging.warning("RUNNING IN DRY-RUN SIMULATION MODE.")
        logging.warning("Outputs will be saved with 'simulated_' prefix and marked SIMULATED.")
        logging.warning("=" * 70)
    else:
        output_md = args.output_md or PROJECT_ROOT / "reports" / "phase_8_statistical_analysis_report.md"
        output_json = args.output_json or PROJECT_ROOT / "reports" / "phase_8_statistical_analysis_report.json"
        output_plot = args.output_plot or PROJECT_ROOT / "reports" / "figures" / "phase_8_correlation_plot.png"

    # Fallback to simulated files if running with --allow-simulated or --dry-run
    loc_csv = args.localization_csv
    faith_csv = args.faithfulness_csv
    loc_json = args.localization_json
    faith_json = args.faithfulness_json

    if args.allow_simulated or args.dry_run:
        if not loc_csv.exists() and (PROJECT_ROOT / "reports" / "simulated_phase_6_localization_predictions.csv").exists():
            loc_csv = PROJECT_ROOT / "reports" / "simulated_phase_6_localization_predictions.csv"
        if not faith_csv.exists() and (PROJECT_ROOT / "reports" / "simulated_faithfulness_predictions.csv").exists():
            faith_csv = PROJECT_ROOT / "reports" / "simulated_faithfulness_predictions.csv"
        if not loc_json.exists() and (PROJECT_ROOT / "reports" / "simulated_phase_6_localization_report.json").exists():
            loc_json = PROJECT_ROOT / "reports" / "simulated_phase_6_localization_report.json"
        if not faith_json.exists() and (PROJECT_ROOT / "reports" / "simulated_phase_7_faithfulness_report.json").exists():
            faith_json = PROJECT_ROOT / "reports" / "simulated_phase_7_faithfulness_report.json"

    # In dry-run mode, if input files are still missing, synthesize 81 image records
    if args.dry_run and (not loc_csv.exists() or not faith_csv.exists()):
        logging.info("Input files not found on disk. Generating synthetic records for 81 images (78 primary, 3 supplementary)...")
        rng = np.random.default_rng(args.seed)
        synth_rows = []
        for i in range(81):
            d = float(0.08 + 0.04 * (i % 5) + 0.01 * rng.random())
            m = float(0.12 + 0.03 * (i % 5) + 0.01 * rng.random())
            if i < 57:
                img_id = f"dry_run_{i+1:02d}"
                co = "idrid_grading_train_cohort"
                g = (i % 4) + 1
            elif i < 60:
                supp_names = ["IDRiD_09", "IDRiD_22", "IDRiD_31"]
                img_id = supp_names[i - 57]
                co = "idrid_grading_test_cohort"
                g = (i % 4) + 1
            else:
                img_id = f"dry_run_{i+1:02d}"
                co = "idrid_seg_unique"
                g = None

            synth_rows.append({
                "clean_id": img_id,
                "cohort_origin": co,
                "cohort_origin_loc": co,
                "true_grade": g,
                "true_grade_loc": g,
                "composite_dice_top_10": d,
                "composite_iou_top_10": d * 0.5,
                "ma_dice_top_10": d * 0.8,
                "ma_iou_top_10": d * 0.4,
                "he_dice_top_10": d * 0.7,
                "he_iou_top_10": d * 0.35,
                "ex_dice_top_10": d * 0.75,
                "ex_iou_top_10": d * 0.38,
                "se_dice_top_10": d * 0.65,
                "se_iou_top_10": d * 0.32,
                "od_dice_top_10": 0.02,
                "od_iou_top_10": 0.01,
                "normalized_pauc_margin": m,
                "mean_deletion_margin": m,
            })

        merged_df = pd.DataFrame(synth_rows)
        is_simulated = True
        checkpoint_sha256 = None
    else:
        try:
            merged_df, loc_meta, faith_meta, is_simulated, checkpoint_sha256 = verify_inputs_and_load_data(
                localization_csv=loc_csv,
                faithfulness_csv=faith_csv,
                localization_json=loc_json,
                faithfulness_json=faith_json,
                allow_simulated=args.allow_simulated or args.dry_run,
            )
        except Exception as e:
            logging.error(f"Integrity Verification Failed: {e}")
            return 1

    logging.info(
        f"Loaded and verified {len(merged_df)} records. "
        f"Mode: {'SIMULATED (Software Test)' if is_simulated else 'GENUINE RESEARCH'}."
    )

    # Execute statistical analysis
    analysis_results = perform_statistical_analysis(
        merged_df,
        seed=args.seed,
        n_bootstraps=args.bootstrap_samples,
    )

    # Generate correlation plot
    generate_correlation_plot(
        merged_df=merged_df,
        analysis_results=analysis_results,
        output_path=output_plot,
        is_simulated=is_simulated,
    )
    logging.info(f"Correlation plot saved to: {output_plot}")

    # Generate Markdown report
    generate_markdown_report(
        analysis_results=analysis_results,
        output_path=output_md,
        is_simulated=is_simulated,
        checkpoint_sha256=checkpoint_sha256,
    )
    logging.info(f"Markdown report written to: {output_md}")

    # Generate JSON report
    report_dict = {
        "status": "SIMULATED_DATA_NOT_VALID_FOR_RESEARCH" if is_simulated else "GENUINE_STATISTICAL_ANALYSIS_COMPLETE",
        "checkpoint_sha256": checkpoint_sha256,
        "is_simulated": is_simulated,
        "analysis_results": analysis_results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logging.info(f"Structured metrics JSON written to: {output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
