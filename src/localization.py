"""Localization Evaluation Module for Explainable AI (Phase 6).

Evaluates spatial alignment between Grad-CAM saliency heatmaps and expert-annotated
retinal lesion ground truths (Microaneurysms, Haemorrhages, Hard Exudates, Soft Exudates).

Provides:
- Pixel-level Intersection over Union (IoU / Jaccard Index) and Dice Similarity Coefficient (DSC).
- Rigorous handling of edge cases (e.g., Grade 0 absence of lesions, unannotated lesion masks).
- Multi-threshold evaluation across saliency cutoffs (top 5%, 10%, 20% salient area).
- Lesion-specific overlap (MA, HE, EX, SE) and composite overlap (MA U HE U EX U SE).
- Optic Disc (OD) anatomical control overlap.
- Grade-wise and cohort-stratified aggregation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.explainability import binarize_heatmap

logger = logging.getLogger("localization")

LESION_CODES = ["MA", "HE", "EX", "SE"]
CONTROL_CODES = ["OD"]


def compute_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    empty_value: Optional[float] = None,
) -> Optional[float]:
    """Compute Intersection over Union (IoU / Jaccard Index) between two binary masks.

    Formula: IoU = |pred ∩ gt| / |pred ∪ gt|

    Args:
        pred_mask: Predicted binary mask array (H, W) in {0, 1}.
        gt_mask: Ground-truth binary mask array (H, W) in {0, 1}.
        empty_value: Value to return if ground-truth mask is all-zeros (e.g. Grade 0).
                     Default None (treated as N/A per clinical specification).

    Returns:
        float IoU in [0.0, 1.0], or empty_value if gt_mask has no positive pixels.

    Raises:
        ValueError: If array dimensions do not match.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Mask shape mismatch: {pred_mask.shape} vs {gt_mask.shape}")

    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0

    gt_sum = int(np.sum(gt_bool))

    # If ground-truth contains zero positive pixels (e.g. no lesions present),
    # overlap metric is mathematically undefined / not applicable (N/A)
    if gt_sum == 0:
        return empty_value

    intersection = int(np.sum(pred_bool & gt_bool))
    union = int(np.sum(pred_bool | gt_bool))

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_dice(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    empty_value: Optional[float] = None,
) -> Optional[float]:
    """Compute Dice Similarity Coefficient (DSC / F1 score) between two binary masks.

    Formula: Dice = (2 * |pred ∩ gt|) / (|pred| + |gt|)

    Args:
        pred_mask: Predicted binary mask array (H, W) in {0, 1}.
        gt_mask: Ground-truth binary mask array (H, W) in {0, 1}.
        empty_value: Value to return if ground-truth mask is all-zeros (e.g. Grade 0).
                     Default None (treated as N/A per clinical specification).

    Returns:
        float Dice in [0.0, 1.0], or empty_value if gt_mask has no positive pixels.

    Raises:
        ValueError: If array dimensions do not match.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Mask shape mismatch: {pred_mask.shape} vs {gt_mask.shape}")

    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0

    gt_sum = int(np.sum(gt_bool))

    if gt_sum == 0:
        return empty_value

    intersection = int(np.sum(pred_bool & gt_bool))
    total_elements = int(np.sum(pred_bool)) + gt_sum

    if total_elements == 0:
        return 1.0

    return float((2.0 * intersection) / total_elements)


def evaluate_single_image_localization(
    heatmap: np.ndarray,
    ground_truth_masks: Dict[str, np.ndarray],
    image_id: str,
    grade: Optional[int] = None,
    thresholds: Optional[List[float]] = None,
    annotation_status: Optional[Dict[str, bool]] = None,
    cohort_origin: str = "idrid_seg_unique",
    fov_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate localization overlap of a Grad-CAM heatmap across all lesion masks.

    Evaluates across multiple percentile thresholds (e.g. top 5%, 10%, 20%).
    When fov_mask is provided, top percentile saliency selection is strictly calculated
    within the valid retinal FOV, preventing spurious border activations from entering
    the candidate set.
    Handles Grade 0 by designating all lesion overlaps as None / N/A.
    For images without verified grading labels (e.g. 21 unique segmentation images),
    grade is set to None (unavailable) and excluded from grade-wise stratification
    while still evaluated for overall lesion localization.

    Args:
        heatmap: 2D normalized Grad-CAM heatmap (H, W) with float values in [0.0, 1.0].
        ground_truth_masks: Dictionary mapping structure codes ('MA', 'HE', 'EX', 'SE',
                            'composite', 'OD') to binary arrays (H, W).
        image_id: Unique image identifier string.
        grade: Ground-truth ICDR DR grade (0-4), or None if grade is unavailable.
        thresholds: List of percentile cutoffs (default: [0.05, 0.10, 0.20]).
        annotation_status: Optional dict flagging whether each lesion type was annotated
                           (e.g., SE is unannotated in 41 IDRiD images).
        cohort_origin: Dataset cohort origin ('idrid_grading_train_cohort',
                       'idrid_grading_test_cohort', or 'idrid_seg_unique').
        fov_mask: Optional binary mask (H, W) defining valid retinal field of view.

    Returns:
        Dictionary containing structured localization scores for each threshold & lesion.
    """
    thresholds = thresholds or [0.05, 0.10, 0.20]
    annotation_status = annotation_status or {}

    is_grade_0 = (int(grade) == 0) if grade is not None else False

    if fov_mask is not None:
        if fov_mask.shape != heatmap.shape:
            raise ValueError(f"FOV mask shape {fov_mask.shape} does not match heatmap shape {heatmap.shape}")
        if not set(np.unique(fov_mask)).issubset({0, 1, 255}):
            raise ValueError("FOV mask must be strictly binary {0, 1} or {0, 255}")
        fov_pixels = int(np.sum(fov_mask > 0))
        if fov_pixels == 0:
            raise ValueError("FOV mask is empty (0 positive pixels).")
        fov_aware = True
    else:
        fov_pixels = heatmap.size
        fov_aware = False

    eval_result: Dict[str, Any] = {
        "image_id": image_id,
        "grade": int(grade) if grade is not None else None,
        "is_grade_0": is_grade_0,
        "cohort_origin": cohort_origin,
        "thresholds": thresholds,
        "fov_aware": fov_aware,
        "fov_pixel_count": fov_pixels,
        "metrics_by_threshold": {},
    }

    # Evaluate each saliency cutoff threshold
    for t in thresholds:
        t_key = f"top_{int(round(t * 100))}_pct"
        pred_binary = binarize_heatmap(heatmap, threshold=t, method="percentile", fov_mask=fov_mask)
        pred_pixel_count = int(np.sum(pred_binary))

        t_metrics: Dict[str, Any] = {
            "salient_pixel_count": pred_pixel_count,
            "lesions": {},
            "control": {},
        }

        # 1. Evaluate individual pathological lesions: MA, HE, EX, SE
        for code in LESION_CODES:
            is_annotated = annotation_status.get(code, True)
            mask = ground_truth_masks.get(code)

            if is_grade_0 or not is_annotated or mask is None:
                iou = None
                dice = None
                gt_pixels = 0 if mask is None else int(np.sum(mask > 0))
            else:
                gt_pixels = int(np.sum(mask > 0))
                iou = compute_iou(pred_binary, mask, empty_value=None)
                dice = compute_dice(pred_binary, mask, empty_value=None)

            t_metrics["lesions"][code] = {
                "iou": iou,
                "dice": dice,
                "gt_pixel_count": gt_pixels,
                "annotated": is_annotated and not is_grade_0,
            }

        # 2. Evaluate composite lesion mask (MA U HE U EX U SE)
        comp_mask = ground_truth_masks.get("composite")
        if is_grade_0 or comp_mask is None:
            comp_iou = None
            comp_dice = None
            comp_gt_pixels = 0 if comp_mask is None else int(np.sum(comp_mask > 0))
        else:
            comp_gt_pixels = int(np.sum(comp_mask > 0))
            comp_iou = compute_iou(pred_binary, comp_mask, empty_value=None)
            comp_dice = compute_dice(pred_binary, comp_mask, empty_value=None)

        t_metrics["lesions"]["composite"] = {
            "iou": comp_iou,
            "dice": comp_dice,
            "gt_pixel_count": comp_gt_pixels,
            "annotated": not is_grade_0,
        }

        # 3. Evaluate normal anatomical control: Optic Disc (OD)
        od_mask = ground_truth_masks.get("OD")
        if od_mask is not None:
            od_gt_pixels = int(np.sum(od_mask > 0))
            od_iou = compute_iou(pred_binary, od_mask, empty_value=0.0)
            od_dice = compute_dice(pred_binary, od_mask, empty_value=0.0)
        else:
            od_iou = None
            od_dice = None
            od_gt_pixels = 0

        t_metrics["control"]["OD"] = {
            "iou": od_iou,
            "dice": od_dice,
            "gt_pixel_count": od_gt_pixels,
        }

        eval_result["metrics_by_threshold"][t_key] = t_metrics

    return eval_result


def aggregate_localization_scores(
    results: List[Dict[str, Any]],
    baseline_threshold_key: str = "top_10_pct",
) -> Dict[str, Any]:
    """Aggregate per-image localization results overall, by grade, and by cohort.

    Properly handles Grade 0 cases as N/A so they do not artificially penalize
    mean IoU/Dice averages across diseased images.

    Args:
        results: List of evaluation dictionaries from evaluate_single_image_localization().
        baseline_threshold_key: Primary threshold key to compute detailed stats for.

    Returns:
        Structured summary with mean, median, std, and grade-wise breakdowns.
    """
    total_images = len(results)
    grade_counts: Dict[Union[int, str], int] = {g: 0 for g in range(5)}
    grade_counts["unavailable"] = 0
    for r in results:
        g = r.get("grade")
        if g is not None and isinstance(g, int) and 0 <= g <= 4:
            grade_counts[g] = grade_counts.get(g, 0) + 1
        else:
            grade_counts["unavailable"] = grade_counts.get("unavailable", 0) + 1

    summary: Dict[str, Any] = {
        "total_images": total_images,
        "grade_distribution": grade_counts,
        "baseline_threshold": baseline_threshold_key,
        "primary_cohort": {},
        "supplementary_test_cohort": {},
        "overall_composite": {},
        "grade_wise_composite": {},
        "cohort_origin_composite": {},
        "lesion_breakdown": {},
        "optic_disc_control": {},
    }

    # Extract all composite IoUs and Dices for baseline threshold (excluding Grade 0 N/As)
    comp_ious = []
    comp_dices = []
    grade_data: Dict[int, Dict[str, List[float]]] = {
        g: {"iou": [], "dice": []} for g in range(1, 5)
    }
    cohort_data: Dict[str, Dict[str, List[float]]] = {}

    # Separate Primary (N=78 non-overlapping) and Supplementary Test Overlap (N=3)
    primary_comp_ious: List[float] = []
    primary_comp_dices: List[float] = []
    primary_grade_data: Dict[int, Dict[str, List[float]]] = {
        g: {"iou": [], "dice": []} for g in range(1, 5)
    }
    primary_lesion_ious: Dict[str, List[float]] = {code: [] for code in LESION_CODES}
    primary_lesion_dices: Dict[str, List[float]] = {code: [] for code in LESION_CODES}
    primary_od_ious: List[float] = []
    primary_od_dices: List[float] = []

    supp_comp_ious: List[float] = []
    supp_comp_dices: List[float] = []
    supp_images_detail: List[Dict[str, Any]] = []

    lesion_ious: Dict[str, List[float]] = {code: [] for code in LESION_CODES}
    lesion_dices: Dict[str, List[float]] = {code: [] for code in LESION_CODES}
    od_ious: List[float] = []
    od_dices: List[float] = []

    for r in results:
        g = r.get("grade")
        co = r.get("cohort_origin", "unspecified")
        is_test_overlap = (co == "idrid_grading_test_cohort")

        if co not in cohort_data:
            cohort_data[co] = {"iou": [], "dice": []}

        t_data = r["metrics_by_threshold"].get(baseline_threshold_key, {})
        lesions_data = t_data.get("lesions", {})
        control_data = t_data.get("control", {})

        # Composite
        comp = lesions_data.get("composite", {})
        c_iou = comp.get("iou")
        c_dice = comp.get("dice")
        if c_iou is not None:
            comp_ious.append(c_iou)
            cohort_data[co]["iou"].append(c_iou)
            if g is not None and g in grade_data:
                grade_data[g]["iou"].append(c_iou)

            if is_test_overlap:
                supp_comp_ious.append(c_iou)
            else:
                primary_comp_ious.append(c_iou)
                if g is not None and g in primary_grade_data:
                    primary_grade_data[g]["iou"].append(c_iou)

        if c_dice is not None:
            comp_dices.append(c_dice)
            cohort_data[co]["dice"].append(c_dice)
            if g is not None and g in grade_data:
                grade_data[g]["dice"].append(c_dice)

            if is_test_overlap:
                supp_comp_dices.append(c_dice)
            else:
                primary_comp_dices.append(c_dice)
                if g is not None and g in primary_grade_data:
                    primary_grade_data[g]["dice"].append(c_dice)

        if is_test_overlap:
            supp_images_detail.append({
                "image_id": r.get("image_id"),
                "grade": g,
                "composite_iou": c_iou,
                "composite_dice": c_dice,
            })

        # Individual lesions
        for code in LESION_CODES:
            l_info = lesions_data.get(code, {})
            l_iou = l_info.get("iou")
            l_dice = l_info.get("dice")
            if l_iou is not None:
                lesion_ious[code].append(l_iou)
                if not is_test_overlap:
                    primary_lesion_ious[code].append(l_iou)
            if l_dice is not None:
                lesion_dices[code].append(l_dice)
                if not is_test_overlap:
                    primary_lesion_dices[code].append(l_dice)

        # Control (OD)
        od_info = control_data.get("OD", {})
        o_iou = od_info.get("iou")
        o_dice = od_info.get("dice")
        if o_iou is not None:
            od_ious.append(o_iou)
            if not is_test_overlap:
                primary_od_ious.append(o_iou)
        if o_dice is not None:
            od_dices.append(o_dice)
            if not is_test_overlap:
                primary_od_dices.append(o_dice)

    def _calc_stats(arr: List[float]) -> Dict[str, Any]:
        if not arr:
            return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
        a = np.array(arr, dtype=np.float64)
        return {
            "count": len(arr),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "iqr_25": float(np.percentile(a, 25)),
            "iqr_75": float(np.percentile(a, 75)),
        }

    # Primary Cohort Summary (Excludes 3 test overlap images)
    primary_total = total_images - len(supp_images_detail)
    summary["primary_cohort"] = {
        "description": "Primary non-overlapping localization cohort (excludes IDRiD grading test overlap)",
        "total_images": primary_total,
        "composite_iou": _calc_stats(primary_comp_ious),
        "composite_dice": _calc_stats(primary_comp_dices),
        "grade_wise": {
            f"grade_{g}": {
                "iou": _calc_stats(primary_grade_data[g]["iou"]),
                "dice": _calc_stats(primary_grade_data[g]["dice"]),
            }
            for g in range(1, 5)
        },
        "lesion_breakdown": {
            code: {
                "iou": _calc_stats(primary_lesion_ious[code]),
                "dice": _calc_stats(primary_lesion_dices[code]),
            }
            for code in LESION_CODES
        },
        "optic_disc_control": {
            "iou": _calc_stats(primary_od_ious),
            "dice": _calc_stats(primary_od_dices),
        },
    }

    # Supplementary Test Cohort Summary (N=3: IDRiD_09, IDRiD_22, IDRiD_31)
    summary["supplementary_test_cohort"] = {
        "description": "Supplementary test-overlap cohort (N=3 images matching IDRiD grading test set)",
        "total_images": len(supp_images_detail),
        "composite_iou": _calc_stats(supp_comp_ious),
        "composite_dice": _calc_stats(supp_comp_dices),
        "images": supp_images_detail,
    }

    summary["overall_composite"]["total_images"] = total_images
    summary["overall_composite"]["iou"] = _calc_stats(comp_ious)
    summary["overall_composite"]["dice"] = _calc_stats(comp_dices)

    for g in range(1, 5):
        summary["grade_wise_composite"][f"grade_{g}"] = {
            "iou": _calc_stats(grade_data[g]["iou"]),
            "dice": _calc_stats(grade_data[g]["dice"]),
        }

    for co, data in cohort_data.items():
        summary["cohort_origin_composite"][co] = {
            "iou": _calc_stats(data["iou"]),
            "dice": _calc_stats(data["dice"]),
        }

    for code in LESION_CODES:
        summary["lesion_breakdown"][code] = {
            "iou": _calc_stats(lesion_ious[code]),
            "dice": _calc_stats(lesion_dices[code]),
        }

    summary["optic_disc_control"] = {
        "iou": _calc_stats(od_ious),
        "dice": _calc_stats(od_dices),
    }

    return summary

