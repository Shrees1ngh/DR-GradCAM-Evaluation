"""Real benchmark dataset browser and ground-truth reference helpers for RetinaScope.

Provides optional access to local processed datasets (IDRiD External Test,
IDRiD Segmentation with expert lesion masks, and APTOS Validation) when present.
Never injects ground truth into model inference; used strictly for image selection
and post-inference reference comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

from src.explainability import binarize_heatmap
from src.localization import compute_dice, compute_iou
from src.preprocessing import extract_retinal_fov_mask


def load_benchmark_catalogs(project_root: Path) -> Dict[str, pd.DataFrame]:
    """Load available real dataset manifests from data/processed/."""
    catalogs: Dict[str, pd.DataFrame] = {}

    idrid_dir = project_root / "data" / "processed" / "idrid"
    aptos_dir = project_root / "data" / "processed" / "aptos"

    # Build grade lookup for segmentation cohorts
    train_grades: Dict[str, int] = {}
    test_grades: Dict[str, int] = {}

    ext_train_csv = idrid_dir / "external_train.csv"
    if ext_train_csv.exists():
        df_tr = pd.read_csv(ext_train_csv)
        train_grades = dict(zip(df_tr["original_id"].astype(str), df_tr["grade"].astype(int)))

    ext_test_csv = idrid_dir / "external_test.csv"
    if ext_test_csv.exists():
        df_te = pd.read_csv(ext_test_csv)
        test_grades = dict(zip(df_te["original_id"].astype(str), df_te["grade"].astype(int)))
        df_te = df_te.copy()
        df_te["true_grade"] = df_te["grade"].astype(int)
        catalogs["IDRiD External Test (103 images)"] = df_te

    seg_csv = idrid_dir / "segmentation_manifest.csv"
    if seg_csv.exists():
        df_seg = pd.read_csv(seg_csv).copy()

        def _resolve_seg_grade(row: pd.Series) -> Optional[int]:
            cohort = str(row.get("cohort_origin", ""))
            match_id = row.get("grading_match_id")
            if pd.isna(match_id) or not str(match_id).strip():
                return None
            stem = str(match_id).replace(".jpg", "").strip()
            if cohort == "idrid_grading_train_cohort":
                return train_grades.get(stem)
            if cohort == "idrid_grading_test_cohort":
                return test_grades.get(stem)
            return None

        df_seg["true_grade"] = df_seg.apply(_resolve_seg_grade, axis=1)
        catalogs["IDRiD Segmentation (81 lesion-masked images)"] = df_seg

    aptos_val_csv = aptos_dir / "val_split.csv"
    if aptos_val_csv.exists():
        df_val = pd.read_csv(aptos_val_csv).copy()
        df_val["true_grade"] = df_val["grade"].astype(int)
        catalogs["APTOS 2019 Validation (733 images)"] = df_val

    return catalogs


def resolve_image_path(row: pd.Series, project_root: Path) -> Path:
    """Resolve relative image_path from manifest row across OS path separators."""
    rel = str(row["image_path"]).replace("\\", "/")
    return (project_root / rel).resolve()


def load_expert_lesion_data(
    row: pd.Series,
    preprocessed_rgb: np.ndarray,
    project_root: Path,
) -> Optional[Dict[str, Any]]:
    """Load real expert binary lesion masks (MA, HE, EX, SE, composite) for a segmentation row."""
    comp_rel = row.get("mask_composite")
    if pd.isna(comp_rel) or not str(comp_rel).strip():
        return None

    comp_path = (project_root / str(comp_rel).replace("\\", "/")).resolve()
    if not comp_path.exists():
        return None

    with Image.open(comp_path) as m_img:
        comp_mask = (np.array(m_img.convert("L")) > 0).astype(np.uint8)

    # Load FOV mask if available, else derive from preprocessed image
    fov_mask = None
    fov_rel = row.get("mask_fov")
    if pd.notna(fov_rel) and str(fov_rel).strip():
        fov_path = (project_root / str(fov_rel).replace("\\", "/")).resolve()
        if fov_path.exists():
            with Image.open(fov_path) as f_img:
                fov_mask = (np.array(f_img.convert("L")) > 0).astype(np.uint8)
    if fov_mask is None:
        fov_mask = extract_retinal_fov_mask(preprocessed_rgb)

    lesion_counts: Dict[str, int] = {}
    for code in ["MA", "HE", "EX", "SE"]:
        if bool(row.get(f"annotated_{code}", False)):
            px = int(row.get(f"pixel_count_{code}", 0))
            if px >= 0:
                lesion_counts[code] = px

    return {
        "composite_mask": comp_mask,
        "fov_mask": fov_mask,
        "lesion_counts": lesion_counts,
        "cohort_origin": str(row.get("cohort_origin", "unknown")),
    }


def evaluate_live_localization(
    grayscale_cam: np.ndarray,
    expert_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute real IoU, Dice, and Pointing-Game Hit against expert composite lesion mask."""
    comp_mask = expert_data["composite_mask"]
    fov_mask = expert_data["fov_mask"]

    # 1. Fixed threshold t=0.5 (used in Model B notebook evaluation)
    cam_t50 = (grayscale_cam >= 0.50).astype(np.uint8)
    iou_50 = compute_iou(cam_t50, comp_mask, empty_value=0.0) or 0.0
    dice_50 = compute_dice(cam_t50, comp_mask, empty_value=0.0) or 0.0

    # 2. Top 10% FOV-restricted threshold (used in Phase 6/8 pipeline)
    cam_top10 = binarize_heatmap(grayscale_cam, threshold=0.10, method="percentile", fov_mask=fov_mask)
    iou_10 = compute_iou(cam_top10, comp_mask, empty_value=0.0) or 0.0
    dice_10 = compute_dice(cam_top10, comp_mask, empty_value=0.0) or 0.0

    # 3. Pointing-game hit (peak activation pixel falls inside expert lesion mask)
    if int(comp_mask.sum()) > 0:
        max_yx = np.unravel_index(int(np.argmax(grayscale_cam)), grayscale_cam.shape)
        pointing_hit = bool(comp_mask[max_yx] > 0)
    else:
        pointing_hit = False

    return {
        "iou_50": iou_50,
        "dice_50": dice_50,
        "iou_10": iou_10,
        "dice_10": dice_10,
        "pointing_hit": pointing_hit,
        "composite_pixels": int(comp_mask.sum()),
    }
