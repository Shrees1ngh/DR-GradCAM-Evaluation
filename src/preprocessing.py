"""Image preprocessing and transformation utilities for Diabetic Retinopathy fundus images.

Preserves native retinal aspect ratio via isotropic scaling and centered zero-padding.
Maintains pixel-perfect spatial alignment between fundus images and lesion masks.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger("preprocessing")


def crop_to_retinal_fov(
    image: np.ndarray,
    threshold: int = 10,
    margin_ratio: float = 0.01,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Detect circular retinal field-of-view and crop uninformative black borders.

    Args:
        image: RGB or grayscale image array (H, W, C) or (H, W).
        threshold: Intensity threshold separating retinal foreground from background.
        margin_ratio: Margin fraction preserved around the bounding box.

    Returns:
        cropped_image: Foreground cropped array.
        bbox: Bounding box tuple (ymin, ymax, xmin, xmax) in original image coordinates.
    """
    if image.ndim == 3:
        # Green channel provides the highest contrast for retinal structures
        gray = image[:, :, 1] if image.shape[2] >= 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    mask = gray > threshold

    if not np.any(mask):
        h, w = gray.shape[:2]
        return image, (0, h, 0, w)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    h, w = gray.shape[:2]
    h_margin = int(h * margin_ratio)
    w_margin = int(w * margin_ratio)

    ymin = max(0, ymin - h_margin)
    ymax = min(h, ymax + h_margin + 1)
    xmin = max(0, xmin - w_margin)
    xmax = min(w, xmax + w_margin + 1)

    cropped = image[ymin:ymax, xmin:xmax]
    return cropped, (ymin, ymax, xmin, xmax)


def extract_retinal_fov_mask(
    image: np.ndarray,
    threshold: int = 10,
    pad_offsets: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Generate a clean binary mask (H, W) where 1 indicates valid retinal foreground.

    Uses green-channel thresholding followed by morphological contour extraction and hole filling
    to ensure:
    1. Retinal lesions (e.g. dense haemorrhages with green intensity <= threshold) remain
       strictly inside the valid retinal field of view (not punched out as background).
    2. Zero-padded canvas borders (if pad_offsets provided) are strictly zeroed out.
    3. The 4 natural black background corners outside the circular optical aperture
       are strictly excluded.

    Args:
        image: RGB uint8 array (H, W, 3) or 2D array (H, W).
        threshold: Green channel intensity cutoff for retinal tissue (default: 10).
        pad_offsets: Optional tuple (pad_top, pad_bottom, pad_left, pad_right) from resize_and_pad.

    Returns:
        Binary uint8 mask of shape (H, W) with values strictly in {0, 1}.
    """
    if image.ndim == 3:
        green = image[:, :, 1] if image.shape[2] >= 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        green = image

    h, w = green.shape[:2]
    raw_mask = (green > threshold).astype(np.uint8)

    if not np.any(raw_mask):
        return np.zeros((h, w), dtype=np.uint8)

    # Find the largest external contour representing the circular retinal aperture
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        fov_mask = np.zeros((h, w), dtype=np.uint8)
        # Draw and fill the largest contour to guarantee no internal holes from dark lesions
        cv2.drawContours(fov_mask, [largest_contour], -1, 1, thickness=-1)
    else:
        fov_mask = raw_mask

    # If pad_offsets are specified, enforce strict zero-padding on canvas margins
    if pad_offsets is not None:
        pad_top, pad_bottom, pad_left, pad_right = pad_offsets
        if pad_top > 0:
            fov_mask[:pad_top, :] = 0
        if pad_bottom > 0:
            fov_mask[h - pad_bottom:, :] = 0
        if pad_left > 0:
            fov_mask[:, :pad_left] = 0
        if pad_right > 0:
            fov_mask[:, w - pad_right:] = 0

    return fov_mask.astype(np.uint8)


def resize_and_pad(
    image: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    interpolation: int = cv2.INTER_AREA,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
    """Resize image to fit within target_size while strictly preserving aspect ratio.

    Applies centered symmetric zero-padding so output is exactly target_size.

    Args:
        image: Input array (H, W, C) or (H, W).
        target_size: Desired (target_height, target_width).
        interpolation: OpenCV interpolation mode.

    Returns:
        padded_image: Array of exact target_size.
        pad_offsets: Tuple (pad_top, pad_bottom, pad_left, pad_right).
        scale: Scaling factor applied.
    """
    target_h, target_w = target_size
    h, w = image.shape[:2]

    # Calculate uniform isotropic scale factor
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Resize image isotropically
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Compute centered padding
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    if image.ndim == 3:
        padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w, :] = resized
    else:
        padded = np.zeros((target_h, target_w), dtype=image.dtype)
        padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    pad_offsets = (pad_top, pad_bottom, pad_left, pad_right)
    return padded, pad_offsets, scale


def preprocess_fundus_image(
    image_path: str,
    target_size: Tuple[int, int] = (512, 512),
    crop_fov: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Complete preprocessing pipeline for a fundus photograph.

    Args:
        image_path: Path to fundus image file.
        target_size: Target square dimensions (height, width).
        crop_fov: Whether to detect and crop black borders first.

    Returns:
        preprocessed: RGB uint8 array of shape (target_h, target_w, 3).
        meta: Transformation metadata dictionary required to replicate spatial
              transforms identically on corresponding segmentation masks.
    """
    with Image.open(image_path) as pil_img:
        img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)

    raw_h, raw_w = img_rgb.shape[:2]

    if crop_fov:
        cropped_img, bbox = crop_to_retinal_fov(img_rgb)
    else:
        cropped_img = img_rgb
        bbox = (0, raw_h, 0, raw_w)

    padded_img, pad_offsets, scale = resize_and_pad(
        cropped_img, target_size=target_size, interpolation=cv2.INTER_AREA
    )

    meta = {
        "raw_shape": (raw_h, raw_w),
        "crop_bbox": bbox,
        "pad_offsets": pad_offsets,
        "scale": scale,
        "target_size": target_size,
    }

    return padded_img, meta


def preprocess_mask_with_transform(
    mask: np.ndarray,
    meta: Dict[str, Any],
) -> np.ndarray:
    """Apply the identical spatial transformation of a fundus image to its lesion mask.

    Uses nearest-neighbor interpolation to preserve strict binary {0, 1} values.

    Args:
        mask: Raw 2D binary mask array (H_raw, W_raw).
        meta: Transformation metadata from preprocess_fundus_image.

    Returns:
        Binary mask of shape (target_h, target_w) with values strictly in {0, 1}.
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    ymin, ymax, xmin, xmax = meta["crop_bbox"]
    pad_top, pad_bottom, pad_left, pad_right = meta["pad_offsets"]
    target_h, target_w = meta["target_size"]

    # 1. Apply identical bounding box crop
    cropped_mask = mask[ymin:ymax, xmin:xmax]

    # Calculate expected intermediate dimensions
    new_h = target_h - (pad_top + pad_bottom)
    new_w = target_w - (pad_left + pad_right)

    # 2. Resize with nearest neighbor
    resized_mask = cv2.resize(
        cropped_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )

    # 3. Apply identical padding offsets
    padded_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    padded_mask[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_mask

    # 4. Strict binary enforcement {0, 1}
    binary_mask = (padded_mask > 0).astype(np.uint8)
    return binary_mask


def build_composite_lesion_mask(
    lesion_masks: Dict[str, np.ndarray],
) -> np.ndarray:
    """Build composite lesion mask as logical union: Composite = MA U HE U EX U SE.

    Optic disc ('OD') is an anatomical landmark and is strictly excluded.

    Args:
        lesion_masks: Dictionary mapping lesion codes ('MA', 'HE', 'EX', 'SE')
                      to binary arrays of shape (H, W).

    Returns:
        Composite binary mask {0, 1} of shape (H, W).
    """
    relevant_codes = ["MA", "HE", "EX", "SE"]
    masks_to_combine = [
        lesion_masks[c] for c in relevant_codes
        if c in lesion_masks and lesion_masks[c] is not None
    ]

    if not masks_to_combine:
        return np.zeros((512, 512), dtype=np.uint8)

    composite = np.bitwise_or.reduce(masks_to_combine)
    return (composite > 0).astype(np.uint8)


def evaluate_fov_mask_quality(
    image_rgb: np.ndarray,
    fov_mask: np.ndarray,
    pad_offsets: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Any]:
    """Evaluate quality and geometry of a generated retinal FOV mask.

    Validates:
    1. Spatial dimension match between image and mask.
    2. Strict binary {0, 1} mask values.
    3. FOV pixel count and active retinal area ratio.
    4. Contour circularity.
    5. Zero encroachment into padded margins (PADDING_LEAK).
    6. Identifies anomalies (EMPTY_FOV, UNUSUALLY_SMALL, UNUSUALLY_LARGE, PADDING_LEAK).

    Args:
        image_rgb: RGB fundus image array (H, W, 3).
        fov_mask: Binary FOV mask array (H, W).
        pad_offsets: Tuple (pad_top, pad_bottom, pad_left, pad_right).

    Returns:
        Dictionary with validation metrics, status ('PASSED' / 'FAILED'), and flags.
    """
    if image_rgb.shape[:2] != fov_mask.shape[:2]:
        raise ValueError(
            f"Image shape {image_rgb.shape[:2]} does not match FOV mask shape {fov_mask.shape[:2]}"
        )

    unique_vals = set(np.unique(fov_mask))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"FOV mask must be strictly binary {{0, 1}}, got values: {unique_vals}")

    h, w = fov_mask.shape[:2]
    total_pixels = h * w
    flags = []

    fov_pixels = int(np.sum(fov_mask == 1))
    fov_area_ratio = float(fov_pixels / total_pixels) if total_pixels > 0 else 0.0

    # Calculate circularity via contours
    circularity = 0.0
    contours, _ = cv2.findContours(fov_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest_contour))
        perimeter = float(cv2.arcLength(largest_contour, True))
        if perimeter > 0:
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))

    # Quality-control screening heuristics for circular retinal aperture geometry on square canvas:
    # An ideal circular aperture inscribed in a square canvas covers pi/4 (~78.54%) of total area.
    # Allowing for FOV centering pad margins (typically <= 3%) and camera aperture framing:
    # - UNUSUALLY_SMALL (< 0.50): operational screening heuristic for aperture truncation or foreground loss.
    # - UNUSUALLY_LARGE (> 0.85): operational screening heuristic for threshold leakage into black canvas corners.
    # - IRREGULAR_GEOMETRY (< 0.60): operational screening heuristic for non-circular contour deformities.
    # Note: These values are operational screening heuristics, not clinical ground truth or empirical cutoffs.
    if fov_pixels == 0:
        flags.append("EMPTY_FOV")
    elif fov_area_ratio < 0.50:
        flags.append("UNUSUALLY_SMALL")
    elif fov_area_ratio > 0.85:
        flags.append("UNUSUALLY_LARGE")

    # Retinal aperture contour circularity check (ideal circle = 1.0)
    if circularity > 0.0 and circularity < 0.60:
        flags.append("IRREGULAR_GEOMETRY")

    # Check for padding leakage: FOV mask must never encroach into zero-padded canvas margins
    if pad_offsets is not None:
        pad_top, pad_bottom, pad_left, pad_right = pad_offsets
        leak = False
        if pad_top > 0 and np.any(fov_mask[:pad_top, :] > 0):
            leak = True
        if pad_bottom > 0 and np.any(fov_mask[h - pad_bottom:, :] > 0):
            leak = True
        if pad_left > 0 and np.any(fov_mask[:, :pad_left] > 0):
            leak = True
        if pad_right > 0 and np.any(fov_mask[:, w - pad_right:] > 0):
            leak = True
        if leak:
            flags.append("PADDING_LEAK")

    is_valid = len(flags) == 0
    if is_valid:
        status = "PASSED"
        flags.append("VALID")
    else:
        status = "FAILED"

    return {
        "status": status,
        "valid": is_valid,
        "fov_pixels": fov_pixels,
        "total_pixels": total_pixels,
        "fov_area_ratio": fov_area_ratio,
        "circularity": circularity,
        "flags": flags,
    }


def create_fov_overlay(
    image_rgb: np.ndarray,
    fov_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Create a visualization overlay of the FOV contour on the fundus image.

    Args:
        image_rgb: RGB fundus image (H, W, 3).
        fov_mask: Binary FOV mask (H, W).
        color: RGB tuple for contour line (default: cyan/yellow (0, 255, 255)).
        thickness: Contour line thickness in pixels.

    Returns:
        RGB image with FOV boundary drawn.
    """
    overlay = image_rgb.copy()
    contours, _ = cv2.findContours(
        fov_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay


def create_fov_contact_sheet(
    overlays: List[np.ndarray],
    image_ids: List[str],
    grid_size: Tuple[int, int] = (9, 9),
    thumb_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Create a contact sheet montage of FOV contour overlays covering all 81 images.

    Args:
        overlays: List of RGB images with FOV contours drawn.
        image_ids: Corresponding list of image IDs (e.g. 'IDRiD_01').
        grid_size: (rows, cols) for the contact sheet grid (default: 9x9 = 81 tiles).
        thumb_size: (width, height) thumbnail tile dimensions.

    Returns:
        RGB numpy array representing the complete 81-image contact sheet.
    """
    rows, cols = grid_size
    tw, th = thumb_size
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)

    for idx in range(min(len(overlays), rows * cols)):
        r = idx // cols
        c = idx % cols
        thumb = cv2.resize(overlays[idx], (tw, th), interpolation=cv2.INTER_AREA)

        # Draw image label at top-left
        label = image_ids[idx].split("/")[-1].replace(".jpg", "")
        cv2.putText(
            thumb,
            label,
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        sheet[r * th : (r + 1) * th, c * tw : (c + 1) * tw, :] = thumb

    return sheet


EXPECTED_IDRID_SEG_IDS: List[str] = [f"IDRiD_{i:02d}" for i in range(1, 82)]


def normalize_idrid_seg_id(val: Any) -> str:
    """Extract and normalize canonical IDRiD segmentation ID (e.g. 'IDRiD_01')."""
    s = str(val).strip()
    m = re.search(r"IDRiD_(\d+)", s, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        return f"IDRiD_{num:02d}"
    return s


def run_fov_quality_control(
    manifest_df: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    output_md: Optional[Path] = None,
    montage_path: Optional[Path] = None,
    expected_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Execute quality control checks across all segmentation FOV masks.

    If genuine segmentation images and masks exist on disk, validates each mask,
    accumulates metrics, generates an 81-image FOV contour contact sheet,
    and outputs comprehensive CSV and Markdown reports.
    If genuine images are absent, explicitly documents that genuine 81-image
    validation remains pending.

    Args:
        manifest_df: DataFrame containing segmentation records (e.g. from idrid_seg_manifest.csv).
        project_root: Project root path.
        output_csv: Target path for QC report CSV.
        output_md: Target path for QC report Markdown.
        montage_path: Optional path to save an overlay montage image (default: reports/fov_montage_81.png).
        expected_ids: Optional custom sequence of expected image IDs (defaults to 81 IDRiD IDs).

    Returns:
        Summary dictionary with QC results and status.
    """
    if project_root is None:
        project_root = Path.cwd()
    if output_csv is None:
        output_csv = project_root / "reports" / "fov_qc_report.csv"
    if output_md is None:
        output_md = project_root / "reports" / "fov_qc_report.md"
    if montage_path is None:
        montage_path = project_root / "reports" / "fov_montage_81.png"
    if expected_ids is None:
        expected_ids = list(EXPECTED_IDRID_SEG_IDS)

    expected_list = list(expected_ids)
    expected_set = set(expected_list)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    pending_md_content = """# Retinal FOV Quality Control Report

## Status: PENDING

> **Notice**: Genuine 81-image validation remains pending. IDRiD segmentation dataset is not present in this local workspace.
> Field of View (FOV) Quality Control validation will execute automatically during genuine execution in Google Colab.

### Quality Control Screening Heuristics
- Expected Images: 81
- Format: Centered square (512x512)
- Screening Area Ratio Heuristic: ~0.60 - 0.78 (theoretical upper limit for an inscribed circle in a square: pi/4 = 78.54%)
- Flags to monitor: `EMPTY_FOV`, `UNUSUALLY_SMALL` (< 0.50), `UNUSUALLY_LARGE` (> 0.85), `PADDING_LEAK`, `IRREGULAR_GEOMETRY` (< 0.60)

> [!NOTE]
> **Methodological Clarification:** The area-ratio and circularity thresholds are operational screening heuristics based on circular aperture geometry on a square canvas. They are not empirically validated diagnostic boundaries and do not replace clinical expert verification of fine anatomic margin delineation.
"""

    if manifest_df is None or manifest_df.empty:
        cols = ["image_id", "fov_pixels", "total_pixels", "fov_area_ratio", "circularity", "status", "flags"]
        empty_df = pd.DataFrame(columns=cols)
        empty_df.to_csv(output_csv, index=False)
        output_md.write_text(pending_md_content, encoding="utf-8")
        return {
            "status": "PENDING",
            "total_evaluated": 0,
            "passed": 0,
            "failed": 0,
            "missing_records": len(expected_list),
            "duplicate_records": 0,
            "unexpected_records": 0,
            "message": "Genuine 81-image validation remains pending. Segmentation manifest missing or empty.",
        }

    # Inspect all records to check if any genuine images exist on disk
    any_images_exist = False
    for _, row in manifest_df.iterrows():
        img_rel = row.get("processed_image_path", row.get("image_path", ""))
        if img_rel and (project_root / img_rel).exists():
            any_images_exist = True
            break

    if not any_images_exist:
        cols = ["image_id", "fov_pixels", "total_pixels", "fov_area_ratio", "circularity", "status", "flags"]
        empty_df = pd.DataFrame(columns=cols)
        empty_df.to_csv(output_csv, index=False)
        output_md.write_text(pending_md_content, encoding="utf-8")
        return {
            "status": "PENDING",
            "total_evaluated": 0,
            "passed": 0,
            "failed": 0,
            "missing_records": len(expected_list),
            "duplicate_records": 0,
            "unexpected_records": 0,
            "message": "Genuine 81-image validation remains pending. Segmentation images not found locally.",
        }

    manifest_id_counts: Dict[str, int] = {}
    manifest_ids: List[str] = []

    for idx, row in manifest_df.iterrows():
        raw_id = row.get("original_id") or row.get("image_id", f"img_{idx}")
        canon_id = normalize_idrid_seg_id(raw_id)
        manifest_ids.append(canon_id)
        manifest_id_counts[canon_id] = manifest_id_counts.get(canon_id, 0) + 1

    duplicate_ids = {cid for cid, cnt in manifest_id_counts.items() if cnt > 1}
    unexpected_ids = {cid for cid in manifest_id_counts.keys() if cid not in expected_set}
    missing_ids = [exp_id for exp_id in expected_list if exp_id not in manifest_id_counts]

    seen_ids = set()
    records = []
    overlays: List[np.ndarray] = []
    image_ids: List[str] = []

    # Evaluate each segmentation image and mask across all manifest records
    for idx, row in manifest_df.iterrows():
        raw_id = row.get("original_id") or row.get("image_id", f"img_{idx}")
        canon_id = normalize_idrid_seg_id(raw_id)
        row_flags: List[str] = []

        if canon_id in seen_ids:
            row_flags.append("DUPLICATE_RECORD")
        seen_ids.add(canon_id)

        if canon_id in unexpected_ids:
            row_flags.append("UNEXPECTED_RECORD")

        img_rel = row.get("processed_image_path", row.get("image_path", ""))
        fov_rel = row.get("fov_mask_path", row.get("mask_fov", ""))

        img_path = project_root / img_rel if img_rel else None
        fov_path = project_root / fov_rel if fov_rel else None

        if not img_path or not img_path.exists() or not fov_path or not fov_path.exists():
            row_flags.append("FILE_MISSING")
            records.append({
                "image_id": canon_id,
                "fov_pixels": 0,
                "total_pixels": 0,
                "fov_area_ratio": 0.0,
                "circularity": 0.0,
                "status": "FAILED",
                "flags": ";".join(row_flags),
            })
            continue

        with Image.open(img_path) as pil_img:
            img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        with Image.open(fov_path) as pil_fov:
            fov_mask = (np.array(pil_fov.convert("L")) > 0).astype(np.uint8)

        pad_offsets = None
        if "pad_offsets" in row and pd.notna(row["pad_offsets"]):
            try:
                import ast
                pad_offsets = ast.literal_eval(str(row["pad_offsets"]))
            except Exception:
                pad_offsets = None

        qc_res = evaluate_fov_mask_quality(img_rgb, fov_mask, pad_offsets=pad_offsets)
        if qc_res["status"] == "FAILED":
            row_flags.extend(qc_res["flags"])
        status = "FAILED" if row_flags else "PASSED"
        final_flags = row_flags if row_flags else ["VALID"]

        records.append({
            "image_id": canon_id,
            "fov_pixels": qc_res["fov_pixels"],
            "total_pixels": qc_res["total_pixels"],
            "fov_area_ratio": round(qc_res["fov_area_ratio"], 4),
            "circularity": round(qc_res["circularity"], 4),
            "status": status,
            "flags": ";".join(final_flags),
        })

        overlay = create_fov_overlay(img_rgb, fov_mask)
        overlays.append(overlay)
        image_ids.append(canon_id)

    # Append any expected IDs completely missing from manifest
    for mis_id in missing_ids:
        records.append({
            "image_id": mis_id,
            "fov_pixels": 0,
            "total_pixels": 0,
            "fov_area_ratio": 0.0,
            "circularity": 0.0,
            "status": "FAILED",
            "flags": "RECORD_MISSING",
        })

    df_qc = pd.DataFrame(records)
    df_qc.to_csv(output_csv, index=False)

    total_eval = len(df_qc)
    passed_cnt = int((df_qc["status"] == "PASSED").sum())
    failed_cnt = total_eval - passed_cnt
    missing_cnt = len(missing_ids)
    duplicate_cnt = len(duplicate_ids)
    unexpected_cnt = len(unexpected_ids)
    mean_ratio = float(df_qc["fov_area_ratio"].mean()) if total_eval > 0 else 0.0

    # Generate complete contact sheet montage covering all available overlay images
    if overlays:
        sheet = create_fov_contact_sheet(overlays, image_ids, grid_size=(9, 9), thumb_size=(128, 128))
        montage_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(montage_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        logging.info(f"Complete FOV contact sheet saved to {montage_path} ({sheet.shape[1]}x{sheet.shape[0]} px, {len(overlays)} images)")

    # Overall execution status: can ONLY be PASSED if 0 failures, exactly 81 records, 0 missing/dup/unexpected
    is_strict_passed = (
        failed_cnt == 0
        and total_eval == len(expected_list)
        and len(expected_list) == 81
        and missing_cnt == 0
        and duplicate_cnt == 0
        and unexpected_cnt == 0
    )
    overall_status = "PASSED" if is_strict_passed else "FAILED"

    def _table_to_markdown(df_subset: pd.DataFrame) -> str:
        try:
            return df_subset.to_markdown(index=False)
        except (ImportError, ModuleNotFoundError):
            if df_subset.empty:
                return "*(empty table)*"
            headers = [str(col) for col in df_subset.columns]
            rows = [[str(val) for val in row] for row in df_subset.itertuples(index=False)]
            return "\n".join([
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ] + ["| " + " | ".join(r) + " |" for r in rows])

    md_lines = [
        "# Retinal FOV Quality Control Report",
        "",
        f"## Execution Status: {overall_status}",
        "",
        f"- **Expected IDRiD Segmentation Records**: {len(expected_list)}",
        f"- **Total Segmentation Records Evaluated**: {total_eval}",
        f"- **Valid FOV Masks**: {passed_cnt}",
        f"- **Flagged / Missing / Anomalous Masks**: {failed_cnt}",
        f"- **Missing Expected Records**: {missing_cnt}",
        f"- **Duplicate Records**: {duplicate_cnt}",
        f"- **Unexpected Records**: {unexpected_cnt}",
        f"- **Mean Active FOV Area Ratio**: {mean_ratio:.4f} (expected ~0.60 - 0.78)",
        f"- **Contact Sheet Montage**: `reports/fov_montage_81.png` (9x9 grid covering all {len(overlays)} available images)",
        "",
        "> [!NOTE]",
        "> **Methodological Clarification:** Area-ratio and circularity thresholds are operational screening heuristics based on circular aperture geometry.",
        "> They are not empirically validated diagnostic boundaries and do not replace clinical expert verification of fine anatomic margin delineation.",
        "",
        "### Quality Control Summary Table",
        "",
        _table_to_markdown(df_qc.head(20)),
        "",
    ]
    output_md.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "status": overall_status,
        "total_evaluated": total_eval,
        "passed": passed_cnt,
        "failed": failed_cnt,
        "missing_records": missing_cnt,
        "duplicate_records": duplicate_cnt,
        "unexpected_records": unexpected_cnt,
        "mean_fov_area_ratio": mean_ratio,
        "montage_path": str(montage_path) if overlays else None,
    }


def _parse_strict_int(val: Any, field_name: str, img_id: str) -> int:
    """Parse integer strictly, rejecting booleans, fractionals, and ambiguous strings."""
    # In Python, bool is a subclass of int: isinstance(True, int) is True!
    if isinstance(val, (bool, np.bool_)):
        raise ValueError(f"Image {img_id}: boolean value not allowed for integer count '{field_name}': {val}")

    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            raise ValueError(f"Image {img_id}: NaN/Inf value not allowed for integer count '{field_name}': {val}")
        if not float(val).is_integer():
            raise ValueError(f"Image {img_id}: fractional pixel count not allowed for '{field_name}': {val}")
        return int(val)

    if isinstance(val, (int, np.integer)):
        return int(val)

    if isinstance(val, str):
        s = val.strip()
        if s.lower() in {"true", "false"}:
            raise ValueError(f"Image {img_id}: boolean string not allowed for integer count '{field_name}': {val}")
        if "." in s:
            try:
                f_val = float(s)
            except ValueError as e:
                raise ValueError(f"Image {img_id}: invalid non-integer value for '{field_name}': {val}") from e
            if not f_val.is_integer():
                raise ValueError(f"Image {img_id}: fractional pixel count not allowed for '{field_name}': {val}")
            return int(f_val)
        try:
            return int(s)
        except ValueError as e:
            raise ValueError(f"Image {img_id}: invalid non-integer value for '{field_name}': {val}") from e

    raise ValueError(f"Image {img_id}: unsupported type {type(val)} for integer count '{field_name}': {val}")


def _parse_strict_bool(val: Any, field_name: str, img_id: str, allow_none: bool = False) -> Optional[bool]:
    """Parse boolean strictly, supporting only documented True/False and 1/0 representations."""
    if pd.isna(val) or val is None:
        if allow_none:
            return None
        raise ValueError(f"Image {img_id}: missing required boolean metadata '{field_name}'!")

    if isinstance(val, (bool, np.bool_)):
        return bool(val)

    if isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_)):
        if val == 1:
            return True
        elif val == 0:
            return False
        raise ValueError(f"Image {img_id}: integer {val} not a valid boolean for '{field_name}' (expected 1 or 0)")

    if isinstance(val, (float, np.floating)):
        if val == 1.0:
            return True
        elif val == 0.0:
            return False
        raise ValueError(f"Image {img_id}: float {val} not a valid boolean for '{field_name}' (expected 1.0 or 0.0)")

    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"true", "1"}:
            return True
        elif s in {"false", "0"}:
            return False
        elif s in {"none", "null", "nan", ""} and allow_none:
            return None
        raise ValueError(f"Image {img_id}: ambiguous or invalid boolean string '{val}' for '{field_name}'")

    raise ValueError(f"Image {img_id}: unsupported type {type(val)} for boolean field '{field_name}'")


def validate_segmentation_masks(
    manifest_df: pd.DataFrame,
    project_root: Optional[Path] = None,
    target_size: Tuple[int, int] = (512, 512),
) -> Dict[str, Any]:
    """Validate completeness, resolution, binarization, and pixel counts of all lesion masks.

    Checks:
    1. Rejects missing, null, or empty manifest DataFrame.
    2. Every expected annotated mask file exists on disk and is non-empty.
    3. Every mask has the exact expected target dimensions (target_size).
    4. Every mask is strictly binary with values in {0, 1} or {0, 255}.
    5. Actual positive pixel count strictly matches manifest pixel_count_{code}.
    6. Rejects fractional pixel counts, booleans in place of counts, and invalid presence flags.
    7. Missing annotations require pixel_count = -1 and lesion_present = null.
    8. Requires valid composite mask path and valid non-negative composite pixel-count metadata for every record.
    9. Compares recorded composite pixel counts with actual decoded composite mask pixel counts.
    10. Enforces that composite mask is exactly the pixel-wise union of MA, HE, EX, and SE,
        and strictly excludes OD.

    Args:
        manifest_df: DataFrame from segmentation_manifest.csv.
        project_root: Project root directory.
        target_size: Expected (H, W) resolution (default: (512, 512)).

    Returns:
        Validation statistics dictionary.

    Raises:
        ValueError: If manifest is empty, any required column/metadata is missing,
                    any mask file is missing/empty/corrupted, dimensions mismatch,
                    values are non-binary, or pixel counts / presence flags are inconsistent.
    """
    if manifest_df is None or manifest_df.empty:
        raise ValueError("Segmentation manifest DataFrame is missing, null, or empty.")

    if project_root is None:
        project_root = Path.cwd()

    lesion_codes = ["MA", "HE", "EX", "SE", "OD"]
    total_checked = 0
    annotated_counts = {c: 0 for c in lesion_codes}
    missing_annotation_counts = {c: 0 for c in lesion_codes}

    for idx, row in manifest_df.iterrows():
        img_id = str(row.get("image_id", f"row_{idx}"))
        decoded_masks: Dict[str, np.ndarray] = {}

        for code in lesion_codes:
            mask_col = f"mask_{code}"
            ann_col = f"annotated_{code}"
            cnt_col = f"pixel_count_{code}"
            pres_col = f"lesion_present_{code}"

            if ann_col not in row or pd.isna(row[ann_col]):
                raise ValueError(
                    f"Image {img_id}: missing required annotation status column '{ann_col}'!"
                )
            if cnt_col not in row or pd.isna(row[cnt_col]):
                raise ValueError(
                    f"Image {img_id}: missing required pixel-count metadata column '{cnt_col}' for lesion {code}!"
                )
            if pres_col not in row:
                raise ValueError(
                    f"Image {img_id}: missing required presence metadata column '{pres_col}' for lesion {code}!"
                )

            is_annotated = _parse_strict_bool(row[ann_col], ann_col, img_id, allow_none=False)

            if not is_annotated:
                missing_annotation_counts[code] += 1
                exp_cnt = _parse_strict_int(row[cnt_col], cnt_col, img_id)

                if exp_cnt != -1:
                    raise ValueError(
                        f"Image {img_id}: unannotated lesion {code} must have pixel_count = -1, got {exp_cnt}"
                    )
                pres_val = _parse_strict_bool(row[pres_col], pres_col, img_id, allow_none=True)
                if pres_val is not None:
                    raise ValueError(
                        f"Image {img_id}: unannotated lesion {code} must have lesion_present = null/None, got {row[pres_col]}"
                    )
                continue

            # Annotated lesion branch
            annotated_counts[code] += 1
            pres_val = _parse_strict_bool(row[pres_col], pres_col, img_id, allow_none=False)

            mask_rel = row.get(mask_col)
            if not mask_rel or pd.isna(mask_rel):
                raise ValueError(f"Image {img_id}: annotated mask_{code} path is missing in manifest!")

            mask_path = project_root / str(mask_rel)
            if not mask_path.exists() or mask_path.stat().st_size == 0:
                raise ValueError(f"Image {img_id}: required annotated mask file missing or empty: {mask_path}")

            with Image.open(mask_path) as m_pil:
                m_arr = np.array(m_pil.convert("L"))

            if m_arr.shape != target_size:
                raise ValueError(
                    f"Image {img_id} mask_{code} resolution {m_arr.shape} does not match expected {target_size}"
                )

            u_vals = set(np.unique(m_arr))
            if not u_vals.issubset({0, 1, 255}):
                raise ValueError(
                    f"Image {img_id} mask_{code} contains non-binary values: {u_vals}"
                )

            actual_pos_pixels = int(np.sum(m_arr > 0))
            exp_cnt = _parse_strict_int(row[cnt_col], cnt_col, img_id)

            if exp_cnt < 0:
                raise ValueError(
                    f"Image {img_id}: annotated mask_{code} has negative pixel count: {exp_cnt}"
                )
            if actual_pos_pixels != exp_cnt:
                raise ValueError(
                    f"Image {img_id}: pixel count mismatch for {code}. "
                    f"Actual positive pixels: {actual_pos_pixels}, manifest value: {exp_cnt}"
                )

            actual_present = bool(actual_pos_pixels > 0)
            if actual_present != pres_val:
                raise ValueError(
                    f"Image {img_id}: lesion_present disagreement for {code}. "
                    f"Actual pixels > 0: {actual_present}, manifest specifies {pres_val}"
                )

            decoded_masks[code] = (m_arr > 0)
            total_checked += 1

        # Composite mask validation (required for every manifest record)
        comp_rel = row.get("mask_composite")
        if not comp_rel or pd.isna(comp_rel):
            raise ValueError(
                f"Image {img_id}: required composite mask path 'mask_composite' is missing or null in manifest!"
            )

        comp_path = project_root / str(comp_rel)
        if not comp_path.exists() or comp_path.stat().st_size == 0:
            raise ValueError(f"Image {img_id}: required composite mask file missing or empty: {comp_path}")

        with Image.open(comp_path) as c_pil:
            c_arr = np.array(c_pil.convert("L"))

        if c_arr.shape != target_size:
            raise ValueError(f"Image {img_id} composite mask resolution {c_arr.shape} != {target_size}")

        if not set(np.unique(c_arr)).issubset({0, 1, 255}):
            raise ValueError(f"Image {img_id} composite mask contains non-binary values: {set(np.unique(c_arr))}")

        # Required composite pixel-count metadata
        comp_val = None
        comp_col_name = None
        for comp_col in ["composite_pixels", "pixel_count_composite"]:
            if comp_col in row and pd.notna(row[comp_col]):
                comp_val = row[comp_col]
                comp_col_name = comp_col
                break

        if comp_val is None:
            raise ValueError(
                f"Image {img_id}: missing required composite pixel-count metadata ('composite_pixels' or 'pixel_count_composite')!"
            )

        exp_comp = _parse_strict_int(comp_val, comp_col_name or "composite_pixels", img_id)

        if exp_comp < 0:
            raise ValueError(f"Image {img_id}: negative composite pixel count: {exp_comp}")

        actual_comp_pixels = int(np.sum(c_arr > 0))
        if actual_comp_pixels != exp_comp:
            raise ValueError(
                f"Image {img_id}: composite pixel count mismatch. "
                f"Actual positive pixels: {actual_comp_pixels}, manifest value: {exp_comp}"
            )

        # Composite mask logical union integrity check:
        # Must strictly equal the pixel-wise union of MA, HE, EX, SE.
        # OD must NEVER be included.
        expected_union = np.zeros(target_size, dtype=bool)
        for code in ["MA", "HE", "EX", "SE"]:
            if code in decoded_masks:
                expected_union |= decoded_masks[code]

        actual_comp_bool = (c_arr > 0)
        if not np.array_equal(actual_comp_bool, expected_union):
            if "OD" in decoded_masks and np.any(decoded_masks["OD"] & actual_comp_bool & (~expected_union)):
                raise ValueError(
                    f"Image {img_id}: composite mask improperly includes Optic Disc (OD) pixels!"
                )
            raise ValueError(
                f"Image {img_id}: composite mask does not match the pixel-wise logical union of MA, HE, EX, and SE!"
            )

    return {
        "status": "PASSED",
        "total_records_checked": len(manifest_df),
        "total_lesion_masks_checked": total_checked,
        "annotated_counts": annotated_counts,
        "missing_annotation_counts": missing_annotation_counts,
    }

