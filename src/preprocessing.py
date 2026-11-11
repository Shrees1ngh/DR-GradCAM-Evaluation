"""Image preprocessing and transformation utilities for Diabetic Retinopathy fundus images.

Preserves native retinal aspect ratio via isotropic scaling and centered zero-padding.
Maintains pixel-perfect spatial alignment between fundus images and lesion masks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def run_fov_quality_control(
    manifest_df: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    output_md: Optional[Path] = None,
    montage_path: Optional[Path] = None,
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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    records = []
    overlays: List[np.ndarray] = []
    image_ids: List[str] = []

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
            "message": "Genuine 81-image validation remains pending. Segmentation images not found locally.",
        }

    # Evaluate each segmentation image and mask across all manifest records
    for idx, row in manifest_df.iterrows():
        img_id = str(row.get("image_id", f"img_{idx}"))
        img_rel = row.get("processed_image_path", row.get("image_path", ""))
        fov_rel = row.get("fov_mask_path", row.get("mask_fov", ""))

        img_path = project_root / img_rel if img_rel else None
        fov_path = project_root / fov_rel if fov_rel else None

        if not img_path or not img_path.exists() or not fov_path or not fov_path.exists():
            records.append({
                "image_id": img_id,
                "fov_pixels": 0,
                "total_pixels": 0,
                "fov_area_ratio": 0.0,
                "circularity": 0.0,
                "status": "FAILED",
                "flags": "FILE_MISSING",
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
        records.append({
            "image_id": img_id,
            "fov_pixels": qc_res["fov_pixels"],
            "total_pixels": qc_res["total_pixels"],
            "fov_area_ratio": round(qc_res["fov_area_ratio"], 4),
            "circularity": round(qc_res["circularity"], 4),
            "status": qc_res["status"],
            "flags": ";".join(qc_res["flags"]),
        })

        overlay = create_fov_overlay(img_rgb, fov_mask)
        overlays.append(overlay)
        image_ids.append(img_id)

    df_qc = pd.DataFrame(records)
    df_qc.to_csv(output_csv, index=False)

    total_eval = len(df_qc)
    passed_cnt = int((df_qc["status"] == "PASSED").sum())
    failed_cnt = total_eval - passed_cnt
    mean_ratio = float(df_qc["fov_area_ratio"].mean()) if total_eval > 0 else 0.0

    # Generate complete contact sheet montage covering all available overlay images
    if overlays:
        sheet = create_fov_contact_sheet(overlays, image_ids, grid_size=(9, 9), thumb_size=(128, 128))
        montage_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(montage_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        logging.info(f"Complete FOV contact sheet saved to {montage_path} ({sheet.shape[1]}x{sheet.shape[0]} px, {len(overlays)} images)")

    # Overall execution status: can ONLY be PASSED if 0 failures and total_eval >= 81
    overall_status = "PASSED" if (failed_cnt == 0 and total_eval >= 81) else "FAILED"

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
        f"- **Total Segmentation Images Evaluated**: {total_eval}",
        f"- **Valid FOV Masks**: {passed_cnt}",
        f"- **Flagged / Missing / Anomalous Masks**: {failed_cnt}",
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
        "mean_fov_area_ratio": mean_ratio,
        "montage_path": str(montage_path) if overlays else None,
    }


def validate_segmentation_masks(
    manifest_df: pd.DataFrame,
    project_root: Optional[Path] = None,
    target_size: Tuple[int, int] = (512, 512),
) -> Dict[str, Any]:
    """Validate completeness, resolution, binarization, and pixel counts of all lesion masks.

    Checks:
    1. Every expected annotated mask file exists on disk and is non-empty.
    2. Every mask has the exact expected target dimensions (target_size).
    3. Every mask is strictly binary with values in {0, 1} or {0, 255}.
    4. Actual positive pixel count strictly matches manifest pixel_count_{code}.
    5. Rejects negative counts for annotated masks and validates lesion_present_{code}.
    6. Missing annotations remain distinguished (pixel_count = -1) from zero-lesion masks.
    7. Validates composite_pixels against actual composite mask count.

    Args:
        manifest_df: DataFrame from segmentation_manifest.csv.
        project_root: Project root directory.
        target_size: Expected (H, W) resolution (default: (512, 512)).

    Returns:
        Validation statistics dictionary.

    Raises:
        ValueError: If any annotated mask is missing, dimensions mismatch, values are non-binary,
                    or pixel counts / presence flags are inconsistent.
    """
    if project_root is None:
        project_root = Path.cwd()

    lesion_codes = ["MA", "HE", "EX", "SE", "OD"]
    total_checked = 0
    annotated_counts = {c: 0 for c in lesion_codes}
    missing_annotation_counts = {c: 0 for c in lesion_codes}

    for idx, row in manifest_df.iterrows():
        img_id = str(row.get("image_id", f"row_{idx}"))

        for code in lesion_codes:
            mask_col = f"mask_{code}"
            ann_col = f"annotated_{code}"
            mask_rel = row.get(mask_col)

            is_annotated = bool(row.get(ann_col, True))

            if not is_annotated:
                missing_annotation_counts[code] += 1
                cnt_col = f"pixel_count_{code}"
                if cnt_col in row and row[cnt_col] != -1:
                    raise ValueError(
                        f"Image {img_id}: unannotated lesion {code} must have pixel_count = -1, got {row[cnt_col]}"
                    )
                pres_col = f"lesion_present_{code}"
                if pres_col in row and pd.notna(row[pres_col]):
                    raise ValueError(
                        f"Image {img_id}: unannotated lesion {code} must have lesion_present = None/NaN, got {row[pres_col]}"
                    )
                continue

            annotated_counts[code] += 1
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

            cnt_col = f"pixel_count_{code}"
            if cnt_col in row and pd.notna(row[cnt_col]):
                exp_cnt = int(row[cnt_col])
                if exp_cnt < 0:
                    raise ValueError(
                        f"Image {img_id}: annotated mask_{code} has negative pixel count: {exp_cnt}"
                    )
                if actual_pos_pixels != exp_cnt:
                    raise ValueError(
                        f"Image {img_id}: pixel count mismatch for {code}. "
                        f"Actual positive pixels: {actual_pos_pixels}, manifest value: {exp_cnt}"
                    )

            pres_col = f"lesion_present_{code}"
            if pres_col in row and pd.notna(row[pres_col]):
                exp_present = bool(row[pres_col])
                actual_present = bool(actual_pos_pixels > 0)
                if actual_present != exp_present:
                    raise ValueError(
                        f"Image {img_id}: lesion_present disagreement for {code}. "
                        f"Actual pixels > 0: {actual_present}, manifest specifies {exp_present}"
                    )

            total_checked += 1

        # Composite mask validation
        comp_rel = row.get("mask_composite")
        if comp_rel and pd.notna(comp_rel):
            comp_path = project_root / str(comp_rel)
            if not comp_path.exists() or comp_path.stat().st_size == 0:
                raise ValueError(f"Image {img_id}: required composite mask file missing or empty: {comp_path}")
            with Image.open(comp_path) as c_pil:
                c_arr = np.array(c_pil.convert("L"))
            if c_arr.shape != target_size:
                raise ValueError(f"Image {img_id} composite mask resolution {c_arr.shape} != {target_size}")
            if not set(np.unique(c_arr)).issubset({0, 1, 255}):
                raise ValueError(f"Image {img_id} composite mask contains non-binary values: {set(np.unique(c_arr))}")

            actual_comp_pixels = int(np.sum(c_arr > 0))
            for comp_col in ["composite_pixels", "pixel_count_composite"]:
                if comp_col in row and pd.notna(row[comp_col]):
                    exp_comp = int(row[comp_col])
                    if actual_comp_pixels != exp_comp:
                        raise ValueError(
                            f"Image {img_id}: composite pixel count mismatch. "
                            f"Actual: {actual_comp_pixels}, manifest {comp_col}: {exp_comp}"
                        )
            total_checked += 1

    return {
        "status": "PASSED",
        "total_masks_validated": total_checked,
        "annotated_counts": annotated_counts,
        "missing_annotation_counts": missing_annotation_counts,
    }

