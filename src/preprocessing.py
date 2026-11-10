"""Image preprocessing and transformation utilities for Diabetic Retinopathy fundus images.

Preserves native retinal aspect ratio via isotropic scaling and centered zero-padding.
Maintains pixel-perfect spatial alignment between fundus images and lesion masks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
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
