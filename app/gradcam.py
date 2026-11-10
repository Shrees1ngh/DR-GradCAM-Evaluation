"""Grad-CAM visualization module for RetinaScope.

Generates Class Activation Maps from ResNet-50 layer4 to explain model predictions
with pixel-accurate alignment on retinal fundus images.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.model import get_gradcam_target_layer

logger = logging.getLogger("gradcam")


def compute_gradcam(
    model: torch.nn.Module,
    tensor_img: torch.Tensor,
    preprocessed_rgb: np.ndarray,
    target_grade: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate Grad-CAM activation heatmap and overlay on the preprocessed image.

    Args:
        model: Trained ResNet-50 instance.
        tensor_img: Preprocessed input tensor of shape (1, 3, 512, 512).
        preprocessed_rgb: uint8 RGB image of shape (512, 512, 3) aligned with tensor.
        target_grade: Target DR grade class index (0-4). If None, defaults to highest scoring class.

    Returns:
        Tuple of:
        - heatmap_rgb: Colorized heatmap as uint8 array (512, 512, 3).
        - overlay_rgb: Blended overlay as uint8 array (512, 512, 3).
        - grayscale_cam: Raw activation values normalized to [0.0, 1.0] of shape (512, 512).
        - stats: Summary statistics (peak salience, coverage at top 10% threshold).

    Raises:
        Exception: If Grad-CAM extraction fails.
    """
    try:
        target_layer = get_gradcam_target_layer(model)
        target_layers = [target_layer]

        # Ensure tensor requires grad for backward pass in GradCAM
        input_tensor = tensor_img.clone().detach()

        # Set up target
        targets = [ClassifierOutputTarget(target_grade)] if target_grade is not None else None

        # Instantiate GradCAM
        with GradCAM(model=model, target_layers=target_layers) as cam:
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Ensure grayscale_cam is float32 in [0, 1]
        grayscale_cam = np.clip(grayscale_cam, 0.0, 1.0).astype(np.float32)

        # 1. Colorized heatmap RGB
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # 2. Composite overlay onto normalized fundus image
        rgb_float = preprocessed_rgb.astype(np.float32) / 255.0
        overlay_rgb = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

        # 3. Calculate salience statistics
        # Peak activation coordinate (y, x)
        peak_y, peak_x = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)

        # Salient area at baseline 10% threshold (consistent with project config)
        salient_mask_10 = grayscale_cam >= 0.10
        coverage_10 = float(np.mean(salient_mask_10) * 100)

        # Highly focused salience (>= 0.50)
        salient_mask_50 = grayscale_cam >= 0.50
        coverage_50 = float(np.mean(salient_mask_50) * 100)

        stats = {
            "peak_activation": float(np.max(grayscale_cam)),
            "mean_activation": float(np.mean(grayscale_cam)),
            "peak_coordinate": (int(peak_x), int(peak_y)),
            "coverage_baseline_10pct": coverage_10,
            "coverage_focused_50pct": coverage_50,
        }

        return heatmap_rgb, overlay_rgb, grayscale_cam, stats

    except Exception as e:
        logger.error(f"Grad-CAM computation failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Grad-CAM visualization generation failed: {str(e)}")

