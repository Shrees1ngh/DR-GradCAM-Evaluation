"""Grad-CAM Explainability Module for Diabetic Retinopathy Classification (Phase 5).

Provides:
- Native PyTorch Grad-CAM implementation hooked to target convolutional layers (e.g. ResNet-50 layer4).
- Multi-threshold binarization for localization evaluation (top 5%, 10%, 20% salient cuts).
- Grad-CAM heatmap generation, normalization, and RGB fundus visual overlay blending.
- Safe hook management and context manager lifecycle to prevent CUDA memory leaks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = object
    F = None
    TORCH_AVAILABLE = False

from src.model import ICDR_GRADES, get_gradcam_target_layer

logger = logging.getLogger("explainability")


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) explainer.

    Hooks forward feature activations and backward gradients of a target
    convolutional layer (default: `model.layer4` for ResNet-50) to compute
    visual saliency heatmaps for specific disease grades.

    Supports usage as a context manager:
        with GradCAM(model) as explainer:
            heatmap, pred_class = explainer.generate_cam(image_tensor)
    """

    def __init__(
        self,
        model: Any,
        target_layer: Optional[Any] = None,
        use_cuda: Optional[bool] = None,
    ):
        """Initialize Grad-CAM hooks on target convolutional layer.

        Args:
            model: PyTorch classification model (e.g. ResNet-50 from build_model()).
            target_layer: Target convolutional layer module. If None, automatically
                          retrieved via get_gradcam_target_layer(model).
            use_cuda: Explicit device flag. If None, inferred from model parameters.

        Raises:
            ImportError: If PyTorch is not available.
            AttributeError: If target_layer cannot be identified or is invalid.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for GradCAM. Please install torch and torchvision."
            )

        self.model = model
        self.target_layer = target_layer or get_gradcam_target_layer(model)

        if not hasattr(self.target_layer, "register_forward_hook"):
            raise AttributeError(
                f"Target layer {type(self.target_layer)} does not support forward hooks."
            )

        if use_cuda is None:
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_hook_handle: Optional[Any] = None
        self._backward_hook_handle: Optional[Any] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target convolutional layer."""
        def forward_hook(module: Any, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: Any, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
            # grad_output is a tuple containing gradients with respect to output feature map
            if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self._forward_hook_handle = self.target_layer.register_forward_hook(forward_hook)

        # Use register_full_backward_hook for modern PyTorch (>=1.8) compatibility
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self._backward_hook_handle = self.target_layer.register_full_backward_hook(backward_hook)
        else:
            self._backward_hook_handle = self.target_layer.register_backward_hook(backward_hook)

        logger.debug(f"GradCAM hooks registered on target layer: {self.target_layer}")

    def remove_hooks(self) -> None:
        """Unregister hooks to prevent memory leaks and dangling tensor references."""
        if getattr(self, "_forward_hook_handle", None) is not None:
            self._forward_hook_handle.remove()
            self._forward_hook_handle = None
        if getattr(self, "_backward_hook_handle", None) is not None:
            self._backward_hook_handle.remove()
            self._backward_hook_handle = None
        self.activations = None
        self.gradients = None
        logger.debug("GradCAM hooks removed.")

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()

    def __del__(self) -> None:
        self.remove_hooks()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
        retain_graph: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """Generate normalized 2D Grad-CAM heatmap for a single fundus image.

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W) or (C, H, W).
            target_class: Integer class index to explain (0-4). If None, explains
                          the model's top predicted class (argmax logits).
            target_size: Desired output (height, width) for spatial upsampling.
                         If None, uses input_tensor's spatial dimensions (H, W).
            retain_graph: Whether to retain the computational graph during backward.

        Returns:
            heatmap: 2D numpy array of shape (target_h, target_w) with float32
                     values normalized to [0.0, 1.0].
            target_class: The integer class index that was explained.

        Raises:
            ValueError: If input tensor dimensions are invalid or hook capture fails.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(
                f"input_tensor must be shape (1, C, H, W) or (C, H, W), got {input_tensor.shape}"
            )

        orig_h, orig_w = input_tensor.shape[2], input_tensor.shape[3]
        out_size = target_size or (orig_h, orig_w)

        with torch.enable_grad():
            input_tensor = input_tensor.to(self.device)
            input_tensor.requires_grad_(True)

            # Set model in evaluation mode
            self.model.eval()
            self.model.zero_grad()

            # Forward pass creates computation graph from input_tensor through target layer
            logits = self.model(input_tensor)

            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())

            if target_class < 0 or target_class >= logits.size(1):
                raise ValueError(
                    f"target_class {target_class} is out of bounds for model with {logits.size(1)} classes."
                )

            # Backward pass on the target class score to compute activation gradients
            score = logits[0, target_class]
            score.backward(retain_graph=retain_graph)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM failed to capture activations or gradients. "
                "Ensure target_layer participates in the forward/backward graph."
            )

        # Global average pooling of gradients: alpha_k = 1/Z * sum_{i,j} dY^c / dA^k_{i,j}
        # shape: (1, C_channels, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Linear combination of weighted feature maps
        # cam shape: (1, 1, H_feat, W_feat)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to retain only positive contributions toward the target class
        cam = F.relu(cam)

        # Bilinear upsampling to target spatial dimensions
        cam = F.interpolate(
            cam,
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )

        heatmap = cam.squeeze().cpu().detach().numpy().astype(np.float32)

        # Min-max normalization into [0.0, 1.0]
        h_min = float(np.min(heatmap))
        h_max = float(np.max(heatmap))
        denom = h_max - h_min

        if denom > 1e-8:
            heatmap = (heatmap - h_min) / denom
        else:
            heatmap = np.zeros(out_size, dtype=np.float32)

        return heatmap, target_class

    def generate_batch_cam(
        self,
        input_tensors: torch.Tensor,
        target_classes: Optional[List[Optional[int]]] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[np.ndarray, int]]:
        """Generate Grad-CAM heatmaps for a batch of images.

        Args:
            input_tensors: Batch tensor of shape (B, C, H, W).
            target_classes: Optional list of target classes per image.
            target_size: Output spatial dimensions.

        Returns:
            List of (heatmap, target_class) tuples for each image in batch.
        """
        batch_size = input_tensors.size(0)
        if target_classes is None:
            target_classes = [None] * batch_size

        results = []
        for i in range(batch_size):
            single_input = input_tensors[i:i + 1]
            t_class = target_classes[i]
            cam_map, cls_idx = self.generate_cam(
                single_input, target_class=t_class, target_size=target_size
            )
            results.append((cam_map, cls_idx))

        return results


def validate_fov_mask(
    fov_mask: Optional[np.ndarray],
    expected_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Validate retinal field of view (FOV) mask geometry and binary values.

    Args:
        fov_mask: Optional 2D array representing the binary retinal FOV mask.
        expected_shape: Expected (height, width) tuple.

    Returns:
        Boolean numpy array of shape (H, W) where True indicates valid retinal FOV,
        or None if fov_mask is None.

    Raises:
        ValueError: If fov_mask shape is invalid, values are non-binary, or mask is empty.
    """
    if fov_mask is None:
        return None
    if fov_mask.ndim != 2:
        raise ValueError(f"FOV mask must be 2-dimensional (H, W), got shape {fov_mask.shape}")
    if fov_mask.shape != expected_shape:
        raise ValueError(f"FOV mask shape {fov_mask.shape} does not match expected shape {expected_shape}")
    u_vals = set(np.unique(fov_mask))
    if not u_vals.issubset({0, 1, 255}):
        raise ValueError(f"FOV mask contains non-binary values: {u_vals}")
    fov_bool = (fov_mask > 0)
    if int(np.sum(fov_bool)) == 0:
        raise ValueError("FOV mask is empty (contains 0 positive pixels).")
    return fov_bool


def check_heatmap_informativeness(
    heatmap: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
) -> Tuple[bool, str]:
    """Check if a heatmap contains meaningful spatial contrast within the valid retinal FOV.

    A heatmap is considered non-informative if:
    - All activations within the FOV are zero or practically zero (max <= 1e-8)
    - The peak-to-peak variation (max - min) within the FOV is <= tolerance (default: 1e-6)
    - The standard deviation within the FOV is <= 1e-7

    Args:
        heatmap: 2D numpy array (H, W).
        fov_mask: Optional binary mask (H, W) restricting evaluation to valid retinal tissue.
        tolerance: Numerical threshold for peak-to-peak variation.

    Returns:
        is_informative: True if heatmap has informative spatial variation, False otherwise.
        status: 'ok' if informative, or 'non_informative_heatmap' if flat/zero/near-constant.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2-dimensional (H, W), got shape {heatmap.shape}")

    fov_bool = validate_fov_mask(fov_mask, heatmap.shape)
    if fov_bool is not None:
        vals = heatmap[fov_bool]
    else:
        vals = heatmap.ravel()

    if len(vals) == 0:
        return False, "non_informative_heatmap"

    # Explicitly detect NaN, positive infinity, and negative infinity within valid FOV
    if not np.all(np.isfinite(vals)):
        nan_cnt = int(np.isnan(vals).sum())
        pos_inf_cnt = int(np.isposinf(vals).sum())
        neg_inf_cnt = int(np.isneginf(vals).sum())
        details = []
        if nan_cnt > 0:
            details.append(f"{nan_cnt} NaN")
        if pos_inf_cnt > 0:
            details.append(f"{pos_inf_cnt} +Inf")
        if neg_inf_cnt > 0:
            details.append(f"{neg_inf_cnt} -Inf")
        detail_str = ", ".join(details)
        raise ValueError(
            f"Heatmap contains non-finite values ({detail_str}) within valid retinal FOV."
        )

    max_val = float(np.max(vals))
    min_val = float(np.min(vals))
    ptp = max_val - min_val

    if max_val <= 1e-8 or ptp <= tolerance or float(np.std(vals)) <= 1e-7:
        return False, "non_informative_heatmap"

    return True, "ok"


def binarize_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.10,
    method: str = "percentile",
    fov_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Binarize a continuous [0, 1] Grad-CAM heatmap into a binary saliency mask {0, 1}.

    When fov_mask is provided, saliency thresholding and pixel selection are strictly
    confined to the valid retinal field of view:
    - Rejects invalid or empty FOV masks.
    - Zeroes out any activations outside the FOV mask.
    - For method='percentile', the selected pixel percentage is computed strictly
      relative to the count of valid retinal FOV pixels (not the full image canvas).
    - Resolves ties deterministically using coordinate raster ordering.
    - If the heatmap is constant, near-constant, or zero within the FOV (non-informative),
      returns an empty mask without selecting arbitrary pixels.

    Args:
        heatmap: 2D numpy array (H, W) with values in [0.0, 1.0].
        threshold: Cutoff value.
            - If method='percentile': Fraction of top salient pixels to keep (e.g. 0.10 = top 10%).
            - If method='relative': Cutoff as fraction of peak activation (T = threshold * max).
            - If method='absolute': Absolute cutoff value in [0, 1].
        method: One of 'percentile' (default), 'relative', or 'absolute'.
        fov_mask: Optional binary mask (H, W) defining valid retinal foreground pixels.

    Returns:
        binary_mask: 2D uint8 numpy array of shape (H, W) strictly with values in {0, 1}.

    Raises:
        ValueError: If method is unrecognized, threshold is out of range, or fov_mask dimensions/values are invalid.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2-dimensional (H, W), got shape {heatmap.shape}")

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Threshold must be within [0.0, 1.0], got {threshold}")

    if method not in ("percentile", "relative", "absolute"):
        raise ValueError(
            f"Unrecognized thresholding method: '{method}'. Choose 'percentile', 'relative', or 'absolute'."
        )

    fov_bool = validate_fov_mask(fov_mask, heatmap.shape)
    fov_pixels = int(np.sum(fov_bool)) if fov_bool is not None else heatmap.size

    # Check informativeness: do not select arbitrary pixels from flat heatmaps
    is_inf, _ = check_heatmap_informativeness(heatmap, fov_mask=fov_mask)
    if not is_inf:
        return np.zeros_like(heatmap, dtype=np.uint8)

    if method == "percentile":
        if fov_bool is not None:
            # Deterministic, FOV-confined top-k selection
            k = int(round(threshold * fov_pixels))
            if k <= 0:
                return np.zeros_like(heatmap, dtype=np.uint8)
            if k >= fov_pixels:
                return fov_bool.astype(np.uint8)

            fov_y, fov_x = np.where(fov_bool)
            fov_vals = heatmap[fov_y, fov_x]

            flat_coords = np.ravel_multi_index((fov_y, fov_x), heatmap.shape)
            # Deterministic tie-breaking: highest activation first, ties broken by coordinate raster order
            sorted_order = np.lexsort((flat_coords, -fov_vals))
            top_k_coords = flat_coords[sorted_order[:k]]

            binary_mask = np.zeros_like(heatmap, dtype=np.uint8)
            binary_mask.flat[top_k_coords] = 1
            return binary_mask

        else:
            # Full canvas percentile thresholding (backward compatible)
            k = int(round(threshold * fov_pixels))
            if k <= 0:
                return np.zeros_like(heatmap, dtype=np.uint8)
            if k >= fov_pixels:
                return np.ones_like(heatmap, dtype=np.uint8)

            flat_vals = heatmap.ravel()
            flat_coords = np.arange(heatmap.size)
            sorted_order = np.lexsort((flat_coords, -flat_vals))
            top_k_coords = flat_coords[sorted_order[:k]]

            binary_mask = np.zeros_like(heatmap, dtype=np.uint8)
            binary_mask.flat[top_k_coords] = 1
            return binary_mask

    elif method == "relative":
        if fov_bool is not None:
            fov_vals = heatmap[fov_bool]
            max_val = float(np.max(fov_vals)) if len(fov_vals) > 0 else 0.0
            if max_val <= 1e-8:
                return np.zeros_like(heatmap, dtype=np.uint8)
            cutoff = threshold * max_val
            return ((heatmap >= cutoff) & fov_bool).astype(np.uint8)
        else:
            max_val = float(np.max(heatmap))
            if max_val <= 1e-8:
                return np.zeros_like(heatmap, dtype=np.uint8)
            cutoff = threshold * max_val
            return (heatmap >= cutoff).astype(np.uint8)

    elif method == "absolute":
        if fov_bool is not None:
            return ((heatmap >= threshold) & fov_bool).astype(np.uint8)
        else:
            return (heatmap >= threshold).astype(np.uint8)

    else:
        raise ValueError(
            f"Unrecognized binarization method '{method}'. Must be 'percentile', 'relative', or 'absolute'."
        )

    return binary_mask


def generate_gradcam_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a continuous Grad-CAM heatmap with an RGB fundus photograph.

    Args:
        image_rgb: RGB image array of shape (H, W, 3) and dtype uint8.
        heatmap: 2D float array of shape (H, W) with values normalized to [0.0, 1.0].
        alpha: Heatmap blend weight in [0.0, 1.0] (default: 0.5).
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).

    Returns:
        overlay: Blended RGB uint8 image of shape (H, W, 3).
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be shape (H, W, 3), got {image_rgb.shape}")

    h, w = image_rgb.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Scale to 0-255 uint8 for colormap application
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blending: overlay = alpha * heatmap + (1 - alpha) * original_image
    overlay = cv2.addWeighted(
        heatmap_colored_rgb, alpha, image_rgb.astype(np.uint8), 1.0 - alpha, 0
    )
    return overlay


def build_explanation_summary(
    image_id: str,
    true_grade: int,
    predicted_grade: int,
    prediction_confidence: float,
    heatmap: np.ndarray,
    percentile_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Package explanation metadata and multi-threshold masks into a structured dict.

    Args:
        image_id: Unique image identifier.
        true_grade: Ground-truth ICDR grade (0-4).
        predicted_grade: Model predicted ICDR grade (0-4).
        prediction_confidence: Softmax probability for predicted grade.
        heatmap: 2D normalized Grad-CAM heatmap.
        percentile_thresholds: List of percentile cutoffs (default: [0.05, 0.10, 0.20]).

    Returns:
        Structured dictionary with saliency metrics and binary masks.
    """
    thresholds = percentile_thresholds or [0.05, 0.10, 0.20]
    binary_masks = {}
    pixel_counts = {}

    for t in thresholds:
        key = f"top_{int(round(t * 100))}_pct"
        mask = binarize_heatmap(heatmap, threshold=t, method="percentile")
        binary_masks[key] = mask
        pixel_counts[key] = int(np.sum(mask))

    return {
        "image_id": image_id,
        "true_grade": int(true_grade),
        "true_grade_name": ICDR_GRADES.get(int(true_grade), "Unknown"),
        "predicted_grade": int(predicted_grade),
        "predicted_grade_name": ICDR_GRADES.get(int(predicted_grade), "Unknown"),
        "prediction_confidence": float(prediction_confidence),
        "heatmap_shape": list(heatmap.shape),
        "heatmap_mean": float(np.mean(heatmap)),
        "heatmap_std": float(np.std(heatmap)),
        "salient_pixel_counts": pixel_counts,
        "binary_masks": binary_masks,
    }

