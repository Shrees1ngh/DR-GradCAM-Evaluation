"""Faithfulness Evaluation Module: Deletion Test and Random-Region Controls (Phase 7).

Evaluates whether Grad-CAM highlighted regions actually drive the CNN's classification
confidence (decision fidelity) via progressive region ablation and area-matched
random controls.

Provides:
- Progressive salient region ablation (e.g. 10%, 20%, 30% of peak activations).
- Area-matched random-region control baseline (with repeated trials).
- Multiple image fill strategies:
  * 'black': Primary zero-fill (0, 0, 0).
  * 'gaussian_blur': Neutral blurred fill to control for high-frequency edge/OOD artifacts.
- Metric calculations: Confidence Drop (delta p), Deletion Curves, and Area Under Deletion Curve (AUDC).
- Faithfulness Margin: delta p_salient - delta p_random.
- Grade-wise and cohort-level aggregation across all ICDR grades 0-4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.explainability import binarize_heatmap
from src.model import ICDR_GRADES

logger = logging.getLogger("faithfulness")


def apply_ablation_mask(
    image_rgb: np.ndarray,
    ablation_mask: np.ndarray,
    fill_strategy: str = "black",
    blur_kernel_size: int = 21,
) -> np.ndarray:
    """Ablate masked pixels in an RGB fundus image using a specified fill strategy.

    Args:
        image_rgb: Input image array of shape (H, W, 3) and dtype uint8.
        ablation_mask: Binary mask of shape (H, W) where 1 indicates pixels to ablate.
        fill_strategy: Method to replace ablated pixels:
            - 'black': Zero out pixels to (0, 0, 0).
            - 'gaussian_blur': Replace with Gaussian-blurred version of the original image
                               to avoid artificial high-frequency edge discontinuities.
            - 'mean': Replace with the image's mean RGB color.
        blur_kernel_size: Odd integer kernel size for Gaussian blur fill (default: 21).

    Returns:
        Ablated image of shape (H, W, 3) and dtype uint8.

    Raises:
        ValueError: If array dimensions do not match or strategy is unrecognized.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be shape (H, W, 3), got {image_rgb.shape}")

    h, w = image_rgb.shape[:2]
    if ablation_mask.shape != (h, w):
        raise ValueError(f"Mask shape {ablation_mask.shape} does not match image {(h, w)}")

    mask_bool = ablation_mask > 0
    ablated = image_rgb.copy()

    if not np.any(mask_bool):
        return ablated

    if fill_strategy == "black":
        ablated[mask_bool] = 0

    elif fill_strategy == "gaussian_blur":
        # Ensure kernel size is odd and positive
        k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        blurred = cv2.GaussianBlur(image_rgb, (k, k), 0)
        ablated[mask_bool] = blurred[mask_bool]

    elif fill_strategy == "mean":
        mean_color = np.mean(image_rgb, axis=(0, 1)).astype(np.uint8)
        ablated[mask_bool] = mean_color

    else:
        raise ValueError(
            f"Unrecognized fill_strategy '{fill_strategy}'. Must be 'black', 'gaussian_blur', or 'mean'."
        )

    return ablated


def generate_random_ablation_mask(
    target_pixel_count: int,
    shape: Tuple[int, int],
    valid_fov_mask: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate an area-matched random binary mask strictly confined to the retinal FOV.

    Ensures that the random ablation control replaces the exact same number of pixels
    as the salient mask without cheating by placing masks in the empty black border padding.

    Args:
        target_pixel_count: Exact number of pixels to activate (value 1).
        shape: Output dimensions (height, width).
        valid_fov_mask: Optional binary mask (H, W) defining valid retinal foreground pixels.
                        If None, any non-border pixel in shape is eligible.
        rng: Optional NumPy random Generator instance for reproducibility.

    Returns:
        Binary mask array of shape (H, W) and dtype uint8 with sum == target_pixel_count.
    """
    h, w = shape
    random_gen = rng or np.random.default_rng()

    if valid_fov_mask is not None:
        eligible_indices = np.flatnonzero(valid_fov_mask > 0)
    else:
        eligible_indices = np.arange(h * w)

    total_eligible = len(eligible_indices)
    actual_count = min(target_pixel_count, total_eligible)

    if actual_count <= 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Randomly sample indices without replacement
    chosen_indices = random_gen.choice(eligible_indices, size=actual_count, replace=False)

    mask = np.zeros(h * w, dtype=np.uint8)
    mask[chosen_indices] = 1
    return mask.reshape((h, w))


def run_single_deletion_step(
    model: Any,
    image_rgb: np.ndarray,
    ablation_mask: np.ndarray,
    target_class: int,
    device: Any,
    fill_strategy: str = "black",
    transform_fn: Optional[Callable] = None,
) -> Tuple[float, np.ndarray]:
    """Execute a single ablation pass and measure model confidence for target_class.

    Args:
        model: PyTorch classification model.
        image_rgb: RGB image array of shape (H, W, 3).
        ablation_mask: Binary mask of pixels to remove.
        target_class: Integer class index to observe.
        device: PyTorch device.
        fill_strategy: 'black' or 'gaussian_blur'.
        transform_fn: Transform pipeline to convert uint8 RGB image to normalized Tensor.

    Returns:
        post_prob: Softmax probability for target_class after ablation.
        ablated_image: The ablated RGB image array.
    """
    ablated_rgb = apply_ablation_mask(image_rgb, ablation_mask, fill_strategy=fill_strategy)

    if transform_fn is not None:
        input_tensor = transform_fn(ablated_rgb).unsqueeze(0).to(device)
    else:
        # Default fallback normalization
        norm = (ablated_rgb / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        input_tensor = torch.tensor(norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    post_prob = float(probs[target_class])
    return post_prob, ablated_rgb


def evaluate_progressive_deletion(
    model: Any,
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    target_class: int,
    device: Any,
    image_id: str,
    true_grade: int,
    deletion_steps: Optional[List[float]] = None,
    fill_strategies: Optional[List[str]] = None,
    num_random_trials: int = 5,
    valid_fov_mask: Optional[np.ndarray] = None,
    transform_fn: Optional[Callable] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute full multi-step progressive deletion test with random-region controls.

    Args:
        model: PyTorch classification model.
        image_rgb: RGB image (H, W, 3).
        heatmap: 2D normalized Grad-CAM heatmap (H, W).
        target_class: Predicted or evaluated class index (0-4).
        device: PyTorch device.
        image_id: Unique image identifier.
        true_grade: Ground-truth ICDR grade (0-4).
        deletion_steps: Fraction cuts to ablate, e.g. [0.10, 0.20, 0.30] (default).
        fill_strategies: List of fills to test, e.g. ['black', 'gaussian_blur'].
        num_random_trials: Number of random area-matched masks per step (default: 5).
        valid_fov_mask: Mask defining retinal boundary to prevent cheating in black borders.
        transform_fn: Image transform callable.
        seed: Random seed for reproducible random controls.

    Returns:
        Structured dictionary containing deletion curves, drops, and faithfulness margins.
    """
    steps = deletion_steps or [0.10, 0.20, 0.30]
    strategies = fill_strategies or ["black", "gaussian_blur"]
    rng = np.random.default_rng(seed)

    # 1. Baseline prediction on unperturbed original image
    if transform_fn is not None:
        base_tensor = transform_fn(image_rgb).unsqueeze(0).to(device)
    else:
        norm = (image_rgb / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        base_tensor = torch.tensor(norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        base_logits = model(base_tensor)
        base_probs = torch.softmax(base_logits, dim=1).cpu().numpy()[0]

    baseline_prob = float(base_probs[target_class])
    predicted_class = int(np.argmax(base_probs))

    results: Dict[str, Any] = {
        "image_id": image_id,
        "true_grade": int(true_grade),
        "predicted_grade": predicted_class,
        "target_class": int(target_class),
        "baseline_confidence": baseline_prob,
        "deletion_steps": steps,
        "evaluations_by_fill": {},
    }

    # Derive retinal FOV mask if not explicitly supplied (pixels > 10 in green channel)
    if valid_fov_mask is None:
        valid_fov_mask = (image_rgb[:, :, 1] > 10).astype(np.uint8)

    for fill in strategies:
        step_records = []
        salient_drops = []
        random_drops = []

        for step in steps:
            # Saliency ablation mask: Top step fraction of hottest activations
            salient_mask = binarize_heatmap(heatmap, threshold=step, method="percentile")
            pixel_count = int(np.sum(salient_mask))

            # Run salient ablation pass
            salient_prob, _ = run_single_deletion_step(
                model=model,
                image_rgb=image_rgb,
                ablation_mask=salient_mask,
                target_class=target_class,
                device=device,
                fill_strategy=fill,
                transform_fn=transform_fn,
            )
            delta_p_salient = float(baseline_prob - salient_prob)
            salient_drops.append(delta_p_salient)

            # Run repeated area-matched random ablation control passes
            rand_probs = []
            for _ in range(num_random_trials):
                rand_mask = generate_random_ablation_mask(
                    target_pixel_count=pixel_count,
                    shape=image_rgb.shape[:2],
                    valid_fov_mask=valid_fov_mask,
                    rng=rng,
                )
                r_prob, _ = run_single_deletion_step(
                    model=model,
                    image_rgb=image_rgb,
                    ablation_mask=rand_mask,
                    target_class=target_class,
                    device=device,
                    fill_strategy=fill,
                    transform_fn=transform_fn,
                )
                rand_probs.append(r_prob)

            mean_rand_prob = float(np.mean(rand_probs))
            delta_p_random = float(baseline_prob - mean_rand_prob)
            random_drops.append(delta_p_random)

            # Faithfulness Margin: Does deleting salient features hurt more than random?
            # Positive margin -> Faithful explanation
            # Negative margin -> Unfaithful / spurious explanation
            margin = float(delta_p_salient - delta_p_random)

            step_records.append({
                "step_fraction": float(step),
                "ablated_pixels": pixel_count,
                "salient_post_prob": salient_prob,
                "delta_p_salient": delta_p_salient,
                "random_mean_post_prob": mean_rand_prob,
                "delta_p_random": delta_p_random,
                "faithfulness_margin": margin,
                "is_faithful_step": bool(margin > 0.0),
            })

        # Calculate Area Under Deletion Curve (mean drop across steps)
        audc_salient = float(np.mean(salient_drops)) if salient_drops else 0.0
        audc_random = float(np.mean(random_drops)) if random_drops else 0.0
        audc_margin = float(audc_salient - audc_random)

        results["evaluations_by_fill"][fill] = {
            "steps": step_records,
            "audc_salient": audc_salient,
            "audc_random": audc_random,
            "audc_margin": audc_margin,
            "overall_is_faithful": bool(audc_margin > 0.0),
        }

    return results


def aggregate_faithfulness_scores(
    results: List[Dict[str, Any]],
    primary_fill: str = "black",
    baseline_step_index: int = 0,  # 0 -> 10% step
) -> Dict[str, Any]:
    """Aggregate per-image faithfulness and deletion metrics overall and by grade.

    Args:
        results: List of result dictionaries from evaluate_progressive_deletion().
        primary_fill: Primary fill strategy to aggregate ('black' or 'gaussian_blur').
        baseline_step_index: Index of the primary deletion step for table reporting.

    Returns:
        Structured summary with mean drops, margins, and grade-wise breakdowns.
    """
    total_images = len(results)
    grade_counts: Dict[int, int] = {g: 0 for g in range(5)}
    for r in results:
        g = r["true_grade"]
        grade_counts[g] = grade_counts.get(g, 0) + 1

    summary: Dict[str, Any] = {
        "total_images": total_images,
        "grade_distribution": grade_counts,
        "primary_fill_strategy": primary_fill,
        "overall": {},
        "grade_wise": {},
    }

    all_audc_salient = []
    all_audc_random = []
    all_audc_margins = []
    faithful_count = 0

    grade_audc_margins: Dict[int, List[float]] = {g: [] for g in range(5)}
    grade_salient_drops: Dict[int, List[float]] = {g: [] for g in range(5)}
    grade_random_drops: Dict[int, List[float]] = {g: [] for g in range(5)}

    for r in results:
        g = r["true_grade"]
        fill_eval = r["evaluations_by_fill"].get(primary_fill, {})

        audc_s = fill_eval.get("audc_salient", 0.0)
        audc_r = fill_eval.get("audc_random", 0.0)
        audc_m = fill_eval.get("audc_margin", 0.0)

        all_audc_salient.append(audc_s)
        all_audc_random.append(audc_r)
        all_audc_margins.append(audc_m)

        if fill_eval.get("overall_is_faithful", False):
            faithful_count += 1

        if g in grade_audc_margins:
            grade_audc_margins[g].append(audc_m)
            grade_salient_drops[g].append(audc_s)
            grade_random_drops[g].append(audc_r)

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

    summary["overall"] = {
        "audc_salient": _calc_stats(all_audc_salient),
        "audc_random": _calc_stats(all_audc_random),
        "audc_margin": _calc_stats(all_audc_margins),
        "faithful_cases_count": faithful_count,
        "faithful_rate_pct": float(faithful_count / max(1, total_images) * 100.0),
    }

    for g in range(5):
        summary["grade_wise"][f"grade_{g}"] = {
            "audc_salient": _calc_stats(grade_salient_drops[g]),
            "audc_random": _calc_stats(grade_random_drops[g]),
            "audc_margin": _calc_stats(grade_audc_margins[g]),
        }

    return summary

