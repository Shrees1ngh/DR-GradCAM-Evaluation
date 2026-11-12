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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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

from src.preprocessing import extract_retinal_fov_mask

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.explainability import (
    binarize_heatmap,
    check_heatmap_informativeness,
    validate_fov_mask,
)
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

    valid_strategies = {"black", "gaussian_blur", "mean"}
    if fill_strategy not in valid_strategies:
        raise ValueError(
            f"Unrecognized fill_strategy '{fill_strategy}'. Must be 'black', 'gaussian_blur', or 'mean'."
        )

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

    return ablated


def generate_fov_restricted_salient_mask(
    heatmap: np.ndarray,
    valid_fov_mask: np.ndarray,
    fraction: float = 0.10,
) -> Tuple[np.ndarray, int]:
    """Generate a binary salient mask strictly confined to the retinal FOV.

    Extracts the top `fraction` of hottest Grad-CAM activations computed relative
    to the valid retinal surface area (excluding empty padding/background).
    Uses the exact same deterministic FOV-restricted percentile selection and
    tie-breaking logic as binarize_heatmap.

    Guarantees:
    1. Exactly round(fraction * N_fov) pixels are selected for informative heatmaps.
    2. Zero pixels outside the valid retinal FOV are selected.
    3. For non-informative heatmaps (flat, zero, or near-constant), returns an all-zero
       mask with 0 count, preventing arbitrary pixel deletions.

    Args:
        heatmap: 2D array of continuous Grad-CAM activations (H, W).
        valid_fov_mask: 2D binary mask (H, W) where 1 indicates valid retinal tissue.
        fraction: Top fraction to select relative to retinal FOV (e.g. 0.10 for top 10%).

    Returns:
        salient_mask: Binary mask array of shape (H, W) and dtype uint8.
        pixel_count: Exact number of activated pixels.
    """
    if valid_fov_mask is None:
        raise ValueError("valid_fov_mask is required.")

    fov_bool = validate_fov_mask(valid_fov_mask, heatmap.shape)
    is_inf, _ = check_heatmap_informativeness(heatmap, fov_mask=valid_fov_mask)
    if not is_inf or fraction <= 0.0:
        h, w = heatmap.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), 0

    salient_mask = binarize_heatmap(
        heatmap=heatmap,
        threshold=fraction,
        method="percentile",
        fov_mask=valid_fov_mask,
    )
    actual_count = int(np.sum(salient_mask))
    return salient_mask, actual_count


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


def compute_normalized_pauc(steps: Sequence[float], drops: Sequence[float]) -> float:
    """Compute normalized trapezoidal partial-AUC anchored at baseline point (0, 0).

    Formula:
        pAUC = sum_{i=1}^M 0.5 * (y_{i-1} + y_i) * (x_i - x_{i-1})
        normalized_pauc = pAUC / x_M

    Normalized by maximum deletion fraction x_M so the resulting metric is dimensionally
    consistent with confidence drop values in [0.0, 1.0].

    Args:
        steps: Sequence of ablation fractions, e.g. [0.10, 0.20, 0.30].
        drops: Sequence of corresponding probability drops.

    Returns:
        float normalized partial-AUC in [0.0, 1.0].
    """
    if not steps and not drops:
        return 0.0

    if len(steps) != len(drops):
        raise ValueError(
            f"steps and drops must have the same non-zero length: got len(steps)={len(steps)}, len(drops)={len(drops)}"
        )

    for i, s in enumerate(steps):
        if s <= 0.0:
            raise ValueError(
                f"steps must be strictly positive and monotonically increasing: non-positive step {s} at index {i}"
            )
        if i > 0 and s <= steps[i - 1]:
            raise ValueError(
                f"steps must be strictly positive and monotonically increasing: step {s} <= previous {steps[i-1]}"
            )

    x_vals = [0.0] + [float(s) for s in steps]
    y_vals = [0.0] + [float(d) for d in drops]

    x_max = x_vals[-1]
    if x_max <= 0.0:
        return 0.0

    pauc = 0.0
    for i in range(1, len(x_vals)):
        dx = x_vals[i] - x_vals[i - 1]
        y_avg = 0.5 * (y_vals[i - 1] + y_vals[i])
        pauc += y_avg * dx

    return float(pauc / x_max)


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
        "true_grade": int(true_grade) if true_grade is not None else None,
        "predicted_grade": predicted_class,
        "target_class": int(target_class),
        "baseline_confidence": baseline_prob,
        "deletion_steps": steps,
        "evaluations_by_fill": {},
    }

    # Derive retinal FOV mask if not explicitly supplied
    if valid_fov_mask is None:
        valid_fov_mask = extract_retinal_fov_mask(image_rgb)

    try:
        is_inf, heatmap_status = check_heatmap_informativeness(heatmap, fov_mask=valid_fov_mask)
    except ValueError as e:
        logger = logging.getLogger("faithfulness")
        logger.error(f"Invalid heatmap encountered for image '{image_id}': {e}")
        raise ValueError(f"Faithfulness evaluation failed for image '{image_id}': {e}") from e
    results["heatmap_status"] = heatmap_status
    results["is_informative"] = is_inf

    if not is_inf:
        for fill in strategies:
            results["evaluations_by_fill"][fill] = {
                "heatmap_status": heatmap_status,
                "is_informative": False,
                "steps": [],
                "mean_deletion_drop_salient": None,
                "mean_deletion_drop_random": None,
                "mean_deletion_margin": None,
                "audc_salient": None,
                "audc_random": None,
                "audc_margin": None,
                "normalized_pauc_salient": None,
                "normalized_pauc_random": None,
                "normalized_pauc_margin": None,
                "overall_is_faithful": None,
            }
        return results

    for fill in strategies:
        step_records = []
        salient_drops = []
        random_drops = []

        for step in steps:
            # Saliency ablation mask: Top step fraction of hottest activations strictly within valid retinal FOV
            salient_mask, pixel_count = generate_fov_restricted_salient_mask(
                heatmap=heatmap,
                valid_fov_mask=valid_fov_mask,
                fraction=step,
            )

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

        # 1. Discrete average confidence drops across evaluated steps
        mean_s = float(np.mean(salient_drops)) if salient_drops else 0.0
        mean_r = float(np.mean(random_drops)) if random_drops else 0.0
        mean_m = float(mean_s - mean_r)

        # 2. Normalized trapezoidal partial-AUC from baseline (0, 0)
        pauc_s = compute_normalized_pauc(steps, salient_drops)
        pauc_r = compute_normalized_pauc(steps, random_drops)
        pauc_m = float(pauc_s - pauc_r)

        results["evaluations_by_fill"][fill] = {
            "steps": step_records,
            # Discrete mean drop metrics
            "mean_deletion_drop_salient": mean_s,
            "mean_deletion_drop_random": mean_r,
            "mean_deletion_margin": mean_m,
            # Backward-compatible aliases
            "audc_salient": mean_s,
            "audc_random": mean_r,
            "audc_margin": mean_m,
            # Normalized trapezoidal partial-AUC metrics
            "normalized_pauc_salient": pauc_s,
            "normalized_pauc_random": pauc_r,
            "normalized_pauc_margin": pauc_m,
            "overall_is_faithful": bool(pauc_m > 0.0),
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
    grade_counts: Dict[Union[int, str], int] = {g: 0 for g in range(5)}
    grade_counts["unavailable"] = 0
    for r in results:
        g = r.get("true_grade")
        if g is not None and isinstance(g, int) and 0 <= g <= 4:
            grade_counts[g] = grade_counts.get(g, 0) + 1
        else:
            grade_counts["unavailable"] = grade_counts.get("unavailable", 0) + 1

    summary: Dict[str, Any] = {
        "total_images": total_images,
        "grade_distribution": grade_counts,
        "primary_fill_strategy": primary_fill,
        "overall": {},
        "grade_wise": {},
    }

    non_informative_count = sum(1 for r in results if not r.get("is_informative", True))
    summary["non_informative_images"] = non_informative_count

    all_audc_salient = []
    all_audc_random = []
    all_audc_margins = []
    all_pauc_salient = []
    all_pauc_random = []
    all_pauc_margins = []
    faithful_count = 0
    informative_count = 0

    grade_pauc_margins: Dict[int, List[float]] = {g: [] for g in range(5)}
    grade_pauc_salient: Dict[int, List[float]] = {g: [] for g in range(5)}
    grade_pauc_random: Dict[int, List[float]] = {g: [] for g in range(5)}
    grade_audc_margins: Dict[int, List[float]] = {g: [] for g in range(5)}
    grade_salient_drops: Dict[int, List[float]] = {g: [] for g in range(5)}
    grade_random_drops: Dict[int, List[float]] = {g: [] for g in range(5)}

    for r in results:
        g = r.get("true_grade")
        fill_eval = r["evaluations_by_fill"].get(primary_fill, {})

        if not r.get("is_informative", True) or fill_eval.get("is_informative") is False:
            continue

        audc_s = fill_eval.get("mean_deletion_drop_salient", fill_eval.get("audc_salient"))
        audc_r = fill_eval.get("mean_deletion_drop_random", fill_eval.get("audc_random"))
        audc_m = fill_eval.get("mean_deletion_margin", fill_eval.get("audc_margin"))

        pauc_s = fill_eval.get("normalized_pauc_salient", audc_s)
        pauc_r = fill_eval.get("normalized_pauc_random", audc_r)
        pauc_m = fill_eval.get("normalized_pauc_margin", audc_m)

        if audc_s is not None and audc_r is not None and audc_m is not None:
            all_audc_salient.append(float(audc_s))
            all_audc_random.append(float(audc_r))
            all_audc_margins.append(float(audc_m))

        if pauc_s is not None and pauc_r is not None and pauc_m is not None:
            all_pauc_salient.append(float(pauc_s))
            all_pauc_random.append(float(pauc_r))
            all_pauc_margins.append(float(pauc_m))

        informative_count += 1
        if fill_eval.get("overall_is_faithful") is True:
            faithful_count += 1

        if g is not None and g in grade_audc_margins:
            if audc_m is not None:
                grade_audc_margins[g].append(float(audc_m))
                grade_salient_drops[g].append(float(audc_s))
                grade_random_drops[g].append(float(audc_r))
            if pauc_m is not None:
                grade_pauc_margins[g].append(float(pauc_m))
                grade_pauc_salient[g].append(float(pauc_s))
                grade_pauc_random[g].append(float(pauc_r))

    def _calc_stats(arr: List[float]) -> Dict[str, Any]:
        valid = [v for v in arr if v is not None and not np.isnan(v)]
        if not valid:
            return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
        a = np.array(valid, dtype=np.float64)
        return {
            "count": len(valid),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "iqr_25": float(np.percentile(a, 25)),
            "iqr_75": float(np.percentile(a, 75)),
        }

    summary["overall"] = {
        "mean_deletion_drop_salient": _calc_stats(all_audc_salient),
        "mean_deletion_drop_random": _calc_stats(all_audc_random),
        "mean_deletion_margin": _calc_stats(all_audc_margins),
        "normalized_pauc_salient": _calc_stats(all_pauc_salient),
        "normalized_pauc_random": _calc_stats(all_pauc_random),
        "normalized_pauc_margin": _calc_stats(all_pauc_margins),
        "audc_salient": _calc_stats(all_audc_salient),
        "audc_random": _calc_stats(all_audc_random),
        "audc_margin": _calc_stats(all_audc_margins),
        "faithful_cases_count": faithful_count,
        "faithful_rate_pct": float(faithful_count / max(1, informative_count) * 100.0) if informative_count > 0 else 0.0,
    }

    for g in range(5):
        summary["grade_wise"][f"grade_{g}"] = {
            "mean_deletion_drop_salient": _calc_stats(grade_salient_drops[g]),
            "mean_deletion_drop_random": _calc_stats(grade_random_drops[g]),
            "mean_deletion_margin": _calc_stats(grade_audc_margins[g]),
            "normalized_pauc_salient": _calc_stats(grade_pauc_salient[g]),
            "normalized_pauc_random": _calc_stats(grade_pauc_random[g]),
            "normalized_pauc_margin": _calc_stats(grade_pauc_margins[g]),
            "audc_salient": _calc_stats(grade_salient_drops[g]),
            "audc_random": _calc_stats(grade_random_drops[g]),
            "audc_margin": _calc_stats(grade_audc_margins[g]),
        }

    return summary

