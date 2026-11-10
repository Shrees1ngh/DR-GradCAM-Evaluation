"""Unit tests for Grad-CAM explainability module and visual saliency binarization (Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.explainability import (
    binarize_heatmap,
    build_explanation_summary,
    generate_gradcam_overlay,
)


# ==============================================================================
# Non-PyTorch Tests (Heatmap Processing, Binarization, Visual Blending)
# ==============================================================================

def test_binarize_heatmap_percentile():
    """Verify that percentile thresholding extracts top X% salient pixels."""
    # Create a 100x100 grid with linearly spaced values 0 to 1
    h, w = 100, 100
    grid = np.linspace(0.0, 1.0, h * w).reshape((h, w)).astype(np.float32)

    # Top 10% cut -> exactly 10% of pixels (1,000 pixels) should be 1
    mask_10 = binarize_heatmap(grid, threshold=0.10, method="percentile")
    assert mask_10.shape == (h, w)
    assert set(np.unique(mask_10)).issubset({0, 1})
    assert mask_10.dtype == np.uint8
    # 10% of 10,000 is 1,000 (allow slight variance due to >= percentile equality)
    assert 990 <= np.sum(mask_10) <= 1010

    # Top 5% cut -> ~500 pixels
    mask_05 = binarize_heatmap(grid, threshold=0.05, method="percentile")
    assert 490 <= np.sum(mask_05) <= 510

    # Top 20% cut -> ~2000 pixels
    mask_20 = binarize_heatmap(grid, threshold=0.20, method="percentile")
    assert 1990 <= np.sum(mask_20) <= 2010

    # Monotonicity check: top 5% must be a strict subset of top 10% and top 20%
    assert np.all((mask_05 == 1) <= (mask_10 == 1))
    assert np.all((mask_10 == 1) <= (mask_20 == 1))


def test_binarize_heatmap_relative_and_absolute():
    """Verify relative (to peak) and absolute binarization modes."""
    heatmap = np.array([
        [0.0, 0.2, 0.4],
        [0.5, 0.8, 1.0],
    ], dtype=np.float32)

    # Relative to max (max is 1.0) with threshold=0.5 -> values >= 0.5 are 1
    mask_rel = binarize_heatmap(heatmap, threshold=0.5, method="relative")
    expected = np.array([
        [0, 0, 0],
        [1, 1, 1],
    ], dtype=np.uint8)
    np.testing.assert_array_equal(mask_rel, expected)

    # Absolute with threshold=0.3
    mask_abs = binarize_heatmap(heatmap, threshold=0.3, method="absolute")
    expected_abs = np.array([
        [0, 0, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    np.testing.assert_array_equal(mask_abs, expected_abs)


def test_binarize_heatmap_flat_edge_case():
    """Verify that a completely flat or zero heatmap safely yields all zeros."""
    flat = np.zeros((50, 50), dtype=np.float32)
    mask = binarize_heatmap(flat, threshold=0.10, method="percentile")
    assert mask.shape == (50, 50)
    assert np.sum(mask) == 0

    flat_ones = np.ones((50, 50), dtype=np.float32)
    mask_ones = binarize_heatmap(flat_ones, threshold=0.10, method="percentile")
    assert np.sum(mask_ones) == 0


def test_binarize_heatmap_invalid_inputs():
    """Verify that invalid shapes or thresholds raise ValueErrors."""
    with pytest.raises(ValueError, match="2-dimensional"):
        binarize_heatmap(np.zeros((10, 10, 3)), threshold=0.1)

    with pytest.raises(ValueError, match="within"):
        binarize_heatmap(np.zeros((10, 10)), threshold=1.5)

    with pytest.raises(ValueError, match="Unrecognized"):
        binarize_heatmap(np.zeros((10, 10)), threshold=0.1, method="invalid_method")


def test_generate_gradcam_overlay():
    """Verify overlay blending shape, dtype, and dimension matching."""
    img_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[40:60, 40:60] = 1.0

    overlay = generate_gradcam_overlay(img_rgb, heatmap, alpha=0.5)

    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8
    # Highlighted area should differ from base gray
    assert not np.array_equal(overlay[50, 50], [128, 128, 128])


def test_generate_gradcam_overlay_resizes_heatmap():
    """Verify overlay automatically resizes a low-res heatmap to match the image."""
    img_rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    low_res_heatmap = np.ones((16, 16), dtype=np.float32)

    overlay = generate_gradcam_overlay(img_rgb, low_res_heatmap)
    assert overlay.shape == (200, 200, 3)


def test_build_explanation_summary():
    """Verify structure and content of build_explanation_summary dictionary."""
    heatmap = np.linspace(0.0, 1.0, 10000).reshape((100, 100)).astype(np.float32)

    summary = build_explanation_summary(
        image_id="test_img_001",
        true_grade=2,
        predicted_grade=2,
        prediction_confidence=0.912,
        heatmap=heatmap,
        percentile_thresholds=[0.05, 0.10, 0.20],
    )

    assert summary["image_id"] == "test_img_001"
    assert summary["true_grade"] == 2
    assert summary["true_grade_name"] == "Moderate"
    assert summary["predicted_grade"] == 2
    assert summary["predicted_grade_name"] == "Moderate"
    assert pytest.approx(summary["prediction_confidence"]) == 0.912
    assert summary["heatmap_shape"] == [100, 100]
    assert "top_5_pct" in summary["binary_masks"]
    assert "top_10_pct" in summary["binary_masks"]
    assert "top_20_pct" in summary["binary_masks"]
    assert summary["salient_pixel_counts"]["top_10_pct"] > 0


# ==============================================================================
# PyTorch-Dependent Tests (GradCAM Class, Hook Lifecycle & Gradient Backprop)
# ==============================================================================

def test_gradcam_initialization_and_hooks():
    """Verify hook attachment and context manager cleanup on ResNet-50."""
    torch = pytest.importorskip("torch")
    from src.explainability import GradCAM
    from src.model import build_model

    model = build_model(pretrained=False, num_classes=5)

    with GradCAM(model) as explainer:
        assert explainer.model is model
        assert explainer._forward_hook_handle is not None
        assert explainer._backward_hook_handle is not None

    # Hooks must be cleanly removed after exiting the context block
    assert explainer._forward_hook_handle is None
    assert explainer._backward_hook_handle is None


def test_gradcam_generate_cam_single_image():
    """Verify GradCAM generates valid normalized 2D heatmap on dummy input."""
    torch = pytest.importorskip("torch")
    from src.explainability import GradCAM
    from src.model import build_model

    model = build_model(pretrained=False, num_classes=5)
    dummy_input = torch.randn(1, 3, 256, 256)

    with GradCAM(model) as explainer:
        heatmap, pred_class = explainer.generate_cam(dummy_input)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (256, 256)
    assert heatmap.dtype == np.float32
    assert 0 <= pred_class < 5
    # Normalization bounds: values must be in [0.0, 1.0]
    assert float(np.min(heatmap)) >= 0.0
    assert float(np.max(heatmap)) <= 1.0


def test_gradcam_specific_target_class():
    """Verify GradCAM allows explicitly targeting a specific class (e.g. Grade 4)."""
    torch = pytest.importorskip("torch")
    from src.explainability import GradCAM
    from src.model import build_model

    model = build_model(pretrained=False, num_classes=5)
    dummy_input = torch.randn(3, 128, 128)  # 3D tensor without batch dim

    with GradCAM(model) as explainer:
        heatmap, explained_class = explainer.generate_cam(
            dummy_input, target_class=4, target_size=(512, 512)
        )

    assert explained_class == 4
    assert heatmap.shape == (512, 512)


def test_gradcam_batch_cam():
    """Verify batch Grad-CAM generation across multiple images."""
    torch = pytest.importorskip("torch")
    from src.explainability import GradCAM
    from src.model import build_model

    model = build_model(pretrained=False, num_classes=5)
    batch_input = torch.randn(3, 3, 64, 64)

    with GradCAM(model) as explainer:
        results = explainer.generate_batch_cam(batch_input, target_size=(128, 128))

    assert len(results) == 3
    for heatmap, cls_idx in results:
        assert heatmap.shape == (128, 128)
        assert 0 <= cls_idx < 5

