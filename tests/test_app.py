"""Unit and integration tests for RetinaScope Streamlit application."""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import torch

from app.inference import (
    load_trained_model,
    validate_image,
    preprocess_retinal_image,
    run_inference,
    get_device,
    GRADE_DESCRIPTIONS,
)
from app.gradcam import compute_gradcam


def test_validate_image_valid():
    """Verify validation on valid PIL and numpy images."""
    arr = np.ones((200, 200, 3), dtype=np.uint8) * 100
    img = Image.fromarray(arr)
    valid_img = validate_image(img)
    assert valid_img.size == (200, 200)
    assert valid_img.mode == "RGB"

    # Bytes input
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    valid_from_bytes = validate_image(buf.getvalue())
    assert valid_from_bytes.size == (200, 200)


def test_validate_image_too_small():
    """Verify rejection of images that are too small."""
    tiny = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        validate_image(tiny)


def test_validate_image_corrupt():
    """Verify rejection of corrupt byte buffers."""
    corrupt = b"not-an-image-file"
    with pytest.raises(ValueError, match="Failed to decode"):
        validate_image(corrupt)


def test_preprocess_retinal_image():
    """Verify complete preprocessing pipeline preserves 512x512 resolution and normalization."""
    # Synthetic fundus image: black background with circular green retina
    synthetic = np.zeros((300, 400, 3), dtype=np.uint8)
    synthetic[40:260, 90:310, 1] = 160  # bright green circle
    pil_img = Image.fromarray(synthetic)

    padded_img, tensor_img, meta = preprocess_retinal_image(
        pil_img, target_size=(512, 512), crop_fov=True
    )

    assert padded_img.shape == (512, 512, 3)
    assert tensor_img.shape == (1, 3, 512, 512)
    assert tensor_img.dtype == torch.float32
    assert "crop_bbox" in meta
    assert "pad_offsets" in meta
    assert meta["target_size"] == (512, 512)


def test_load_trained_model_missing():
    """Verify explicit error when checkpoint is missing."""
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_trained_model("non_existent_dir/missing_model.pth")


def test_full_pipeline_unit_mock(tmp_path: Path):
    """Unit test: full pipeline (load checkpoint, run inference, Grad-CAM) using an independent mock checkpoint."""
    from src.model import build_model

    # Construct mock model and save as checkpoint in tmp_path
    base_model = build_model(num_classes=5, pretrained=False, dropout_rate=0.3)
    mock_ckpt_path = tmp_path / "mock_best_model.pth"
    torch.save(
        {
            "epoch": 8,
            "best_val_qwk": 0.85,
            "model_state_dict": base_model.state_dict(),
        },
        mock_ckpt_path,
    )

    device = torch.device("cpu")
    model, meta = load_trained_model(mock_ckpt_path, device=device)

    assert model is not None
    assert meta["best_val_qwk"] == 0.85
    assert meta["epoch"] == 8

    # Create dummy retinal image
    synthetic = np.zeros((400, 400, 3), dtype=np.uint8)
    synthetic[50:350, 50:350, 1] = 150
    pil_img = Image.fromarray(synthetic)

    preprocessed_rgb, tensor_img, _ = preprocess_retinal_image(
        pil_img, target_size=(512, 512), crop_fov=True, device=device
    )

    # 1. Run forward inference
    result = run_inference(model, tensor_img)
    assert "predicted_grade" in result
    assert result["predicted_grade"] in [0, 1, 2, 3, 4]
    assert len(result["probabilities"]) == 5
    prob_sum = sum(result["probabilities"].values())
    assert pytest.approx(prob_sum, rel=1e-3) == 1.0

    # 2. Run Grad-CAM
    heatmap, overlay, grayscale_cam, stats = compute_gradcam(
        model, tensor_img, preprocessed_rgb, target_grade=result["predicted_grade"]
    )

    assert heatmap.shape == (512, 512, 3)
    assert overlay.shape == (512, 512, 3)
    assert grayscale_cam.shape == (512, 512)
    assert grayscale_cam.min() >= 0.0
    assert grayscale_cam.max() <= 1.0
    assert "coverage_baseline_10pct" in stats
    assert "peak_coordinate" in stats


@pytest.mark.integration
def test_full_pipeline_with_checkpoint():
    """Integration test: load real Model B checkpoint, run inference on real image, compute Grad-CAM."""
    ckpt_path = Path("experiments/checkpoints/model_b_best.pth")
    if not ckpt_path.exists():
        pytest.skip(
            "Integration test skipped: genuine trained checkpoint "
            "'experiments/checkpoints/model_b_best.pth' not found on disk. "
            "Place genuine trained weights to execute."
        )

    device = torch.device("cpu")
    model, meta = load_trained_model(ckpt_path, device=device)

    assert model is not None
    assert meta["best_val_qwk"] is not None
    assert meta["epoch"] is not None
    assert meta["epoch"] == 10

    # Use a real fundus image if available, otherwise fallback to synthetic
    real_img_path = Path("data/processed/idrid/images/external_test_IDRiD_001.jpg")
    if real_img_path.exists():
        with Image.open(real_img_path) as img:
            pil_img = img.convert("RGB")
    else:
        synthetic = np.zeros((400, 400, 3), dtype=np.uint8)
        synthetic[50:350, 50:350, 1] = 150
        pil_img = Image.fromarray(synthetic)

    preprocessed_rgb, tensor_img, _ = preprocess_retinal_image(
        pil_img, target_size=(512, 512), crop_fov=True, device=device
    )

    # 1. Run forward inference
    result = run_inference(model, tensor_img)
    assert "predicted_grade" in result
    assert result["predicted_grade"] in [0, 1, 2, 3, 4]
    assert len(result["probabilities"]) == 5
    prob_sum = sum(result["probabilities"].values())
    assert pytest.approx(prob_sum, rel=1e-3) == 1.0

    # 2. Run Grad-CAM
    heatmap, overlay, grayscale_cam, stats = compute_gradcam(
        model, tensor_img, preprocessed_rgb, target_grade=result["predicted_grade"]
    )

    assert heatmap.shape == (512, 512, 3)
    assert overlay.shape == (512, 512, 3)
    assert grayscale_cam.shape == (512, 512)
    assert grayscale_cam.min() >= 0.0
    assert grayscale_cam.max() <= 1.0
    assert "coverage_baseline_10pct" in stats
    assert "peak_coordinate" in stats


@pytest.mark.integration
def test_benchmark_real_data_integration():
    """Verify real benchmark catalog loading and live localization evaluation on IDRiD segmentation."""
    from app.benchmark import (
        load_benchmark_catalogs,
        load_expert_lesion_data,
        evaluate_live_localization,
        resolve_image_path,
    )

    project_root = Path(".")
    catalogs = load_benchmark_catalogs(project_root)
    if "IDRiD Segmentation (81 lesion-masked images)" not in catalogs:
        pytest.skip("IDRiD segmentation manifest not present locally.")

    seg_df = catalogs["IDRiD Segmentation (81 lesion-masked images)"]
    assert len(seg_df) == 81

    # Test first image with non-zero composite mask
    pos_rows = seg_df[seg_df["composite_pixels"] > 0]
    assert len(pos_rows) > 0
    row = pos_rows.iloc[0]

    img_path = resolve_image_path(row, project_root)
    assert img_path.exists()
    with Image.open(img_path) as im:
        rgb = np.array(im.convert("RGB"), dtype=np.uint8)

    expert_data = load_expert_lesion_data(row, rgb, project_root)
    assert expert_data is not None
    assert expert_data["composite_mask"].shape == (512, 512)
    assert expert_data["fov_mask"].shape == (512, 512)

    # Evaluate against a real Model B Grad-CAM heatmap
    ckpt_path = Path("experiments/checkpoints/model_b_best.pth")
    if not ckpt_path.exists():
        pytest.skip("model_b_best.pth not present locally.")

    device = torch.device("cpu")
    model, _ = load_trained_model(ckpt_path, device=device)
    preprocessed_rgb, tensor_img, _ = preprocess_retinal_image(
        Image.fromarray(rgb), target_size=(512, 512), crop_fov=True, device=device
    )
    res = run_inference(model, tensor_img)
    _, _, grayscale_cam, _ = compute_gradcam(
        model, tensor_img, preprocessed_rgb, target_grade=res["predicted_grade"]
    )
    metrics = evaluate_live_localization(grayscale_cam, expert_data)
    assert 0.0 <= metrics["iou_50"] <= 1.0
    assert 0.0 <= metrics["dice_50"] <= 1.0
    assert 0.0 <= metrics["dice_10"] <= 1.0
    assert isinstance(metrics["pointing_hit"], bool)


