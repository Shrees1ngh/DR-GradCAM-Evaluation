"""Unit tests for Phase 2 preprocessing and dataset pipeline."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.preprocessing import (
    build_composite_lesion_mask,
    crop_to_retinal_fov,
    preprocess_fundus_image,
    preprocess_mask_with_transform,
    resize_and_pad,
)


def test_crop_to_retinal_fov():
    """Verify FOV detector crops black margins around a circular retina."""
    # Create 200x300 image with black borders and a bright circle in the center
    canvas = np.zeros((200, 300, 3), dtype=np.uint8)
    canvas[50:150, 80:220, 1] = 180  # Green channel circle
    cropped, bbox = crop_to_retinal_fov(canvas, threshold=10, margin_ratio=0.0)

    ymin, ymax, xmin, xmax = bbox
    assert ymin >= 48 and ymin <= 50
    assert ymax >= 150 and ymax <= 152
    assert xmin >= 78 and xmin <= 80
    assert xmax >= 220 and xmax <= 222
    assert cropped.shape[0] == ymax - ymin
    assert cropped.shape[1] == xmax - xmin


def test_resize_and_pad_aspect_ratio():
    """Verify isotropic scaling and centered zero padding to target dimensions."""
    # Rectangular 100x200 image (aspect ratio 2:1)
    img = np.ones((100, 200, 3), dtype=np.uint8) * 128
    padded, pad_offsets, scale = resize_and_pad(img, target_size=(512, 512))

    assert padded.shape == (512, 512, 3)
    pad_top, pad_bottom, pad_left, pad_right = pad_offsets

    # Since width is 2x height, target width 512 should have 0 left/right pad
    # and height should be 256 with 128 top and 128 bottom pad
    assert pad_left == 0
    assert pad_right == 0
    assert pad_top == 128
    assert pad_bottom == 128
    assert np.isclose(scale, 512 / 200)

    # Check that padded borders are black (zeros)
    assert np.all(padded[:pad_top, :, :] == 0)
    assert np.all(padded[512 - pad_bottom :, :, :] == 0)
    # Check that inner image contains the original pixels
    assert np.all(padded[pad_top : 512 - pad_bottom, :, :] == 128)


def test_mask_spatial_alignment_and_binarization(tmp_path: Path):
    """Verify lesion mask matches the image transform identically and remains binary."""
    # 1. Create a dummy fundus image
    raw_img = np.zeros((300, 400, 3), dtype=np.uint8)
    raw_img[50:250, 50:350, :] = 120  # Retinal area
    img_path = tmp_path / "test_fundus.jpg"
    Image.fromarray(raw_img).save(img_path)

    # 2. Create a lesion mask with a lesion spot at row 100, col 100
    raw_mask = np.zeros((300, 400), dtype=np.uint8)
    raw_mask[100:120, 100:120] = 255  # Lesion

    # Preprocess image
    processed_img, meta = preprocess_fundus_image(str(img_path), target_size=(512, 512), crop_fov=True)
    assert processed_img.shape == (512, 512, 3)

    # Preprocess mask with identical transform
    processed_mask = preprocess_mask_with_transform(raw_mask, meta)
    assert processed_mask.shape == (512, 512)
    # Must be strictly binary {0, 1}
    unique_vals = set(np.unique(processed_mask))
    assert unique_vals.issubset({0, 1})
    assert 1 in unique_vals  # The lesion must be present


def test_build_composite_lesion_mask():
    """Verify composite lesion mask combines MA, HE, EX, SE and excludes OD."""
    ma = np.zeros((512, 512), dtype=np.uint8)
    he = np.zeros((512, 512), dtype=np.uint8)
    od = np.zeros((512, 512), dtype=np.uint8)

    ma[10:30, 10:30] = 1
    he[20:40, 20:40] = 1
    od[200:300, 200:300] = 1  # Should be ignored

    masks_dict = {"MA": ma, "HE": he, "EX": None, "SE": None, "OD": od}
    composite = build_composite_lesion_mask(masks_dict)

    assert composite.shape == (512, 512)
    assert np.all(composite[10:30, 10:30] == 1)
    assert np.all(composite[20:40, 20:40] == 1)
    assert np.all(composite[200:300, 200:300] == 0)  # OD not in composite


def test_dr_grading_dataset(tmp_path: Path):
    """Verify PyTorch DRGradingDataset loads samples and applies normalization."""
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from src.dataset import DRGradingDataset, get_default_transforms

    # Create 2 sample images
    img1 = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 100)
    img2 = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 200)

    p1 = tmp_path / "img1.jpg"
    p2 = tmp_path / "img2.jpg"
    img1.save(p1)
    img2.save(p2)

    df = pd.DataFrame({
        "image_id": ["APTOS_001", "APTOS_002"],
        "image_path": [str(p1), str(p2)],
        "grade": [0, 3],
        "dme_risk": [-1, 1],
    })

    ds = DRGradingDataset(
        manifest_df=df,
        project_root=tmp_path,
        transform=get_default_transforms(split="val", image_size=(512, 512)),
    )

    assert len(ds) == 2
    item0 = ds[0]
    assert item0["image"].shape == (3, 512, 512)
    assert item0["image"].dtype == torch.float32
    assert item0["grade"] == 0
    assert item0["image_id"] == "APTOS_001"

    item1 = ds[1]
    assert item1["grade"] == 3
    assert item1["image_id"] == "APTOS_002"


def test_segmentation_lesion_dataset(tmp_path: Path):
    """Verify SegmentationLesionDataset handles cohort origin and missing vs zero masks."""
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from src.dataset import SegmentationLesionDataset

    # Create dummy image and masks
    img = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 128)
    p_img = tmp_path / "IDRiD_01.jpg"
    img.save(p_img)

    # MA mask has lesions
    mask_ma = Image.fromarray(np.ones((512, 512), dtype=np.uint8) * 255)
    p_ma = tmp_path / "IDRiD_01_MA.png"
    mask_ma.save(p_ma)

    # HE mask exists but is all-zero (lesion absent)
    mask_he = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    p_he = tmp_path / "IDRiD_01_HE.png"
    mask_he.save(p_he)

    # SE mask is missing (annotated=False)
    p_se = tmp_path / "IDRiD_01_SE.png"  # does not exist

    df = pd.DataFrame([{
        "image_id": "idrid/seg/IDRiD_01",
        "image_path": str(p_img),
        "mask_MA": str(p_ma),
        "annotated_MA": True,
        "pixel_count_MA": 500,
        "mask_HE": str(p_he),
        "annotated_HE": True,
        "pixel_count_HE": 0,  # All-zero annotated mask
        "mask_EX": "",
        "annotated_EX": False,
        "pixel_count_EX": -1,
        "mask_SE": str(p_se),
        "annotated_SE": False,  # Missing annotation file
        "pixel_count_SE": -1,
        "mask_OD": "",
        "annotated_OD": False,
        "pixel_count_OD": -1,
        "mask_composite": "",
        "cohort_origin": "idrid_grading_train_cohort",
        "model_seen_status": "unseen",
        "grading_match_id": "IDRiD_102.jpg",
    }])

    ds = SegmentationLesionDataset(manifest_df=df, project_root=tmp_path)
    assert len(ds) == 1
    sample = ds[0]

    assert sample["cohort_origin"] == "idrid_grading_train_cohort"
    assert sample["model_seen_status"] == "unseen"
    assert sample["annotation_status"]["MA"] is True
    assert sample["annotation_status"]["HE"] is True
    assert sample["annotation_status"]["SE"] is False

    # MA has positive pixels
    assert torch.sum(sample["masks"]["MA"]).item() > 0
    # HE is all-zero mask
    assert torch.sum(sample["masks"]["HE"]).item() == 0
    # SE is zero mask because missing
    assert torch.sum(sample["masks"]["SE"]).item() == 0

