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


def test_preprocess_cli_missing_inputs_validation(tmp_path: Path):
    """Verify preprocess CLI returns error 1 when inputs are missing."""
    import yaml
    from scripts.preprocess_data import main as preprocess_main

    empty_config = tmp_path / "empty_config.yaml"
    with open(empty_config, "w", encoding="utf-8") as f:
        yaml.dump({"dataset": {}}, f)

    # Missing IDRiD inputs with --idrid-only
    ret_idrid = preprocess_main(["--config", str(empty_config), "--idrid-only"])
    assert ret_idrid == 1

    # Missing APTOS inputs without --idrid-only
    ret_aptos = preprocess_main(["--config", str(empty_config)])
    assert ret_aptos == 1


def test_preprocess_cli_idrid_only_execution(tmp_path: Path):
    """Verify --idrid-only flag executes and prepares IDRiD cohorts without APTOS raw data."""
    import yaml
    from scripts.preprocess_data import main as preprocess_main

    # Build mock IDRiD raw directory structure
    raw_dir = tmp_path / "raw_idrid"
    grading_train_dir = raw_dir / "grading" / "train"
    grading_test_dir = raw_dir / "grading" / "test"
    grading_train_dir.mkdir(parents=True, exist_ok=True)
    grading_test_dir.mkdir(parents=True, exist_ok=True)

    # Dummy images
    dummy_img = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 128)
    dummy_img.save(grading_train_dir / "IDRiD_001.jpg")
    dummy_img.save(grading_test_dir / "IDRiD_103.jpg")

    # Labels CSVs
    labels_dir = raw_dir / "grading" / "groundtruth"
    labels_dir.mkdir(parents=True, exist_ok=True)
    train_labels = labels_dir / "train_labels.csv"
    test_labels = labels_dir / "test_labels.csv"
    pd.DataFrame({
        "Image name": ["IDRiD_001"],
        "Retinopathy grade": [2],
        "Risk of macular edema ": [0],
    }).to_csv(train_labels, index=False)
    pd.DataFrame({
        "Image name": ["IDRiD_103"],
        "Retinopathy grade": [3],
        "Risk of macular edema ": [1],
    }).to_csv(test_labels, index=False)

    # Segmentation directory
    seg_train_dir = raw_dir / "seg" / "train"
    seg_test_dir = raw_dir / "seg" / "test"
    seg_train_dir.mkdir(parents=True, exist_ok=True)
    seg_test_dir.mkdir(parents=True, exist_ok=True)
    dummy_img.save(seg_train_dir / "IDRiD_01.jpg")

    masks_root = raw_dir / "seg" / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)
    for c in ["1. Microaneurysms", "2. Haemorrhages", "3. Hard Exudates", "4. Soft Exudates", "5. Optic Disc"]:
        (masks_root / c).mkdir(parents=True, exist_ok=True)
    dummy_mask = Image.fromarray(np.ones((200, 200), dtype=np.uint8) * 255)
    dummy_mask.save(masks_root / "1. Microaneurysms" / "IDRiD_01_MA.tif")

    processed_dir = tmp_path / "processed"
    cfg = {
        "project": {"random_seed": 42},
        "dataset": {
            "processed_dir": str(processed_dir),
            "disease_grading": {
                "train_images_dir": str(grading_train_dir),
                "test_images_dir": str(grading_test_dir),
                "train_labels_file": str(train_labels),
                "test_labels_file": str(test_labels),
                "image_id_col": "Image name",
                "grade_col": "Retinopathy grade",
                "dme_col": "Risk of macular edema ",
            },
            "segmentation": {
                "train_images_dir": str(seg_train_dir),
                "test_images_dir": str(seg_test_dir),
                "masks_root_dir": str(masks_root),
                "lesion_types": {
                    "microaneurysms": {"code": "MA", "subdir": "1. Microaneurysms"},
                    "haemorrhages": {"code": "HE", "subdir": "2. Haemorrhages"},
                    "hard_exudates": {"code": "EX", "subdir": "3. Hard Exudates"},
                    "soft_exudates": {"code": "SE", "subdir": "4. Soft Exudates"},
                    "optic_disc": {"code": "OD", "subdir": "5. Optic Disc"},
                },
            },
            "aptos": {
                "raw_dir": "nonexistent_aptos",
                "labels_file": "nonexistent_aptos/train.csv",
                "images_dir": "nonexistent_aptos/train_images",
            },
        },
    }

    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    mock_report = tmp_path / "mock_report.md"
    ret = preprocess_main([
        "--config", str(cfg_file),
        "--idrid-only",
        "--validation-mode", "test",
        "--dry-run",
        "--output-report", str(mock_report),
    ])
    assert ret == 0
    assert mock_report.exists()

    assert (processed_dir / "idrid" / "external_train.csv").exists()
    assert (processed_dir / "idrid" / "external_test.csv").exists()
    assert (processed_dir / "idrid" / "segmentation_manifest.csv").exists()


def test_extract_retinal_fov_mask_holes_filled_and_padding_zeroed():
    """Verify extract_retinal_fov_mask fills interior dark lesions and zeros padded borders."""
    from src.preprocessing import extract_retinal_fov_mask

    # 100x100 canvas with a circular retina of radius 35 centered at (50, 50)
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    y, x = np.ogrid[:100, :100]
    circle_mask = (y - 50)**2 + (x - 50)**2 <= 35**2
    canvas[circle_mask] = [120, 160, 40]  # Green channel bright

    # Introduce a dark internal lesion (e.g. dense haemorrhage where green channel <= 5)
    canvas[48:53, 48:53] = [5, 2, 0]

    pad_offsets = (10, 10, 12, 12)
    fov_mask = extract_retinal_fov_mask(canvas, threshold=10, pad_offsets=pad_offsets)

    assert fov_mask.shape == (100, 100)
    assert set(np.unique(fov_mask)).issubset({0, 1})

    # The dark lesion inside the retina must be FILLED (1, not punched out as 0)
    assert np.all(fov_mask[48:53, 48:53] == 1)

    # Padding borders must be strictly 0
    assert np.all(fov_mask[:10, :] == 0)
    assert np.all(fov_mask[90:, :] == 0)
    assert np.all(fov_mask[:, :12] == 0)
    assert np.all(fov_mask[:, 88:] == 0)

    # The outer 4 corners must be strictly 0
    assert fov_mask[1, 1] == 0
    assert fov_mask[1, 98] == 0
    assert fov_mask[98, 1] == 0
    assert fov_mask[98, 98] == 0


def test_verify_splits_leakage_idrid_only_skipped_and_overlap_detection():
    """Verify verify_splits_leakage distinguishes SKIPPED from PASSED and detects IDRiD overlap."""
    from scripts.preprocess_data import verify_splits_leakage

    # Scenario A: --idrid-only mode (APTOS empty) in test validation mode
    empty_aptos = pd.DataFrame()
    idrid_train_clean = pd.DataFrame({"image_id": ["IDRiD_001", "IDRiD_002"], "original_id": ["IDRiD_001", "IDRiD_002"]})
    idrid_test_clean = pd.DataFrame({"image_id": ["IDRiD_414", "IDRiD_415"], "original_id": ["IDRiD_414", "IDRiD_415"]})

    res_partial = verify_splits_leakage(
        empty_aptos, empty_aptos, idrid_train_clean, idrid_test_clean, validation_mode="test"
    )
    assert res_partial["is_leakage_free"] is True
    assert "PARTIALLY EVALUATED" in res_partial["overall_status"]
    assert "Zero Leakage" not in res_partial["overall_status"]
    assert res_partial["checks"][2]["status"] == "SKIPPED"
    assert res_partial["checks"][3]["status"] == "SKIPPED"
    assert res_partial["checks"][4]["status"] == "PASSED"

    # Scenario B: Critical Leakage between IDRiD Train and Test
    idrid_train_leak = pd.DataFrame({"image_id": ["IDRiD_001", "IDRiD_414"]})
    idrid_test_leak = pd.DataFrame({"image_id": ["IDRiD_414", "IDRiD_415"]})

    res_leak = verify_splits_leakage(
        empty_aptos, empty_aptos, idrid_train_leak, idrid_test_leak, validation_mode="test"
    )
    assert res_leak["is_leakage_free"] is False
    assert "FAILED" in res_leak["overall_status"]
    assert any(c["name"] == "IDRiD Cohort Isolation" and c["status"] == "FAILED" for c in res_leak["checks"])


def test_verify_splits_leakage_strict_exact_counts():
    """Verify verify_splits_leakage strictly enforces exact counts in strict mode."""
    from scripts.preprocess_data import verify_splits_leakage

    # Exactly 413 train, 103 test, 81 seg
    idrid_train = pd.DataFrame({"image_id": [f"IDRiD_train_{i:03d}" for i in range(413)]})
    idrid_test = pd.DataFrame({"image_id": [f"IDRiD_test_{i:03d}" for i in range(103)]})
    idrid_seg = pd.DataFrame({"image_id": [f"IDRiD_seg_{i:02d}" for i in range(81)]})

    # When APTOS is omitted in idrid-only
    empty_aptos = pd.DataFrame()
    res = verify_splits_leakage(
        empty_aptos, empty_aptos, idrid_train, idrid_test, idrid_seg=idrid_seg, validation_mode="strict"
    )
    assert res["is_leakage_free"] is True
    assert "PARTIALLY EVALUATED" in res["overall_status"]

    # When exact count mismatches (e.g. 412 instead of 413)
    idrid_train_bad = pd.DataFrame({"image_id": [f"IDRiD_train_{i:03d}" for i in range(412)]})
    res_bad = verify_splits_leakage(
        empty_aptos, empty_aptos, idrid_train_bad, idrid_test, idrid_seg=idrid_seg, validation_mode="strict"
    )
    assert res_bad["is_leakage_free"] is False
    assert "FAILED" in res_bad["overall_status"]
    assert any(c["name"] == "IDRiD Training Set Integrity" and c["status"] == "FAILED" for c in res_bad["checks"])


def test_verify_splits_leakage_duplicate_ids_detected():
    """Verify duplicate IDs within a cohort cause immediate validation failure."""
    from scripts.preprocess_data import verify_splits_leakage

    idrid_train_dup = pd.DataFrame({"image_id": ["IDRiD_01", "IDRiD_01", "IDRiD_02"]})
    idrid_test = pd.DataFrame({"image_id": ["IDRiD_10", "IDRiD_11"]})

    res = verify_splits_leakage(
        pd.DataFrame(), pd.DataFrame(), idrid_train_dup, idrid_test, validation_mode="test"
    )
    assert res["is_leakage_free"] is False
    assert "FAILED" in res["overall_status"]
    assert any("Duplicate image identities detected in IDRiD Train" in c["details"] for c in res["checks"])


def test_verify_splits_leakage_empty_datasets_strict_mode(tmp_path: Path):
    """Verify empty IDRiD datasets in strict mode trigger FAILED status and report."""
    from scripts.preprocess_data import generate_report, verify_splits_leakage

    valid_train = pd.DataFrame({"image_id": [f"IDRiD_train_{i:03d}" for i in range(413)]})
    valid_test = pd.DataFrame({"image_id": [f"IDRiD_test_{i:03d}" for i in range(103)]})
    valid_seg = pd.DataFrame({"image_id": [f"IDRiD_seg_{i:02d}" for i in range(81)]})
    empty_df = pd.DataFrame()

    # 1. Empty IDRiD Train in strict mode must FAIL
    res_empty_tr = verify_splits_leakage(
        empty_df, empty_df, empty_df, valid_test, idrid_seg=valid_seg, validation_mode="strict"
    )
    assert res_empty_tr["is_leakage_free"] is False
    assert "FAILED" in res_empty_tr["overall_status"]
    assert any(c["name"] == "IDRiD Training Set Integrity" and c["status"] == "FAILED" for c in res_empty_tr["checks"])

    # 2. Empty IDRiD Test in strict mode must FAIL
    res_empty_te = verify_splits_leakage(
        empty_df, empty_df, valid_train, empty_df, idrid_seg=valid_seg, validation_mode="strict"
    )
    assert res_empty_te["is_leakage_free"] is False
    assert "FAILED" in res_empty_te["overall_status"]
    assert any(c["name"] == "IDRiD Test Set Integrity" and c["status"] == "FAILED" for c in res_empty_te["checks"])

    # 3. Empty IDRiD Seg in strict mode must FAIL
    res_empty_seg = verify_splits_leakage(
        empty_df, empty_df, valid_train, valid_test, idrid_seg=empty_df, validation_mode="strict"
    )
    assert res_empty_seg["is_leakage_free"] is False
    assert "FAILED" in res_empty_seg["overall_status"]
    assert any(c["name"] == "IDRiD Segmentation Set Integrity" and c["status"] == "FAILED" for c in res_empty_seg["checks"])

    # 4. generate_report must reflect FAILED pipeline status when is_leakage_free is False
    out_rep = tmp_path / "failed_report.md"
    generate_report(empty_df, empty_df, empty_df, valid_test, valid_seg, res_empty_tr, out_rep)
    rep_text = out_rep.read_text(encoding="utf-8")
    assert "- **Pipeline Status:** `FAILED` ❌" in rep_text
    assert "`SUCCESS`" not in rep_text


def test_evaluate_fov_mask_quality():
    """Verify evaluate_fov_mask_quality detects valid masks and flags anomalies."""
    from src.preprocessing import evaluate_fov_mask_quality

    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    y, x = np.ogrid[:100, :100]

    # 1. Valid circular FOV mask (~70% area ratio)
    valid_mask = ((y - 50)**2 + (x - 50)**2 <= 47**2).astype(np.uint8)
    res_valid = evaluate_fov_mask_quality(img, valid_mask)
    assert res_valid["status"] == "PASSED"
    assert "VALID" in res_valid["flags"]

    # 2. Empty FOV
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    res_empty = evaluate_fov_mask_quality(img, empty_mask)
    assert res_empty["status"] == "FAILED"
    assert "EMPTY_FOV" in res_empty["flags"]

    # 3. Unusually small FOV (< 50%)
    small_mask = ((y - 50)**2 + (x - 50)**2 <= 10**2).astype(np.uint8)
    res_small = evaluate_fov_mask_quality(img, small_mask)
    assert res_small["status"] == "FAILED"
    assert "UNUSUALLY_SMALL" in res_small["flags"]

    # 4. Padding leak
    pad_offsets = (10, 10, 10, 10)
    leak_mask = valid_mask.copy()
    leak_mask[5, 50] = 1  # Leaks into top pad (top 10 rows)
    res_leak = evaluate_fov_mask_quality(img, leak_mask, pad_offsets=pad_offsets)
    assert res_leak["status"] == "FAILED"
    assert "PADDING_LEAK" in res_leak["flags"]

    # 5. Dimension mismatch raises ValueError
    dim_mismatch = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match FOV mask shape"):
        evaluate_fov_mask_quality(img, dim_mismatch)

    # 6. Non-binary values raise ValueError
    bad_vals = valid_mask.copy().astype(np.uint8)
    bad_vals[50, 50] = 128  # neither 0 nor 1
    with pytest.raises(ValueError, match="must be strictly binary"):
        evaluate_fov_mask_quality(img, bad_vals)


def test_run_fov_quality_control(tmp_path: Path):
    """Verify run_fov_quality_control generates valid reports in both pending and active modes."""
    from src.preprocessing import run_fov_quality_control

    # Pending mode (no images on disk)
    dummy_manifest = pd.DataFrame([{"image_id": "IDRiD_01", "processed_image_path": "nonexistent.jpg"}])
    csv_pending = tmp_path / "pending_qc.csv"
    md_pending = tmp_path / "pending_qc.md"
    res_pend = run_fov_quality_control(
        dummy_manifest, project_root=tmp_path, output_csv=csv_pending, output_md=md_pending
    )
    assert res_pend["status"] == "PENDING"
    assert csv_pending.exists()
    assert "Genuine 81-image validation remains pending" in md_pending.read_text(encoding="utf-8")


def test_create_fov_contact_sheet():
    """Verify create_fov_contact_sheet arranges overlays into a valid montage grid."""
    from src.preprocessing import create_fov_contact_sheet

    overlays = [np.ones((64, 64, 3), dtype=np.uint8) * (i * 20) for i in range(12)]
    ids = [f"IDRiD_{i:02d}" for i in range(12)]

    # 3x4 grid, 64x64 thumbnails
    sheet = create_fov_contact_sheet(overlays, ids, grid_size=(3, 4), thumb_size=(64, 64))

    assert sheet.shape == (3 * 64, 4 * 64, 3)
    assert sheet.dtype == np.uint8
    assert np.any(sheet > 0)


def test_validate_segmentation_masks_valid_and_invalid(tmp_path: Path):
    """Verify validate_segmentation_masks checks existence, resolution, binary values, and pixel counts."""
    from src.preprocessing import validate_segmentation_masks

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    # 1. Exactly 500 positive pixels (50 x 10)
    m_ma_arr = np.zeros((512, 512), dtype=np.uint8)
    m_ma_arr[0:50, 0:10] = 255
    m_ma = Image.fromarray(m_ma_arr)
    p_ma = mask_dir / "m_ma.png"
    m_ma.save(p_ma)

    # 2. All-zero annotated mask (0 positive pixels, lesion absent)
    m_he = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    p_he = mask_dir / "m_he.png"
    m_he.save(p_he)

    # 3. Composite mask with matching 500 pixels
    p_comp = mask_dir / "m_comp.png"
    m_ma.save(p_comp)

    # Valid manifest row
    row_valid = {
        "image_id": "seg_01",
        "mask_MA": str(p_ma.relative_to(tmp_path)),
        "annotated_MA": True,
        "pixel_count_MA": 500,
        "lesion_present_MA": True,
        "mask_HE": str(p_he.relative_to(tmp_path)),
        "annotated_HE": True,
        "pixel_count_HE": 0,
        "lesion_present_HE": False,
        "mask_EX": "",
        "annotated_EX": False,
        "pixel_count_EX": -1,
        "lesion_present_EX": None,
        "mask_SE": "",
        "annotated_SE": False,
        "pixel_count_SE": -1,
        "lesion_present_SE": None,
        "mask_OD": "",
        "annotated_OD": False,
        "pixel_count_OD": -1,
        "lesion_present_OD": None,
        "mask_composite": str(p_comp.relative_to(tmp_path)),
        "composite_pixels": 500,
    }
    df_valid = pd.DataFrame([row_valid])

    res = validate_segmentation_masks(df_valid, project_root=tmp_path, target_size=(512, 512))
    assert res["status"] == "PASSED"
    assert res["annotated_counts"]["MA"] == 1
    assert res["annotated_counts"]["HE"] == 1
    assert res["missing_annotation_counts"]["EX"] == 1

    # Case: Pixel count mismatch raises ValueError
    df_mismatch = pd.DataFrame([{**row_valid, "pixel_count_MA": 400}])
    with pytest.raises(ValueError, match="pixel count mismatch for MA"):
        validate_segmentation_masks(df_mismatch, project_root=tmp_path, target_size=(512, 512))

    # Case: Negative pixel count for annotated lesion raises ValueError
    df_neg = pd.DataFrame([{**row_valid, "pixel_count_MA": -10}])
    with pytest.raises(ValueError, match="negative pixel count"):
        validate_segmentation_masks(df_neg, project_root=tmp_path, target_size=(512, 512))

    # Case: lesion_present disagreement raises ValueError
    df_pres_disagree = pd.DataFrame([{**row_valid, "lesion_present_MA": False}])
    with pytest.raises(ValueError, match="lesion_present disagreement"):
        validate_segmentation_masks(df_pres_disagree, project_root=tmp_path, target_size=(512, 512))

    # Case: Composite pixel count mismatch raises ValueError
    df_comp_mismatch = pd.DataFrame([{**row_valid, "composite_pixels": 999}])
    with pytest.raises(ValueError, match="composite pixel count mismatch"):
        validate_segmentation_masks(df_comp_mismatch, project_root=tmp_path, target_size=(512, 512))

    # Case: Non-binary values raise ValueError
    m_bad_vals = Image.fromarray(np.ones((512, 512), dtype=np.uint8) * 128)
    p_bad = mask_dir / "bad.png"
    m_bad_vals.save(p_bad)
    df_bad_vals = pd.DataFrame([{**row_valid, "mask_MA": str(p_bad.relative_to(tmp_path))}])
    with pytest.raises(ValueError, match="contains non-binary values"):
        validate_segmentation_masks(df_bad_vals, project_root=tmp_path, target_size=(512, 512))

    # Case: Resolution mismatch raises ValueError
    m_bad_res = Image.fromarray(np.ones((256, 256), dtype=np.uint8))
    p_res = mask_dir / "bad_res.png"
    m_bad_res.save(p_res)
    df_bad_res = pd.DataFrame([{**row_valid, "mask_MA": str(p_res.relative_to(tmp_path))}])
    with pytest.raises(ValueError, match="does not match expected"):
        validate_segmentation_masks(df_bad_res, project_root=tmp_path, target_size=(512, 512))


def test_run_fov_quality_control_first_missing_regression(tmp_path: Path):
    """Verify FOV QC evaluates all records and fails safely when first image is missing."""
    from src.preprocessing import run_fov_quality_control

    # Image 1 is available
    img1_p = tmp_path / "img1.jpg"
    fov1_p = tmp_path / "fov1.png"

    y, x = np.ogrid[:100, :100]
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    fov1 = ((y - 50)**2 + (x - 50)**2 <= 47**2).astype(np.uint8)

    Image.fromarray(img1).save(img1_p)
    Image.fromarray(fov1 * 255).save(fov1_p)

    # Manifest with 2 rows: row 0 is missing, row 1 is present
    df = pd.DataFrame([
        {
            "image_id": "IDRiD_01",
            "processed_image_path": str((tmp_path / "missing_01.jpg").relative_to(tmp_path)),
            "fov_mask_path": str((tmp_path / "missing_01_fov.png").relative_to(tmp_path)),
        },
        {
            "image_id": "IDRiD_02",
            "processed_image_path": str(img1_p.relative_to(tmp_path)),
            "fov_mask_path": str(fov1_p.relative_to(tmp_path)),
        },
    ])

    out_csv = tmp_path / "qc_reg.csv"
    out_md = tmp_path / "qc_reg.md"
    out_montage = tmp_path / "qc_reg_montage.png"

    res = run_fov_quality_control(
        df,
        project_root=tmp_path,
        output_csv=out_csv,
        output_md=out_md,
        montage_path=out_montage,
    )

    # Must NOT report PENDING
    assert res["status"] != "PENDING"
    # Overall status cannot be PASSED when an image is missing
    assert res["status"] == "FAILED"
    assert res["total_evaluated"] == 2
    assert res["passed"] == 1
    assert res["failed"] == 1

    # Check that contact sheet montage was created for the available image
    assert out_montage.exists()

    df_res = pd.read_csv(out_csv)
    assert len(df_res) == 2
    assert df_res.loc[0, "flags"] == "FILE_MISSING"
    assert df_res.loc[0, "status"] == "FAILED"
    assert df_res.loc[1, "status"] == "PASSED"


