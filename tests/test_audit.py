"""Unit and integration tests for dataset audit pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.audit import (
    DatasetAuditResult,
    DatasetAuditor,
    generate_markdown_report,
)


@pytest.fixture
def empty_config(tmp_path: Path) -> dict:
    return {
        "dataset": {
            "raw_dir": str(tmp_path / "non_existent_idrid"),
            "processed_dir": str(tmp_path / "processed"),
        }
    }


def test_audit_missing_dataset(empty_config: dict, tmp_path: Path):
    auditor = DatasetAuditor(config=empty_config, project_root=tmp_path)
    result = auditor.run_audit()

    assert not result.dataset_found
    assert len(result.warnings) > 0
    assert "does not exist" in result.warnings[0]

    md = generate_markdown_report(result)
    assert "PENDING DATASET ACQUISITION" in md
    assert "Required Directory Placement" in md


@pytest.fixture
def mock_idrid_dataset(tmp_path: Path) -> Path:
    """Create a minimal synthetic IDRiD folder layout for audit validation."""
    raw_root = tmp_path / "IDRiD"
    train_img_dir = raw_root / "1. Original Images" / "a. Training Set"
    test_img_dir = raw_root / "1. Original Images" / "b. Testing Set"
    labels_dir = raw_root / "2. Groundtruths" / "a. DR AC Groundtruths"
    seg_img_dir = raw_root / "Segmentation" / "Original Images"
    seg_masks_dir = raw_root / "Segmentation" / "Groundtruths"

    for d in [train_img_dir, test_img_dir, labels_dir, seg_img_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Create Train Images (4 valid, 1 corrupt 0-byte file)
    for i in range(1, 5):
        img = Image.new("RGB", (100, 100), color=(i * 20, 50, 100))
        img.save(train_img_dir / f"IDRiD_{i:03d}.jpg")

    # Corrupt 0-byte image
    corrupt_file = train_img_dir / "IDRiD_005.jpg"
    corrupt_file.touch()

    # 2. Create Test Images (2 valid)
    for i in range(6, 8):
        img = Image.new("RGB", (100, 100), color=(150, 80, 20))
        img.save(test_img_dir / f"IDRiD_{i:03d}.jpg")

    # 3. Create Labels
    train_df = pd.DataFrame({
        "Image name": [f"IDRiD_{i:03d}" for i in range(1, 6)],
        "Retinopathy grade": [0, 1, 2, 3, 4],
        "Risk of macular edema": [0, 0, 1, 1, 2],
    })
    train_df.to_csv(labels_dir / "train.csv", index=False)

    test_df = pd.DataFrame({
        "Image name": [f"IDRiD_{i:03d}" for i in [6, 7]],
        "Retinopathy grade": [0, 2],
        "Risk of macular edema": [0, 1],
    })
    test_df.to_csv(labels_dir / "test.csv", index=False)

    # 4. Create Segmentation Images (2 images: IDRiD_001 and IDRiD_006)
    # Note: IDRiD_001 is from Train, IDRiD_006 is from Test
    for i in [1, 6]:
        img = Image.new("RGB", (100, 100), color=(30, 90, 40))
        img.save(seg_img_dir / f"IDRiD_{i:03d}.jpg")

    # 5. Create Lesion Masks
    lesions = ["1. Microaneurysms", "2. Haemorrhages", "3. Hard Exudates", "4. Soft Exudates", "5. Optic Disc"]
    for l in lesions:
        ldir = seg_masks_dir / l
        ldir.mkdir(parents=True, exist_ok=True)
        # Create mask for IDRiD_001
        mask_arr = np.zeros((100, 100), dtype=np.uint8)
        mask_arr[40:60, 40:60] = 255
        mask_img = Image.fromarray(mask_arr, mode="L")
        code = l.split(". ")[1][:2].upper()
        mask_img.save(ldir / f"IDRiD_001_{code}.tif")

    return raw_root


def test_audit_synthetic_dataset(mock_idrid_dataset: Path, tmp_path: Path):
    config = {
        "dataset": {
            "raw_dir": str(mock_idrid_dataset),
            "processed_dir": str(tmp_path / "processed"),
            "disease_grading": {
                "train_images_dir": str(mock_idrid_dataset / "1. Original Images" / "a. Training Set"),
                "test_images_dir": str(mock_idrid_dataset / "1. Original Images" / "b. Testing Set"),
                "train_labels_file": str(mock_idrid_dataset / "2. Groundtruths" / "a. DR AC Groundtruths" / "train.csv"),
                "test_labels_file": str(mock_idrid_dataset / "2. Groundtruths" / "a. DR AC Groundtruths" / "test.csv"),
            },
            "segmentation": {
                "images_dir": str(mock_idrid_dataset / "Segmentation" / "Original Images"),
                "masks_root_dir": str(mock_idrid_dataset / "Segmentation" / "Groundtruths"),
            },
        }
    }

    auditor = DatasetAuditor(config=config, project_root=tmp_path)
    result = auditor.run_audit()

    assert result.dataset_found
    # Grading train: 5 total, 4 valid, 1 corrupt
    assert result.grading_train_images.total_found == 5
    assert result.grading_train_images.valid_count == 4
    assert result.grading_train_images.corrupt_count == 1
    assert "IDRiD_005.jpg" in result.grading_train_images.corrupt_files

    # Grading test: 2 valid
    assert result.grading_test_images.valid_count == 2
    assert result.grading_test_images.corrupt_count == 0

    # Class distributions
    assert result.train_grading_distribution.total_labeled == 5
    for g in range(5):
        assert result.train_grading_distribution.grade_counts[g] == 1
    assert result.test_grading_distribution.total_labeled == 2
    assert result.combined_grading_distribution.total_labeled == 7

    # Segmentation
    assert result.segmentation_images.valid_count == 2
    assert result.mask_integrity.total_segmentation_images == 2
    assert result.mask_integrity.lesion_availability.get("MA") == 1

    # Overlap: IDRiD_001 is in train, IDRiD_006 is in test
    assert result.overlap_and_leakage.seg_in_grading_train == 1
    assert result.overlap_and_leakage.seg_in_grading_test == 1

    # Serialization check
    d = result.to_dict()
    assert isinstance(d, dict)
    assert json.dumps(d)  # Ensure JSON serializable

    # Markdown generation check
    md = generate_markdown_report(result)
    assert "VERIFIED ✅" in md
    assert "Grade 0" in md
    assert "Microaneurysms" in md


def test_audit_aptos_synthetic(tmp_path: Path):
    """Test auditing APTOS dataset on synthetic folder structure."""
    aptos_root = tmp_path / "APTOS"
    img_dir = aptos_root / "train_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 valid PNG images and 1 corrupt image
    for i in range(1, 4):
        img = Image.new("RGB", (64, 64), color=(i * 30, i * 40, i * 50))
        img.save(img_dir / f"img_{i:03d}.png")

    corrupt_f = img_dir / "img_004.png"
    corrupt_f.touch()

    # Create train.csv with 4 rows (1 matching the corrupt image)
    df = pd.DataFrame({
        "id_code": [f"img_{i:03d}" for i in range(1, 5)],
        "diagnosis": [0, 1, 2, 4],
    })
    df.to_csv(aptos_root / "train.csv", index=False)

    config = {
        "dataset": {
            "raw_dir": str(tmp_path / "non_existent_idrid"),
            "processed_dir": str(tmp_path / "processed"),
            "aptos": {
                "raw_dir": str(aptos_root),
                "images_dir": str(img_dir),
                "labels_file": str(aptos_root / "train.csv"),
            },
        }
    }

    auditor = DatasetAuditor(config=config, project_root=tmp_path)
    result = auditor.run_audit()

    assert result.aptos is not None
    assert result.aptos.dataset_found
    assert result.aptos.total_csv_rows == 4
    assert result.aptos.total_disk_images == 4
    assert result.aptos.unreadable_images_count == 1
    assert "img_004.png" in result.aptos.unreadable_files
    assert result.aptos.distribution.total_labeled == 4
    assert result.aptos.distribution.grade_counts[0] == 1
    assert result.aptos.distribution.grade_counts[1] == 1
    assert result.aptos.distribution.grade_counts[2] == 1
    assert result.aptos.distribution.grade_counts[4] == 1

    md = generate_markdown_report(result)
    assert "APTOS 2019 Dataset Audit" in md
    assert "Cross-Dataset Comparison" not in md  # IDRiD is missing here


