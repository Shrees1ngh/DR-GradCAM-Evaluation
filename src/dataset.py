"""PyTorch Dataset implementations and DataLoader builders for Diabetic Retinopathy.

Provides DRGradingDataset for 5-class ICDR classification (APTOS & IDRiD) and
SegmentationLesionDataset for the 81 IDRiD lesion-annotated images.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    DataLoader = None
    Dataset = object
    transforms = None
    TORCH_AVAILABLE = False

logger = logging.getLogger("dataset")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transforms(
    split: str = "train",
    image_size: Tuple[int, int] = (512, 512),
):
    """Build torchvision transformation pipeline for train, val, or test splits.

    Args:
        split: One of 'train', 'val', 'test'.
        image_size: Target (height, width).

    Returns:
        torchvision.transforms.Compose instance.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch and torchvision are required for transformation pipelines.")

    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # Strictly deterministic for validation, external testing, and XAI evaluation
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class DRGradingDataset(Dataset):
    """PyTorch Dataset for Diabetic Retinopathy 5-class grading."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        project_root: Optional[Path] = None,
        transform: Optional[Callable] = None,
        return_raw_image: bool = False,
    ):
        """
        Args:
            manifest_df: DataFrame containing at least 'image_id', 'image_path', 'grade'.
            project_root: Project root path to resolve relative image paths.
            transform: Callable transformation pipeline.
            return_raw_image: If True, also returns unnormalized uint8 image array for visualization.
        """
        self.manifest = manifest_df.reset_index(drop=True)
        self.project_root = project_root or Path.cwd()
        self.transform = transform
        self.return_raw_image = return_raw_image

        required_cols = {"image_id", "image_path", "grade"}
        missing = required_cols - set(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.manifest.iloc[idx]
        raw_path_str = str(row["image_path"])
        p = Path(raw_path_str)
        img_path = p if p.is_absolute() else (self.project_root / p).resolve()

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        with Image.open(img_path) as pil_img:
            raw_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)

        if self.transform is not None:
            tensor_img = self.transform(raw_rgb)
        else:
            tensor_img = transforms.functional.to_tensor(raw_rgb)

        grade = int(row["grade"])
        image_id = str(row["image_id"])
        dme_risk = int(row["dme_risk"]) if "dme_risk" in row and not pd.isna(row["dme_risk"]) else -1

        item = {
            "image": tensor_img,
            "grade": torch.tensor(grade, dtype=torch.long),
            "image_id": image_id,
            "dme_risk": torch.tensor(dme_risk, dtype=torch.long),
        }

        if self.return_raw_image:
            item["raw_image"] = raw_rgb

        return item


class SegmentationLesionDataset(Dataset):
    """PyTorch Dataset for the 81 IDRiD segmentation images and their multi-lesion masks."""

    LESION_CODES = ["MA", "HE", "EX", "SE", "OD"]

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        project_root: Optional[Path] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            manifest_df: DataFrame with columns 'image_id', 'image_path', and mask path columns
                         ('mask_MA', 'mask_HE', 'mask_EX', 'mask_SE', 'mask_OD', 'mask_composite').
            project_root: Project root path to resolve relative file paths.
            transform: Image transformation pipeline (Normalize).
        """
        self.manifest = manifest_df.reset_index(drop=True)
        self.project_root = project_root or Path.cwd()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.manifest.iloc[idx]
        raw_path_str = str(row["image_path"])
        p = Path(raw_path_str)
        img_path = p if p.is_absolute() else (self.project_root / p).resolve()

        with Image.open(img_path) as pil_img:
            raw_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)

        h, w = raw_rgb.shape[:2]

        if self.transform is not None:
            tensor_img = self.transform(raw_rgb)
        else:
            tensor_img = transforms.functional.to_tensor(raw_rgb)

        masks: Dict[str, torch.Tensor] = {}
        annotation_status: Dict[str, bool] = {}

        for code in self.LESION_CODES:
            col_name = f"mask_{code}"
            annotated_col = f"annotated_{code}"

            is_annotated = bool(row.get(annotated_col, True))
            annotation_status[code] = is_annotated

            mask_path_str = row.get(col_name)
            if pd.notna(mask_path_str) and mask_path_str:
                mp = Path(str(mask_path_str))
                full_mask_path = mp if mp.is_absolute() else (self.project_root / mp).resolve()
                if full_mask_path.exists():
                    with Image.open(full_mask_path) as m_img:
                        m_arr = (np.array(m_img) > 0).astype(np.float32)
                    masks[code] = torch.tensor(m_arr, dtype=torch.float32)
                else:
                    masks[code] = torch.zeros((h, w), dtype=torch.float32)
            else:
                masks[code] = torch.zeros((h, w), dtype=torch.float32)

        # Composite mask (Union of MA, HE, EX, SE)
        comp_path_str = row.get("mask_composite")
        if pd.notna(comp_path_str) and comp_path_str:
            cp = Path(str(comp_path_str))
            full_comp_path = cp if cp.is_absolute() else (self.project_root / cp).resolve()
            if full_comp_path.exists():
                with Image.open(full_comp_path) as c_img:
                    c_arr = (np.array(c_img) > 0).astype(np.float32)
                masks["composite"] = torch.tensor(c_arr, dtype=torch.float32)
            else:
                lesion_stack = torch.stack([masks["MA"], masks["HE"], masks["EX"], masks["SE"]], dim=0)
                masks["composite"] = (torch.sum(lesion_stack, dim=0) > 0).float()
        else:
            lesion_stack = torch.stack([masks["MA"], masks["HE"], masks["EX"], masks["SE"]], dim=0)
            masks["composite"] = (torch.sum(lesion_stack, dim=0) > 0).float()

        return {
            "image": tensor_img,
            "raw_image": raw_rgb,
            "image_id": str(row["image_id"]),
            "masks": masks,
            "annotation_status": annotation_status,
        }


def build_dataloaders(
    train_manifest_path: Union[str, Path],
    val_manifest_path: Union[str, Path],
    test_manifest_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Path] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (512, 512),
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Factory to construct train, val, and optional test DataLoaders.

    Args:
        train_manifest_path: CSV path for training split.
        val_manifest_path: CSV path for validation split.
        test_manifest_path: Optional CSV path for test split.
        project_root: Root directory for relative path resolution.
        batch_size: Mini-batch size.
        num_workers: PyTorch DataLoader workers.
        image_size: Target image dimensions.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_df = pd.read_csv(train_manifest_path)
    val_df = pd.read_csv(val_manifest_path)

    train_ds = DRGradingDataset(
        manifest_df=train_df,
        project_root=project_root,
        transform=get_default_transforms(split="train", image_size=image_size),
    )
    val_ds = DRGradingDataset(
        manifest_df=val_df,
        project_root=project_root,
        transform=get_default_transforms(split="val", image_size=image_size),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    test_loader = None
    if test_manifest_path is not None and Path(test_manifest_path).exists():
        test_df = pd.read_csv(test_manifest_path)
        test_ds = DRGradingDataset(
            manifest_df=test_df,
            project_root=project_root,
            transform=get_default_transforms(split="test", image_size=image_size),
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )

    return train_loader, val_loader, test_loader
