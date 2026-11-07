"""PyTorch Dataset implementations and DataLoader builders for Diabetic Retinopathy.

Provides:
- DRGradingDataset: 5-class ICDR classification (APTOS 2019 & IDRiD external benchmarks)
- SegmentationLesionDataset: IDRiD 81 lesion-annotated images with multi-lesion masks
- SynchronizedJointTransform: Synchronous spatial transforms for paired image and masks
- build_dataloaders: DataLoader construction with reproducible batching and split handling
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    import torchvision.transforms.functional as TF
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    DataLoader = None
    Dataset = object
    transforms = None
    TF = None
    TORCH_AVAILABLE = False

logger = logging.getLogger("dataset")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transforms(
    split: str = "train",
    image_size: Tuple[int, int] = (512, 512),
):
    """Build torchvision transformation pipeline for train, val, or test splits.

    Applied to classification images. Augmentations are applied only to 'train';
    'val' and 'test' splits are strictly deterministic.

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


class SynchronizedJointTransform:
    """Synchronous spatial augmentation for paired image and segmentation masks.

    Applies identical random spatial transformations (horizontal flip, vertical flip,
    affine rotation) to both the fundus image and all corresponding lesion masks.
    Uses nearest-neighbor interpolation on masks to strictly preserve binary {0, 1} values.
    Photometric jitter is applied strictly to the image, never to the masks.

    Applied exclusively to training data; never to validation or test data.
    """

    def __init__(
        self,
        hflip_p: float = 0.5,
        vflip_p: float = 0.5,
        rotation_degrees: float = 15.0,
        brightness_jitter: float = 0.1,
        contrast_jitter: float = 0.1,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torchvision are required.")
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotation_degrees = rotation_degrees
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter

    def __call__(
        self, image: Image.Image, masks: Dict[str, Image.Image]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply synchronous transform to PIL Image and dict of PIL mask images."""
        # 1. Random horizontal flip
        if random.random() < self.hflip_p:
            image = TF.hflip(image)
            masks = {k: TF.hflip(m) for k, m in masks.items()}

        # 2. Random vertical flip
        if random.random() < self.vflip_p:
            image = TF.vflip(image)
            masks = {k: TF.vflip(m) for k, m in masks.items()}

        # 3. Random rotation
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            masks = {
                k: TF.rotate(m, angle, interpolation=transforms.InterpolationMode.NEAREST)
                for k, m in masks.items()
            }

        # 4. Color jitter strictly for image
        if self.brightness_jitter > 0 or self.contrast_jitter > 0:
            b_factor = random.uniform(1.0 - self.brightness_jitter, 1.0 + self.brightness_jitter)
            c_factor = random.uniform(1.0 - self.contrast_jitter, 1.0 + self.contrast_jitter)
            image = TF.adjust_brightness(image, b_factor)
            image = TF.adjust_contrast(image, c_factor)

        # 5. Convert to Tensors and normalize image
        tensor_img = TF.to_tensor(image)
        norm_img = TF.normalize(tensor_img, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        tensor_masks = {
            k: (torch.from_numpy(np.array(m)) > 0).float()
            for k, m in masks.items()
        }

        return norm_img, tensor_masks


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
    """PyTorch Dataset for the 81 IDRiD segmentation images and their multi-lesion masks.

    Metadata preserves cohort origin:
    - 'idrid_grading_train_cohort' (57 images)
    - 'idrid_grading_test_cohort' (3 images)
    - 'idrid_seg_unique' (21 images)

    Note: All 81 images are strictly UNSEEN by the primary classification model,
    which is trained exclusively on APTOS 2019.
    """

    LESION_CODES = ["MA", "HE", "EX", "SE", "OD"]

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        project_root: Optional[Path] = None,
        transform: Optional[Callable] = None,
        joint_transform: Optional[SynchronizedJointTransform] = None,
    ):
        """
        Args:
            manifest_df: DataFrame with image and mask paths.
            project_root: Project root path to resolve relative file paths.
            transform: Image transformation pipeline (Normalize).
            joint_transform: Synchronized joint spatial transform (training only).
        """
        self.manifest = manifest_df.reset_index(drop=True)
        self.project_root = project_root or Path.cwd()
        self.transform = transform
        self.joint_transform = joint_transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.manifest.iloc[idx]
        raw_path_str = str(row["image_path"])
        p = Path(raw_path_str)
        img_path = p if p.is_absolute() else (self.project_root / p).resolve()

        with Image.open(img_path) as pil_img:
            pil_rgb = pil_img.convert("RGB")
            raw_rgb = np.array(pil_rgb, dtype=np.uint8)

        h, w = raw_rgb.shape[:2]

        pil_masks: Dict[str, Image.Image] = {}
        annotation_status: Dict[str, bool] = {}
        pixel_counts: Dict[str, int] = {}

        for code in self.LESION_CODES:
            col_name = f"mask_{code}"
            annotated_col = f"annotated_{code}"

            is_annotated = bool(row.get(annotated_col, True))
            annotation_status[code] = is_annotated
            pixel_counts[code] = int(row.get(f"pixel_count_{code}", -1 if not is_annotated else 0))

            mask_path_str = row.get(col_name)
            if pd.notna(mask_path_str) and mask_path_str:
                mp = Path(str(mask_path_str))
                full_mask_path = mp if mp.is_absolute() else (self.project_root / mp).resolve()
                if full_mask_path.exists():
                    pil_masks[code] = Image.open(full_mask_path).convert("L")
                else:
                    pil_masks[code] = Image.new("L", (w, h), 0)
            else:
                pil_masks[code] = Image.new("L", (w, h), 0)

        # Composite mask (MA U HE U EX U SE) - OD strictly excluded
        comp_path_str = row.get("mask_composite")
        if pd.notna(comp_path_str) and comp_path_str:
            cp = Path(str(comp_path_str))
            full_comp_path = cp if cp.is_absolute() else (self.project_root / cp).resolve()
            if full_comp_path.exists():
                pil_masks["composite"] = Image.open(full_comp_path).convert("L")
            else:
                pil_masks["composite"] = Image.new("L", (w, h), 0)
        else:
            pil_masks["composite"] = Image.new("L", (w, h), 0)

        # Apply transformations
        if self.joint_transform is not None:
            tensor_img, masks = self.joint_transform(pil_rgb, pil_masks)
        else:
            if self.transform is not None:
                tensor_img = self.transform(raw_rgb)
            else:
                tensor_img = transforms.functional.to_tensor(raw_rgb)
            masks = {
                k: (torch.from_numpy(np.array(m)) > 0).float()
                for k, m in pil_masks.items()
            }

        cohort_origin = str(row.get("cohort_origin", "idrid_seg_unique"))
        model_seen_status = str(row.get("model_seen_status", "unseen"))
        grading_match_id = str(row.get("grading_match_id", ""))

        return {
            "image": tensor_img,
            "raw_image": raw_rgb,
            "image_id": str(row["image_id"]),
            "masks": masks,
            "annotation_status": annotation_status,
            "pixel_counts": pixel_counts,
            "cohort_origin": cohort_origin,
            "model_seen_status": model_seen_status,
            "grading_match_id": grading_match_id,
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
    """Factory to construct train, val, and optional test DataLoaders."""
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
