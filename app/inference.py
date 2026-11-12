"""Inference module for Diabetic Retinopathy classification.

Integrates with the trained ResNet-50 checkpoint, applies verified
preprocessing, and provides deterministic model evaluation.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.model import ICDR_GRADES, build_model
from src.preprocessing import crop_to_retinal_fov, resize_and_pad

# Standard ImageNet normalization parameters used during training
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Grade descriptions following International Clinical Diabetic Retinopathy (ICDR) criteria
GRADE_DESCRIPTIONS = {
    0: {
        "title": "No Diabetic Retinopathy",
        "severity": "Normal",
        "description": "No visible retinal microvascular abnormalities. Absence of microaneurysms, hemorrhages, and exudates.",
        "badge_color": "#0F766E",
    },
    1: {
        "title": "Mild Non-Proliferative DR",
        "severity": "Mild",
        "description": "Microaneurysms only. Earliest clinically detectable stage of diabetic retinopathy.",
        "badge_color": "#0D9488",
    },
    2: {
        "title": "Moderate Non-Proliferative DR",
        "severity": "Moderate",
        "description": "More than microaneurysms but less severe than Severe NPDR. May present with microaneurysms, dot-and-blot hemorrhages, venous beading, or hard exudates.",
        "badge_color": "#D97706",
    },
    3: {
        "title": "Severe Non-Proliferative DR",
        "severity": "Severe",
        "description": "Any of the following criteria without signs of proliferative retinopathy: >20 intraretinal hemorrhages in each of 4 quadrants, definite venous beading in 2+ quadrants, or prominent IRMA in 1+ quadrant (4-2-1 rule).",
        "badge_color": "#DC2626",
    },
    4: {
        "title": "Proliferative Diabetic Retinopathy",
        "severity": "Proliferative",
        "description": "Advanced stage characterized by pathological neovascularization of the retina or optic disc, preretinal or vitreous hemorrhage, and fibrovascular proliferation.",
        "badge_color": "#991B1B",
    },
}


def get_device() -> torch.device:
    """Return the optimal available hardware device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load the trained ResNet-50 model from the specified checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (e.g., model_b_best.pth).
        device: Torch device (defaults to get_device()).

    Returns:
        Tuple of (model in eval mode, checkpoint metadata dict).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint fails to load or state dict is incompatible.
    """
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {ckpt_path}\n"
            "Please ensure 'experiments/checkpoints/model_b_best.pth' is present."
        )

    if device is None:
        device = get_device()

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config", {})
    num_classes = model_config.get("num_classes", 5)
    dropout_rate = model_config.get("dropout_rate", 0.3)

    # Initialize model architecture matching the saved configuration
    model = build_model(
        num_classes=num_classes,
        pretrained=False,
        dropout_rate=dropout_rate,
    )

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError("Checkpoint missing 'model_state_dict' key.")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_qwk = checkpoint.get("best_val_qwk")
    if val_qwk is None:
        val_qwk = checkpoint.get("val_qwk")

    meta = {
        "epoch": checkpoint.get("epoch"),
        "best_val_qwk": val_qwk,
        "timestamp": checkpoint.get("timestamp"),
        "val_metrics": checkpoint.get("val_metrics", {}),
        "device": str(device),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_name": ckpt_path.name,
    }

    return model, meta


def validate_image(image_input: Union[bytes, io.BytesIO, Image.Image, np.ndarray]) -> Image.Image:
    """Validate uploaded image integrity, dimensions, and format.

    Args:
        image_input: Image bytes, BytesIO buffer, PIL Image, or numpy array.

    Returns:
        Validated PIL.Image instance converted to RGB mode.

    Raises:
        ValueError: If image cannot be read or is invalid.
    """
    try:
        if isinstance(image_input, Image.Image):
            pil_img = image_input
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, (bytes, io.BytesIO)):
            if isinstance(image_input, bytes):
                image_input = io.BytesIO(image_input)
            pil_img = Image.open(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Convert to RGB mode (handles RGBA, grayscale, or CMYK uploads)
        rgb_img = pil_img.convert("RGB")
        w, h = rgb_img.size

        if w < 64 or h < 64:
            raise ValueError(f"Image dimensions ({w}x{h}) are too small for retinal analysis (minimum 64x64 required).")

        return rgb_img

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to decode fundus image: {str(e)}")


def preprocess_retinal_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    crop_fov: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
    """Execute the exact preprocessing pipeline used during model training and evaluation.

    Steps:
    1. Retinal FOV detection and black border cropping (green channel thresholding > 10).
    2. Isotropic resize maintaining native retinal aspect ratio.
    3. Symmetric centered zero-padding to target_size (512, 512).
    4. ImageNet normalization: (x / 255 - mean) / std.

    Args:
        image: Validated PIL Image in RGB format.
        target_size: Desired dimensions (512, 512).
        crop_fov: Detect and crop uninformative borders.
        device: Device to place tensor on.

    Returns:
        preprocessed_rgb: uint8 array (512, 512, 3) for display and Grad-CAM overlay.
        tensor_img: Normalized float32 tensor of shape (1, 3, 512, 512).
        meta: Transformation bounding box and padding metadata.
    """
    raw_rgb = np.array(image, dtype=np.uint8)
    raw_h, raw_w = raw_rgb.shape[:2]

    if crop_fov:
        cropped_img, bbox = crop_to_retinal_fov(raw_rgb, threshold=10, margin_ratio=0.01)
    else:
        cropped_img = raw_rgb
        bbox = (0, raw_h, 0, raw_w)

    padded_img, pad_offsets, scale = resize_and_pad(
        cropped_img, target_size=target_size, interpolation=cv2.INTER_AREA
    )

    # Convert to float32 [0.0, 1.0] and apply ImageNet normalization
    rgb_float = padded_img.astype(np.float32) / 255.0
    norm_img = (rgb_float - IMAGENET_MEAN) / IMAGENET_STD

    # Reshape from (H, W, C) to (1, C, H, W) PyTorch batch tensor
    tensor_img = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0)

    if device is not None:
        tensor_img = tensor_img.to(device)

    meta = {
        "raw_shape": (raw_h, raw_w),
        "crop_bbox": bbox,
        "pad_offsets": pad_offsets,
        "scale": scale,
        "target_size": target_size,
    }

    return padded_img, tensor_img, meta


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    tensor_img: torch.Tensor,
) -> Dict[str, Any]:
    """Execute forward inference on preprocessed tensor and compute probabilities.

    Args:
        model: Loaded model in eval mode.
        tensor_img: 4D tensor (1, 3, 512, 512).

    Returns:
        Dictionary containing:
        - predicted_grade: int (0 to 4)
        - predicted_label: str ("No DR", "Mild", etc.)
        - confidence: float (0.0 to 1.0)
        - probabilities: dict mapping grade int to probability float
        - raw_logits: list of float values
    """
    model.eval()
    logits = model(tensor_img)
    probs_tensor = torch.softmax(logits, dim=1)

    probs = probs_tensor.detach().cpu().numpy()[0]
    raw_logits = logits.detach().cpu().numpy()[0]

    predicted_grade = int(np.argmax(probs))
    confidence = float(probs[predicted_grade])

    probabilities = {grade: float(probs[grade]) for grade in range(len(probs))}

    return {
        "predicted_grade": predicted_grade,
        "predicted_label": ICDR_GRADES[predicted_grade],
        "confidence": confidence,
        "probabilities": probabilities,
        "raw_logits": [float(val) for val in raw_logits],
        "grade_info": GRADE_DESCRIPTIONS[predicted_grade],
    }

