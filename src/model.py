"""CNN Model Architecture for Diabetic Retinopathy 5-Class Grading.

Implements ResNet-50 backbone with ImageNet pretraining, configurable dropout,
custom classification head for ICDR grades 0-4, and Grad-CAM layer extraction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger("model")

# ICDR 5-class Diabetic Retinopathy Grading standard mapping
ICDR_GRADES: Dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = object
    models = None
    ResNet50_Weights = None
    TORCH_AVAILABLE = False


def build_model(
    num_classes: Optional[int] = None,
    pretrained: Optional[bool] = None,
    dropout_rate: Optional[float] = None,
    config: Optional[dict] = None,
) -> Any:
    """Build and configure ResNet-50 for Diabetic Retinopathy classification.

    Replaces the default 1000-class ImageNet fully connected head with a custom
    dropout layer and linear classifier matching `num_classes`.

    Args:
        num_classes: Number of output DR grade classes (default: 5 for ICDR).
        pretrained: Whether to initialize with ImageNet pretrained weights.
                    If explicitly False, avoids downloading ImageNet weights.
        dropout_rate: Dropout probability applied before the final linear layer (default: 0.3).
        config: Optional project configuration dictionary. If provided, values under
                `config["model"]` supply defaults for arguments not explicitly provided.

    Returns:
        Configured PyTorch nn.Module (ResNet-50 instance).

    Raises:
        ImportError: If PyTorch or torchvision is not installed.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch and torchvision are required to build the CNN model. "
            "Please install torch and torchvision."
        )

    # Resolve arguments: explicit arguments take precedence over config defaults
    model_cfg = config.get("model", {}) if config is not None else {}
    if num_classes is None:
        num_classes = model_cfg.get("num_classes", 5)
    if pretrained is None:
        pretrained = model_cfg.get("pretrained", True)
    if dropout_rate is None:
        dropout_rate = model_cfg.get("dropout_rate", 0.3)

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features  # 2048 for ResNet-50

    if dropout_rate > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )
    else:
        model.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    # Attach model metadata for reproducibility
    model.num_classes = num_classes
    model.dropout_rate = dropout_rate
    model.pretrained = pretrained
    model.class_names = ICDR_GRADES

    logger.info(
        f"Built ResNet-50: pretrained={pretrained}, num_classes={num_classes}, "
        f"dropout_rate={dropout_rate}, in_features={in_features}"
    )
    return model


def get_gradcam_target_layer(model: Any) -> Any:
    """Return the primary convolutional target layer for Grad-CAM.

    For ResNet-50, this is `model.layer4` (the final convolutional block consisting
    of 3 bottleneck blocks, outputting 2048-channel high-level feature maps).

    Args:
        model: ResNet-50 instance built via `build_model()`.

    Returns:
        The target convolutional layer module (`model.layer4`).

    Raises:
        AttributeError: If `model` does not have a `layer4` attribute.
    """
    if not hasattr(model, "layer4"):
        raise AttributeError("Provided model does not contain expected 'layer4' attribute.")
    return model.layer4

