"""Unit tests for ResNet-50 model architecture and Grad-CAM layer extraction (Phase 3)."""

from __future__ import annotations

import pytest

from src.model import ICDR_GRADES, build_model, get_gradcam_target_layer


def test_icdr_grades_mapping():
    """Verify standard ICDR 5-class mapping."""
    assert len(ICDR_GRADES) == 5
    assert ICDR_GRADES[0] == "No DR"
    assert ICDR_GRADES[1] == "Mild"
    assert ICDR_GRADES[2] == "Moderate"
    assert ICDR_GRADES[3] == "Severe"
    assert ICDR_GRADES[4] == "Proliferative DR"


def test_build_model_default():
    """Verify default ResNet-50 construction with 5 classes and 0.3 dropout."""
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    model = build_model(num_classes=5, pretrained=False, dropout_rate=0.3)
    assert model is not None
    assert model.num_classes == 5
    assert model.dropout_rate == 0.3

    # Check fc structure: Sequential(Dropout(p=0.3), Linear(2048, 5))
    assert isinstance(model.fc, nn.Sequential)
    assert isinstance(model.fc[0], nn.Dropout)
    assert pytest.approx(model.fc[0].p) == 0.3
    assert isinstance(model.fc[1], nn.Linear)
    assert model.fc[1].in_features == 2048
    assert model.fc[1].out_features == 5

    # Test forward pass with dummy tensor
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 5)


def test_build_model_zero_dropout():
    """Verify that dropout_rate=0 creates a direct Linear layer without Dropout."""
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    model = build_model(num_classes=5, pretrained=False, dropout_rate=0.0)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 2048
    assert model.fc.out_features == 5


def test_build_model_config_override():
    """Verify that configuration dictionary overrides default parameters."""
    pytest.importorskip("torch")

    config = {
        "model": {
            "num_classes": 3,
            "dropout_rate": 0.45,
            "pretrained": False,
        }
    }
    model = build_model(config=config)
    assert model.num_classes == 3
    assert pytest.approx(model.dropout_rate) == 0.45
    assert model.fc[1].out_features == 3


def test_build_model_pretrained_false_overrides_config():
    """Verify that explicitly passing pretrained=False overrides config with pretrained=True."""
    pytest.importorskip("torch")

    config = {"model": {"pretrained": True, "num_classes": 5}}
    model = build_model(config=config, pretrained=False)
    assert model.pretrained is False
    assert model.num_classes == 5


def test_get_gradcam_target_layer():
    """Verify get_gradcam_target_layer correctly extracts model.layer4."""
    pytest.importorskip("torch")

    model = build_model(pretrained=False)
    target_layer = get_gradcam_target_layer(model)

    assert target_layer is model.layer4

    # Test invalid object raises AttributeError
    class DummyObj:
        pass

    with pytest.raises(AttributeError, match="layer4"):
        get_gradcam_target_layer(DummyObj())

