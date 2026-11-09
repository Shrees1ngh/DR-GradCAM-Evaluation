"""Unit tests for training metrics, class weights, and checkpointing (Phase 3)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.train import (
    compute_class_weights,
    compute_metrics,
    save_checkpoint,
    seed_everything,
)


def test_seed_everything():
    """Verify seed_everything sets reproducible random states."""
    seed_everything(123)
    val1 = np.random.rand()
    seed_everything(123)
    val2 = np.random.rand()
    assert val1 == val2


def test_compute_metrics_perfect():
    """Perfect predictions must yield accuracy=1.0, macro_f1=1.0, qwk=1.0."""
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["qwk"] == 1.0
    assert len(metrics["per_class"]) == 5
    assert len(metrics["confusion_matrix"]) == 5


def test_compute_metrics_qwk_quadratic_penalty():
    """Off-by-one errors must yield higher QWK than distant off-by-four errors."""
    y_true = np.array([0, 0, 0, 0, 4, 4, 4, 4])

    # Minor off-by-one error: predicting 1 instead of 0
    y_pred_close = np.array([1, 1, 0, 0, 3, 3, 4, 4])
    metrics_close = compute_metrics(y_true, y_pred_close)

    # Severe error: predicting 4 instead of 0, and 0 instead of 4
    y_pred_far = np.array([4, 4, 0, 0, 0, 0, 4, 4])
    metrics_far = compute_metrics(y_true, y_pred_far)

    assert metrics_close["qwk"] > metrics_far["qwk"]


def test_compute_class_weights(tmp_path: Path):
    """Verify class weights are inversely proportional to class frequencies."""
    pytest.importorskip("torch")

    # Create dummy manifest: 10 of Grade 0, 2 of Grade 1, 1 of Grade 2, 1 of Grade 3, 1 of Grade 4
    # Total = 15 samples across 5 classes
    rows = []
    for _ in range(10):
        rows.append({"image_id": "img", "grade": 0})
    for _ in range(2):
        rows.append({"image_id": "img", "grade": 1})
    for g in [2, 3, 4]:
        rows.append({"image_id": "img", "grade": g})

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "dummy_train.csv"
    df.to_csv(csv_path, index=False)

    weights = compute_class_weights(csv_path, num_classes=5)

    assert weights.shape == (5,)
    # Class 0 has the most samples -> lowest weight
    # Class 2, 3, 4 have 1 sample -> highest weight
    assert weights[0] < weights[1]
    assert weights[1] < weights[2]
    # Check exact formula: w_0 = 15 / (5 * 10) = 0.3
    assert pytest.approx(weights[0].item()) == 0.3
    # w_1 = 15 / (5 * 2) = 1.5
    assert pytest.approx(weights[1].item()) == 1.5


def test_save_and_load_checkpoint(tmp_path: Path):
    """Verify checkpoint serialization and metadata preservation."""
    torch = pytest.importorskip("torch")

    ckpt_path = tmp_path / "checkpoints" / "test_model.pth"
    state = {
        "epoch": 5,
        "best_val_qwk": 0.852,
        "model_state_dict": {"dummy_weight": torch.tensor([1.0, 2.0])},
        "optimizer_state_dict": {"param_groups": []},
        "model_config": {"backbone": "resnet50", "num_classes": 5},
        "random_seed": 42,
    }

    save_checkpoint(state, ckpt_path)
    assert ckpt_path.exists()

    loaded = torch.load(ckpt_path, weights_only=False)
    assert loaded["epoch"] == 5
    assert loaded["best_val_qwk"] == 0.852
    assert loaded["random_seed"] == 42
    assert torch.equal(loaded["model_state_dict"]["dummy_weight"], torch.tensor([1.0, 2.0]))

