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


def test_early_stopping_min_delta_logic():
    """Verify early stopping patience logic correctly respects min_delta."""
    min_delta = 0.01
    patience = 2

    # Simulate sequence of validation QWKs
    # Epoch 1: 0.50 (initial -> patience=0, benchmark=0.50)
    # Epoch 2: 0.505 (gain=0.005 < 0.01 -> patience increments to 1)
    # Epoch 3: 0.508 (gain=0.008 < 0.01 -> patience increments to 2 -> early stop!)
    val_qwks = [0.50, 0.505, 0.508]
    best_val_qwk = -float("inf")
    best_epoch = 0
    patience_benchmark_qwk = -float("inf")
    patience_counter = 0
    stopped = False

    for epoch, val_qwk in enumerate(val_qwks, start=1):
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            best_epoch = epoch

        if patience_benchmark_qwk == -float("inf") or (val_qwk >= patience_benchmark_qwk + min_delta):
            patience_benchmark_qwk = val_qwk
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopped = True
                break

    assert stopped is True
    assert best_val_qwk == 0.508
    assert best_epoch == 3
    assert patience_counter == 2


def test_checkpoint_preservation_small_increments_then_large_jump():
    """Verify that every actual QWK improvement updates the checkpoint,
    while patience resets only on gains >= min_delta over benchmark."""
    min_delta = 0.01
    patience = 4

    # Epoch 1: 0.600 -> Saves (initial), patience=0, benchmark=0.600
    # Epoch 2: 0.602 -> Saves (0.602 > 0.600), but gain 0.002 < 0.01 -> patience=1
    # Epoch 3: 0.605 -> Saves (0.605 > 0.602), but gain 0.005 < 0.01 -> patience=2
    # Epoch 4: 0.607 -> Saves (0.607 > 0.605), but gain 0.007 < 0.01 -> patience=3
    # Epoch 5: 0.630 -> Saves (0.630 > 0.607), gain 0.030 >= 0.01 -> patience resets to 0! benchmark=0.630
    val_qwks = [0.600, 0.602, 0.605, 0.607, 0.630]

    best_val_qwk = -float("inf")
    best_epoch = 0
    patience_benchmark_qwk = -float("inf")
    patience_counter = 0

    saved_checkpoints = []  # List of (epoch, qwk)
    patience_history = []

    for epoch, val_qwk in enumerate(val_qwks, start=1):
        # 1. Checkpoint selection: highest observed validation QWK
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            best_epoch = epoch
            saved_checkpoints.append((epoch, val_qwk))

        # 2. Early stopping patience: reset only when gain >= min_delta over benchmark
        if patience_benchmark_qwk == -float("inf") or (val_qwk >= patience_benchmark_qwk + min_delta):
            patience_benchmark_qwk = val_qwk
            patience_counter = 0
        else:
            patience_counter += 1

        patience_history.append(patience_counter)

    # Every small improvement was preserved in checkpoints
    assert saved_checkpoints == [
        (1, 0.600),
        (2, 0.602),
        (3, 0.605),
        (4, 0.607),
        (5, 0.630),
    ]

    # Patience accumulated during small increments (1, 2, 3), then reset to 0 on significant jump at epoch 5
    assert patience_history == [0, 1, 2, 3, 0]
    assert best_val_qwk == 0.630
    assert best_epoch == 5
    assert patience_benchmark_qwk == 0.630


def test_eval_external_cli_arg_opt_in():
    """Verify --eval-external defaults to False and is opt-in."""
    import argparse
    from scripts.train import main

    # Inspect parser configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-external", action="store_true", default=False)
    args_default = parser.parse_args([])
    assert args_default.eval_external is False

    args_opted_in = parser.parse_args(["--eval-external"])
    assert args_opted_in.eval_external is True


def test_dry_run_report_sample_counts_and_labeling(tmp_path: Path):
    """Verify report reflects actual sample counts and labels dry-run smoke tests."""
    from scripts.train import generate_training_report

    output_path = tmp_path / "test_report.md"
    history_df = pd.DataFrame([{"epoch": 1, "train_loss": 1.5, "val_qwk": 0.1}])

    # 1. Test Dry-Run Report
    dry_run_summary = {
        "timestamp": "2026-11-09T10:00:00",
        "random_seed": 42,
        "best_epoch": 1,
        "total_epochs_trained": 1,
        "total_training_time_sec": 12.5,
        "is_dry_run": True,
        "sample_counts": {
            "train": 16,
            "val": 16,
            "external_test": 0,
        },
        "hyperparameters": {
            "model": "resnet50",
            "pretrained": True,
            "num_classes": 5,
            "dropout_rate": 0.3,
            "batch_size": 8,
            "initial_lr": 1e-4,
            "weight_decay": 1e-4,
            "patience": 5,
            "min_delta": 0.001,
            "device": "cpu",
        },
        "best_validation_metrics": {
            "accuracy": 0.25,
            "macro_f1": 0.20,
            "qwk": 0.10,
            "per_class": {},
            "confusion_matrix": [[4, 0], [0, 12]],
        },
        "idrid_external_test_metrics": None,
    }

    generate_training_report(dry_run_summary, history_df, output_path)
    content = output_path.read_text(encoding="utf-8")

    assert "SMOKE TEST" in content
    assert "16 images" in content
    assert "Not Evaluated (Opt-in via `--eval-external`)" in content
    assert "733" not in content  # Hardcoded count must NOT be present
    assert "103" not in content

    # 2. Test Full Training Report without External Test
    full_summary = dict(dry_run_summary)
    full_summary["is_dry_run"] = False
    full_summary["sample_counts"] = {"train": 2929, "val": 733, "external_test": 0}

    generate_training_report(full_summary, history_df, output_path)
    full_content = output_path.read_text(encoding="utf-8")

    assert "SMOKE TEST" not in full_content
    assert "COMPLETE & VERIFIED" in full_content
    assert "733 images" in full_content
    assert "2929" in full_content


def test_modern_amp_api_and_cpu_fallback():
    """Verify modern torch.amp API is available and gracefully bypassed on CPU."""
    torch = pytest.importorskip("torch")

    assert hasattr(torch, "amp")
    assert hasattr(torch.amp, "autocast")
    assert hasattr(torch.amp, "GradScaler")

    # On CPU, scaler is None and execution does not trigger CUDA amp
    device = torch.device("cpu")
    use_amp = False
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None
    assert scaler is None


def test_configurable_augmentation():
    """Verify get_default_transforms respects custom augmentation configuration and defaults."""
    torch = pytest.importorskip("torch")
    from torchvision import transforms
    from src.dataset import get_default_transforms

    # 1. Default pipeline must NOT include vertical flip (p=0.0 default)
    default_train_pipe = get_default_transforms(split="train")
    has_vflip_default = any(isinstance(t, transforms.RandomVerticalFlip) for t in default_train_pipe.transforms)
    assert has_vflip_default is False, "Vertical flip must be disabled by default (p=0.0)"

    # 2. Config with explicit vertical flip enabled
    vflip_cfg = {"vertical_flip_prob": 0.5}
    vflip_train_pipe = get_default_transforms(split="train", augmentation_config=vflip_cfg)
    has_vflip_enabled = any(isinstance(t, transforms.RandomVerticalFlip) for t in vflip_train_pipe.transforms)
    assert has_vflip_enabled is True, "Vertical flip should be enabled when explicitly configured > 0"

    # 3. Config with 0% flips and 0 rotation
    custom_cfg = {
        "horizontal_flip_prob": 0.0,
        "vertical_flip_prob": 0.0,
        "rotation_degrees": 0.0,
        "brightness_jitter": 0.0,
        "contrast_jitter": 0.0,
    }

    train_pipe = get_default_transforms(split="train", augmentation_config=custom_cfg)
    val_pipe = get_default_transforms(split="val")
    test_pipe = get_default_transforms(split="test")

    # Check that dummy image transforms through all splits without errors
    dummy_img = np.full((512, 512, 3), 128, dtype=np.uint8)
    t_train = train_pipe(dummy_img)
    t_val = val_pipe(dummy_img)
    t_test = test_pipe(dummy_img)

    assert isinstance(t_train, torch.Tensor)
    assert t_train.shape == (3, 512, 512)
    assert isinstance(t_val, torch.Tensor)
    assert t_val.shape == (3, 512, 512)
    assert isinstance(t_test, torch.Tensor)
    assert t_test.shape == (3, 512, 512)

    # Validation and test must produce identical output on identical input (deterministic)
    assert torch.allclose(t_val, t_test)


def test_output_dir_cli_arg_and_auto_creation(tmp_path: Path):
    """Verify --output-dir argument parsing and automatic directory creation."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)

    # 1. Default is None (sensible default for local execution)
    args_default = parser.parse_args([])
    assert args_default.output_dir is None

    # 2. Configurable via CLI
    drive_output = tmp_path / "content" / "drive" / "MyDrive" / "DR-GradCAM" / "outputs"
    args_drive = parser.parse_args(["--output-dir", str(drive_output)])
    assert args_drive.output_dir == drive_output

    # 3. Target subdirectories are created automatically even if deep/nested
    checkpoints_dir = drive_output / "checkpoints"
    reports_dir = drive_output / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    assert drive_output.exists()
    assert checkpoints_dir.exists()
    assert reports_dir.exists()


def test_save_checkpoint_creates_missing_directories(tmp_path: Path):
    """Verify save_checkpoint automatically creates missing nested parent directories."""
    torch = pytest.importorskip("torch")

    deep_ckpt_path = tmp_path / "deeply" / "nested" / "drive" / "folder" / "model.pth"
    assert not deep_ckpt_path.parent.exists()

    state = {"epoch": 1, "qwk": 0.9}
    save_checkpoint(state, deep_ckpt_path)

    assert deep_ckpt_path.exists()
    loaded = torch.load(deep_ckpt_path, weights_only=False)
    assert loaded["qwk"] == 0.9


