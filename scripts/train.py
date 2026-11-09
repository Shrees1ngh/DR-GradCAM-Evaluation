#!/usr/bin/env python3
"""CNN Model Training and Evaluation Pipeline for Diabetic Retinopathy (Phase 3).

Trains ResNet-50 backbone on the primary APTOS 2019 dataset using:
- Stratified APTOS train (80%, N=2,929) for training.
- Stratified APTOS val (20%, N=733) for validation, model selection, and early stopping.
- Class-weighted CrossEntropyLoss to address class imbalance across ICDR Grades 0-4.
- AdamW optimizer with Cosine Annealing learning rate schedule.
- Mixed precision training (torch.cuda.amp) when CUDA is available.
- Early stopping and model checkpointing monitored strictly on Validation Quadratic Weighted Kappa (QWK).
- Strictly held-out IDRiD External Test Set (N=103) evaluated ONLY post-training.

Usage:
    python scripts/train.py [--config configs/config.yaml] [--epochs 25] [--batch-size 8] [--lr 1e-4]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = object
    AdamW = None
    CosineAnnealingLR = None
    ReduceLROnPlateau = None
    DataLoader = None
    TORCH_AVAILABLE = False

from src.dataset import DRGradingDataset, get_default_transforms
from src.model import ICDR_GRADES, build_model


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def seed_everything(seed: int = 42) -> None:
    """Set random seeds across python, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_class_weights(manifest_path: Path, num_classes: int = 5) -> torch.Tensor:
    """Compute balanced inverse class weights from the training split manifest.

    Formula: w_c = N / (num_classes * count_c)

    Args:
        manifest_path: Path to train_split.csv containing 'grade' column.
        num_classes: Total number of discrete grade classes (5).

    Returns:
        torch.Tensor of shape (num_classes,) with float32 class weights.
    """
    df = pd.read_csv(manifest_path)
    counts = df["grade"].value_counts().to_dict()
    total = len(df)

    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        if cnt > 0:
            w = total / (num_classes * cnt)
        else:
            w = 1.0
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    logging.info(
        f"Computed class weights from {manifest_path.name} (N={total}): "
        + ", ".join([f"G{c}: {w:.3f}" for c, w in enumerate(weights)])
    )
    return weights_tensor


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive multiclass classification metrics for DR grading.

    Calculates: Accuracy, Macro F1, Quadratic Weighted Kappa (QWK),
    per-class precision, recall (sensitivity), F1, and confusion matrix.
    """
    class_names = class_names or ICDR_GRADES
    num_classes = len(class_names)
    labels = list(range(num_classes))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    per_class = {}
    for c in labels:
        per_class[f"grade_{c}_{class_names[c]}"] = {
            "precision": float(prec[c]),
            "recall": float(rec[c]),
            "f1": float(f1[c]),
            "support": int(sup[c]),
        }

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[class_names[c] for c in labels],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "qwk": qwk,
        "per_class": per_class,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 1,
) -> Tuple[float, float, float]:
    """Execute one training epoch with optional mixed precision."""
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["grade"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    acc = float(accuracy_score(all_targets, all_preds))
    qwk = float(cohen_kappa_score(all_targets, all_preds, weights="quadratic"))

    return epoch_loss, acc, qwk


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "Val",
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate model on validation or external test dataloader."""
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    pbar = tqdm(dataloader, desc=f"Evaluating [{split_name}]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["grade"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())
        all_probs.extend(probs)

    eval_loss = running_loss / len(dataloader.dataset)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_probs)
    return eval_loss, metrics, y_true, y_pred


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_path: Path,
) -> None:
    """Save PyTorch model checkpoint atomically."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(checkpoint_path)
    logging.info(f"Checkpoint successfully saved: {checkpoint_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ResNet-50 for Diabetic Retinopathy Grading.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay (overrides config)")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability (overrides config)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience epochs")
    parser.add_argument("--min-delta", type=float, default=None, help="Minimum validation QWK improvement required (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 quick epoch on small subset for verification")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; evaluate existing checkpoint")
    parser.add_argument("--eval-external", action="store_true", help="Explicitly evaluate strictly held-out IDRiD external test set")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Custom checkpoint path to load or save")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not TORCH_AVAILABLE:
        logging.error("PyTorch is not available. Please install torch and torchvision.")
        return 1

    config = load_config(args.config)

    # 1. Resolve Hyperparameters
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    project_cfg = config.get("project", {})

    seed = args.seed if args.seed is not None else project_cfg.get("random_seed", 42)
    seed_everything(seed)

    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 25)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 8)
    lr = args.lr if args.lr is not None else train_cfg.get("learning_rate", 1e-4)
    weight_decay = args.weight_decay if args.weight_decay is not None else train_cfg.get("weight_decay", 1e-4)
    dropout_rate = args.dropout if args.dropout is not None else model_cfg.get("dropout_rate", 0.3)
    patience = args.patience if args.patience is not None else train_cfg.get("patience", 5)
    min_delta = args.min_delta if args.min_delta is not None else float(train_cfg.get("min_delta", 0.001))
    num_classes = model_cfg.get("num_classes", 5)
    pretrained = model_cfg.get("pretrained", True)
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0)

    # 2. Select Device & Mixed Precision (Modern PyTorch AMP API)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_amp = bool(train_cfg.get("mixed_precision", True) and device.type == "cuda" and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logging.info(
        f"Training configuration: device={device.type.upper()}, batch_size={batch_size}, "
        f"lr={lr}, weight_decay={weight_decay}, dropout={dropout_rate}, epochs={epochs}, "
        f"patience={patience}, min_delta={min_delta}, amp={use_amp}, seed={seed}"
    )

    # 3. Resolve Dataset Manifest Paths
    processed_dir = PROJECT_ROOT / config.get("dataset", {}).get("processed_dir", "data/processed")
    train_csv = processed_dir / "aptos" / "train_split.csv"
    val_csv = processed_dir / "aptos" / "val_split.csv"
    idrid_test_csv = processed_dir / "idrid" / "external_test.csv"

    required_manifests = [train_csv, val_csv]
    if args.eval_external:
        required_manifests.append(idrid_test_csv)

    for path in required_manifests:
        if not path.exists():
            logging.error(f"Required manifest not found at: {path}. Run scripts/preprocess_data.py first.")
            return 1

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    idrid_test_df = pd.read_csv(idrid_test_csv) if args.eval_external else None

    if args.dry_run:
        logging.info("DRY-RUN mode active: truncating datasets to 16 samples for smoke testing.")
        train_df = train_df.head(16)
        val_df = val_df.head(16)
        if idrid_test_df is not None:
            idrid_test_df = idrid_test_df.head(16)
        epochs = 1

    # 4. Build Datasets and DataLoaders
    aug_cfg = config.get("preprocessing", {}).get("augmentation", {})
    train_dataset = DRGradingDataset(
        manifest_df=train_df,
        project_root=PROJECT_ROOT,
        transform=get_default_transforms(split="train", augmentation_config=aug_cfg),
    )
    val_dataset = DRGradingDataset(
        manifest_df=val_df,
        project_root=PROJECT_ROOT,
        transform=get_default_transforms(split="val"),
    )
    idrid_test_dataset = (
        DRGradingDataset(
            manifest_df=idrid_test_df,
            project_root=PROJECT_ROOT,
            transform=get_default_transforms(split="test"),
        )
        if idrid_test_df is not None
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    idrid_test_loader = (
        DataLoader(
            idrid_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if idrid_test_dataset is not None
        else None
    )

    ext_status = f", IDRiD External Test={len(idrid_test_dataset)}" if idrid_test_dataset is not None else " (IDRiD External Test: Skipped; opt-in with --eval-external)"
    logging.info(
        f"DataLoaders initialized: Train={len(train_dataset)}, Val={len(val_dataset)}{ext_status}"
    )

    # 5. Build Model
    model = build_model(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        config=config,
    )
    model.to(device)

    # 6. Loss Function with Inverse Class Weights
    class_weights = compute_class_weights(train_csv, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 7. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Output paths
    checkpoints_dir = PROJECT_ROOT / config.get("outputs", {}).get("checkpoints_dir", "experiments/checkpoints")
    experiments_dir = PROJECT_ROOT / "experiments"
    reports_dir = PROJECT_ROOT / config.get("outputs", {}).get("reports_dir", "reports")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = args.checkpoint_path or (checkpoints_dir / "best_model_qwk.pth")

    # Handle eval-only mode
    if args.eval_only:
        logging.info(f"Evaluation-only mode. Loading checkpoint from: {best_checkpoint_path}")
        if not best_checkpoint_path.exists():
            logging.error(f"Checkpoint not found at: {best_checkpoint_path}")
            return 1
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device, "Validation")
        logging.info(f"Validation QWK: {val_metrics['qwk']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

        if args.eval_external and idrid_test_loader is not None:
            test_loss, test_metrics, _, _ = evaluate(model, idrid_test_loader, criterion, device, "IDRiD External Test")
            logging.info(f"IDRiD Test QWK: {test_metrics['qwk']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}")
        else:
            logging.info("External test evaluation not requested. Pass --eval-external to evaluate on IDRiD.")
        return 0

    # 8. Training Loop with Early Stopping on Validation QWK
    logging.info(f"Starting model training for up to {epochs} epochs (min_delta={min_delta}, patience={patience})...")
    best_val_qwk = -float("inf")           # Highest actual validation QWK observed across all epochs
    best_val_metrics: Dict[str, Any] = {}  # Metrics corresponding to highest validation QWK
    best_epoch = 0                         # Epoch that produced the highest validation QWK
    patience_benchmark_qwk = -float("inf") # Benchmark QWK for early-stopping patience evaluation
    patience_counter = 0                   # Consecutive epochs without >= min_delta improvement over benchmark
    history: List[Dict[str, Any]] = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_qwk = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
        )

        val_loss, val_metrics, _, _ = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            split_name="Val",
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_time = time.time() - t0

        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        val_qwk = val_metrics["qwk"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 4),
            "train_qwk": round(train_qwk, 4),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "val_qwk": round(val_qwk, 4),
            "lr": f"{current_lr:.2e}",
            "epoch_time_sec": round(epoch_time, 2),
        }
        history.append(epoch_record)

        logging.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, QWK: {train_qwk:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_f1:.4f}, QWK: {val_qwk:.4f} | "
            f"lr: {current_lr:.2e}"
        )

        # 1. Checkpoint Selection: Always preserve the checkpoint with the highest actual validation QWK
        if val_qwk > best_val_qwk:
            actual_gain = val_qwk - best_val_qwk if best_val_qwk != -float("inf") else val_qwk
            best_val_qwk = val_qwk
            best_val_metrics = val_metrics
            best_epoch = epoch

            checkpoint_state = {
                "epoch": epoch,
                "best_val_qwk": best_val_qwk,
                "min_delta": min_delta,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": {
                    "backbone": "resnet50",
                    "num_classes": num_classes,
                    "pretrained": pretrained,
                    "dropout_rate": dropout_rate,
                },
                "class_mapping": ICDR_GRADES,
                "random_seed": seed,
                "val_metrics": val_metrics,
                "timestamp": datetime.now().isoformat(),
            }
            save_checkpoint(checkpoint_state, best_checkpoint_path)
            logging.info(
                f"--> Saved best model checkpoint at epoch {epoch:02d} "
                f"(Highest observed QWK: {best_val_qwk:.4f}, gain: {actual_gain:+.4f})"
            )

        # 2. Early-Stopping Patience: Reset patience only when QWK improves by at least min_delta over benchmark
        if patience_benchmark_qwk == -float("inf") or (val_qwk >= patience_benchmark_qwk + min_delta):
            benchmark_gain = val_qwk - patience_benchmark_qwk if patience_benchmark_qwk != -float("inf") else val_qwk
            patience_benchmark_qwk = val_qwk
            patience_counter = 0
            logging.info(
                f"Patience reset to 0 at epoch {epoch:02d} "
                f"(QWK: {val_qwk:.4f}, benchmark gain: {benchmark_gain:+.4f} >= min_delta {min_delta:.4f})"
            )
        else:
            patience_counter += 1
            logging.info(
                f"Patience: {patience_counter}/{patience} "
                f"(Current QWK: {val_qwk:.4f}, Benchmark: {patience_benchmark_qwk:.4f}, "
                f"Target for reset: >={patience_benchmark_qwk + min_delta:.4f}, "
                f"Best saved QWK: {best_val_qwk:.4f} at epoch {best_epoch:02d})"
            )
            if patience_counter >= patience:
                logging.info(
                    f"Early stopping triggered after {patience} epochs without QWK improvement >= {min_delta} "
                    f"(Preserving best checkpoint from epoch {best_epoch:02d} with QWK {best_val_qwk:.4f})."
                )
                break

    total_training_time = time.time() - start_time
    logging.info(f"Training completed in {total_training_time/60:.2f} minutes.")

    # 9. Save History CSV
    history_df = pd.DataFrame(history)
    history_csv = experiments_dir / "training_history.csv"
    history_df.to_csv(history_csv, index=False)
    logging.info(f"Training history saved: {history_csv}")

    # 10. Final Evaluation using Selected Best Checkpoint
    best_ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    # Re-evaluate validation with best model to confirm exact reproduction
    _, final_val_metrics, _, _ = evaluate(model, val_loader, criterion, device, "Validation [Best Checkpoint]")

    idrid_metrics = None
    if args.eval_external:
        if idrid_test_loader is not None:
            logging.info("=" * 70)
            logging.info("EXTERNAL EVALUATION: Evaluating best checkpoint on IDRiD External Test...")
            logging.info("=" * 70)
            idrid_loss, idrid_metrics, _, _ = evaluate(
                model, idrid_test_loader, criterion, device, "IDRiD External Test"
            )
            logging.info(f"IDRiD External Test QWK: {idrid_metrics['qwk']:.4f}, Acc: {idrid_metrics['accuracy']:.4f}")
        else:
            logging.warning("--eval-external was requested, but external test loader was not initialized.")
    else:
        logging.info("IDRiD External Test evaluation skipped (opt-in via --eval-external).")

    # 11. Save Full Metrics Summary JSON
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "random_seed": seed,
        "best_epoch": best_epoch,
        "total_epochs_trained": len(history),
        "total_training_time_sec": round(total_training_time, 2),
        "is_dry_run": bool(args.dry_run),
        "evaluated_external_test": bool(args.eval_external and idrid_metrics is not None),
        "sample_counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "external_test": len(idrid_test_dataset) if (args.eval_external and idrid_test_dataset is not None) else 0,
        },
        "hyperparameters": {
            "model": "resnet50",
            "pretrained": pretrained,
            "num_classes": num_classes,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "initial_lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "min_delta": min_delta,
            "device": device.type,
            "mixed_precision": use_amp,
        },
        "best_validation_metrics": final_val_metrics,
        "idrid_external_test_metrics": idrid_metrics,
    }

    metrics_json = experiments_dir / "evaluation_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logging.info(f"Evaluation metrics JSON saved: {metrics_json}")

    # 12. Generate Phase 3 Markdown Report
    report_md_path = reports_dir / "phase_3_training_report.md"
    generate_training_report(summary_data, history_df, report_md_path)
    logging.info(f"Phase 3 training report written to: {report_md_path}")

    banner_title = "PHASE 3: CNN MODEL DEVELOPMENT & TRAINING SUMMARY"
    if args.dry_run:
        banner_title += " [DRY-RUN / SMOKE TEST]"

    print("\n" + "=" * 70)
    print(banner_title)
    print("=" * 70)
    print(f"Architecture:            ResNet-50 (Pretrained={pretrained}, Dropout={dropout_rate})")
    print(f"Best Epoch:              {best_epoch} / {len(history)}")
    print(f"Validation QWK:          {final_val_metrics['qwk']:.4f} (Accuracy: {final_val_metrics['accuracy']*100:.2f}%) [N={len(val_dataset)}]")
    if idrid_metrics is not None:
        print(f"IDRiD External Test QWK: {idrid_metrics['qwk']:.4f} (Accuracy: {idrid_metrics['accuracy']*100:.2f}%) [N={len(idrid_test_dataset)}]")
    else:
        print("IDRiD External Test:     Skipped (opt-in with --eval-external)")
    print(f"Best Checkpoint:         {best_checkpoint_path}")
    print(f"Metrics JSON:            {metrics_json}")
    print(f"Report:                  {report_md_path}")
    if args.dry_run:
        print("NOTE: Dry-run mode was used. Results are for pipeline verification only.")
    print("=" * 70 + "\n")

    return 0


def generate_training_report(
    summary: dict, history_df: pd.DataFrame, output_path: Path
) -> None:
    """Generate comprehensive Markdown report for Phase 3 CNN model development."""
    val_m = summary["best_validation_metrics"]
    test_m = summary.get("idrid_external_test_metrics")
    hp = summary["hyperparameters"]
    counts = summary.get("sample_counts", {})
    train_n = counts.get("train", "N/A")
    val_n = counts.get("val", "N/A")
    test_n = counts.get("external_test", 0)
    is_dry_run = summary.get("is_dry_run", False)

    status_badge = "SMOKE TEST / PIPELINE VERIFICATION ONLY ⚠️" if is_dry_run else "COMPLETE & VERIFIED ✅"
    title_suffix = " [SMOKE TEST / DRY RUN]" if is_dry_run else ""

    dry_run_notice = ""
    if is_dry_run:
        dry_run_notice = f"""
> [!WARNING]
> **SMOKE TEST EXECUTION ONLY**
> This report was generated from a dry-run execution on a truncated sample subset
> (Train: {train_n}, Val: {val_n}).
> These metrics verify pipeline execution only and **DO NOT** represent full model convergence or valid research results.
> Full model training must be executed on GPU to obtain research-grade results.
"""

    ext_col_header = "External Benchmark (IDRiD Test)"
    if test_m is not None:
        ext_count_str = f"{test_n} images"
        ext_qwk_str = f"**{test_m['qwk']:.4f}**"
        ext_acc_str = f"**{test_m['accuracy']*100:.2f}%**"
        ext_f1_str = f"**{test_m['macro_f1']:.4f}**"
    else:
        ext_count_str = "Not Evaluated (Opt-in via `--eval-external`)"
        ext_qwk_str = "N/A"
        ext_acc_str = "N/A"
        ext_f1_str = "N/A"

    md = f"""# Phase 3 Training & Evaluation Report: CNN Classification Model{title_suffix}

**Project:** DR-GradCAM-Evaluation  
**Phase:** Phase 3 — CNN Model Development (ResNet-50)  
**Generated:** {summary['timestamp']}  
**Status:** {status_badge}  
{dry_run_notice}
---

## 1. Executive Summary

A 5-class Diabetic Retinopathy classification model based on ResNet-50 was trained on the primary APTOS 2019 dataset and evaluated on the internal validation split.

| Metric | Primary Validation (APTOS Val) | {ext_col_header} |
|:---|:---:|:---:|
| **Sample Count ($N$)** | {val_n} images | {ext_count_str} |
| **Quadratic Weighted Kappa (QWK)** | **{val_m['qwk']:.4f}** | {ext_qwk_str} |
| **Accuracy** | **{val_m['accuracy']*100:.2f}%** | {ext_acc_str} |
| **Macro F1** | **{val_m['macro_f1']:.4f}** | {ext_f1_str} |

> [!IMPORTANT]
> **Dataset Separation Guarantee:**
> - The primary model was trained exclusively on APTOS 2019 ($N={train_n}$).
> - Model selection and early stopping were monitored strictly on APTOS validation ($N={val_n}$).
> - IDRiD External Test was never used during training, tuning, or checkpoint selection.

---

## 2. Model & Training Hyperparameters

- **Backbone Architecture:** ResNet-50 (`torchvision.models.resnet50`)
- **Pretrained Initialization:** ImageNet (`ResNet50_Weights.DEFAULT`)
- **Dropout Rate:** {hp['dropout_rate']}
- **Output Classes:** {hp['num_classes']} (ICDR Grades 0–4)
- **Target Conv Layer for Grad-CAM:** `model.layer4` (2048-dim feature maps)
- **Batch Size:** {hp['batch_size']} (Optimized for Tesla T4 16GB GPU)
- **Optimizer:** AdamW (Initial LR: {hp['initial_lr']:.1e}, Weight Decay: {hp['weight_decay']:.1e})
- **LR Schedule:** CosineAnnealingLR
- **Loss Function:** Class-Weighted CrossEntropyLoss (Inversely proportional to class frequencies)
- **Early Stopping:** Patience = {hp['patience']} epochs on validation QWK (min_delta = {hp.get('min_delta', 0.001)})
- **Best Epoch:** {summary['best_epoch']} (Total epochs trained: {summary['total_epochs_trained']})
- **Random Seed:** {summary['random_seed']}

---

## 3. Class-Level Validation Performance (APTOS Val)

| Grade | Clinical Diagnosis | Precision | Recall (Sensitivity) | F1-Score | Support |
|:---:|:---|:---:|:---:|:---:|:---:|
"""
    for g, name in ICDR_GRADES.items():
        key = f"grade_{g}_{name}"
        if key in val_m.get("per_class", {}):
            stats = val_m["per_class"][key]
            md += f"| **Grade {g}** | {name} | {stats['precision']:.3f} | {stats['recall']:.3f} | {stats['f1']:.3f} | {stats['support']} |\n"

    md += f"""
### Confusion Matrix (APTOS Validation)
Rows: Ground Truth, Columns: Prediction

```
{np.array(val_m.get('confusion_matrix', []))}
```

---

## 4. External Test Performance (IDRiD External Test)
"""
    if test_m is not None:
        md += """
| Grade | Clinical Diagnosis | Precision | Recall (Sensitivity) | F1-Score | Support |
|:---:|:---|:---:|:---:|:---:|:---:|
"""
        for g, name in ICDR_GRADES.items():
            key = f"grade_{g}_{name}"
            if key in test_m.get("per_class", {}):
                stats = test_m["per_class"][key]
                md += f"| **Grade {g}** | {name} | {stats['precision']:.3f} | {stats['recall']:.3f} | {stats['f1']:.3f} | {stats['support']} |\n"

        md += f"""
### Confusion Matrix (IDRiD External Test)

```
{np.array(test_m.get('confusion_matrix', []))}
```
"""
    else:
        md += """
> [!NOTE]
> External test evaluation was not requested for this run.
> To benchmark the saved best checkpoint against the held-out IDRiD test cohort, execute:
> ```bash
> python scripts/train.py --eval-only --eval-external
> ```
"""

    md += """
---

## 5. Artifacts & Reproducibility Files

- **Best Checkpoint:** `experiments/checkpoints/best_model_qwk.pth`
- **Training History CSV:** `experiments/training_history.csv`
- **Full Metrics JSON:** `experiments/evaluation_metrics.json`
- **Source Module:** `src/model.py` (`build_model()`, `get_gradcam_target_layer()`)
- **Training CLI:** `scripts/train.py`

---
*Report auto-generated by `scripts/train.py` for Phase 3.*
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    sys.exit(main())

