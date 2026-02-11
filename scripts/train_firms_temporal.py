#!/usr/bin/env python3
"""
Train U-Net on FIRMS temporal progression dataset (3-day history).
Input: data/firms_progressions/firms_progressions_temporal_*.npz
Output: outputs/checkpoints/unet_firms_temporal.pt
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
import time
from datetime import datetime

from src.model import build_unet

DATASET_PATH = ROOT / "data" / "firms_progressions" / "firms_progressions_temporal_256.npz"
OUTPUT_MODEL = ROOT / "outputs" / "checkpoints" / "unet_firms_temporal.pt"
METRICS_FILE = ROOT / "outputs" / "firms_temporal_training_metrics.json"

EPOCHS = 60  # Increased epochs since temporal data is more complex
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
VAL_SPLIT = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ERROR_LOG = ROOT / "outputs" / "training_errors.json"
LIVE_PROGRESS = ROOT / "outputs" / "training_firms_progress.json"


def _write_error(msg: str):
    """Write error to JSON for dashboard."""
    try:
        with open(ERROR_LOG, "w") as f:
            json.dump({"error": msg, "timestamp": datetime.now().isoformat()}, f)
    except:
        pass


def _write_progress(epoch, total_epochs, train_loss, val_loss, val_iou, val_dice, eta_min, status):
    """Write live progress to JSON for dashboard."""
    try:
        with open(LIVE_PROGRESS, "w") as f:
            json.dump({
                "current_epoch": epoch,
                "total_epochs": total_epochs,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_iou": float(val_iou),
                "val_dice": float(val_dice),
                "eta_minutes": float(eta_min),
                "status": status,
                "timestamp": datetime.now().isoformat()
            }, f)
    except:
        pass


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(logits, target):
    return 0.5 * nn.BCEWithLogitsLoss()(logits, target) + 0.5 * dice_loss(logits, target)


def calculate_metrics(pred, target):
    pred_binary = (pred > 0.5).float()
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return {"iou": iou.item(), "dice": dice.item()}


def main():
    print("=" * 80)
    print("TRAIN FIRMS TEMPORAL PROGRESSION MODEL")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not DATASET_PATH.exists():
        print("Temporal dataset not found.")
        print(f"Expected: {DATASET_PATH}")
        print("Run: python scripts/process_firms_progressions_temporal.py")
        sys.exit(1)

    print(f"Loading: {DATASET_PATH}")
    data = np.load(DATASET_PATH)
    X = data["X"]  # (N, 10, 256, 256) - 7 terrain + 3 fire history
    Y = data["Y"]  # (N, 1, 256, 256)
    print(f"Dataset: X={X.shape}, Y={Y.shape}")
    print(f"Input channels: {X.shape[1]}")

    # Train/val split
    n = len(X)
    n_val = max(1, int(n * VAL_SPLIT))
    indices = np.random.RandomState(42).permutation(n)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build U-Net with 10 input channels (7 terrain + 3 fire history)
    model = build_unet(in_channels=10, out_channels=1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
    }
    best_val_iou = 0.0
    start_time = time.time()

    print("=" * 80)
    print("TRAINING START")
    print("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = combined_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_loss += combined_loss(logits, y).item()
                metrics = calculate_metrics(torch.sigmoid(logits), y)
                val_iou += metrics["iou"]
                val_dice += metrics["dice"]

        if len(val_loader) > 0:
            val_loss /= len(val_loader)
            val_iou /= len(val_loader)
            val_dice /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"IoU: {val_iou:.4f} | Dice: {val_dice:.4f}"
        )

        # Write live progress for dashboard
        elapsed = (time.time() - start_time) / 60
        eta_min = (elapsed / epoch) * (EPOCHS - epoch) if epoch > 0 else 0
        _write_progress(epoch, EPOCHS, train_loss, val_loss, val_iou, val_dice, eta_min, "training")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
                "iou": val_iou,
                "dice": val_dice,
            }, OUTPUT_MODEL)
            print(f"  ✓ New best: {val_iou:.4f}")

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "best_val_iou": best_val_iou,
            "training_time_minutes": (time.time() - start_time) / 60,
            "input_channels": 10,
            "epochs": EPOCHS,
            "history": history,
        }, f, indent=2)

    # Mark as completed
    _write_progress(EPOCHS, EPOCHS, history["train_loss"][-1], history["val_loss"][-1], 
                   best_val_iou, history["val_dice"][history["val_iou"].index(best_val_iou)], 0, "completed")

    print("=" * 80)
    print(f"TRAINING COMPLETE - Best IoU: {best_val_iou:.4f}")
    print(f"Model saved: {OUTPUT_MODEL}")
    print(f"Metrics: {METRICS_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"\n❌ TRAINING ERROR: {error_msg}")
        _write_error(error_msg)
        _write_progress(0, EPOCHS, 0, 0, 0, 0, 0, "error")
        sys.exit(1)
