#!/usr/bin/env python3
"""
Resume training from checkpoint - continue from epoch 29 to 80.
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

DATASET_PATH = ROOT / "data" / "firms_progressions" / "firms_progressions_augmented.npz"
CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "unet_firms_augmented.pt"
OUTPUT_MODEL = ROOT / "outputs" / "checkpoints" / "unet_firms_augmented.pt"
METRICS_FILE = ROOT / "outputs" / "firms_augmented_training_metrics.json"

START_EPOCH = 29  # Resume from here
TOTAL_EPOCHS = 80
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
VAL_SPLIT = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LIVE_PROGRESS = ROOT / "outputs" / "training_augmented_progress.json"


def _write_progress(epoch, total_epochs, train_loss, val_loss, val_iou, val_dice, eta_min, status):
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
    print(f"RESUME AUGMENTED TRAINING FROM EPOCH {START_EPOCH}")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    data = np.load(DATASET_PATH)
    X = torch.from_numpy(data["X"]).float()
    Y = torch.from_numpy(data["Y"]).float()

    print(f"Samples: {len(X)} (8x augmented)")
    
    # Same split as original
    n_val = max(1, int(VAL_SPLIT * len(X)))
    indices = torch.randperm(len(X))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load checkpoint
    print(f"\nLoading checkpoint from epoch 28...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    model = build_unet(in_channels=8, out_channels=1, dropout_rate=0.3).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Adjust scheduler to start from epoch 28
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    for _ in range(START_EPOCH - 1):
        scheduler.step()
    
    best_val_iou = checkpoint['iou']
    print(f"Resuming from: Epoch {checkpoint['epoch']}, IoU {best_val_iou:.4f}")
    
    # Initialize history (we'll start fresh for resumed epochs)
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

    print("\n" + "=" * 80)
    print(f"RESUMING TRAINING: EPOCHS {START_EPOCH}-{TOTAL_EPOCHS}")
    print("=" * 80)

    start_time = time.time()
    for epoch in range(START_EPOCH, TOTAL_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}", leave=False):
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

        improvement = ""
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
            improvement = f"  ✓ New best: {val_iou:.4f}"

        print(
            f"Epoch {epoch:3d}/{TOTAL_EPOCHS} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"IoU: {val_iou:.4f} | Dice: {val_dice:.4f}{improvement}"
        )

        elapsed = (time.time() - start_time) / 60
        eta_min = (elapsed / (epoch - START_EPOCH + 1)) * (TOTAL_EPOCHS - epoch) if epoch > START_EPOCH else 0
        _write_progress(epoch, TOTAL_EPOCHS, train_loss, val_loss, val_iou, val_dice, eta_min, "training")

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "best_val_iou": best_val_iou,
            "training_time_minutes": (time.time() - start_time) / 60,
            "total_samples": len(X),
            "augmentation_factor": 8,
            "resumed_from_epoch": START_EPOCH - 1,
            "history": history,
        }, f, indent=2)

    best_dice = history["val_dice"][history["val_iou"].index(max(history["val_iou"]))] if history["val_iou"] else 0
    _write_progress(TOTAL_EPOCHS, TOTAL_EPOCHS, history["train_loss"][-1], history["val_loss"][-1], 
                   best_val_iou, best_dice, 0, "completed")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best IoU: {best_val_iou:.4f} ({best_val_iou*100:.2f}%)")
    print(f"Model saved: {OUTPUT_MODEL}")
    print(f"\nComparison:")
    print(f"  Baseline (300 samples): 6.62%")
    print(f"  Augmented (2400 samples): {best_val_iou*100:.2f}%")
    if best_val_iou > 0.0662:
        print(f"  ✓ Improvement: +{(best_val_iou - 0.0662)*100:.2f}%")


if __name__ == "__main__":
    main()
