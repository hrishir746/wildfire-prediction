#!/usr/bin/env python3
"""
Strategy: Train ONLY on larger, learnable fires (>= 5 pixels)

Problem with original 6.62% IoU:
- 82% of training data has <5 fire pixels (pure noise)
- Model wastes capacity learning to ignore tiny detections
- Can't form meaningful fire patterns from 2-3 pixel targets

Solution:
- Filter to 55 samples with >= 5 fire pixels
- Train dedicated "large fire predictor" 
- These fires are actually learnable
- Expected: 15-25% IoU (5 pixels is at the edge of learnable)
"""
import sys
from pathlib import Path
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.unet import UNet

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = ROOT / "data" / "firms_progressions" / "firms_progressions_256.npz"
OUTPUT_MODEL = ROOT / "outputs" / "checkpoints" / "unet_large_fires.pt"
METRICS_FILE = ROOT / "outputs" / "large_fires_training_metrics.json"
LIVE_PROGRESS = ROOT / "outputs" / "training_large_fires_progress.json"

EPOCHS = 100  # More epochs since less data
BATCH_SIZE = 4
LEARNING_RATE = 1e-3  # Higher LR for smaller dataset
VAL_SPLIT = 0.2  # Need more validation with small dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_FIRE_PIXELS = 5  # Only train on fires with >= 5 pixels

# ============================================================================
# LOSS & METRICS
# ============================================================================

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(logits, target):
    bce = nn.BCEWithLogitsLoss()(logits, target)
    dice = dice_loss(logits, target)
    return 0.5 * bce + 0.5 * dice


def calculate_metrics(pred, target):
    """Calculate IoU and Dice coefficient."""
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-6)
    
    return {"iou": iou.item(), "dice": dice.item()}


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


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("=" * 80)
    print("LARGE FIRES ONLY: Training on >= 5 pixel fires")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_PATH}")
    print(f"Min fire pixels: {MIN_FIRE_PIXELS}")
    print("=" * 80)
    
    # Load and filter data
    print("Loading FIRMS progression data...")
    data = np.load(DATA_PATH)
    X = torch.FloatTensor(data['X'])  # (N, 8, 256, 256)
    Y = torch.FloatTensor(data['Y'])  # (N, 1, 256, 256)
    
    # Count fire pixels
    fire_pixels = (Y > 0).sum(dim=(2, 3)).squeeze()  # (N,)
    
    # Filter to larger fires
    mask = fire_pixels >= MIN_FIRE_PIXELS
    X_filtered = X[mask]
    Y_filtered = Y[mask]
    
    print(f"✓ Loaded {len(Y)} total samples")
    print(f"✓ Filtered to {len(Y_filtered)} samples with >= {MIN_FIRE_PIXELS} fire pixels")
    print(f"  Removed: {(~mask).sum()} small/noisy fires")
    print(f"  Fire pixel range: {fire_pixels[mask].min().int()}-{fire_pixels[mask].max().int()}")
    
    if len(Y_filtered) < 10:
        print(f"\n❌ ERROR: Only {len(Y_filtered)} samples - too small to train!")
        return
    
    N = X_filtered.shape[0]
    
    # Train/val split
    val_size = int(N * VAL_SPLIT)
    indices = torch.randperm(N)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    train_dataset = TensorDataset(X_filtered[train_idx], Y_filtered[train_idx])
    val_dataset = TensorDataset(X_filtered[val_idx], Y_filtered[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Model setup
    model = UNet(in_channels=8, out_channels=1, base_channels=64).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print(f"✓ Model: U-Net (8 → 1 channels)")
    print(f"✓ Optimizer: AdamW (LR={LEARNING_RATE})")
    
    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    best_iou = 0.0
    
    print("\n" + "=" * 80)
    print("TRAINING START")
    print("=" * 80)
    
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = combined_loss(model(x), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for x, y_batch in val_loader:
                x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
                logits = model(x)
                val_loss += combined_loss(logits, y_batch).item()
                metrics = calculate_metrics(torch.sigmoid(logits), y_batch)
                val_iou += metrics["iou"]
                val_dice += metrics["dice"]
        
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
            val_iou /= len(val_loader)
            val_dice /= len(val_loader)
        
        scheduler.step()
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': val_iou,
                'dice': val_dice,
                'loss': val_loss
            }, OUTPUT_MODEL)
        
        # Log progress
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)
        
        elapsed = (time.time() - start_time) / 60
        eta = (elapsed / epoch) * (EPOCHS - epoch)
        _write_progress(epoch, EPOCHS, train_loss, val_loss, val_iou, val_dice, eta, "training")
        
        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"IoU: {val_iou*100:.2f}% | Dice: {val_dice*100:.2f}% | Best: {best_iou*100:.2f}%")
    
    # Final summary
    elapsed = (time.time() - start_time) / 60
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best IoU: {best_iou*100:.2f}%")
    print(f"Time: {elapsed:.1f} minutes")
    print(f"Model saved: {OUTPUT_MODEL}")
    
    # Save metrics
    metrics_data = {
        "model": "UNet - Large Fires Only",
        "dataset": f"FIRMS Progressions (filtered to >={MIN_FIRE_PIXELS} fire pixels)",
        "total_samples": int(N),
        "epochs": EPOCHS,
        "best_val_iou": float(best_iou),
        "best_val_dice": float(history["val_dice"][history["val_iou"].index(best_iou)]),
        "training_time_minutes": float(elapsed),
        "history": history
    }
    
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    _write_progress(EPOCHS, EPOCHS, train_loss, val_loss, val_iou, val_dice, 0, "completed")
    
    print(f"✓ Metrics saved: {METRICS_FILE}")
    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print(f"Training on {N} learnable fires (>={MIN_FIRE_PIXELS} pixels) instead of 300.")
    print("Small fires (2-3 pixels) are unlearnable noise.")
    print(f"This model is specialized for larger fire progressions.")
    print("Expected improvement: 15-25% IoU (vs 6.62% on mixed data)")
    print("=" * 80)


if __name__ == "__main__":
    main()
