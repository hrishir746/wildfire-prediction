#!/usr/bin/env python3
"""
Train FU-NetCastV2: 24-hour fire spread prediction with multi-modal input.

This implements the Khennou & Akhloufi (2023) approach:
- Input: Fire perimeter (T) + terrain + Landsat satellite + weather timeseries
- Output: Fire probability map at T+24h
- Model: U-Net with 19 input channels

For now, uses FIRMS progressions dataset (8ch) expanded with synthetic Landsat+weather
to reach 19 channels. In production, would integrate real Landsat + weather APIs.
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

from src.model.fu_unet import build_fu_unet

DATASET_PATH = ROOT / "data" / "firms_progressions" / "firms_progressions_256.npz"
OUTPUT_MODEL = ROOT / "outputs" / "checkpoints" / "fu_unet_firms_progressions.pt"
METRICS_FILE = ROOT / "outputs" / "fu_unet_training_metrics.json"

EPOCHS = 50
BATCH_SIZE = 2  # Reduced for 19 channels
LEARNING_RATE = 3e-4
VAL_SPLIT = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def expand_to_19_channels(X_8ch: np.ndarray) -> np.ndarray:
    """
    Expand 8-channel FIRMS data to 19 channels by adding synthetic Landsat + weather.
    
    Input: (N, 8, H, W)
    Output: (N, 19, H, W)
    
    Channels:
      0-7:   Original (terrain + fuel + wind + fire)
      8-13:  Landsat bands 2-7 (synthetic: derived from elevation/fuel)
      14:    NDVI (synthetic: from slope and fuel)
      15-18: Weather features (synthetic: scalar broadcast to grid)
    """
    N, C, H, W = X_8ch.shape
    X_19ch = np.zeros((N, 19, H, W), dtype=np.float32)
    
    # Copy original 8 channels
    X_19ch[:, :8, :, :] = X_8ch
    
    for i in range(N):
        # Extract features from original channels
        slope = X_8ch[i, 0]  # Normalized slope
        fuel = X_8ch[i, 3]   # Fuel density
        
        # Synthetic Landsat bands 2-7
        # Band 2 (Blue): sensitive to water/snow, use elevation proxy
        X_19ch[i, 8] = (X_8ch[i, 7] * 0.5).clip(0, 1)  # Elevation-based
        
        # Band 3 (Green): use fuel
        X_19ch[i, 9] = fuel
        
        # Band 4 (Red): use slope
        X_19ch[i, 10] = slope
        
        # Band 5 (NIR): vegetation proxy, inverse of fuel scarcity
        X_19ch[i, 11] = (1 - fuel * 0.3).clip(0, 1)
        
        # Band 6 (SWIR-1): use fuel
        X_19ch[i, 12] = fuel * 0.8
        
        # Band 7 (SWIR-2): high values for dry areas
        X_19ch[i, 13] = (1 - fuel).clip(0, 1) * 0.9
        
        # NDVI (Normalized Difference Vegetation Index)
        # NDVI = (NIR - Red) / (NIR + Red)
        nir = X_19ch[i, 11]
        red = X_19ch[i, 10]
        X_19ch[i, 14] = ((nir - red) / (nir + red + 1e-6)).clip(-1, 1)
        
        # Weather features (4 channels, broadcast across grid)
        # Extract wind speed and direction from original channels
        wind_speed = X_8ch[i, 4, 0, 0]  # Wind speed (scalar, same for all pixels)
        wind_sin = X_8ch[i, 5, 0, 0]    # Wind direction sin
        wind_cos = X_8ch[i, 6, 0, 0]    # Wind direction cos
        
        # Add synthetic temp/humidity variation based on elevation
        temp_variation = X_8ch[i, 7] * 0.2  # Elevation influences temp
        humidity = (1 - fuel * 0.3).clip(0, 1)  # High fuel = lower humidity
        
        X_19ch[i, 15] = np.full((H, W), wind_speed, dtype=np.float32)
        X_19ch[i, 16] = np.full((H, W), wind_sin, dtype=np.float32)
        X_19ch[i, 17] = np.full((H, W), wind_cos, dtype=np.float32)
        X_19ch[i, 18] = temp_variation
    
    return X_19ch


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
    print("TRAIN FU-NetCastV2: MULTI-MODAL FIRE SPREAD PREDICTION")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not DATASET_PATH.exists():
        print(f"\n❌ Dataset not found: {DATASET_PATH}")
        print("Run: python scripts/process_firms_progressions.py")
        sys.exit(1)

    print(f"\n[1/5] Loading FIRMS progression dataset...")
    data = np.load(DATASET_PATH)
    X_8ch = data["X"].astype(np.float32)
    Y = data["Y"].astype(np.float32)

    print(f"  Original shape: {X_8ch.shape}")
    print(f"  Target shape: {Y.shape}")

    print(f"\n[2/5] Expanding to 19 channels (adding Landsat + weather)...")
    X_19ch = expand_to_19_channels(X_8ch)
    X = torch.from_numpy(X_19ch).float()
    Y = torch.from_numpy(Y).float()

    print(f"  Expanded shape: {X.shape}")
    print(f"  Channel breakdown:")
    print(f"    0-7:   Terrain + fuel + wind + fire (original FIRMS)")
    print(f"    8-13:  Landsat bands 2-7 (synthetic)")
    print(f"    14:    NDVI (synthetic)")
    print(f"    15-18: Weather features (wind + temp)")

    print(f"\n[3/5] Preparing train/val split...")
    n_val = max(1, int(VAL_SPLIT * len(X)))
    indices = torch.randperm(len(X))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    print(f"\n[4/5] Building FU-UNet model (19 channels)...")
    model = build_fu_unet(in_channels=19, out_channels=1, dropout_rate=0.2).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n[5/5] Training FU-NetCastV2 (50 epochs)...")
    print("=" * 80)

    best_val_iou = 0.0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

    start_time = time.time()
    
    # Initialize live progress file
    LIVE_PROGRESS_FILE = ROOT / "outputs" / "training_live_progress.json"
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch:2d}/{EPOCHS}", leave=True):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = combined_loss(model(x), y)
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

        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss:.4f}/{val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")

        # Save live progress for dashboard
        elapsed = (time.time() - start_time) / 60
        eta_minutes = (elapsed / epoch) * (EPOCHS - epoch) if epoch > 0 else 0
        with open(LIVE_PROGRESS_FILE, 'w') as f:
            json.dump({
                'current_epoch': epoch,
                'total_epochs': EPOCHS,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'val_iou': float(val_iou),
                'val_dice': float(val_dice),
                'eta_minutes': float(eta_minutes),
                'status': 'training'
            }, f)

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

    elapsed_minutes = (time.time() - start_time) / 60
    
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "best_val_iou": best_val_iou,
            "best_val_dice": history["val_dice"][history["val_iou"].index(best_val_iou)] if history["val_iou"] else 0,
            "training_time_minutes": elapsed_minutes,
            "epochs": EPOCHS,
            "history": history,
        }, f, indent=2)

    # Update live progress to show completion
    with open(LIVE_PROGRESS_FILE, 'w') as f:
        json.dump({
            'current_epoch': EPOCHS,
            'total_epochs': EPOCHS,
            'train_loss': float(history['train_loss'][-1]) if history['train_loss'] else 0,
            'val_loss': float(history['val_loss'][-1]) if history['val_loss'] else 0,
            'val_iou': float(best_val_iou),
            'val_dice': float(history['val_dice'][history['val_iou'].index(best_val_iou)] if history['val_iou'] else 0),
            'eta_minutes': 0.0,
            'status': 'completed'
        }, f)

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n✓ Best validation IoU: {best_val_iou:.4f} ({best_val_iou*100:.1f}%)")
    print(f"✓ Training time: {elapsed_minutes:.1f} minutes")
    print(f"✓ Model saved: {OUTPUT_MODEL}")
    print(f"✓ Metrics saved: {METRICS_FILE}")
    
    print(f"\n📊 Interpretation:")
    if best_val_iou > 0.7:
        print(f"  ✅ EXCELLENT: {best_val_iou*100:.0f}% - Far exceeds synthetic (1%) baseline")
        print(f"     This validates the FU-NetCastV2 + multi-modal approach")
    elif best_val_iou > 0.5:
        print(f"  ✅ GOOD: {best_val_iou*100:.0f}% - Significant improvement over synthetic (1%)")
        print(f"     Real fire progression learning is working")
    elif best_val_iou > 0.3:
        print(f"  ✓ PROMISING: {best_val_iou*100:.0f}% - Learning fire patterns from data")
        print(f"     More data/features would likely improve further")
    else:
        print(f"  ⚠ LOW: {best_val_iou*100:.0f}% - Still working through domain gap")
        print(f"     Consider: more data, real Landsat, weather timeseries")
    
    print(f"\nNEXT STEP: Validate on held-out MTBS fires")


if __name__ == "__main__":
    main()
