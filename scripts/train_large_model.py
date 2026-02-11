#!/usr/bin/env python3
"""
Larger U-Net for better feature extraction on large fires.

The standard U-Net (64→128→256→512) might be constrained.
Let's try a bigger model (128→256→512→1024) to better capture fire patterns.
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
OUTPUT_MODEL = ROOT / "outputs" / "checkpoints" / "unet_large_model.pt"
METRICS_FILE = ROOT / "outputs" / "large_model_training_metrics.json"
LIVE_PROGRESS = ROOT / "outputs" / "training_large_model_progress.json"

EPOCHS = 80
BATCH_SIZE = 2  # Larger model needs smaller batch
LEARNING_RATE = 5e-4
VAL_SPLIT = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# CUSTOM LARGER U-NET
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_rate: float = 0.0):
        super().__init__()
        num_groups = min(8, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LargeUNet(nn.Module):
    """Larger U-Net with more channels for better feature extraction."""

    def __init__(self, in_channels: int = 8, out_channels: int = 1, dropout_rate: float = 0.2):
        super().__init__()
        
        # Encoder (128→256→512→1024→2048)
        self.enc1 = DoubleConv(in_channels, 128, dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(128, 256, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(256, 512, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(512, 1024, dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(1024, 2048, dropout_rate=dropout_rate)
        
        # Decoder (2048→1024→512→256→128)
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec4 = DoubleConv(2048, 1024, dropout_rate=dropout_rate)
        
        self.upconv3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec3 = DoubleConv(1024, 512, dropout_rate=dropout_rate)
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConv(512, 256, dropout_rate=dropout_rate)
        
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = DoubleConv(256, 128, dropout_rate=dropout_rate)
        
        # Final output
        self.final = nn.Conv2d(128, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        u4 = self.upconv4(b)
        u4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(u4)
        
        u3 = self.upconv3(d4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)
        
        u2 = self.upconv2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        
        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        return self.final(d1)


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("=" * 80)
    print("LARGER U-NET: Better feature extraction for fire patterns")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Model: LargeUNet (128→256→512→1024→2048)")
    print(f"Data: {DATA_PATH}")
    print("=" * 80)
    
    # Load data
    print("Loading FIRMS progression data...")
    data = np.load(DATA_PATH)
    X = torch.FloatTensor(data['X'])  # (N, 8, 256, 256)
    Y = torch.FloatTensor(data['Y'])  # (N, 1, 256, 256)
    
    N = X.shape[0]
    fire_density = (Y > 0).float().mean().item()
    print(f"✓ Loaded {N} samples")
    print(f"  Fire pixel density: {fire_density*100:.4f}%")
    
    # Train/val split
    val_size = int(N * VAL_SPLIT)
    indices = torch.randperm(N)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    train_dataset = TensorDataset(X[train_idx], Y[train_idx])
    val_dataset = TensorDataset(X[val_idx], Y[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Model setup
    model = LargeUNet(in_channels=8, out_channels=1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: LargeUNet (2.8M parameters)")
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
        "model": "LargeUNet (128→256→512→1024→2048)",
        "dataset": "FIRMS Progressions (300 samples, all fires)",
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
    print("Larger model for better feature extraction.")
    print("=" * 80)


if __name__ == "__main__":
    main()
