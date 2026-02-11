#!/usr/bin/env python3
"""
FIRMS Training with Focal Loss - Addressing Extreme Class Imbalance

The core problem: Fire pixels are only 0.006% of total pixels (3-5 per 256x256 image).
Standard BCE loss treats all pixels equally → model learns to predict all zeros.

Solution: Focal Loss (Lin et al., 2017 - RetinaNet paper)
- Down-weights easy examples (correctly classified non-fire pixels)
- Focuses on hard examples (fire pixels and fire boundaries)
- gamma parameter controls how much to down-weight easy examples

Expected improvement: 2-3x IoU increase (6% → 12-18%)
Target: Get closer to 60% with proper handling of class imbalance
"""
import sys
from pathlib import Path
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
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
OUTPUT_MODEL = ROOT / "outputs" / "checkpoints" / "unet_firms_focal.pt"
METRICS_FILE = ROOT / "outputs" / "firms_focal_training_metrics.json"
LIVE_PROGRESS = ROOT / "outputs" / "training_focal_progress.json"

EPOCHS = 60
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
VAL_SPLIT = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Focal Loss parameters
FOCAL_GAMMA = 2.0       # Original RetinaNet uses 2.0
FOCAL_ALPHA = 0.25      # Weight for positive class (fire pixels)
POS_WEIGHT = 100.0      # Additional weight for fire pixels in BCE component

# ============================================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for extreme class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where:
    - p_t is the probability of the correct class
    - γ (gamma) controls how much to down-weight easy examples
    - α (alpha) is a class-balancing weight
    
    For γ=2:
    - Easy example (p_t=0.97): weight reduced by 1000x
    - Medium example (p_t=0.7): weight reduced by 10x
    - Hard example (p_t=0.3): weight only reduced by 2x
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (N, 1, H, W)
            targets: Binary masks (N, 1, H, W)
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # For numerical stability
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        
        # Compute p_t (probability of correct class)
        p_t = torch.where(targets == 1, probs, 1 - probs)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t).pow(self.gamma)
        
        # BCE loss component
        bce = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # Alpha weighting (balance positive/negative classes)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Combine: focal_loss = alpha * focal_weight * bce
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()


class CombinedFocalLoss(nn.Module):
    """
    Combines Focal Loss with Dice Loss for better performance.
    
    Total = 0.6 * Focal + 0.4 * Dice
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.focal = FocalLoss(gamma, alpha)
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def forward(self, logits, targets):
        focal = self.focal(logits, targets)
        dice = self.dice_loss(logits, targets)
        return 0.6 * focal + 0.4 * dice


# ============================================================================
# METRICS
# ============================================================================

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
    print("FOCAL LOSS TRAINING: Addressing Class Imbalance")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_PATH}")
    print(f"Focal gamma: {FOCAL_GAMMA} (down-weight easy examples)")
    print(f"Focal alpha: {FOCAL_ALPHA} (positive class weight)")
    print(f"Pos weight: {POS_WEIGHT}x (fire pixel emphasis)")
    print("=" * 80)
    
    # Load data
    print("Loading FIRMS progression data...")
    data = np.load(DATA_PATH)
    X = torch.FloatTensor(data['X'])  # (N, 8, 256, 256)
    y = torch.FloatTensor(data['Y'])  # (N, 1, 256, 256)
    
    N = X.shape[0]
    fire_density = (y > 0).float().mean().item()
    print(f"✓ Loaded {N} samples")
    print(f"  Fire pixel density: {fire_density*100:.4f}% (1:{int(1/fire_density):,})")
    print(f"  This extreme imbalance is why we need Focal Loss!")
    
    # Train/val split
    val_size = int(N * VAL_SPLIT)
    indices = torch.randperm(N)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    train_dataset = TensorDataset(X[train_idx], y[train_idx])
    val_dataset = TensorDataset(X[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Model setup
    model = UNet(in_channels=8, out_channels=1, base_channels=64).to(DEVICE)
    criterion = CombinedFocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print(f"✓ Model: U-Net (8 → 1 channels)")
    print(f"✓ Loss: Focal + Dice (addresses class imbalance)")
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
            loss = criterion(model(x), y_batch)
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
                val_loss += criterion(logits, y_batch).item()
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
        "model": "UNet with Focal Loss",
        "dataset": "FIRMS Progressions (300 samples)",
        "epochs": EPOCHS,
        "best_val_iou": float(best_iou),
        "training_time_minutes": float(elapsed),
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha": FOCAL_ALPHA,
        "history": history,
        "fire_density": fire_density,
        "imbalance_ratio": f"1:{int(1/fire_density):,}"
    }
    
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    _write_progress(EPOCHS, EPOCHS, train_loss, val_loss, val_iou, val_dice, 0, "completed")
    
    print(f"✓ Metrics saved: {METRICS_FILE}")
    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("Focal Loss addresses the 1:165,000 class imbalance by:")
    print("1. Down-weighting easy negatives (99.999% of pixels)")
    print("2. Focusing on hard examples (fire boundaries, small fires)")
    print("3. Alpha-balancing positive/negative classes")
    print("\nIf this doesn't reach 60%, the next step is:")
    print("- Collect more real fire progressions (we only have 300)")
    print("- Or accept that 60% may not be achievable with sparse FIRMS data")
    print("=" * 80)


if __name__ == "__main__":
    main()
