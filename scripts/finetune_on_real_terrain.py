#!/usr/bin/env python3
"""
Fine-tune the synthetic-trained model on real California terrain data.
This bridges the domain gap from synthetic → real terrain.
"""
import sys
import os
from pathlib import Path

# UTF-8 encoding fix
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.model import build_unet

print("=" * 80)
print("FINE-TUNING ON REAL CALIFORNIA TERRAIN")
print("=" * 80)

# Configuration
DATASET_PATH = ROOT / "data" / "real_terrain_synthetic_fires" / "real_terrain_dataset.npz"
PRETRAINED_MODEL = ROOT / "outputs" / "checkpoints" / "unet_best.pt"
OUTPUT_MODEL = ROOT / "outputs" / "checkpoints" / "unet_finetuned_real_terrain.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4  # Lower LR for fine-tuning

print(f"\nConfiguration:")
print(f"  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Dataset: {DATASET_PATH.name}")
print(f"  Pretrained: {PRETRAINED_MODEL.name}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")

# Load dataset
print(f"\n[1/4] Loading real terrain dataset...")
data = np.load(DATASET_PATH)
X = torch.from_numpy(data['X'])  # (N, 8, H, W)
Y = torch.from_numpy(data['Y'])  # (N, 1, H, W)
print(f"  Loaded: {X.shape[0]} samples")
print(f"  Input shape: {X.shape}")
print(f"  Target shape: {Y.shape}")

# Train/val split (80/20)
n_train = int(0.8 * len(X))
n_val = len(X) - n_train

indices = torch.randperm(len(X))
train_indices = indices[:n_train]
val_indices = indices[n_train:]

X_train, Y_train = X[train_indices], Y[train_indices]
X_val, Y_val = X[val_indices], Y[val_indices]

print(f"  Train: {len(X_train)} samples")
print(f"  Val: {len(X_val)} samples")

# Create dataloaders
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pretrained model
print(f"\n[2/4] Loading pretrained model...")
model = build_unet(in_channels=8, out_channels=1, dropout_rate=0.2)

if PRETRAINED_MODEL.exists():
    checkpoint = torch.load(PRETRAINED_MODEL, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Loaded pretrained weights from synthetic training")
    print(f"    Original IoU: {checkpoint.get('iou', 0):.4f}")
    print(f"    Original Dice: {checkpoint.get('dice', 0):.4f}")
else:
    print(f"  ⚠ No pretrained model found, training from scratch")

model = model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

# Setup optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(logits, target):
    bce = nn.BCEWithLogitsLoss()(logits, target)
    dice = dice_loss(logits, target)
    return 0.5 * bce + 0.5 * dice

def calculate_metrics(pred, target):
    pred_binary = (pred > 0.5).float()
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()
    
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    
    return {'iou': iou.item(), 'dice': dice.item()}

# Training loop
print(f"\n[3/4] Fine-tuning on real terrain...")
print("=" * 80)

best_val_loss = float('inf')
best_val_iou = 0.0

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = combined_loss(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validate
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            logits = model(x)
            loss = combined_loss(logits, y)
            pred = torch.sigmoid(logits)
            
            metrics = calculate_metrics(pred, y)
            
            val_loss += loss.item()
            val_iou += metrics['iou']
            val_dice += metrics['dice']
    
    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_dice /= len(val_loader)
    
    # Step scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print progress
    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val IoU: {val_iou:.4f} | "
          f"Val Dice: {val_dice:.4f} | "
          f"LR: {current_lr:.2e}")
    
    # Save best model
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        best_val_loss = val_loss
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'iou': val_iou,
            'dice': val_dice,
        }, OUTPUT_MODEL)

print("\n" + "=" * 80)
print("FINE-TUNING COMPLETE")
print("=" * 80)
print(f"\nBest Validation IoU: {best_val_iou:.4f}")
print(f"Best Validation Dice: {val_dice:.4f}")
print(f"\n✓ Fine-tuned model saved: {OUTPUT_MODEL}")

print(f"\n[4/4] Next steps:")
print("  1. Validate on real Camp Fire 2018 data")
print("  2. Run: python scripts/validate_camp_2018.py")
print("=" * 80)
