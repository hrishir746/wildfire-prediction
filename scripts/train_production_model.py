#!/usr/bin/env python3
"""
PRODUCTION MODEL TRAINING
Train on 500 real CA terrain samples at 512x512 for maximum accuracy.
"""
import sys
import os
from pathlib import Path

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
import time
from datetime import datetime

from src.model import build_unet

print("=" * 80)
print("PRODUCTION MODEL TRAINING")
print("=" * 80)

# Configuration
DATASET_PATH = ROOT / "data" / "production_real_terrain" / "production_dataset.npz"
PRETRAINED_MODEL = ROOT / "outputs" / "checkpoints" / "unet_finetuned_real_terrain.pt"
OUTPUT_MODEL = ROOT / "outputs" / "checkpoints" / "unet_production_v1.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training config
EPOCHS = 50
BATCH_SIZE = 2  # 512x512 is memory-intensive, use small batch
LEARNING_RATE = 5e-5  # Lower LR for fine-tuning large model
VAL_SPLIT = 0.15  # 15% validation

print(f"\nConfiguration:")
print(f"  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  Dataset: {DATASET_PATH.name}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Val split: {VAL_SPLIT:.0%}")

# Load dataset
print(f"\n[1/5] Loading production dataset...")
if not DATASET_PATH.exists():
    print(f"  ERROR: Dataset not found at {DATASET_PATH}")
    print(f"  Run: python scripts/generate_production_dataset.py")
    sys.exit(1)

data = np.load(DATASET_PATH)
X = torch.from_numpy(data['X'])
Y = torch.from_numpy(data['Y'])

print(f"  Loaded: {X.shape[0]} samples")
print(f"  Input: {X.shape}")
print(f"  Target: {Y.shape}")
print(f"  Memory: {(X.element_size() * X.nelement() + Y.element_size() * Y.nelement()) / 1e9:.2f} GB")

# Train/val split
n_val = int(VAL_SPLIT * len(X))
n_train = len(X) - n_val

indices = torch.randperm(len(X))
train_indices = indices[:n_train]
val_indices = indices[n_train:]

X_train, Y_train = X[train_indices], Y[train_indices]
X_val, Y_val = X[val_indices], Y[val_indices]

print(f"  Train: {len(X_train)} samples")
print(f"  Val: {len(X_val)} samples")

# Dataloaders
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

# Load model
print(f"\n[2/5] Initializing model...")
model = build_unet(in_channels=8, out_channels=1, dropout_rate=0.2)

# Try to load pretrained weights
if PRETRAINED_MODEL.exists():
    checkpoint = torch.load(PRETRAINED_MODEL, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Loaded pretrained weights")
    print(f"    Previous IoU: {checkpoint.get('iou', 0):.4f}")
else:
    print(f"  ⚠ Training from scratch (no pretrained model)")

model = model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

# Optimizer & loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

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

# Training
print(f"\n[3/5] Training production model...")
print("=" * 80)

start_time = time.time()
best_val_iou = 0.0
history = {
    'train_loss': [],
    'val_loss': [],
    'val_iou': [],
    'val_dice': []
}

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    
    # Train
    model.train()
    train_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = combined_loss(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_loss /= len(train_loader)
    history['train_loss'].append(train_loss)
    
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
    
    history['val_loss'].append(val_loss)
    history['val_iou'].append(val_iou)
    history['val_dice'].append(val_dice)
    
    # Step scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Time estimates
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    eta = (elapsed / epoch) * (EPOCHS - epoch)
    
    # Print
    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
          f"IoU: {val_iou:.4f} | "
          f"Dice: {val_dice:.4f} | "
          f"LR: {current_lr:.2e} | "
          f"Time: {epoch_time:.1f}s | "
          f"ETA: {eta/60:.0f}m")
    
    # Save best
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'iou': val_iou,
            'dice': val_dice,
            'history': history
        }, OUTPUT_MODEL)
        print(f"  ✓ New best IoU: {val_iou:.4f}")
    
    # Periodic saves
    if epoch % 10 == 0:
        periodic_path = OUTPUT_MODEL.parent / f"unet_production_epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'iou': val_iou,
            'dice': val_dice,
        }, periodic_path)

# Final save
final_path = OUTPUT_MODEL.parent / "unet_production_final.pt"
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': val_loss,
    'iou': val_iou,
    'dice': val_dice,
    'history': history
}, final_path)

total_time = time.time() - start_time

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nBest Validation:")
print(f"  IoU: {best_val_iou:.4f}")
print(f"  Dice: {max(history['val_dice']):.4f}")
print(f"\nTotal training time: {total_time/3600:.2f} hours")
print(f"\n✓ Production model saved: {OUTPUT_MODEL}")
print(f"✓ Final model saved: {final_path}")

print(f"\n[4/5] Next steps:")
print("  1. Test on Camp Fire: python scripts/test_with_ignition.py")
print("  2. Create visualizations")
print("  3. Document results")

print("\n" + "=" * 80)
