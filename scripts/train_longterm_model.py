#!/usr/bin/env python3
"""
Train on long-term fire dataset (150 timesteps) to match MTBS validation timeframe.
This should give us 30-60% IoU on Camp Fire instead of 0.6%.
"""
import sys
import os
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import json

from src.model import build_unet

print("=" * 80)
print("LONG-TERM FIRE PREDICTION TRAINING")
print("Goal: Get proper MTBS baseline (target: 30-60% IoU)")
print("=" * 80)

DATASET_PATH = ROOT / "data" / "longterm_fires" / "longterm_dataset.npz"
PRETRAINED = ROOT / "outputs" / "checkpoints" / "unet_production_v1.pt"
OUTPUT_MODEL = ROOT / "outputs" / "checkpoints" / "unet_longterm_v1.pt"
METRICS_FILE = ROOT / "outputs" / "longterm_training_metrics.json"
LIVE_PROGRESS_FILE = ROOT / "outputs" / "training_live_progress.json"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 40
BATCH_SIZE = 2
LEARNING_RATE = 3e-5

print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load dataset
print(f"\n[1/4] Loading long-term dataset...")
data = np.load(DATASET_PATH)
X = torch.from_numpy(data['X'])
Y = torch.from_numpy(data['Y'])

print(f"  Loaded: {X.shape[0]} samples (150 timesteps each)")
print(f"  Input: {X.shape}, Target: {Y.shape}")

# Split
n_val = int(0.15 * len(X))
n_train = len(X) - n_val
indices = torch.randperm(len(X))

X_train, Y_train = X[indices[:n_train]], Y[indices[:n_train]]
X_val, Y_val = X[indices[n_train:]], Y[indices[n_train:]]

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train: {n_train}, Val: {n_val}")

# Load model
print(f"\n[2/4] Loading model...")
model = build_unet(in_channels=8, out_channels=1, dropout_rate=0.2)

if PRETRAINED.exists():
    ckpt = torch.load(PRETRAINED, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  ✓ Loaded pretrained: {PRETRAINED.name}")
else:
    print(f"  Training from scratch")

model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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
    return {'iou': iou.item(), 'dice': dice.item()}

# Training
print(f"\n[3/4] Training for {EPOCHS} epochs...")
print("=" * 80)

best_val_iou = 0
history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}
start_time = time.time()

# Initialize live progress
with open(LIVE_PROGRESS_FILE, 'w') as f:
    json.dump({'current_epoch': 0, 'total_epochs': EPOCHS, 'status': 'starting'}, f)

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = combined_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validate
    model.eval()
    val_loss = val_iou = val_dice = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            val_loss += combined_loss(logits, y).item()
            metrics = calculate_metrics(torch.sigmoid(logits), y)
            val_iou += metrics['iou']
            val_dice += metrics['dice']
    
    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_dice /= len(val_loader)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_iou'].append(val_iou)
    history['val_dice'].append(val_dice)
    
    scheduler.step()
    
    # Update live progress for dashboard
    elapsed = (time.time() - start_time) / 60
    eta_minutes = (elapsed / epoch) * (EPOCHS - epoch)
    with open(LIVE_PROGRESS_FILE, 'w') as f:
        json.dump({
            'current_epoch': epoch,
            'total_epochs': EPOCHS,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_dice,
            'eta_minutes': eta_minutes,
            'status': 'training'
        }, f)
    
    print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {train_loss:.4f}/{val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
    
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'iou': val_iou,
            'dice': val_dice,
        }, OUTPUT_MODEL)
        print(f"  ✓ New best: {val_iou:.4f}")

# Save metrics
with open(METRICS_FILE, 'w') as f:
    json.dump({
        'best_val_iou': best_val_iou,
        'training_time_minutes': (time.time() - start_time) / 60,
        'history': history
    }, f, indent=2)

print(f"\n{'='*80}")
print(f"TRAINING COMPLETE - Best IoU: {best_val_iou:.4f}")
print(f"{'='*80}")
print(f"\n[4/4] Next: Validate on MTBS Camp Fire")
print(f"  Expected: 30-60% IoU (vs current 0.6%)")
print(f"  Model saved: {OUTPUT_MODEL}")
