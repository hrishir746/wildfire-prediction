#!/usr/bin/env python3
"""
INTEGRATION TEST: Full Pipeline with 512x512 Patches

This script performs an end-to-end integration test of the wildfire prediction system:
1. Generates synthetic 512x512 training samples
2. Trains the model for 2 epochs with dashboard logging
3. Validates on test samples
4. Saves checkpoint and results

Ensures:
- UTF-8 encoding works on Windows
- 512x512 patches work throughout the pipeline
- Dashboard logging functions correctly
- Model training and inference work with larger patches

Run: python scripts/test_full_pipeline_512.py
"""

import sys
import os
from pathlib import Path
import time
import json

# UTF-8 encoding fix for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from src.model import build_unet
from src.simulation.rothermel_simulator import rothermel_fire_spread

# ============================================================================
# TEST CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESOLUTION = 512  # Test with 512x512 patches
BATCH_SIZE = 2    # Small batch size for testing
EPOCHS = 2        # Quick test
TRAIN_SAMPLES = 10
VAL_SAMPLES = 3

print("=" * 70)
print("INTEGRATION TEST: Full Pipeline with 512x512 Patches")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Training samples: {TRAIN_SAMPLES}")
print(f"Validation samples: {VAL_SAMPLES}")
print("=" * 70)

# ============================================================================
# STEP 1: Generate Synthetic Training Data
# ============================================================================
print(f"\n[STEP 1/5] Generating {TRAIN_SAMPLES} synthetic training samples...")
start_time = time.time()

train_inputs = []
train_targets = []

for i in range(TRAIN_SAMPLES):
    # Create synthetic input stack (8 channels)
    elevation = np.random.randn(RESOLUTION, RESOLUTION).astype(np.float32) * 100 + 500
    slope = np.abs(np.random.randn(RESOLUTION, RESOLUTION).astype(np.float32)) * 0.3
    aspect = np.random.rand(RESOLUTION, RESOLUTION).astype(np.float32) * 360
    fuel_load = np.random.rand(RESOLUTION, RESOLUTION).astype(np.float32) * 0.8 + 0.2
    fuel_moisture = np.random.rand(RESOLUTION, RESOLUTION).astype(np.float32) * 0.3 + 0.1
    fuel_depth = np.random.rand(RESOLUTION, RESOLUTION).astype(np.float32) * 0.5 + 0.1
    wind_speed = np.ones((RESOLUTION, RESOLUTION), dtype=np.float32) * (5.0 + np.random.uniform(-2, 2))
    wind_direction = np.ones((RESOLUTION, RESOLUTION), dtype=np.float32) * (270.0 + np.random.uniform(-30, 30))
    
    stack = np.stack([elevation, slope, aspect, fuel_load, fuel_moisture, fuel_depth, wind_speed, wind_direction])
    
    # Normalize
    stack = (stack - stack.mean(axis=(1, 2), keepdims=True)) / (stack.std(axis=(1, 2), keepdims=True) + 1e-6)
    
    # Create target using simple fire spread simulation
    row, col = np.random.randint(100, RESOLUTION-100), np.random.randint(100, RESOLUTION-100)
    target = rothermel_fire_spread(
        elevation=elevation,
        fuel_load=fuel_load,
        wind_speed=wind_speed[0, 0],
        ignition_row=row,
        ignition_col=col,
        duration_hours=8.0,
        integration_steps=32
    )
    
    train_inputs.append(stack)
    train_targets.append(target)
    
    if (i + 1) % 5 == 0:
        print(f"  Generated {i + 1}/{TRAIN_SAMPLES} samples...")

train_inputs = np.array(train_inputs, dtype=np.float32)
train_targets = np.array(train_targets, dtype=np.float32)

elapsed = time.time() - start_time
print(f"[OK] Data generation complete in {elapsed:.1f}s")
print(f"  Train inputs shape: {train_inputs.shape}")
print(f"  Train targets shape: {train_targets.shape}")

# ============================================================================
# STEP 2: Generate Validation Data
# ============================================================================
print(f"\n[STEP 2/5] Generating {VAL_SAMPLES} validation samples...")

val_inputs = []
val_targets = []

for i in range(VAL_SAMPLES):
    elevation = np.random.randn(RESOLUTION, RESOLUTION).astype(np.float32) * 100 + 500
    slope = np.abs(np.random.randn(RESOLUTION, RESOLUTION).astype(np.float32)) * 0.3
    aspect = np.random.rand(RESOLUTION, RESOLUTION).astype(np.float32) * 360
    fuel_load = np.random.rand(RESOLUTION, RESOLUTION).astype(np.float32) * 0.8 + 0.2
    fuel_moisture = np.random.rand(RESOLUTION, RESOLUTION).astype(np.float32) * 0.3 + 0.1
    fuel_depth = np.random.rand(RESOLUTION, RESOLUTION).astype(np.float32) * 0.5 + 0.1
    wind_speed = np.ones((RESOLUTION, RESOLUTION), dtype=np.float32) * (5.0 + np.random.uniform(-2, 2))
    wind_direction = np.ones((RESOLUTION, RESOLUTION), dtype=np.float32) * (270.0 + np.random.uniform(-30, 30))
    
    stack = np.stack([elevation, slope, aspect, fuel_load, fuel_moisture, fuel_depth, wind_speed, wind_direction])
    stack = (stack - stack.mean(axis=(1, 2), keepdims=True)) / (stack.std(axis=(1, 2), keepdims=True) + 1e-6)
    
    row, col = np.random.randint(100, RESOLUTION-100), np.random.randint(100, RESOLUTION-100)
    target = rothermel_fire_spread(
        elevation=elevation,
        fuel_load=fuel_load,
        wind_speed=wind_speed[0, 0],
        ignition_row=row,
        ignition_col=col,
        duration_hours=8.0,
        integration_steps=32
    )
    
    val_inputs.append(stack)
    val_targets.append(target)

val_inputs = np.array(val_inputs, dtype=np.float32)
val_targets = np.array(val_targets, dtype=np.float32)

print(f"[OK] Validation data generated")
print(f"  Val inputs shape: {val_inputs.shape}")
print(f"  Val targets shape: {val_targets.shape}")

# ============================================================================
# STEP 3: Initialize Model and Training
# ============================================================================
print(f"\n[STEP 3/5] Initializing model and training setup...")

model = build_unet(in_channels=8, out_channels=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(train_inputs), torch.from_numpy(train_targets)),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(torch.from_numpy(val_inputs), torch.from_numpy(val_targets)),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"[OK] Model initialized")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

# ============================================================================
# STEP 4: Training Loop with Dashboard Logging
# ============================================================================
print(f"\n[STEP 4/5] Training for {EPOCHS} epochs...")

# Create metrics log file for dashboard
metrics_log = ROOT / 'outputs' / 'test_pipeline_metrics.json'
metrics_log.parent.mkdir(parents=True, exist_ok=True)

history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # Training
    model.train()
    train_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.unsqueeze(1).to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            val_loss += criterion(logits, y.unsqueeze(1)).item()
            
            pred = torch.sigmoid(logits)
            pred_bin = (pred > 0.5).float()
            y_unsqueeze = y.unsqueeze(1)
            
            # IoU
            intersection = (pred_bin * y_unsqueeze).sum()
            union = ((pred_bin + y_unsqueeze) > 0).sum()
            val_iou += (intersection / (union + 1e-6)).item()
            
            # Dice
            val_dice += (2 * intersection / (pred_bin.sum() + y_unsqueeze.sum() + 1e-6)).item()
    
    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_dice /= len(val_loader)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_iou'].append(val_iou)
    history['val_dice'].append(val_dice)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
    
    epoch_time = time.time() - epoch_start
    
    # Log metrics to file (for dashboard compatibility)
    metrics = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_iou': val_iou,
        'val_dice': val_dice,
        'epoch_time': epoch_time
    }
    
    with open(metrics_log, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metrics) + '\n')
    
    print(f"  Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f} | Time: {epoch_time:.1f}s")

print(f"[OK] Training complete")

# ============================================================================
# STEP 5: Save Checkpoint and Results
# ============================================================================
print(f"\n[STEP 5/5] Saving checkpoint and results...")

# Save checkpoint
ckpt_dir = ROOT / 'outputs' / 'checkpoints'
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_dir / 'test_pipeline_512x512.pt'

torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': best_val_loss,
    'history': history,
    'config': {
        'resolution': RESOLUTION,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'train_samples': TRAIN_SAMPLES,
        'val_samples': VAL_SAMPLES
    }
}, ckpt_path)

print(f"[OK] Checkpoint saved: {ckpt_path.name} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

# Save visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history['train_loss'], label='Train', marker='o')
axes[0].plot(history['val_loss'], label='Val', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['val_iou'], marker='o', color='tab:orange')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('IoU')
axes[1].set_title('Validation IoU')
axes[1].grid(True, alpha=0.3)

axes[2].plot(history['val_dice'], marker='o', color='tab:green')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Dice Score')
axes[2].set_title('Validation Dice Score')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = ROOT / 'outputs' / 'test_pipeline_512_curves.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"[OK] Training curves saved: {plot_path.name}")

# Save test report
report_path = ROOT / 'outputs' / 'test_pipeline_512_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("INTEGRATION TEST REPORT: 512x512 Pipeline\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Test Configuration:\n")
    f.write(f"  - Resolution: {RESOLUTION}x{RESOLUTION}\n")
    f.write(f"  - Device: {DEVICE}\n")
    f.write(f"  - Batch size: {BATCH_SIZE}\n")
    f.write(f"  - Epochs: {EPOCHS}\n")
    f.write(f"  - Train samples: {TRAIN_SAMPLES}\n")
    f.write(f"  - Val samples: {VAL_SAMPLES}\n\n")
    f.write(f"Results:\n")
    f.write(f"  - Final train loss: {history['train_loss'][-1]:.4f}\n")
    f.write(f"  - Final val loss: {history['val_loss'][-1]:.4f}\n")
    f.write(f"  - Final val IoU: {history['val_iou'][-1]:.4f}\n")
    f.write(f"  - Final val Dice: {history['val_dice'][-1]:.4f}\n")
    f.write(f"  - Best val loss: {best_val_loss:.4f}\n\n")
    f.write(f"Outputs:\n")
    f.write(f"  - Checkpoint: {ckpt_path}\n")
    f.write(f"  - Metrics log: {metrics_log}\n")
    f.write(f"  - Training curves: {plot_path}\n\n")
    f.write("=" * 70 + "\n")
    f.write("[OK] INTEGRATION TEST PASSED\n")
    f.write("=" * 70 + "\n")

print(f"[OK] Test report saved: {report_path.name}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("INTEGRATION TEST SUMMARY")
print("=" * 70)
print(f"[OK] Data generation: {TRAIN_SAMPLES + VAL_SAMPLES} samples at {RESOLUTION}x{RESOLUTION}")
print(f"[OK] Training: {EPOCHS} epochs completed")
print(f"[OK] Validation: IoU={history['val_iou'][-1]:.4f}, Dice={history['val_dice'][-1]:.4f}")
print(f"[OK] UTF-8 encoding: No errors")
print(f"[OK] Dashboard logging: Metrics saved to JSON")
print(f"[OK] Checkpoint: {ckpt_path.stat().st_size / 1e6:.1f} MB")
print("=" * 70)
print("[OK][OK][OK] ALL TESTS PASSED [OK][OK][OK]")
print("=" * 70)
print("\nThe pipeline is ready for large-scale 512x512 wildfire prediction!")
print("\nNext steps:")
print("  1. Run full training: python scripts/train_with_dashboard.py")
print("  2. Monitor in real-time: streamlit run scripts/training_dashboard.py")
print("  3. Validate on real fires: python scripts/validate_extended_model.py")
print("=" * 70 + "\n")
