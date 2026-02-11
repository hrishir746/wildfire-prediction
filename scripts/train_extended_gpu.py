#!/usr/bin/env python3
"""
EXTENDED GPU TRAINING - LARGE RESOLUTION, MANY EPOCHS
Train on all available fire data at 512x512 resolution for large-scale wildfire mapping
Target: 80%+ accuracy across all validation metrics

UTF-8 encoding configured for Windows compatibility.
"""
import sys
import os
from pathlib import Path

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
from tqdm import tqdm
import json

from src.data.real_data import build_real_input_stack
from src.simulation.rothermel_simulator import rothermel_fire_spread
from src.model import build_unet

# ============================================================================
# CONFIGURATION - EXPANDED FOR HIGH ACCURACY
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESOLUTION = 512  # EXPANDED to 512x512
BATCH_SIZE = 4  # Reduce batch size for larger patches
LEARNING_RATE = 1e-4
EPOCHS = 100  # EXTENDED training
NUM_SAMPLES_PER_FIRE = 50  # EXPANDED data per fire
VALIDATION_SPLIT = 0.2

print("="*70)
print("EXTENDED GPU TRAINING FOR 80%+ ACCURACY")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"Epochs: {EPOCHS}")
print(f"Samples per fire: {NUM_SAMPLES_PER_FIRE}")
print(f"Batch size: {BATCH_SIZE}")
print("="*70)

# ============================================================================
# FIRE EVENTS - EXPANDED DATASET
# ============================================================================
FIRE_EVENTS = [
    {
        'name': 'Camp Fire 2018',
        'bbox': (-121.3, 38.3, -120.8, 38.8),
        'wind_speed': 15.0,
        'wind_direction': 270.0,
        'samples': NUM_SAMPLES_PER_FIRE
    },
    {
        'name': 'Dixie Fire 2021',
        'bbox': (-121.7, 39.8, -121.0, 40.5),
        'wind_speed': 12.0,
        'wind_direction': 180.0,
        'samples': NUM_SAMPLES_PER_FIRE
    },
    {
        'name': 'Creek Fire 2020',
        'bbox': (-119.4, 36.8, -118.8, 37.4),
        'wind_speed': 14.0,
        'wind_direction': 270.0,
        'samples': NUM_SAMPLES_PER_FIRE
    },
    {
        'name': 'August Complex 2020',
        'bbox': (-122.8, 38.5, -122.0, 39.3),
        'wind_speed': 11.0,
        'wind_direction': 270.0,
        'samples': NUM_SAMPLES_PER_FIRE
    },
    {
        'name': 'Woolsey Fire 2018',
        'bbox': (-118.8, 34.0, -118.3, 34.4),
        'wind_speed': 13.0,
        'wind_direction': 90.0,
        'samples': NUM_SAMPLES_PER_FIRE
    }
]

print(f"\nLoading data from {len(FIRE_EVENTS)} major fire events...")
print("-" * 70)

all_inputs = []
all_targets = []

for fire_event in FIRE_EVENTS:
    print(f"\n[{fire_event['name']}]")
    fire_samples = 0
    
    for sample_idx in range(fire_event['samples']):
        try:
            # Fetch real environmental data for this fire region
            stack = build_real_input_stack(
                west=fire_event['bbox'][0],
                south=fire_event['bbox'][1],
                east=fire_event['bbox'][2],
                north=fire_event['bbox'][3],
                size=(RESOLUTION, RESOLUTION),
                dem_source='3dep',
                wind_speed=fire_event['wind_speed'],
                wind_direction_deg=fire_event['wind_direction']
            )
            
            # Generate pseudo ground truth using Rothermel simulator
            # Vary ignition point and duration for diversity
            random_row = np.random.randint(20, RESOLUTION-20)
            random_col = np.random.randint(20, RESOLUTION-20)
            duration_hours = np.random.uniform(4, 24)
            
            elevation = stack[0]
            fuel_load = stack[3]
            wind_speed = stack[6]
            
            target = rothermel_fire_spread(
                elevation=elevation,
                fuel_load=fuel_load,
                wind_speed=wind_speed,
                ignition_row=random_row,
                ignition_col=random_col,
                duration_hours=duration_hours,
                integration_steps=int(duration_hours * 4)
            )
            
            all_inputs.append(stack)
            all_targets.append(target)
            fire_samples += 1
            
            if (sample_idx + 1) % 10 == 0:
                print(f"  [OK] {sample_idx + 1}/{fire_event['samples']} samples loaded")
                
        except Exception as e:
            print(f"  [FAIL] Sample {sample_idx + 1} failed: {e}")
            continue
    
    print(f"  Total from {fire_event['name']}: {fire_samples} samples")

print(f"\n{'='*70}")
print(f"TOTAL SAMPLES LOADED: {len(all_inputs)}")
print(f"Expected: {len(FIRE_EVENTS) * NUM_SAMPLES_PER_FIRE}")
print(f"Success rate: {100*len(all_inputs)/(len(FIRE_EVENTS)*NUM_SAMPLES_PER_FIRE):.1f}%")
print(f"{'='*70}")

if len(all_inputs) < 10:
    print("ERROR: Not enough training data loaded!")
    sys.exit(1)

# Convert to tensors
all_inputs = np.array(all_inputs, dtype=np.float32)
all_targets = np.array(all_targets, dtype=np.float32)

print(f"\nData shapes:")
print(f"  Inputs: {all_inputs.shape}")
print(f"  Targets: {all_targets.shape}")

# Create train/validation split
n_samples = len(all_inputs)
n_train = int(n_samples * (1 - VALIDATION_SPLIT))
indices = np.random.permutation(n_samples)

train_indices = indices[:n_train]
val_indices = indices[n_train:]

train_inputs = torch.from_numpy(all_inputs[train_indices])
train_targets = torch.from_numpy(all_targets[train_indices])
val_inputs = torch.from_numpy(all_inputs[val_indices])
val_targets = torch.from_numpy(all_targets[val_indices])

train_dataset = TensorDataset(train_inputs, train_targets)
val_dataset = TensorDataset(val_inputs, val_targets)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTrain set: {len(train_loader) * BATCH_SIZE} samples in {len(train_loader)} batches")
print(f"Val set: {len(val_loader) * BATCH_SIZE} samples in {len(val_loader)} batches")

# ============================================================================
# MODEL & TRAINING
# ============================================================================
print(f"\n{'='*70}")
print("INITIALIZING MODEL")
print(f"{'='*70}")

model = build_unet(in_channels=8, out_channels=1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Optimizer: Adam (lr={LEARNING_RATE})")
print(f"Loss: BCEWithLogitsLoss")

# Training loop
history = {
    'train_loss': [],
    'train_iou': [],
    'train_dice': [],
    'val_loss': [],
    'val_iou': [],
    'val_dice': [],
    'best_val_loss': float('inf'),
    'best_epoch': 0
}

def compute_metrics(pred, target):
    """Compute IoU and Dice coefficient"""
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = ((pred_binary + target_binary) > 0).sum()
    pred_sum = pred_binary.sum()
    target_sum = target_binary.sum()
    
    iou = intersection / (union + 1e-6)
    dice = 2 * intersection / (pred_sum + target_sum + 1e-6)
    
    return iou.item(), dice.item()

print(f"\n{'='*70}")
print("STARTING TRAINING")
print(f"{'='*70}\n")

best_model_state = None

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_iou_list = []
    train_dice_list = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
    for inputs, targets in pbar:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets.unsqueeze(1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        with torch.no_grad():
            pred_prob = torch.sigmoid(logits)
            iou, dice = compute_metrics(pred_prob, targets.unsqueeze(1))
            train_iou_list.append(iou)
            train_dice_list.append(dice)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.4f}',
            'dice': f'{dice:.4f}'
        })
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_iou = np.mean(train_iou_list)
    avg_train_dice = np.mean(train_dice_list)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_iou_list = []
    val_dice_list = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]")
        for inputs, targets in pbar:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            logits = model(inputs)
            loss = criterion(logits, targets.unsqueeze(1))
            val_loss += loss.item()
            
            pred_prob = torch.sigmoid(logits)
            iou, dice = compute_metrics(pred_prob, targets.unsqueeze(1))
            val_iou_list.append(iou)
            val_dice_list.append(dice)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'dice': f'{dice:.4f}'
            })
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = np.mean(val_iou_list)
    avg_val_dice = np.mean(val_dice_list)
    
    # Record history
    history['train_loss'].append(avg_train_loss)
    history['train_iou'].append(avg_train_iou)
    history['train_dice'].append(avg_train_dice)
    history['val_loss'].append(avg_val_loss)
    history['val_iou'].append(avg_val_iou)
    history['val_dice'].append(avg_val_dice)
    
    # Learning rate scheduler
    scheduler.step(avg_val_loss)
    
    # Save best model
    if avg_val_loss < history['best_val_loss']:
        history['best_val_loss'] = avg_val_loss
        history['best_epoch'] = epoch + 1
        best_model_state = model.state_dict().copy()
        print(f"\n[OK] NEW BEST MODEL at epoch {epoch+1} (val_loss={avg_val_loss:.6f})\n")
    
    # Print progress
    if (epoch + 1) % 5 == 0:
        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f} | IoU: {avg_train_iou:.4f} | Dice: {avg_train_dice:.4f}")
        print(f"  Val Loss: {avg_val_loss:.6f} | IoU: {avg_val_iou:.4f} | Dice: {avg_val_dice:.4f}")
        print(f"  Best: Epoch {history['best_epoch']} (Loss={history['best_val_loss']:.6f})")

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\nLoaded best model from epoch {history['best_epoch']}")

# ============================================================================
# SAVE CHECKPOINT
# ============================================================================
ckpt_dir = ROOT / 'outputs' / 'checkpoints'
ckpt_dir.mkdir(parents=True, exist_ok=True)

checkpoint = {
    'epoch': history['best_epoch'],
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': history['best_val_loss'],
    'config': {
        'resolution': RESOLUTION,
        'epochs_trained': EPOCHS,
        'total_samples': len(all_inputs),
        'fire_events': len(FIRE_EVENTS),
        'learning_rate': LEARNING_RATE
    }
}

ckpt_path = ckpt_dir / 'unet_extended_512x512.pt'
torch.save(checkpoint, ckpt_path)
print(f"\n[OK] Checkpoint saved: {ckpt_path.name}")
print(f"  Size: {ckpt_path.stat().st_size / 1e6:.1f} MB")

# ============================================================================
# SAVE TRAINING HISTORY
# ============================================================================
history_file = ROOT / 'outputs' / 'training_history_extended.json'
history_to_save = {k: v for k, v in history.items() if k not in ['best_val_loss']}
for key in ['train_loss', 'train_iou', 'train_dice', 'val_loss', 'val_iou', 'val_dice']:
    if isinstance(history_to_save[key], list):
        history_to_save[key] = [float(x) for x in history_to_save[key]]

with open(history_file, 'w') as f:
    json.dump(history_to_save, f, indent=2)

print(f"[OK] Training history saved: {history_file.name}")

# ============================================================================
# PLOT TRAINING CURVES
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
axes[0].axvline(history['best_epoch']-1, color='red', linestyle='--', alpha=0.7, label=f'Best (Epoch {history["best_epoch"]})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (BCEWithLogits)')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_iou'], label='Train', linewidth=2)
axes[1].plot(history['val_iou'], label='Validation', linewidth=2)
axes[1].axhline(0.8, color='green', linestyle='--', alpha=0.5, label='80% Target')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('IoU (Jaccard Index)')
axes[1].set_title('IoU Metric')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1)

axes[2].plot(history['train_dice'], label='Train', linewidth=2)
axes[2].plot(history['val_dice'], label='Validation', linewidth=2)
axes[2].axhline(0.8, color='green', linestyle='--', alpha=0.5, label='80% Target')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Dice Coefficient')
axes[2].set_title('Dice Metric')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 1)

plt.suptitle(f'Extended Training: {RESOLUTION}x{RESOLUTION} Resolution, {EPOCHS} Epochs, {len(all_inputs)} Samples', 
             fontsize=12, fontweight='bold')
plt.tight_layout()

plot_file = ROOT / 'outputs' / 'training_curves_extended_512.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"[OK] Training curves saved: {plot_file.name}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("TRAINING COMPLETE - SUMMARY")
print(f"{'='*70}")
print(f"Best Epoch: {history['best_epoch']}/{EPOCHS}")
print(f"Final Validation Loss: {history['best_val_loss']:.6f}")
print(f"Final Validation IoU: {history['val_iou'][-1]:.4f}")
print(f"Final Validation Dice: {history['val_dice'][-1]:.4f}")
print(f"\nCheckpoint: {ckpt_path.name}")
print(f"Model size: {ckpt_path.stat().st_size / 1e6:.1f} MB")
print(f"Training samples: {len(train_loader) * BATCH_SIZE}")
print(f"Validation samples: {len(val_loader) * BATCH_SIZE}")
print(f"{'='*70}\n")
