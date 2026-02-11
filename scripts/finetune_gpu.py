#!/usr/bin/env python3
"""
GPU-optimized fine-tuning script for WSL2 with ROCm support.
Automatically detects GPU and uses it for training.
Supports 512x512 patches for large-scale wildfire mapping.

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

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '1100'  # AMD 9070 XT
os.environ['HIP_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from src.data.real_data import build_real_input_stack
from src.model import build_unet
from src.simulation.rothermel_simulator import SimulationParams, FireSpreadSimulator

# ============================================================================
# CONFIGURATION
# ============================================================================
BBOX = (-121.3, 38.3, -120.8, 38.8)  # Camp 2018 region
SIZE = (512, 512)  # 512x512 for GPU training
N_SAMPLES = 15
BATCH_SIZE = 2  # Reduce for memory
EPOCHS = 3
LEARNING_RATE = 1e-4

# ============================================================================
# GPU SETUP
# ============================================================================
print("="*70)
print("GPU-OPTIMIZED FINE-TUNING")
print("="*70)

# Auto-detect device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Devices: {torch.cuda.device_count()}")
else:
    device = torch.device('cpu')
    print("WARNING: CUDA/GPU not detected, using CPU (slow)")

print(f"Using device: {device}\n")

# ============================================================================
# DATASET BUILDING
# ============================================================================
print(f"Building dataset: {N_SAMPLES} samples at {SIZE}")

X_list, Y_list = [], []
wind_speed = 5.0
wind_direction = 270.0

def compute_baseline(stack):
    """Generate pseudo-label from Rothermel simulator"""
    sim_params = SimulationParams(grid_rows=SIZE[0], grid_cols=SIZE[1], cell_size_m=30.0)
    sim = FireSpreadSimulator(sim_params)
    
    slope = stack[0] * 45.0
    fuel = stack[3]
    initial = stack[7]
    
    fire_state = (initial > 0.5).astype(np.int32)
    
    for _ in range(10):
        if not np.any(fire_state == 1):
            break
        fire_state = sim.step(fire_state, slope, np.zeros_like(slope), 
                             wind_speed, wind_direction, fuel)
    
    return (fire_state > 0).astype(np.float32)

for i in range(N_SAMPLES):
    try:
        print(f"  {i+1}/{N_SAMPLES}...", end=' ', flush=True)
        
        # Fetch real data
        stack = build_real_input_stack(
            west=BBOX[0], south=BBOX[1], east=BBOX[2], north=BBOX[3],
            size=SIZE,
            dem_source='3dep',
            wind_speed=wind_speed,
            wind_direction_deg=wind_direction
        )
        
        # Compute baseline
        label = compute_baseline(stack)
        
        X_list.append(stack.astype(np.float32))
        Y_list.append(label[np.newaxis, :, :].astype(np.float32))
        print("OK")
        
    except Exception as e:
        print(f"SKIP ({str(e)[:30]})")
        continue

if not X_list:
    print("ERROR: No samples generated!")
    sys.exit(1)

X = np.stack(X_list, axis=0)
Y = np.stack(Y_list, axis=0)
print(f"Dataset ready: X={X.shape}, Y={Y.shape}\n")

# ============================================================================
# MODEL SETUP
# ============================================================================
print("Loading model...")
model = build_unet(in_channels=8, out_channels=1)
model = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

# ============================================================================
# TRAINING SETUP
# ============================================================================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

dataset = TensorDataset(
    torch.from_numpy(X),
    torch.from_numpy(Y)
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================================
# TRAINING LOOP
# ============================================================================
print(f"Training: {EPOCHS} epochs, {len(loader)} batches/epoch, {BATCH_SIZE} batch size")
print(f"Total batches: {EPOCHS * len(loader)}\n")

best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(xb)
        loss = criterion(logits, yb)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if (batch_idx + 1) % max(1, len(loader) // 3) == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.6f}")
    
    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} Complete - Avg Loss: {avg_loss:.6f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss

# ============================================================================
# SAVE CHECKPOINT
# ============================================================================
checkpoint_path = ROOT / 'outputs' / 'checkpoints' / 'unet_finetune_optimized_512.pt'
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

checkpoint = {
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': best_loss,
    'config': {
        'model': {'input_channels': 8, 'output_channels': 1},
        'data': {'size': SIZE, 'n_samples': N_SAMPLES, 'bbox': BBOX},
        'training': {
            'lr': LEARNING_RATE,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'device': str(device)
        }
    }
}

torch.save(checkpoint, checkpoint_path)
file_size_mb = checkpoint_path.stat().st_size / 1e6

print(f"\n{'='*70}")
print(f"TRAINING COMPLETE")
print(f"Checkpoint: {checkpoint_path.name}")
print(f"Size: {file_size_mb:.1f} MB")
print(f"Best Loss: {best_loss:.6f}")
print(f"Device: {device}")
print(f"{'='*70}\n")
