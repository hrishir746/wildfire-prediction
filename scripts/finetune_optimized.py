#!/usr/bin/env python
"""
Optimized fine-tuning script - CPU-friendly with automatic GPU detection.
Uses reduced precision (mixed precision when possible) to speed up training.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from src.data.real_data import build_real_input_stack
from src.model import build_unet
from src.simulation.rothermel_simulator import SimulationParams, FireSpreadSimulator

# Parameters - optimized for whatever device is available
bbox = (-121.3, 38.3, -120.8, 38.8)
size = (48, 48)  # Small spatial size
n_samples = 10  # Minimal for CPU speed
batch_size = 2
epochs = 2
learning_rate = 1e-4

# Auto-detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Fixed wind parameters (no network calls)
wind_speed = 5.0
wind_direction = 270.0

def compute_baseline_label(stack):
    """Compute Rothermel baseline prediction"""
    rows, cols = size
    sim_params = SimulationParams(grid_rows=rows, grid_cols=cols, cell_size_m=30.0)
    sim = FireSpreadSimulator(sim_params)
    
    slope = stack[0] * 45.0
    fuel = stack[3]
    initial_fire = stack[7]
    
    fire_state = (initial_fire > 0.5).astype(np.int32)
    
    # Run simulation for 8 steps
    for _ in range(8):
        if not np.any(fire_state == 1):
            break
        fire_state = sim.step(
            fire_state, slope, np.zeros_like(slope),
            wind_speed, wind_direction, fuel
        )
    
    return (fire_state > 0).astype(np.float32)

# Build dataset
print(f"Building pseudo-dataset: n={n_samples}, size={size}")
X_list, Y_list = [], []

for i in range(n_samples):
    try:
        print(f"  Sample {i+1}/{n_samples}...")
        
        # Fetch real data using correct function signature
        west, south, east, north = bbox
        stack = build_real_input_stack(
            west=west, south=south, east=east, north=north,
            size=size,
            dem_source='3dep',  # Use 3DEP REST (Windows-compatible)
            wind_speed=wind_speed,
            wind_direction_deg=wind_direction
        )
        
        # Compute baseline label
        label = compute_baseline_label(stack)
        
        X_list.append(stack.astype(np.float32))
        Y_list.append(label[np.newaxis, :, :].astype(np.float32))
        
    except Exception as e:
        print(f"    Warning: {e}")
        continue

X = np.stack(X_list, axis=0)  # (n, 8, h, w)
Y = np.stack(Y_list, axis=0)  # (n, 1, h, w)

print(f"Dataset: X={X.shape}, Y={Y.shape}")

# Create data loader
dataset = TensorDataset(
    torch.from_numpy(X),
    torch.from_numpy(Y)
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Build model
model = build_unet(in_channels=8, out_channels=1)
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Mixed precision if CUDA available
use_amp = device == 'cuda'
scaler = GradScaler() if use_amp else None

# Training loop
print(f"\nTraining for {epochs} epochs...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
        
        if (batch_idx + 1) % max(1, len(loader) // 2) == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.6f}")
    
    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} Complete - Avg Loss: {avg_loss:.6f}")

# Save checkpoint
checkpoint_path = ROOT / 'outputs' / 'checkpoints' / 'unet_finetune_optimized.pt'
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

checkpoint = {
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
    'config': {
        'model': {'input_channels': 8, 'output_channels': 1},
        'data': {'size': size, 'n_samples': n_samples},
        'training': {'lr': learning_rate, 'epochs': epochs, 'batch_size': batch_size}
    }
}

torch.save(checkpoint, checkpoint_path)
print(f"\nCheckpoint saved: {checkpoint_path}")
print(f"File size: {checkpoint_path.stat().st_size / 1e6:.1f} MB")
