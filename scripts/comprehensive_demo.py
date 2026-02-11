#!/usr/bin/env python
"""
Comprehensive Demo: Wildfire Prediction System End-to-End
Shows: real data fetch, baseline computation, model inference, visualization.

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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.data.real_data import build_real_input_stack
from src.model import build_unet
from src.simulation.rothermel_simulator import SimulationParams, FireSpreadSimulator

print("="*70)
print("WILDFIRE PREDICTION SYSTEM - COMPREHENSIVE DEMO")
print("="*70)

# Parameters for Camp 2018 fire
bbox = (-121.3, 38.3, -120.8, 38.8)  # California, USA
size = (128, 128)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n1. REAL DATA FETCHING")
print(f"   Bbox: {bbox}")
print(f"   Size: {size}")
print(f"   Device: {device}")

try:
    print(f"\n   Fetching DEM (elevation, slope, aspect)...")
    stack = build_real_input_stack(
        west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3],
        size=size,
        dem_source='3dep',
        wind_speed=5.0,
        wind_direction_deg=270.0
    )
    print(f"   SUCCESS - Input stack shape: {stack.shape}")
    print(f"   Channel means: {stack.mean(axis=(1,2))}")
except Exception as e:
    print(f"   SKIPPED: {e}")
    # Create synthetic data for demo
    print(f"   Creating synthetic data for demo...")
    stack = np.random.randn(8, *size).astype(np.float32)
    stack = np.clip(stack, 0, 1)
    print(f"   Synthetic stack created: {stack.shape}")

# Load trained model
print(f"\n2. MODEL LOADING & INFERENCE")
try:
    checkpoint_path = ROOT / 'outputs/checkpoints/unet_final.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = build_unet(in_channels=8, out_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   Checkpoint loaded: {checkpoint_path.name}")
    print(f"   Epoch trained: {checkpoint['epoch']}")
    
    # Run inference
    with torch.no_grad():
        x_torch = torch.from_numpy(stack[np.newaxis, :, :, :]).to(device)
        y_logits = model(x_torch)
        y_prob = torch.sigmoid(y_logits)
        y_pred = y_prob.cpu().numpy()[0, 0, :, :]
    
    print(f"   Inference complete")
    print(f"   Output shape: {y_pred.shape}")
    print(f"   Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"   Mean confidence: {y_pred.mean():.4f}")
    
except Exception as e:
    print(f"   ERROR: {e}")
    y_pred = np.random.rand(*size).astype(np.float32)
    print(f"   Using random predictions for demo")

# Compute Rothermel baseline
print(f"\n3. BASELINE (ROTHERMEL SIMULATOR)")
try:
    sim_params = SimulationParams(grid_rows=size[0], grid_cols=size[1], cell_size_m=30.0)
    sim = FireSpreadSimulator(sim_params)
    
    slope_deg = stack[0] * 45.0
    fuel = np.clip(stack[3], 0, 1)
    initial_fire = np.zeros(size)
    initial_fire[size[0]//2, size[1]//2] = 1.0  # Fire starts at center
    
    fire_state = initial_fire.astype(np.int32)
    
    # Run simulation for 5 timesteps
    for step in range(5):
        fire_state = sim.step(
            fire_state, slope_deg, np.zeros(size),
            5.0, 270.0, fuel
        )
    
    baseline = (fire_state > 0).astype(np.float32)
    print(f"   Baseline computed: {baseline.shape}")
    print(f"   Burned area: {baseline.sum():.0f} cells")
    
except Exception as e:
    print(f"   WARNING: {e}")
    baseline = np.zeros(size, dtype=np.float32)
    baseline[size[0]//2-5:size[0]//2+5, size[1]//2-5:size[1]//2+5] = 1.0

# Create visualizations
print(f"\n4. VISUALIZATION")
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Plot 1: Model prediction
im1 = axes[0, 0].imshow(y_pred, cmap='YlOrRd', vmin=0, vmax=1)
axes[0, 0].set_title('Model Prediction (Confidence)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('X (pixels)')
axes[0, 0].set_ylabel('Y (pixels)')
plt.colorbar(im1, ax=axes[0, 0], label='Burn Probability')

# Plot 2: Baseline
im2 = axes[0, 1].imshow(baseline, cmap='YlOrRd', vmin=0, vmax=1)
axes[0, 1].set_title('Baseline (Rothermel Simulator)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('X (pixels)')
axes[0, 1].set_ylabel('Y (pixels)')
plt.colorbar(im2, ax=axes[0, 1], label='Burn Extent')

# Plot 3: DEM slope
slope_vis = (stack[0] * 45).clip(0, 45) / 45
im3 = axes[1, 0].imshow(slope_vis, cmap='terrain')
axes[1, 0].set_title('Terrain Slope', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('X (pixels)')
axes[1, 0].set_ylabel('Y (pixels)')
plt.colorbar(im3, ax=axes[1, 0], label='Slope (0-45 deg)')

# Plot 4: Fuel density
fuel_vis = np.clip(stack[3], 0, 1)
im4 = axes[1, 1].imshow(fuel_vis, cmap='Greens')
axes[1, 1].set_title('Fuel Density', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('X (pixels)')
axes[1, 1].set_ylabel('Y (pixels)')
plt.colorbar(im4, ax=axes[1, 1], label='Fuel Load')

plt.suptitle('Wildfire Prediction - Camp 2018 Region', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()

output_path = ROOT / 'outputs' / 'comprehensive_demo.png'
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   Visualization saved: {output_path}")

# Compute metrics (for demo - would use true labels in production)
print(f"\n5. EVALUATION METRICS (BASELINE COMPARISON)")
pred_binary = (y_pred > 0.5).astype(np.float32)
intersection = (pred_binary * baseline).sum()
union = ((pred_binary + baseline) > 0).sum()
iou = intersection / union if union > 0 else 0.0
dice = 2 * intersection / (pred_binary.sum() + baseline.sum()) if (pred_binary.sum() + baseline.sum()) > 0 else 0.0

print(f"   IoU (vs baseline): {iou:.4f}")
print(f"   Dice (vs baseline): {dice:.4f}")
print(f"   Model burn area: {pred_binary.sum():.0f} cells")
print(f"   Baseline burn area: {baseline.sum():.0f} cells")

print(f"\n{'='*70}")
print(f"DEMO COMPLETE")
print(f"{'='*70}\n")
