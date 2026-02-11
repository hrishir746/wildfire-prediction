#!/usr/bin/env python3
"""
1-HOUR AUTONOMOUS IMPROVEMENT
Quick test: Train with 300 timesteps instead of 150, validate on RIDGE fire.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import subprocess
import time
import json
from datetime import datetime

print("=" * 80)
print("1-HOUR AUTONOMOUS IMPROVEMENT")
print("Baseline: 2.1% IoU on RIDGE fire (150 timesteps)")
print("Strategy: Try 300 timesteps (2x longer)")
print("=" * 80)

BASELINE_IOU = 0.021
START_TIME = time.time()

# Strategy: 300 timesteps
print("\n[STRATEGY] 300 timesteps (double current length)")
print("  - Should match ~12 day fires better")
print("  - Smaller dataset (100 samples) for speed")
print("  - 20 epochs to keep training fast")

# Step 1: Generate dataset with 300 timesteps
print("\n" + "=" * 80)
print("[1/3] GENERATING DATASET (300 timesteps)")
print("=" * 80)

gen_script = """
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm
from src.data.real_data import build_real_input_stack
from src.simulation.rothermel_simulator import FireSpreadSimulator, SimulationParams

N_SAMPLES = 100
GRID_SIZE = (512, 512)
TIMESTEPS = 300  # 2x the original 150
OUTPUT_DIR = ROOT / "data" / "auto_300t"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CA_REGIONS = [
    (-121.8, 39.5, -121.3, 39.9),
    (-121.7, 39.6, -121.2, 40.0),
    (-121.6, 39.7, -121.1, 40.1),
]

WIND_SCENARIOS = [
    (25.0, 45.0), (30.0, 45.0), (35.0, 45.0), (40.0, 60.0)
]

print(f"Generating {N_SAMPLES} samples, {TIMESTEPS} timesteps...")

sim_params = SimulationParams(grid_rows=GRID_SIZE[0], grid_cols=GRID_SIZE[1])
simulator = FireSpreadSimulator(sim_params)

X_data = []
Y_data = []
rng = np.random.default_rng(42)

for i in tqdm(range(N_SAMPLES)):
    try:
        region_idx = rng.integers(0, len(CA_REGIONS))
        wind_idx = rng.integers(0, len(WIND_SCENARIOS))
        
        west, south, east, north = CA_REGIONS[region_idx]
        wind_speed, wind_direction = WIND_SCENARIOS[wind_idx]
        
        input_stack = build_real_input_stack(
            west=west, south=south, east=east, north=north,
            size=GRID_SIZE, dem_source='3dep',
            wind_speed=wind_speed, wind_direction_deg=wind_direction
        )
        
        slope_deg = input_stack[0] * 45.0
        aspect_deg = np.degrees(np.arctan2(input_stack[1], input_stack[2])) % 360
        fuel_density = input_stack[3]
        
        margin = 30
        cx = rng.integers(margin, GRID_SIZE[0] - margin)
        cy = rng.integers(margin, GRID_SIZE[1] - margin)
        ignition_mask = np.zeros(GRID_SIZE, dtype=bool)
        ignition_mask[cx, cy] = True
        
        input_stack[7] = ignition_mask.astype(np.float32)
        
        fire_state = np.zeros(GRID_SIZE, dtype=np.int32)
        fire_state[ignition_mask] = 1
        
        for step in range(TIMESTEPS):
            if not np.any(fire_state == 1):
                break
            fire_state = simulator.step(
                fire_state, slope_deg, aspect_deg,
                wind_speed, wind_direction, fuel_density
            )
        
        target = (fire_state > 0).astype(np.float32)[np.newaxis, ...]
        X_data.append(input_stack)
        Y_data.append(target)
    except:
        pass

if len(X_data) > 0:
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)
    output_file = OUTPUT_DIR / "dataset.npz"
    np.savez_compressed(output_file, X=X_data, Y=Y_data)
    print(f"Saved {len(X_data)} samples")
"""

with open(ROOT / "scripts" / "tmp_gen_300t.py", 'w') as f:
    f.write(gen_script)

subprocess.run([
    str(ROOT / ".venv-gpu" / "Scripts" / "python.exe"),
    str(ROOT / "scripts" / "tmp_gen_300t.py")
], cwd=str(ROOT))

# Step 2: Train
print("\n" + "=" * 80)
print("[2/3] TRAINING MODEL (20 epochs)")
print("=" * 80)

subprocess.run([
    str(ROOT / ".venv-gpu" / "Scripts" / "python.exe"),
    "-c",
    """
import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from src.model import build_unet

data = np.load('data/auto_300t/dataset.npz')
X, Y = torch.from_numpy(data['X']), torch.from_numpy(data['Y'])
n_val = int(0.15 * len(X))
indices = torch.randperm(len(X))

train_loader = DataLoader(TensorDataset(X[indices[:-n_val]], Y[indices[:-n_val]]), batch_size=2, shuffle=True)
val_loader = DataLoader(TensorDataset(X[indices[-n_val:]], Y[indices[-n_val:]]), batch_size=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_unet(8, 1, 0.2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-5)

def loss_fn(logits, y):
    bce = nn.BCEWithLogitsLoss()(logits, y)
    p = torch.sigmoid(logits)
    i = (p*y).sum()
    return 0.5*bce + 0.5*(1-(2*i+1e-6)/(p.sum()+y.sum()+1e-6))

best_iou = 0
for ep in range(20):
    model.train()
    for x, y in tqdm(train_loader, desc=f'Epoch {ep+1}', leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    
    model.eval()
    iou = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            p = (torch.sigmoid(model(x)) > 0.5).float()
            tp = (p*y).sum()
            iou += (tp/(tp+(p*(1-y)).sum()+((1-p)*y).sum()+1e-6)).item()
    iou /= len(val_loader)
    if iou > best_iou:
        best_iou = iou
        torch.save({'model_state_dict': model.state_dict(), 'iou': iou}, 'outputs/checkpoints/auto_300t.pt')
    print(f'Epoch {ep+1}: IoU={iou:.4f}')
print(f'Best: {best_iou:.4f}')
    """
], cwd=str(ROOT))

# Step 3: Validate on RIDGE fire
print("\n" + "=" * 80)
print("[3/3] VALIDATING ON RIDGE FIRE")
print("=" * 80)

# Use existing validation script but update model path
with open(ROOT / "scripts" / "validate_cedar_fire.py") as f:
    val_script = f.read()

val_script = val_script.replace(
    'model_path = ROOT / "outputs" / "checkpoints" / "unet_longterm_v1.pt"',
    'model_path = ROOT / "outputs" / "checkpoints" / "auto_300t.pt"'
)

with open(ROOT / "scripts" / "tmp_val_300t.py", 'w') as f:
    f.write(val_script)

result = subprocess.run([
    str(ROOT / ".venv-gpu" / "Scripts" / "python.exe"),
    str(ROOT / "scripts" / "tmp_val_300t.py")
], cwd=str(ROOT), capture_output=True, text=True)

# Parse IoU
new_iou = BASELINE_IOU
for line in result.stdout.split('\n'):
    if 'BEST PERFORMANCE: IoU' in line:
        new_iou = float(line.split('IoU = ')[1].split(' ')[0])
        break

elapsed = (time.time() - START_TIME) / 60

# Results
print("\n" + "=" * 80)
print("1-HOUR TEST COMPLETE")
print("=" * 80)
print(f"\nTime: {elapsed:.1f} minutes")
print(f"\nBaseline (150 timesteps): {BASELINE_IOU*100:.1f}% IoU")
print(f"New (300 timesteps):      {new_iou*100:.1f}% IoU")
print(f"Change: {(new_iou - BASELINE_IOU)*100:+.1f}%")

if new_iou > BASELINE_IOU:
    print(f"\n✅ IMPROVEMENT! {new_iou/BASELINE_IOU:.1f}x better")
    print("Strategy worked - longer timesteps help!")
else:
    print(f"\n❌ No improvement")
    print("May need different approach")

# Save result
with open(ROOT / "outputs" / "auto_improvement_1hr.json", 'w') as f:
    json.dump({
        'baseline_iou': BASELINE_IOU,
        'new_iou': new_iou,
        'improvement': new_iou - BASELINE_IOU,
        'strategy': '300_timesteps',
        'time_minutes': elapsed,
        'timestamp': datetime.now().isoformat()
    }, f, indent=2)

print(f"\n✓ Results saved: outputs/auto_improvement_1hr.json")
