#!/usr/bin/env python3
"""
AUTONOMOUS IMPROVEMENT LOOP - Trackmania-style reinforcement learning approach.

Continuously improves the model by:
1. Testing different hyperparameters/strategies
2. Training with each configuration
3. Validating on MTBS Camp Fire
4. Keeping improvements, discarding failures
5. Repeating until target IoU is reached

Like the Trackmania AI, this learns what works through trial and error!
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import subprocess
import time
from datetime import datetime
import numpy as np

print("=" * 80)
print("AUTONOMOUS IMPROVEMENT LOOP")
print("Goal: Achieve 30% IoU on MTBS Camp Fire 2018")
print("=" * 80)

# Configuration
TARGET_IOU = 0.30  # 30% IoU target
MAX_ITERATIONS = 10  # Safety limit
HISTORY_FILE = ROOT / "outputs" / "improvement_history.json"

# Improvement strategies to try
STRATEGIES = [
    {
        'name': '300_timesteps',
        'description': 'Increase to 300 timesteps (2x current)',
        'dataset_samples': 150,
        'timesteps': 300,
        'epochs': 30,
        'grid_size': 512,
    },
    {
        'name': '500_timesteps',
        'description': 'Increase to 500 timesteps (extreme)',
        'dataset_samples': 100,
        'timesteps': 500,
        'epochs': 25,
        'grid_size': 512,
    },
    {
        'name': 'more_data_150t',
        'description': 'Keep 150 timesteps, 2x more training samples',
        'dataset_samples': 400,
        'timesteps': 150,
        'epochs': 40,
        'grid_size': 512,
    },
    {
        'name': 'higher_res_150t',
        'description': 'Higher resolution (1024x1024)',
        'dataset_samples': 100,
        'timesteps': 150,
        'epochs': 30,
        'grid_size': 1024,
    },
]

def load_history():
    """Load improvement history."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {
        'baseline': {'iou': 0.012, 'name': '150_timesteps', 'timestamp': datetime.now().isoformat()},
        'iterations': []
    }

def save_history(history):
    """Save improvement history."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def generate_dataset(strategy):
    """Generate dataset with given strategy."""
    print(f"\n[1/3] Generating dataset: {strategy['description']}")
    
    # Create temporary generation script
    gen_script = f"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm
from src.data.real_data import build_real_input_stack
from src.simulation.rothermel_simulator import FireSpreadSimulator, SimulationParams

N_SAMPLES = {strategy['dataset_samples']}
GRID_SIZE = ({strategy['grid_size']}, {strategy['grid_size']})
TIMESTEPS = {strategy['timesteps']}
OUTPUT_DIR = ROOT / "data" / "auto_improve"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Same regions as before
CA_REGIONS = [
    (-121.8, 39.5, -121.3, 39.9),
    (-121.7, 39.6, -121.2, 40.0),
    (-121.6, 39.7, -121.1, 40.1),
    (-122.5, 40.5, -122.0, 40.9),
    (-122.0, 40.0, -121.5, 40.4),
]

WIND_SCENARIOS = [
    (25.0, 45.0, "Strong Santa Ana"),
    (30.0, 45.0, "Very strong Santa Ana"),
    (35.0, 45.0, "Extreme Santa Ana"),
    (40.0, 60.0, "Severe NE winds"),
]

print(f"Generating {{N_SAMPLES}} samples, {{TIMESTEPS}} timesteps, {{GRID_SIZE[0]}}x{{GRID_SIZE[1]}}...")

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
        wind_speed, wind_direction, _ = WIND_SCENARIOS[wind_idx]
        
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
    except Exception as e:
        print(f"Sample {{i}} failed: {{str(e)[:60]}}")

if len(X_data) > 0:
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)
    
    output_file = OUTPUT_DIR / "dataset.npz"
    np.savez_compressed(output_file, X=X_data, Y=Y_data)
    print(f"Saved: {{output_file}}")
"""
    
    script_path = ROOT / "scripts" / "tmp_auto_gen.py"
    with open(script_path, 'w') as f:
        f.write(gen_script)
    
    # Run generation
    result = subprocess.run(
        [str(ROOT / ".venv-gpu" / "Scripts" / "python.exe"), str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Generation failed: {result.stderr[:500]}")
        return False
    
    return True

def train_model(strategy):
    """Train model on generated dataset."""
    print(f"\n[2/3] Training model: {strategy['epochs']} epochs")
    
    # Training script (simplified, inline)
    train_cmd = f"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from src.model import build_unet

DATASET = ROOT / "data" / "auto_improve" / "dataset.npz"
OUTPUT = ROOT / "outputs" / "checkpoints" / "auto_improve_model.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = np.load(DATASET)
X = torch.from_numpy(data['X'])
Y = torch.from_numpy(data['Y'])

n_val = int(0.15 * len(X))
indices = torch.randperm(len(X))
X_train, Y_train = X[indices[:-n_val]], Y[indices[:-n_val]]
X_val, Y_val = X[indices[-n_val:]], Y[indices[-n_val:]]

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=2, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=2)

model = build_unet(8, 1, 0.2).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

def combined_loss(logits, target):
    bce = nn.BCEWithLogitsLoss()(logits, target)
    pred = torch.sigmoid(logits)
    intersection = (pred * target).sum()
    dice = 1 - (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return 0.5 * bce + 0.5 * dice

best_iou = 0
for epoch in range({strategy['epochs']}):
    model.train()
    for x, y in tqdm(train_loader, desc=f"Epoch {{epoch+1}}", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        combined_loss(model(x), y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Quick val
    model.eval()
    val_iou = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = torch.sigmoid(model(x))
            pred_bin = (pred > 0.5).float()
            tp = (pred_bin * y).sum()
            fp = (pred_bin * (1-y)).sum()
            fn = ((1-pred_bin) * y).sum()
            val_iou += (tp / (tp + fp + fn + 1e-6)).item()
    val_iou /= len(val_loader)
    
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save({{'model_state_dict': model.state_dict(), 'iou': val_iou}}, OUTPUT)
    
    print(f"Epoch {{epoch+1}}: Val IoU = {{val_iou:.4f}}")

print(f"Best Val IoU: {{best_iou:.4f}}")
"""
    
    script_path = ROOT / "scripts" / "tmp_auto_train.py"
    with open(script_path, 'w') as f:
        f.write(train_cmd)
    
    result = subprocess.run(
        [str(ROOT / ".venv-gpu" / "Scripts" / "python.exe"), str(script_path)],
        cwd=str(ROOT)
    )
    
    return result.returncode == 0

def validate_mtbs():
    """Validate on MTBS and return IoU score."""
    print(f"\n[3/3] Validating on MTBS Camp Fire...")
    
    # Use existing validation script but with auto_improve model
    # Just run validate_with_real_mtbs.py and parse output
    result = subprocess.run(
        [str(ROOT / ".venv-gpu" / "Scripts" / "python.exe"), 
         str(ROOT / "scripts" / "validate_with_real_mtbs.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True
    )
    
    # Parse IoU from output
    for line in result.stdout.split('\n'):
        if 'BEST PERFORMANCE: IoU' in line:
            # Extract number
            iou_str = line.split('IoU = ')[1].split(' ')[0]
            return float(iou_str)
    
    return 0.0

# Main loop
def main():
    history = load_history()
    current_best = history['baseline']['iou']
    
    print(f"\nBaseline IoU: {current_best*100:.1f}%")
    print(f"Target IoU: {TARGET_IOU*100:.1f}%")
    print(f"\n{len(STRATEGIES)} strategies to try\n")
    
    for iteration in range(MAX_ITERATIONS):
        if current_best >= TARGET_IOU:
            print(f"\n🎉 TARGET REACHED! IoU: {current_best*100:.1f}%")
            break
        
        if iteration >= len(STRATEGIES):
            print(f"\n⚠️ Ran out of strategies. Best: {current_best*100:.1f}%")
            break
        
        strategy = STRATEGIES[iteration]
        
        print("=" * 80)
        print(f"ITERATION {iteration + 1}: {strategy['name']}")
        print(f"Strategy: {strategy['description']}")
        print("=" * 80)
        
        start_time = time.time()
        
        # Try this strategy
        if not generate_dataset(strategy):
            print("❌ Dataset generation failed, skipping")
            continue
        
        if not train_model(strategy):
            print("❌ Training failed, skipping")
            continue
        
        mtbs_iou = validate_mtbs()
        elapsed = (time.time() - start_time) / 60
        
        # Record result
        result = {
            'iteration': iteration + 1,
            'strategy': strategy['name'],
            'mtbs_iou': mtbs_iou,
            'improvement': mtbs_iou - current_best,
            'time_minutes': elapsed,
            'timestamp': datetime.now().isoformat()
        }
        
        history['iterations'].append(result)
        save_history(history)
        
        print(f"\n📊 RESULTS:")
        print(f"   MTBS IoU: {mtbs_iou*100:.1f}% (was {current_best*100:.1f}%)")
        print(f"   Change: {(mtbs_iou - current_best)*100:+.1f}%")
        print(f"   Time: {elapsed:.1f} minutes")
        
        if mtbs_iou > current_best:
            current_best = mtbs_iou
            print(f"   ✅ NEW BEST! Keeping this model.")
        else:
            print(f"   ❌ No improvement, continuing search...")
        
        print()
    
    print("=" * 80)
    print("AUTONOMOUS LOOP COMPLETE")
    print("=" * 80)
    print(f"Final IoU: {current_best*100:.1f}%")
    print(f"Iterations: {len(history['iterations'])}")
    print(f"\nHistory saved: {HISTORY_FILE}")

if __name__ == "__main__":
    main()
