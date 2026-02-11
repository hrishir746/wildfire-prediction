#!/usr/bin/env python3
"""
Generate dataset with LONGER fire simulations (100 timesteps instead of 10).
This matches multi-day fire events like Camp Fire better.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

from src.data.real_data import build_real_input_stack
from src.simulation.rothermel_simulator import FireSpreadSimulator, SimulationParams

print("=" * 80)
print("LONG-TERM FIRE SIMULATION DATASET")
print("100 timesteps = ~4+ days of fire spread")
print("=" * 80)

N_SAMPLES = 200  # Increased for better training
GRID_SIZE = (512, 512)
TIMESTEPS = 150  # Even longer - matches multi-day fires better
OUTPUT_DIR = ROOT / "data" / "longterm_fires"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Focus on Camp Fire region for better validation
CA_REGIONS = [
    # Butte County (Camp Fire area) - multiple tiles
    (-121.8, 39.5, -121.3, 39.9),
    (-121.7, 39.6, -121.2, 40.0),
    (-121.6, 39.7, -121.1, 40.1),
    # Similar terrain areas
    (-122.5, 40.5, -122.0, 40.9),  # Shasta
    (-122.0, 40.0, -121.5, 40.4),  # Tehama
]

# Extreme wind conditions (like actual Camp Fire)
WIND_SCENARIOS = [
    (25.0, 45.0, "Strong Santa Ana"),
    (30.0, 45.0, "Very strong Santa Ana"),
    (35.0, 45.0, "Extreme Santa Ana"),
    (40.0, 60.0, "Severe NE winds"),
]

print(f"\nConfiguration:")
print(f"  Samples: {N_SAMPLES}")
print(f"  Grid: {GRID_SIZE}")
print(f"  Timesteps: {TIMESTEPS} (vs 10 before)")
print(f"  Estimated time: {N_SAMPLES * 15 / 60:.0f}-{N_SAMPLES * 25 / 60:.0f} minutes")
print(f"  This will create a PROPER baseline for MTBS validation")

sim_params = SimulationParams(grid_rows=GRID_SIZE[0], grid_cols=GRID_SIZE[1])
simulator = FireSpreadSimulator(sim_params)

X_data = []
Y_data = []

# Create progress file for dashboard
progress_file = OUTPUT_DIR / "generation_progress.json"

print(f"\nGenerating long-term simulations...")

for i in tqdm(range(N_SAMPLES), desc="Generating"):
    try:
        region_idx = np.random.randint(0, len(CA_REGIONS))
        wind_idx = np.random.randint(0, len(WIND_SCENARIOS))
        
        west, south, east, north = CA_REGIONS[region_idx]
        wind_speed, wind_direction, wind_desc = WIND_SCENARIOS[wind_idx]
        
        input_stack = build_real_input_stack(
            west=west, south=south, east=east, north=north,
            size=GRID_SIZE, dem_source='3dep',
            wind_speed=wind_speed, wind_direction_deg=wind_direction
        )
        
        slope_deg = input_stack[0] * 45.0
        aspect_deg = np.degrees(np.arctan2(input_stack[1], input_stack[2])) % 360
        fuel_density = input_stack[3]
        
        # Single point ignition
        margin = 30
        cx = np.random.randint(margin, GRID_SIZE[0] - margin)
        cy = np.random.randint(margin, GRID_SIZE[1] - margin)
        ignition_mask = np.zeros(GRID_SIZE, dtype=bool)
        ignition_mask[cx, cy] = True
        
        input_stack[7] = ignition_mask.astype(np.float32)
        
        # LONG simulation (100 steps)
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
        
        # Update progress for dashboard (every sample for live updates)
        import json
        with open(progress_file, 'w') as f:
            json.dump({
                'completed': len(X_data),
                'total': N_SAMPLES,
                'percentage': len(X_data) / N_SAMPLES * 100,
                'status': 'generating',
                'current_sample': i + 1
            }, f)
        
    except Exception as e:
        tqdm.write(f"Sample {i} failed: {str(e)[:60]}")
        continue

print(f"\nGenerated: {len(X_data)} samples")

if len(X_data) > 0:
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)
    
    output_file = OUTPUT_DIR / "longterm_dataset.npz"
    np.savez_compressed(output_file, X=X_data, Y=Y_data)
    
    print(f"\n✓ Dataset saved: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1e6:.1f} MB")
    print(f"\nNext: Fine-tune production model on this long-term data")
    print(f"  This should give MUCH better MTBS validation!")
