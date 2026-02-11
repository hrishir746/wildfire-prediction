#!/usr/bin/env python3
"""
Generate training dataset using REAL California terrain with SIMULATED fire spread.
This bridges the domain gap by training on actual terrain distributions.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm import tqdm

from src.data.real_data import build_real_input_stack
from src.simulation.rothermel_simulator import FireSpreadSimulator, SimulationParams

print("=" * 80)
print("REAL TERRAIN + SIMULATED FIRE DATASET GENERATION")
print("=" * 80)

# Configuration
N_SAMPLES = 50  # Start small, can increase later
GRID_SIZE = (256, 256)  # Use 256 for speed, can do 512 later
OUTPUT_DIR = ROOT / "data" / "real_terrain_synthetic_fires"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# California wildfire regions (sample from multiple areas)
CA_REGIONS = [
    # Butte County (Camp Fire area)
    (-121.6, 39.5, -121.4, 39.7),
    (-121.5, 39.6, -121.3, 39.8),
    (-121.4, 39.7, -121.2, 39.9),
    
    # Sonoma County (Wine Country fires)
    (-122.9, 38.4, -122.7, 38.6),
    (-122.8, 38.3, -122.6, 38.5),
    
    # Santa Barbara (Thomas Fire area)
    (-120.0, 34.4, -119.8, 34.6),
    (-119.8, 34.3, -119.6, 34.5),
    
    # Shasta County
    (-122.5, 40.5, -122.3, 40.7),
    
    # Lake County
    (-122.8, 39.0, -122.6, 39.2),
    
    # San Diego backcountry
    (-117.0, 32.8, -116.8, 33.0),
]

# Wind conditions (typical CA wildfire scenarios)
WIND_CONDITIONS = [
    (15.0, 270.0),  # 15 mph westerly (normal)
    (25.0, 270.0),  # 25 mph westerly (elevated)
    (35.0, 270.0),  # 35 mph westerly (high)
    (20.0, 45.0),   # 20 mph NE (Santa Ana-like)
    (40.0, 45.0),   # 40 mph NE (Extreme Santa Ana)
    (30.0, 90.0),   # 30 mph easterly
]

print(f"\nConfiguration:")
print(f"  Samples: {N_SAMPLES}")
print(f"  Grid size: {GRID_SIZE}")
print(f"  CA regions: {len(CA_REGIONS)}")
print(f"  Wind scenarios: {len(WIND_CONDITIONS)}")
print(f"  Output: {OUTPUT_DIR}")

# Initialize simulator
sim_params = SimulationParams(grid_rows=GRID_SIZE[0], grid_cols=GRID_SIZE[1])
simulator = FireSpreadSimulator(sim_params)

# Storage
X_data = []
Y_data = []

print(f"\n{'='*80}")
print("GENERATING DATASET")
print(f"{'='*80}\n")

for i in tqdm(range(N_SAMPLES), desc="Generating samples"):
    try:
        # Randomly select region and wind conditions
        region_idx = np.random.randint(0, len(CA_REGIONS))
        wind_idx = np.random.randint(0, len(WIND_CONDITIONS))
        
        west, south, east, north = CA_REGIONS[region_idx]
        wind_speed, wind_direction = WIND_CONDITIONS[wind_idx]
        
        # Fetch real terrain data
        input_stack = build_real_input_stack(
            west=west, south=south, east=east, north=north,
            size=GRID_SIZE,
            dem_source='3dep',
            wind_speed=wind_speed,
            wind_direction_deg=wind_direction
        )
        
        # Extract terrain for simulation
        slope_deg = input_stack[0] * 45.0  # Denormalize
        aspect_deg = np.degrees(np.arctan2(input_stack[1], input_stack[2])) % 360
        fuel_density = input_stack[3]
        
        # Random ignition point (avoid edges)
        margin = 20
        cx = np.random.randint(margin, GRID_SIZE[0] - margin)
        cy = np.random.randint(margin, GRID_SIZE[1] - margin)
        
        # Small ignition area (2-4 pixel radius)
        ignition_radius = np.random.randint(2, 5)
        ignition_mask = np.zeros(GRID_SIZE, dtype=bool)
        for dx in range(-ignition_radius, ignition_radius + 1):
            for dy in range(-ignition_radius, ignition_radius + 1):
                if dx*dx + dy*dy <= ignition_radius*ignition_radius:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < GRID_SIZE[0] and 0 <= y < GRID_SIZE[1]:
                        ignition_mask[x, y] = True
        
        # Update input stack with ignition
        input_stack[7] = ignition_mask.astype(np.float32)
        
        # Simulate fire spread (10 steps)
        fire_state = np.zeros(GRID_SIZE, dtype=np.int32)
        fire_state[ignition_mask] = 1
        
        for step in range(10):
            if not np.any(fire_state == 1):
                break
            fire_state = simulator.step(
                fire_state, slope_deg, aspect_deg,
                wind_speed, wind_direction, fuel_density
            )
        
        # Target: final burned area
        target = (fire_state > 0).astype(np.float32)[np.newaxis, ...]  # (1, H, W)
        
        # Store
        X_data.append(input_stack)
        Y_data.append(target)
        
    except Exception as e:
        print(f"\n  Sample {i} failed: {str(e)[:60]}")
        continue

print(f"\n{'='*80}")
print(f"DATASET GENERATION COMPLETE")
print(f"{'='*80}")
print(f"\nSuccessfully generated: {len(X_data)} / {N_SAMPLES} samples")

if len(X_data) > 0:
    # Convert to arrays
    X_data = np.array(X_data, dtype=np.float32)  # (N, 8, H, W)
    Y_data = np.array(Y_data, dtype=np.float32)  # (N, 1, H, W)
    
    print(f"\nDataset shapes:")
    print(f"  X (input): {X_data.shape}")
    print(f"  Y (target): {Y_data.shape}")
    
    # Save dataset
    output_file = OUTPUT_DIR / "real_terrain_dataset.npz"
    np.savez_compressed(
        output_file,
        X=X_data,
        Y=Y_data,
        regions=CA_REGIONS,
        wind_conditions=WIND_CONDITIONS
    )
    
    print(f"\n✓ Dataset saved: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1e6:.1f} MB")
    
    # Save metadata
    metadata_file = OUTPUT_DIR / "dataset_info.txt"
    with open(metadata_file, 'w') as f:
        f.write("Real Terrain + Simulated Fire Dataset\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Samples: {len(X_data)}\n")
        f.write(f"Grid size: {GRID_SIZE}\n")
        f.write(f"Regions sampled: {len(CA_REGIONS)} CA wildfire areas\n")
        f.write(f"Wind scenarios: {len(WIND_CONDITIONS)}\n")
        f.write(f"\nInput channels (8):\n")
        f.write("  0: Slope (normalized)\n")
        f.write("  1: Aspect sin\n")
        f.write("  2: Aspect cos\n")
        f.write("  3: Fuel density\n")
        f.write("  4: Wind speed (normalized)\n")
        f.write("  5: Wind direction sin\n")
        f.write("  6: Wind direction cos\n")
        f.write("  7: Initial fire mask\n")
        f.write(f"\nTarget: Binary fire presence after 10 simulation steps\n")
    
    print(f"✓ Metadata saved: {metadata_file}")
    
    print(f"\n{'='*80}")
    print("NEXT STEP: Fine-tune model on this dataset")
    print(f"{'='*80}")
    print("\nRun:")
    print("  python scripts/finetune_on_real_terrain.py")
    
else:
    print("\n⚠ No samples generated. Check internet connection and API access.")
