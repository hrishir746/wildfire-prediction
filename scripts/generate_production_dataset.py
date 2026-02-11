#!/usr/bin/env python3
"""
Generate PRODUCTION dataset: 500 samples at 512x512 resolution
Uses real California terrain with diverse fire scenarios.
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
print("PRODUCTION DATASET GENERATION")
print("Real California Terrain + Diverse Fire Scenarios")
print("=" * 80)

# PRODUCTION Configuration
N_SAMPLES = 500  # More samples for better generalization
GRID_SIZE = (512, 512)  # Full resolution
OUTPUT_DIR = ROOT / "data" / "production_real_terrain"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Expanded CA regions (more diverse terrain)
CA_REGIONS = [
    # Northern CA (Camp Fire, Carr Fire regions)
    (-122.5, 40.5, -122.0, 40.9),  # Shasta County
    (-121.8, 39.5, -121.3, 39.9),  # Butte County (Camp Fire)
    (-121.5, 39.7, -121.0, 40.1),  # Butte/Plumas
    (-122.0, 40.0, -121.5, 40.4),  # Tehama County
    
    # Wine Country / Sonoma
    (-123.0, 38.3, -122.5, 38.7),  # Sonoma County (Tubbs Fire)
    (-122.8, 38.5, -122.3, 38.9),  # Napa County
    (-122.5, 38.7, -122.0, 39.1),  # Lake County
    
    # Central CA
    (-120.5, 37.5, -120.0, 37.9),  # Mariposa (Ferguson Fire area)
    (-121.0, 38.0, -120.5, 38.4),  # Amador/Calaveras
    
    # Southern CA (Thomas Fire, Woolsey regions)
    (-120.0, 34.3, -119.5, 34.7),  # Santa Barbara (Thomas Fire)
    (-119.5, 34.0, -119.0, 34.4),  # Ventura County
    (-119.0, 34.1, -118.5, 34.5),  # Los Angeles backcountry
    (-117.2, 33.0, -116.7, 33.4),  # San Diego backcountry
    
    # Sierra Nevada
    (-120.0, 39.0, -119.5, 39.4),  # Tahoe area
    (-119.5, 37.5, -119.0, 37.9),  # Yosemite region
]

# Diverse wind scenarios (covering all major CA fire weather patterns)
WIND_SCENARIOS = [
    # Normal conditions
    (10.0, 270.0, "Normal westerly"),
    (15.0, 270.0, "Moderate westerly"),
    
    # Elevated fire weather
    (20.0, 270.0, "Strong westerly"),
    (25.0, 270.0, "Very strong westerly"),
    (30.0, 270.0, "Extreme westerly"),
    
    # Santa Ana winds (NE, hot, dry - most dangerous)
    (20.0, 45.0, "Moderate Santa Ana"),
    (30.0, 45.0, "Strong Santa Ana"),
    (40.0, 45.0, "Extreme Santa Ana"),
    (45.0, 60.0, "Severe Santa Ana"),
    
    # Diablo winds (Northern CA equivalent)
    (25.0, 90.0, "Diablo winds"),
    (35.0, 90.0, "Strong Diablo"),
    
    # Variable directions
    (20.0, 180.0, "Southerly"),
    (20.0, 0.0, "Northerly"),
    (15.0, 315.0, "NW wind"),
]

print(f"\nConfiguration:")
print(f"  Samples: {N_SAMPLES}")
print(f"  Resolution: {GRID_SIZE[0]}×{GRID_SIZE[1]}")
print(f"  CA regions: {len(CA_REGIONS)}")
print(f"  Wind scenarios: {len(WIND_SCENARIOS)}")
print(f"  Output: {OUTPUT_DIR}")
print(f"\n⏱️  Estimated time: {N_SAMPLES * 3 / 60:.0f}-{N_SAMPLES * 5 / 60:.0f} minutes")

# Initialize simulator
sim_params = SimulationParams(grid_rows=GRID_SIZE[0], grid_cols=GRID_SIZE[1])
simulator = FireSpreadSimulator(sim_params)

# Storage
X_data = []
Y_data = []
metadata = []

print(f"\n{'='*80}")
print("GENERATING PRODUCTION DATASET")
print(f"{'='*80}\n")

successful = 0
failed = 0

for i in tqdm(range(N_SAMPLES), desc="Generating samples"):
    try:
        # Randomly select region and wind
        region_idx = np.random.randint(0, len(CA_REGIONS))
        wind_idx = np.random.randint(0, len(WIND_SCENARIOS))
        
        west, south, east, north = CA_REGIONS[region_idx]
        wind_speed, wind_direction, wind_desc = WIND_SCENARIOS[wind_idx]
        
        # Fetch real terrain
        input_stack = build_real_input_stack(
            west=west, south=south, east=east, north=north,
            size=GRID_SIZE,
            dem_source='3dep',
            wind_speed=wind_speed,
            wind_direction_deg=wind_direction
        )
        
        # Extract terrain
        slope_deg = input_stack[0] * 45.0
        aspect_deg = np.degrees(np.arctan2(input_stack[1], input_stack[2])) % 360
        fuel_density = input_stack[3]
        
        # Random ignition (more diverse patterns)
        margin = 30
        ignition_type = np.random.choice(['point', 'small', 'line'])
        
        if ignition_type == 'point':
            # Single point ignition
            cx = np.random.randint(margin, GRID_SIZE[0] - margin)
            cy = np.random.randint(margin, GRID_SIZE[1] - margin)
            ignition_mask = np.zeros(GRID_SIZE, dtype=bool)
            ignition_mask[cx, cy] = True
            
        elif ignition_type == 'small':
            # Small cluster (2-5 pixel radius)
            cx = np.random.randint(margin, GRID_SIZE[0] - margin)
            cy = np.random.randint(margin, GRID_SIZE[1] - margin)
            radius = np.random.randint(2, 6)
            ignition_mask = np.zeros(GRID_SIZE, dtype=bool)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        x, y = cx + dx, cy + dy
                        if 0 <= x < GRID_SIZE[0] and 0 <= y < GRID_SIZE[1]:
                            ignition_mask[x, y] = True
        
        else:  # line
            # Power line ignition (linear pattern)
            cx = np.random.randint(margin, GRID_SIZE[0] - margin)
            cy = np.random.randint(margin, GRID_SIZE[1] - margin)
            length = np.random.randint(5, 15)
            angle = np.random.uniform(0, 2 * np.pi)
            ignition_mask = np.zeros(GRID_SIZE, dtype=bool)
            for t in range(length):
                x = int(cx + t * np.cos(angle))
                y = int(cy + t * np.sin(angle))
                if 0 <= x < GRID_SIZE[0] and 0 <= y < GRID_SIZE[1]:
                    ignition_mask[x, y] = True
        
        # Update input stack
        input_stack[7] = ignition_mask.astype(np.float32)
        
        # Simulate fire (10 steps)
        fire_state = np.zeros(GRID_SIZE, dtype=np.int32)
        fire_state[ignition_mask] = 1
        
        for step in range(10):
            if not np.any(fire_state == 1):
                break
            fire_state = simulator.step(
                fire_state, slope_deg, aspect_deg,
                wind_speed, wind_direction, fuel_density
            )
        
        # Target
        target = (fire_state > 0).astype(np.float32)[np.newaxis, ...]
        
        # Store
        X_data.append(input_stack)
        Y_data.append(target)
        metadata.append({
            'region_idx': region_idx,
            'wind_scenario': wind_desc,
            'ignition_type': ignition_type,
            'ignition_pixels': int(ignition_mask.sum())
        })
        
        successful += 1
        
    except Exception as e:
        failed += 1
        if failed < 5:  # Only print first few errors
            tqdm.write(f"  Sample {i} failed: {str(e)[:60]}")
        continue

print(f"\n{'='*80}")
print(f"DATASET GENERATION COMPLETE")
print(f"{'='*80}")
print(f"\n✓ Successful: {successful} / {N_SAMPLES}")
print(f"✗ Failed: {failed} / {N_SAMPLES}")
print(f"Success rate: {100*successful/N_SAMPLES:.1f}%")

if successful > 0:
    # Convert to arrays
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)
    
    print(f"\nDataset shapes:")
    print(f"  X (input): {X_data.shape}")
    print(f"  Y (target): {Y_data.shape}")
    print(f"  Memory: {(X_data.nbytes + Y_data.nbytes) / 1e9:.2f} GB")
    
    # Save
    output_file = OUTPUT_DIR / "production_dataset.npz"
    np.savez_compressed(
        output_file,
        X=X_data,
        Y=Y_data,
        metadata=metadata
    )
    
    print(f"\n✓ Dataset saved: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1e9:.2f} GB")
    
    # Stats
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(X_data)}")
    print(f"  Resolution: {GRID_SIZE[0]}×{GRID_SIZE[1]}")
    print(f"  CA regions covered: {len(set(m['region_idx'] for m in metadata))}")
    print(f"  Wind scenarios: {len(set(m['wind_scenario'] for m in metadata))}")
    print(f"  Ignition types: point, small cluster, line")
    
    print(f"\n{'='*80}")
    print("READY FOR PRODUCTION TRAINING")
    print(f"{'='*80}")
    print("\nNext step:")
    print("  python scripts/train_production_model.py")
    print(f"\nEstimated training time: 2-4 hours (50 epochs, GPU)")
    
else:
    print("\n⚠ No samples generated successfully.")
