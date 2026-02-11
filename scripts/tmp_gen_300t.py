
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
