import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.real_data import build_real_input_stack
from src.model import build_unet

# Parameters
bbox = (-121.3, 38.3, -120.8, 38.8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device:', device)

# Generate pseudo-dataset using Rothermel baseline as pseudo-labels
def compute_baseline_label(stack):
    # reuse simple baseline: simulate using initial_fire channel
    from src.simulation.rothermel_simulator import SimulationParams
    from src.simulation.rothermel_simulator import FireSpreadSimulator as FSS
    rows, cols = size
    sim_params = SimulationParams(grid_rows=rows, grid_cols=cols, cell_size_m=30.0)
    sim = FSS(sim_params)
    slope = stack[0]
    aspect_sin = stack[1]
    aspect_cos = stack[2]
    fuel = stack[3]
    wind_speed = stack[4,0,0]
    wind_sin = stack[5,0,0]
    wind_cos = stack[6,0,0]
    initial_fire = stack[7]
    slope_deg = slope * 45.0
    wind_direction_deg = np.degrees(np.arctan2(wind_sin, wind_cos)) % 360
    wind_speed_m_s = wind_speed * 20.0
    fire_state = (initial_fire > 0.5).astype(np.int32)
    for _ in range(10):
        if not np.any(fire_state==1):
            break
        fire_state = sim.step(fire_state, slope_deg, np.zeros_like(slope_deg), wind_speed_m_s, wind_direction_deg, fuel)
    return (fire_state>0).astype(np.float32)

# Build dataset
stacks = []
labels = []
print('Building pseudo-dataset...')
for i in range(n_samples):
    # random ignition: single point or small cluster
    init = np.zeros(size, dtype=np.float32)
    cx = np.random.randint(size[1]//4, 3*size[1]//4)
    cy = np.random.randint(size[0]//4, 3*size[0]//4)
    rr = np.random.randint(1,4)
    for dy in range(-rr, rr+1):
        for dx in range(-rr, rr+1):
            x = np.clip(cx+dx, 0, size[1]-1)
            y = np.clip(cy+dy, 0, size[0]-1)
            init[y,x]=1.0
    try:
        stack = build_real_input_stack(bbox[0], bbox[1], bbox[2], bbox[3], size, initial_fire=init, dem_source='3dep', fetch_wind_from_nws=True)
    except Exception:
        # fallback to srtm if 3dep fails
        stack = build_real_input_stack(bbox[0], bbox[1], bbox[2], bbox[3], size, initial_fire=init, dem_source='srtm', fetch_wind_from_nws=False)
    label = compute_baseline_label(stack)
    stacks.append(stack)
    labels.append(label)

X = np.stack(stacks).astype(np.float32)  # (N, 8, H, W)
Y = np.stack(labels)[:, None, :, :].astype(np.float32)

# Convert to torch dataset
tensor_x = torch.from_numpy(X)
tensor_y = torch.from_numpy(Y)
dataset = TensorDataset(tensor_x, tensor_y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load pretrained model
ckpt_path = ROOT / 'outputs' / 'checkpoints' / 'unet_final.pt'
if not ckpt_path.exists():
    print('Checkpoint not found:', ckpt_path)
    sys.exit(1)
ckpt = torch.load(ckpt_path, map_location=device)
model = build_unet(in_channels=8, out_channels=1, dropout_rate=0.2).to(device)
model.load_state_dict(ckpt['model_state_dict'])

# Fine-tune
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
criterion = nn.BCEWithLogitsLoss()

size = (128, 128)

# Moderate dataset for better fine-tuning (may take several minutes on CPU)
n_samples = 80

epochs = 3
print('Fine-tuning for', epochs, 'epochs on pseudo-labels...')
for epoch in range(epochs):
    model.train()
    total=0.0
    for xb, yb in loader:
        xb=xb.to(device)
        yb=yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f'Epoch {epoch+1}/{epochs} loss={total/len(loader):.4f}')

# Save fine-tuned model
out_ckpt = ROOT / 'outputs' / 'checkpoints' / 'unet_finetune_camp2018.pt'
torch.save({'epoch':epochs,'model_state_dict':model.state_dict()}, out_ckpt)
print('Saved fine-tuned checkpoint to', out_ckpt)
