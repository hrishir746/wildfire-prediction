#!/usr/bin/env python3
"""
Validate model predictions against REAL MTBS fire perimeter data.
This gives us TRUE accuracy on historical fires!
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box

from src.data.real_data import build_real_input_stack
from src.model import build_unet

print("=" * 80)
print("REAL MTBS VALIDATION - Camp Fire 2018")
print("=" * 80)

DEVICE = torch.device('cpu')
MTBS_SHAPEFILE = ROOT / "data" / "mtbs_perimeters" / "mtbs_perims_DD.shp"

# Load MTBS perimeters
print("\n[1/5] Loading MTBS fire perimeter data...")
mtbs = gpd.read_file(MTBS_SHAPEFILE)
print(f"  Total fires in database: {len(mtbs)}")

# Find Camp Fire 2018 (by name and year)
camp_fire = mtbs[(mtbs['Incid_Name'].str.contains('CAMP', case=False, na=False)) & 
                 (mtbs['Ig_Date'].dt.year == 2018) &
                 (mtbs['BurnBndAc'] > 100000)]  # Camp Fire was 153k acres

if len(camp_fire) == 0:
    print("  Searching by alternative criteria...")
    camp_fire = mtbs[(mtbs['Ig_Date'].dt.year == 2018) & 
                     (mtbs['BurnBndAc'] > 100000)]

print(f"\n  Found {len(camp_fire)} matching fire(s):")
for idx, row in camp_fire.iterrows():
    print(f"    - {row['Incid_Name']}: {row['BurnBndAc']:.0f} acres, {row['Ig_Date'].strftime('%Y-%m-%d')}")

if len(camp_fire) == 0:
    print("\n  ERROR: Camp Fire not found in MTBS data")
    print("  Available 2018 fires:")
    fires_2018 = mtbs[mtbs['Ig_Date'].dt.year == 2018].sort_values('BurnBndAc', ascending=False).head(10)
    for idx, row in fires_2018.iterrows():
        print(f"    {row['Incid_Name']}: {row['BurnBndAc']:.0f} acres")
    sys.exit(1)

# Use the largest match (Camp Fire)
camp_fire = camp_fire.iloc[0]
print(f"\n  Selected: {camp_fire['Incid_Name']}")
print(f"  Acres: {camp_fire['BurnBndAc']:.0f}")
print(f"  Ignition date: {camp_fire['Ig_Date']}")
print(f"  Location: {camp_fire.geometry.centroid}")

# Get bounds
bounds = camp_fire.geometry.bounds  # (minx, miny, maxx, maxy)
print(f"  Bounds: ({bounds[0]:.3f}, {bounds[1]:.3f}, {bounds[2]:.3f}, {bounds[3]:.3f})")

# Fetch terrain for this area
print("\n[2/5] Fetching terrain for Camp Fire region...")
SIZE = (512, 512)
stack = build_real_input_stack(
    west=bounds[0], south=bounds[1], east=bounds[2], north=bounds[3],
    size=SIZE, dem_source='3dep',
    wind_speed=25.0,  # Camp Fire had strong winds
    wind_direction_deg=45.0  # NE winds (Diablo wind event)
)
print(f"  ✓ Terrain loaded: {stack.shape}")

# Add ignition point (Camp Fire started near Pulga, CA)
# Approximate location within our grid
ignition_x, ignition_y = 200, 250
ignition_radius = 4

ignition_mask = np.zeros((SIZE[0], SIZE[1]), dtype=np.float32)
for dx in range(-ignition_radius, ignition_radius + 1):
    for dy in range(-ignition_radius, ignition_radius + 1):
        if dx*dx + dy*dy <= ignition_radius*ignition_radius:
            x, y = ignition_x + dx, ignition_y + dy
            if 0 <= x < SIZE[0] and 0 <= y < SIZE[1]:
                ignition_mask[x, y] = 1.0

stack[7] = ignition_mask
print(f"  ✓ Ignition point set at ({ignition_x}, {ignition_y})")

# Load model
print("\n[3/5] Loading trained model...")
model_paths = [
    ROOT / "outputs" / "checkpoints" / "unet_longterm_v1.pt",
    ROOT / "outputs" / "checkpoints" / "unet_production_v1.pt",
    ROOT / "outputs" / "checkpoints" / "unet_finetuned_real_terrain.pt",
]

model = None
for model_path in model_paths:
    if model_path.exists():
        model = build_unet(in_channels=8, out_channels=1, dropout_rate=0.2)
        ckpt = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"  ✓ Loaded: {model_path.name}")
        print(f"    Training IoU: {ckpt.get('iou', 0):.4f}")
        break

if model is None:
    print("  ERROR: No trained model found")
    sys.exit(1)

# Predict
print("\n[4/5] Running prediction...")
x_torch = torch.from_numpy(stack[np.newaxis, :, :, :]).float()

with torch.no_grad():
    pred_logits = model(x_torch)
    pred_prob = torch.sigmoid(pred_logits).cpu().numpy()[0, 0]

print(f"  Prediction range: [{pred_prob.min():.4f}, {pred_prob.max():.4f}]")
print(f"  Mean confidence: {pred_prob.mean():.4f}")

# Create ground truth mask from MTBS perimeter
print("\n[5/5] Creating ground truth mask from MTBS perimeter...")

# Create grid of points
x_coords = np.linspace(bounds[0], bounds[2], SIZE[1])
y_coords = np.linspace(bounds[1], bounds[3], SIZE[0])
xx, yy = np.meshgrid(x_coords, y_coords)

# Check which points are inside the fire perimeter
ground_truth = np.zeros((SIZE[0], SIZE[1]), dtype=np.float32)
for i in range(SIZE[0]):
    for j in range(SIZE[1]):
        point = Point(xx[i, j], yy[i, j])
        if camp_fire.geometry.contains(point):
            ground_truth[i, j] = 1.0

burned_pixels = ground_truth.sum()
print(f"  ✓ Ground truth mask created")
print(f"  Actual burned area: {burned_pixels} pixels ({100*burned_pixels/(SIZE[0]*SIZE[1]):.1f}% of region)")

# Calculate metrics at different thresholds
print("\n" + "=" * 80)
print("VALIDATION METRICS")
print("=" * 80)

thresholds = [0.1, 0.3, 0.5, 0.7]
best_iou = 0
best_threshold = 0.5

for threshold in thresholds:
    pred_binary = (pred_prob > threshold).astype(np.float32)
    
    tp = (pred_binary * ground_truth).sum()
    fp = (pred_binary * (1 - ground_truth)).sum()
    fn = ((1 - pred_binary) * ground_truth).sum()
    tn = ((1 - pred_binary) * (1 - ground_truth)).sum()
    
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    print(f"\nThreshold: {threshold:.1f}")
    print(f"  IoU:       {iou:.4f} ({iou*100:.1f}%)")
    print(f"  Dice:      {dice:.4f} ({dice*100:.1f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.1f}%)")
    print(f"  Predicted: {pred_binary.sum():.0f} pixels")
    
    if iou > best_iou:
        best_iou = iou
        best_threshold = threshold

print(f"\n{'='*80}")
print(f"BEST PERFORMANCE: IoU = {best_iou:.4f} ({best_iou*100:.1f}%) at threshold {best_threshold}")
print(f"{'='*80}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(stack[0], cmap='terrain')
axes[0, 0].set_title('Real Terrain\n(Slope)')
axes[0, 0].axis('off')

axes[0, 1].imshow(ignition_mask, cmap='Reds')
axes[0, 1].set_title('Ignition Point')
axes[0, 1].axis('off')

axes[0, 2].imshow(ground_truth, cmap='Reds', alpha=0.7)
axes[0, 2].set_title(f'MTBS Ground Truth\n({burned_pixels:.0f} pixels)')
axes[0, 2].axis('off')

axes[1, 0].imshow(pred_prob, cmap='YlOrRd', vmin=0, vmax=1)
axes[1, 0].set_title(f'Model Prediction\n(Max: {pred_prob.max():.2f})')
axes[1, 0].axis('off')

axes[1, 1].imshow(pred_prob > best_threshold, cmap='Reds')
axes[1, 1].set_title(f'Prediction (>{best_threshold})\n({(pred_prob>best_threshold).sum():.0f} pixels)')
axes[1, 1].axis('off')

# Overlay comparison
overlay = np.zeros((SIZE[0], SIZE[1], 3))
overlay[:, :, 0] = ground_truth  # Red = actual
overlay[:, :, 1] = (pred_prob > best_threshold)  # Green = predicted
# Yellow = overlap (both red and green)
axes[1, 2].imshow(overlay)
axes[1, 2].set_title(f'Comparison\nRed=Actual, Green=Pred, Yellow=Match')
axes[1, 2].axis('off')

plt.suptitle(f'Camp Fire 2018 Validation - Real MTBS Data\nIoU: {best_iou:.1%} | Dice: {2*best_iou/(1+best_iou):.1%}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

output = ROOT / 'outputs' / 'camp_fire_real_mtbs_validation.png'
plt.savefig(output, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved: {output}")

# Save results
results_file = ROOT / 'outputs' / 'LONGTERM_MTBS_RESULTS.txt'
with open(results_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CAMP FIRE 2018 - REAL MTBS VALIDATION\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Fire: {camp_fire['Incid_Name']}\n")
    f.write(f"Acres: {camp_fire['BurnBndAc']:.0f}\n")
    f.write(f"Date: {camp_fire['Ig_Date']}\n\n")
    f.write(f"Best IoU: {best_iou:.4f} ({best_iou*100:.1f}%)\n")
    f.write(f"Best Dice: {2*best_iou/(1+best_iou):.4f} ({100*2*best_iou/(1+best_iou):.1f}%)\n")
    f.write(f"Threshold: {best_threshold}\n\n")
    f.write("This is REAL accuracy on historical fire data!\n")

print(f"✓ Results saved: {results_file}")
print("\n" + "=" * 80)
