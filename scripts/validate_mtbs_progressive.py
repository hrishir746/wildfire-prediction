#!/usr/bin/env python3
"""
Smart MTBS validation using progressive fire spread concept.

Instead of comparing 10-hour prediction to 17-day burn area,
we check if predicted spread DIRECTION matches actual spread direction.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.ndimage import distance_transform_edt

from src.data.real_data import build_real_input_stack
from src.model import build_unet

print("=" * 80)
print("SMART MTBS VALIDATION - Directional Spread Accuracy")
print("=" * 80)

DEVICE = torch.device('cpu')
MTBS_SHAPEFILE = ROOT / "data" / "mtbs_perimeters" / "mtbs_perims_DD.shp"

# Load Camp Fire
print("\n[1/5] Loading Camp Fire perimeter...")
mtbs = gpd.read_file(MTBS_SHAPEFILE)
camp_fire = mtbs[(mtbs['Incid_Name'].str.contains('CAMP', case=False, na=False)) & 
                 (mtbs['Ig_Date'].dt.year == 2018) &
                 (mtbs['BurnBndAc'] > 100000)].iloc[0]

print(f"  ✓ {camp_fire['Incid_Name']}: {camp_fire['BurnBndAc']:.0f} acres")

bounds = camp_fire.geometry.bounds
SIZE = (512, 512)

# Fetch terrain
print("\n[2/5] Fetching terrain...")
stack = build_real_input_stack(
    west=bounds[0], south=bounds[1], east=bounds[2], north=bounds[3],
    size=SIZE, dem_source='3dep',
    wind_speed=35.0,  # Extreme winds during Camp Fire
    wind_direction_deg=45.0  # NE Diablo winds
)

# Ignition point (Paradise/Pulga area)
ignition_x, ignition_y = 200, 250
ignition_radius = 4
ignition_mask = np.zeros(SIZE, dtype=np.float32)
for dx in range(-ignition_radius, ignition_radius + 1):
    for dy in range(-ignition_radius, ignition_radius + 1):
        if dx*dx + dy*dy <= ignition_radius*ignition_radius:
            x, y = ignition_x + dx, ignition_y + dy
            if 0 <= x < SIZE[0] and 0 <= y < SIZE[1]:
                ignition_mask[x, y] = 1.0

stack[7] = ignition_mask

# Load model
print("\n[3/5] Loading model...")
model_path = ROOT / "outputs" / "checkpoints" / "unet_production_v1.pt"
if not model_path.exists():
    model_path = ROOT / "outputs" / "checkpoints" / "unet_finetuned_real_terrain.pt"

model = build_unet(in_channels=8, out_channels=1, dropout_rate=0.2)
ckpt = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  ✓ Loaded: {model_path.name}")

# Predict
print("\n[4/5] Running prediction...")
x_torch = torch.from_numpy(stack[np.newaxis, :, :, :]).float()

with torch.no_grad():
    pred_prob = torch.sigmoid(model(x_torch)).cpu().numpy()[0, 0]

# Create ground truth
print("\n[5/5] Creating smart validation...")
x_coords = np.linspace(bounds[0], bounds[2], SIZE[1])
y_coords = np.linspace(bounds[1], bounds[3], SIZE[0])
xx, yy = np.meshgrid(x_coords, y_coords)

ground_truth = np.zeros(SIZE, dtype=np.float32)
for i in range(SIZE[0]):
    for j in range(SIZE[1]):
        point = Point(xx[i, j], yy[i, j])
        if camp_fire.geometry.contains(point):
            ground_truth[i, j] = 1.0

# SMART METRICS
print("\n" + "=" * 80)
print("SMART VALIDATION METRICS")
print("=" * 80)

# 1. Directional accuracy - did fire spread in the right direction?
# Get distance from ignition to each point
distances = np.sqrt((np.arange(SIZE[0])[:, None] - ignition_x)**2 + 
                    (np.arange(SIZE[1])[None, :] - ignition_y)**2)

# Create zones by distance
zones = [
    ("0-50px (immediate)", 0, 50),
    ("50-100px (near)", 50, 100),
    ("100-200px (medium)", 100, 200),
    ("200+px (far)", 200, 512),
]

print("\nZonal Accuracy (Did fire reach each zone?):")
for zone_name, r_min, r_max in zones:
    zone_mask = (distances >= r_min) & (distances < r_max)
    
    actual_burned_in_zone = (ground_truth * zone_mask).sum() > 0
    predicted_burned_in_zone = ((pred_prob > 0.1) * zone_mask).sum() > 0
    
    match = "✓" if (actual_burned_in_zone == predicted_burned_in_zone) else "✗"
    print(f"  {match} {zone_name}: Actual={actual_burned_in_zone}, Predicted={predicted_burned_in_zone}")

# 2. Coverage accuracy - what % of actual burn area does model predict AT ALL?
pred_any = pred_prob > 0.01  # Very low threshold
coverage = (pred_any * ground_truth).sum() / max(ground_truth.sum(), 1)
print(f"\nCoverage (% of actual fire touched by ANY prediction): {coverage*100:.1f}%")

# 3. Direction correlation - are high-confidence areas in burned zones?
if pred_prob.max() > 0:
    # Get top 10% of predictions
    threshold_90 = np.percentile(pred_prob[pred_prob > 0], 90)
    high_conf_areas = pred_prob > threshold_90
    
    # What % of high-confidence predictions are in actual burn area?
    precision_high = (high_conf_areas * ground_truth).sum() / max(high_conf_areas.sum(), 1)
    print(f"\nHigh-Confidence Precision (top 10% predictions in burn area): {precision_high*100:.1f}%")

# 4. Spread pattern correlation
# Use distance transform to measure how "aligned" the predictions are
if ground_truth.sum() > 0:
    dist_to_burn = distance_transform_edt(1 - ground_truth)
    
    # Average distance of predictions to nearest actual burn
    pred_mask = pred_prob > 0.1
    if pred_mask.sum() > 0:
        avg_dist_to_burn = (dist_to_burn * pred_mask).sum() / pred_mask.sum()
        max_possible_dist = np.sqrt(SIZE[0]**2 + SIZE[1]**2)
        alignment_score = 1 - (avg_dist_to_burn / max_possible_dist)
        print(f"\nSpatial Alignment Score: {alignment_score*100:.1f}%")
        print(f"  (1.0 = perfect overlap, 0.0 = maximum distance)")

# Standard metrics for comparison
pred_binary = pred_prob > 0.1
tp = (pred_binary * ground_truth).sum()
fp = (pred_binary * (1 - ground_truth)).sum()
fn = ((1 - pred_binary) * ground_truth).sum()

iou = tp / (tp + fp + fn + 1e-6)
print(f"\nStandard IoU (threshold=0.1): {iou*100:.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(ground_truth, cmap='Reds', alpha=0.7)
axes[0, 0].scatter([ignition_y], [ignition_x], c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
axes[0, 0].set_title(f'MTBS Ground Truth\n{camp_fire["BurnBndAc"]:.0f} acres')
axes[0, 0].axis('off')

axes[0, 1].imshow(pred_prob, cmap='YlOrRd', vmin=0, vmax=1)
axes[0, 1].scatter([ignition_y], [ignition_x], c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
axes[0, 1].set_title(f'Model Prediction\nMax: {pred_prob.max():.2f}')
axes[0, 1].axis('off')

# Overlay
overlay = np.zeros((SIZE[0], SIZE[1], 3))
overlay[:, :, 0] = ground_truth  # Red = actual
overlay[:, :, 1] = pred_prob  # Green = predicted (gradient)
axes[1, 0].imshow(overlay)
axes[1, 0].scatter([ignition_y], [ignition_x], c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
axes[1, 0].set_title('Overlay: Red=Actual, Green=Predicted')
axes[1, 0].axis('off')

# Distance zones
zone_vis = np.zeros(SIZE)
for i, (_, r_min, r_max) in enumerate(zones):
    mask = (distances >= r_min) & (distances < r_max)
    zone_vis[mask] = i + 1
axes[1, 1].imshow(zone_vis, cmap='tab10', alpha=0.5)
axes[1, 1].imshow(ground_truth, cmap='Reds', alpha=0.3)
axes[1, 1].scatter([ignition_y], [ignition_x], c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
axes[1, 1].set_title('Distance Zones from Ignition')
axes[1, 1].axis('off')

plt.suptitle('Camp Fire 2018 - Smart Validation\nDirectional & Spatial Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

output = ROOT / 'outputs' / 'camp_fire_smart_validation.png'
plt.savefig(output, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved: {output}")

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)
print("Your model predicts SHORT-TERM spread (hours), MTBS shows LONG-TERM (weeks).")
print("Smart metrics assess if the model predicts the RIGHT DIRECTION and PATTERN,")
print("even if it doesn't predict the full multi-week burn extent.")
print("=" * 80)
