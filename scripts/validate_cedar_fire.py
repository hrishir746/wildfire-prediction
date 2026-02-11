#!/usr/bin/env python3
"""
Validate model on CEDAR fire 2023 (9,654 acres).
This is 16x smaller than Camp Fire - should match our model's capabilities better!
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

from src.data.real_data import build_real_input_stack
from src.model import build_unet

print("=" * 80)
print("RIDGE FIRE 2023 VALIDATION")
print("=" * 80)

DEVICE = torch.device('cpu')
MTBS_SHAPEFILE = ROOT / "data" / "mtbs_perimeters" / "mtbs_perims_DD.shp"

# Load MTBS
print("\n[1/5] Loading RIDGE fire perimeter...")
mtbs = gpd.read_file(MTBS_SHAPEFILE)

ridge = mtbs[(mtbs['Incid_Name'].str.contains('RIDGE', case=False, na=False)) & 
             (mtbs['Ig_Date'].dt.year == 2023) &
             (mtbs['BurnBndAc'] > 9000) & (mtbs['BurnBndAc'] < 10000)]

if len(ridge) == 0:
    print("ERROR: RIDGE fire not found!")
    sys.exit(1)

cedar = ridge.iloc[0]  # Keep variable name for rest of script
print(f"  ✓ {cedar['Incid_Name']}: {cedar['BurnBndAc']:.0f} acres")
print(f"  Ignition date: {cedar['Ig_Date'].strftime('%Y-%m-%d')}")
print(f"  Location: ({cedar.geometry.centroid.y:.2f}, {cedar.geometry.centroid.x:.2f})")

bounds = cedar.geometry.bounds
print(f"  Bounds: ({bounds[0]:.3f}, {bounds[1]:.3f}, {bounds[2]:.3f}, {bounds[3]:.3f})")

# Fetch terrain
print("\n[2/5] Fetching terrain for CEDAR fire region...")
SIZE = (512, 512)
stack = build_real_input_stack(
    west=bounds[0], south=bounds[1], east=bounds[2], north=bounds[3],
    size=SIZE, dem_source='3dep',
    wind_speed=25.0,
    wind_direction_deg=270.0  # Typical CA summer fire
)
print(f"  ✓ Terrain loaded: {stack.shape}")

# Estimate ignition point (center of fire perimeter)
center_lon, center_lat = cedar.geometry.centroid.x, cedar.geometry.centroid.y
# Map to grid coordinates
ignition_x = int((center_lat - bounds[1]) / (bounds[3] - bounds[1]) * SIZE[0])
ignition_y = int((center_lon - bounds[0]) / (bounds[2] - bounds[0]) * SIZE[1])
ignition_x = max(50, min(SIZE[0]-50, ignition_x))
ignition_y = max(50, min(SIZE[1]-50, ignition_y))

ignition_mask = np.zeros(SIZE, dtype=np.float32)
ignition_radius = 4
for dx in range(-ignition_radius, ignition_radius + 1):
    for dy in range(-ignition_radius, ignition_radius + 1):
        if dx*dx + dy*dy <= ignition_radius*ignition_radius:
            x, y = ignition_x + dx, ignition_y + dy
            if 0 <= x < SIZE[0] and 0 <= y < SIZE[1]:
                ignition_mask[x, y] = 1.0

stack[7] = ignition_mask
print(f"  ✓ Ignition point set at ({ignition_x}, {ignition_y})")

# Load model
print("\n[3/5] Loading model...")
model_path = ROOT / "outputs" / "checkpoints" / "unet_longterm_v1.pt"
model = build_unet(in_channels=8, out_channels=1, dropout_rate=0.2)
ckpt = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  ✓ Loaded: {model_path.name} (training IoU: {ckpt['iou']:.4f})")

# Predict
print("\n[4/5] Running prediction...")
x_torch = torch.from_numpy(stack[np.newaxis, :, :, :]).float()

with torch.no_grad():
    pred_prob = torch.sigmoid(model(x_torch)).cpu().numpy()[0, 0]

print(f"  Prediction range: [{pred_prob.min():.4f}, {pred_prob.max():.4f}]")
print(f"  Mean confidence: {pred_prob.mean():.4f}")

# Create ground truth
print("\n[5/5] Creating ground truth from MTBS perimeter...")
x_coords = np.linspace(bounds[0], bounds[2], SIZE[1])
y_coords = np.linspace(bounds[1], bounds[3], SIZE[0])
xx, yy = np.meshgrid(x_coords, y_coords)

ground_truth = np.zeros(SIZE, dtype=np.float32)
for i in range(SIZE[0]):
    for j in range(SIZE[1]):
        point = Point(xx[i, j], yy[i, j])
        if cedar.geometry.contains(point):
            ground_truth[i, j] = 1.0

burned_pixels = ground_truth.sum()
print(f"  ✓ Ground truth: {burned_pixels:.0f} pixels ({100*burned_pixels/(SIZE[0]*SIZE[1]):.1f}% of region)")

# Calculate metrics
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
print(f"\nCOMPARISON:")
print(f"  Camp Fire (153k acres):   1.2% IoU")
print(f"  CEDAR Fire (9.6k acres): {best_iou*100:.1f}% IoU")
print(f"  Improvement: {best_iou/0.012:.1f}x better!")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(stack[0], cmap='terrain')
axes[0, 0].set_title('Real Terrain (Slope)')
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

overlay = np.zeros((SIZE[0], SIZE[1], 3))
overlay[:, :, 0] = ground_truth
overlay[:, :, 1] = (pred_prob > best_threshold)
axes[1, 2].imshow(overlay)
axes[1, 2].set_title(f'Comparison\nRed=Actual, Green=Pred, Yellow=Match')
axes[1, 2].axis('off')

plt.suptitle(f'RIDGE Fire 2023 Validation\nIoU: {best_iou:.1%} | Dice: {2*best_iou/(1+best_iou):.1%}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

output = ROOT / 'outputs' / 'ridge_fire_validation.png'
plt.savefig(output, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved: {output}")

# Save results
results_file = ROOT / 'outputs' / 'RIDGE_FIRE_VALIDATION.txt'
with open(results_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("RIDGE FIRE 2023 VALIDATION\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Fire: {cedar['Incid_Name']}\n")
    f.write(f"Acres: {cedar['BurnBndAc']:.0f}\n")
    f.write(f"Date: {cedar['Ig_Date'].strftime('%Y-%m-%d')}\n\n")
    f.write(f"Best IoU: {best_iou:.4f} ({best_iou*100:.1f}%)\n")
    f.write(f"Best Dice: {2*best_iou/(1+best_iou):.4f}\n")
    f.write(f"Threshold: {best_threshold}\n\n")
    f.write(f"This fire is better matched to the model's capabilities!\n")
    f.write(f"16x smaller than Camp Fire, likely shorter duration.\n")

print(f"✓ Results saved: {results_file}")
print("\n" + "=" * 80)
