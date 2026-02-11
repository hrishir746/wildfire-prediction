#!/usr/bin/env python3
"""
Test model on fires with SMALL ignition (few pixels) vs LARGE ignition.
Does the model work better when starting from a small fire?
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.model import build_unet

print("="*80)
print("TESTING: Small vs Large Ignition Points")
print("="*80)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = ROOT / "outputs" / "checkpoints" / "unet_dense_firms.pt"
DATASET_PATH = ROOT / "data" / "dense_firms_training" / "dense_firms_fast_256.npz"
OUTPUT_DIR = ROOT / "outputs" / "ignition_test"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load model
model = build_unet(in_channels=8, out_channels=1, dropout_rate=0.3).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load data
data = np.load(DATASET_PATH)
X = torch.from_numpy(data['X']).float()
Y = torch.from_numpy(data['Y']).float()

print(f"\nAnalyzing {len(X)} samples...")

# Categorize by initial fire size
initial_fire_sizes = X[:, 7, :, :].sum(dim=(1, 2)).numpy()

print(f"\nInitial fire sizes:")
print(f"  Min: {initial_fire_sizes.min():.0f} pixels")
print(f"  Max: {initial_fire_sizes.max():.0f} pixels")
print(f"  Mean: {initial_fire_sizes.mean():.0f} pixels")
print(f"  Median: {np.median(initial_fire_sizes):.0f} pixels")

# Define categories
small_threshold = np.percentile(initial_fire_sizes, 33)
large_threshold = np.percentile(initial_fire_sizes, 67)

small_fires = initial_fire_sizes < small_threshold
medium_fires = (initial_fire_sizes >= small_threshold) & (initial_fire_sizes < large_threshold)
large_fires = initial_fire_sizes >= large_threshold

print(f"\nCategories:")
print(f"  Small ignition (<{small_threshold:.0f} px): {small_fires.sum()} samples")
print(f"  Medium ignition: {medium_fires.sum()} samples")
print(f"  Large ignition (>{large_threshold:.0f} px): {large_fires.sum()} samples")

# Test on each category
def test_category(mask, name):
    X_cat = X[mask].to(DEVICE)
    Y_cat = Y[mask].to(DEVICE)
    
    if len(X_cat) == 0:
        return 0, []
    
    with torch.no_grad():
        preds = model(X_cat)
        pred_binary = (torch.sigmoid(preds) > 0.5).float()
    
    ious = []
    for i in range(len(X_cat)):
        intersection = (pred_binary[i] * Y_cat[i]).sum()
        union = pred_binary[i].sum() + Y_cat[i].sum() - intersection
        iou = (intersection / (union + 1e-7)).item()
        ious.append(iou)
    
    return np.mean(ious), ious

small_iou, small_ious = test_category(small_fires, "Small")
medium_iou, medium_ious = test_category(medium_fires, "Medium")
large_iou, large_ious = test_category(large_fires, "Large")

print(f"\n{'='*80}")
print(f"RESULTS BY INITIAL FIRE SIZE")
print(f"{'='*80}")
print(f"\nSmall ignition (<{small_threshold:.0f} px):  {small_iou:.4f} ({small_iou*100:.2f}% IoU)")
print(f"Medium ignition:                {medium_iou:.4f} ({medium_iou*100:.2f}% IoU)")
print(f"Large ignition (>{large_threshold:.0f} px): {large_iou:.4f} ({large_iou*100:.2f}% IoU)")

print(f"\nStandard deviations:")
print(f"  Small: ±{np.std(small_ious):.4f}")
print(f"  Medium: ±{np.std(medium_ious):.4f}")
print(f"  Large: ±{np.std(large_ious):.4f}")

# Find best small ignition example
if len(small_ious) > 0:
    best_small_idx = np.where(small_fires)[0][np.argmax(small_ious)]
    
    X_sample = X[best_small_idx].to(DEVICE)
    Y_sample = Y[best_small_idx].to(DEVICE)
    
    with torch.no_grad():
        pred = torch.sigmoid(model(X_sample.unsqueeze(0)))[0]
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fire_cmap = LinearSegmentedColormap.from_list('fire', ['black', 'red', 'orange', 'yellow'])
    
    initial = X_sample[7].cpu().numpy()
    truth = Y_sample[0].cpu().numpy()
    prediction = pred[0].cpu().numpy()
    
    axes[0].imshow(initial, cmap='Reds')
    axes[0].set_title(f'Initial Fire\n({initial.sum():.0f} pixels)', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(truth, cmap=fire_cmap)
    axes[1].set_title(f'Ground Truth (Next Day)\n({truth.sum():.0f} pixels)', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(prediction, cmap=fire_cmap, vmin=0, vmax=1)
    axes[2].set_title('Prediction', fontsize=12)
    axes[2].axis('off')
    
    # Overlay
    overlay = np.zeros((*truth.shape, 3))
    pred_binary = (prediction > 0.5)
    overlay[truth == 1, 1] = 1
    overlay[pred_binary, 0] = 1
    overlay[(truth == 1) & pred_binary] = [1, 1, 0]
    
    iou = small_ious[np.argmax(small_ious)]
    axes[3].imshow(overlay)
    axes[3].set_title(f'Overlay\nIoU: {iou:.2%}', fontsize=12)
    axes[3].axis('off')
    
    plt.suptitle(f'BEST Small Ignition Example - IoU: {iou:.4f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / f"small_ignition_example_iou{iou:.3f}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()

print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")

if small_iou > large_iou * 1.1:
    print("\n✅ Model works BETTER with small ignition!")
    print("   → This is ideal for early fire detection")
    print("   → Model can predict spread from just a few starting pixels")
elif large_iou > small_iou * 1.1:
    print("\n⚠️  Model works BETTER with large ignition")
    print("   → Model needs context from existing fire")
    print("   → Small ignitions are harder to predict (less spatial info)")
else:
    print("\n✓ Model works EQUALLY well regardless of ignition size")
    print("  → Robust to different starting conditions")
    print("  → Good generalization!")

print(f"\n{'='*80}")
print("ANSWER TO YOUR QUESTION")
print(f"{'='*80}")
print(f"\nYes, we can test on fires with small ignition!")
print(f"Small ignition fires (<{small_threshold:.0f} pixels) achieve {small_iou*100:.2f}% IoU")

if small_iou > 0.35:
    print(f"\n✅ This is EXCELLENT - model predicts spread from small fires!")
    print(f"   → Useful for early warning systems")
else:
    print(f"\n⚠️  Model struggles with small ignitions")
    print(f"   → Needs more context from larger fire perimeters")

print(f"{'='*80}")
