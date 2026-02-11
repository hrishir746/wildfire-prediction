#!/usr/bin/env python3
"""
Quick test: How well does the model predict spread from SMALL ignitions?
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = build_unet(in_channels=8, out_channels=1, dropout_rate=0.3).to(DEVICE)
checkpoint = torch.load('outputs/checkpoints/unet_dense_firms.pt', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load data
data = np.load('data/dense_firms_training/dense_firms_fast_256.npz')
X = torch.from_numpy(data['X']).float()
Y = torch.from_numpy(data['Y']).float()

# Find small ignition fires
initial_sizes = X[:, 7, :, :].sum(dim=(1,2)).numpy()
small_mask = initial_sizes < 1000
large_mask = initial_sizes >= 3000

small_indices = np.where(small_mask)[0]
large_indices = np.where(large_mask)[0]

print("="*80)
print("TESTING: Small vs Large Ignition Performance")
print("="*80)

# Test function
def test_fires(indices, name):
    ious = []
    for idx in indices:
        x = X[idx].unsqueeze(0).to(DEVICE)
        y = Y[idx].to(DEVICE)
        
        with torch.no_grad():
            pred = torch.sigmoid(model(x))[0]
            pred_binary = (pred > 0.5).float()
        
        intersection = (pred_binary * y).sum()
        union = pred_binary.sum() + y.sum() - intersection
        iou = (intersection / (union + 1e-7)).item()
        ious.append(iou)
    
    return ious

print(f"\nTesting {len(small_indices)} SMALL ignition fires (<1000 pixels)...")
small_ious = test_fires(small_indices, "Small")

print(f"Testing {len(large_indices)} LARGE ignition fires (>3000 pixels)...")
large_ious = test_fires(large_indices, "Large")

# Results
print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")

print(f"\nSMALL Ignition (<1000 pixels):")
print(f"  Samples: {len(small_ious)}")
print(f"  Avg IoU: {np.mean(small_ious):.4f} ({np.mean(small_ious)*100:.2f}%)")
print(f"  Min IoU: {np.min(small_ious):.4f}")
print(f"  Max IoU: {np.max(small_ious):.4f}")
print(f"  Std Dev: ±{np.std(small_ious):.4f}")

print(f"\nLARGE Ignition (>3000 pixels):")
print(f"  Samples: {len(large_ious)}")
print(f"  Avg IoU: {np.mean(large_ious):.4f} ({np.mean(large_ious)*100:.2f}%)")
print(f"  Min IoU: {np.min(large_ious):.4f}")
print(f"  Max IoU: {np.max(large_ious):.4f}")
print(f"  Std Dev: ±{np.std(large_ious):.4f}")

# Comparison
diff = np.mean(small_ious) - np.mean(large_ious)
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

if abs(diff) < 0.03:
    print(f"\n✅ NO SIGNIFICANT DIFFERENCE ({diff*100:+.1f}%)")
    print("   → Model works equally well on small and large ignitions!")
    print("   → Good for early fire detection!")
elif diff > 0:
    print(f"\n🎉 SMALL fires are BETTER! ({diff*100:+.1f}%)")
    print("   → Model excels at predicting from small ignitions!")
else:
    print(f"\n⚠️  LARGE fires are better ({abs(diff)*100:.1f}% higher)")
    print("   → Model needs more context from existing fire")

# Visualize best small ignition
best_small = small_indices[np.argmax(small_ious)]
worst_small = small_indices[np.argmin(small_ious)]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fire_cmap = LinearSegmentedColormap.from_list('fire', ['black', 'red', 'orange', 'yellow'])

for row, (idx, label) in enumerate([(best_small, 'BEST'), (worst_small, 'WORST')]):
    x = X[idx]
    y = Y[idx]
    
    with torch.no_grad():
        pred = torch.sigmoid(model(x.unsqueeze(0).to(DEVICE)))[0].cpu()
    
    initial = x[7].numpy()
    truth = y[0].numpy()
    prediction = pred[0].numpy()
    pred_binary = (prediction > 0.5)
    
    iou = small_ious[np.where(small_indices == idx)[0][0]]
    
    axes[row, 0].imshow(initial, cmap='Reds')
    axes[row, 0].set_title(f'Initial ({initial.sum():.0f} px)', fontsize=10)
    axes[row, 0].axis('off')
    
    axes[row, 1].imshow(truth, cmap=fire_cmap)
    axes[row, 1].set_title(f'Ground Truth ({truth.sum():.0f} px)', fontsize=10)
    axes[row, 1].axis('off')
    
    axes[row, 2].imshow(prediction, cmap=fire_cmap, vmin=0, vmax=1)
    axes[row, 2].set_title('Prediction', fontsize=10)
    axes[row, 2].axis('off')
    
    overlay = np.zeros((*truth.shape, 3))
    overlay[truth == 1, 1] = 1
    overlay[pred_binary, 0] = 1
    overlay[(truth == 1) & pred_binary] = [1, 1, 0]
    
    axes[row, 3].imshow(overlay)
    axes[row, 3].set_title(f'{label}: IoU={iou:.2%}', fontsize=10)
    axes[row, 3].axis('off')

plt.suptitle('Small Ignition Examples (<1000 pixels)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/small_ignition_test.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization: outputs/small_ignition_test.png")

print(f"\n{'='*80}")
print("BOTTOM LINE")
print(f"{'='*80}")
print(f"\nYour model achieves {np.mean(small_ious)*100:.2f}% IoU on small ignition fires!")

if np.mean(small_ious) > 0.35:
    print("✅ This is EXCELLENT for early fire prediction!")
    print("   Model can predict spread from just a few starting pixels.")
else:
    print("⚠️  Model struggles with small ignitions.")
    print("   Consider training specifically on early-stage fires.")

print(f"{'='*80}")
