#!/usr/bin/env python3
"""
COMPREHENSIVE MULTI-FIRE VALIDATION
Test model accuracy across 80%+ threshold on multiple major fire events.
Supports 512x512 patches for large-scale wildfire mapping.

UTF-8 encoding configured for Windows compatibility.
"""
import sys
import os
from pathlib import Path

# UTF-8 encoding fix for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from matplotlib.path import Path as MplPath

from src.data.real_data import build_real_input_stack
from src.simulation.rothermel_simulator import rothermel_fire_spread
from src.model import build_unet

print("="*70)
print("COMPREHENSIVE MULTI-FIRE VALIDATION")
print("Testing 80%+ accuracy on major wildfire events")
print("="*70)

# Load the extended trained model
ckpt_path = ROOT / 'outputs' / 'checkpoints' / 'unet_extended_256x256.pt'
if not ckpt_path.exists():
    print(f"ERROR: Checkpoint not found at {ckpt_path}")
    print("Please run train_extended_gpu.py first")
    sys.exit(1)

print(f"\nLoading model: {ckpt_path.name}")
checkpoint = torch.load(ckpt_path, map_location='cpu')

model = build_unet(in_channels=8, out_channels=1)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)

print(f"  Device: {DEVICE}")
print(f"  Best epoch: {checkpoint['epoch']}")
print(f"  Training loss: {checkpoint['loss']:.6f}")

# Test fires with actual ground truth approximations
TEST_FIRES = [
    {
        'name': 'Camp Fire 2018',
        'bbox': (-121.3, 38.3, -120.8, 38.8),
        'fire_points': [
            (-121.35, 38.35), (-121.30, 38.35), (-121.25, 38.35),
            (-121.20, 38.35), (-121.18, 38.40), (-121.20, 38.45),
            (-121.25, 38.48), (-121.30, 38.50), (-121.35, 38.48),
            (-121.38, 38.43), (-121.38, 38.39), (-121.35, 38.35)
        ],
        'wind_speed': 15.0,
        'wind_direction': 270.0,
        'acres': 153336
    },
    {
        'name': 'Dixie Fire 2021',
        'bbox': (-121.7, 39.8, -121.0, 40.5),
        'fire_points': [
            (-121.55, 39.85), (-121.35, 39.85), (-121.15, 39.85),
            (-121.10, 39.95), (-121.15, 40.25), (-121.40, 40.45),
            (-121.60, 40.40), (-121.70, 40.10), (-121.65, 39.95), (-121.55, 39.85)
        ],
        'wind_speed': 12.0,
        'wind_direction': 180.0,
        'acres': 963309
    },
    {
        'name': 'Creek Fire 2020',
        'bbox': (-119.4, 36.8, -118.8, 37.4),
        'fire_points': [
            (-119.25, 36.85), (-119.05, 36.85), (-118.95, 36.95),
            (-118.90, 37.15), (-119.00, 37.35), (-119.25, 37.35),
            (-119.35, 37.15), (-119.30, 36.95), (-119.25, 36.85)
        ],
        'wind_speed': 14.0,
        'wind_direction': 270.0,
        'acres': 379895
    },
    {
        'name': 'August Complex 2020',
        'bbox': (-122.8, 38.5, -122.0, 39.3),
        'fire_points': [
            (-122.65, 38.60), (-122.35, 38.60), (-122.10, 38.70),
            (-122.05, 38.95), (-122.15, 39.20), (-122.50, 39.25),
            (-122.75, 39.05), (-122.75, 38.75), (-122.65, 38.60)
        ],
        'wind_speed': 11.0,
        'wind_direction': 270.0,
        'acres': 1032648
    },
    {
        'name': 'Woolsey Fire 2018',
        'bbox': (-118.8, 34.0, -118.3, 34.4),
        'fire_points': [
            (-118.65, 34.05), (-118.50, 34.05), (-118.35, 34.10),
            (-118.35, 34.25), (-118.50, 34.35), (-118.70, 34.35),
            (-118.75, 34.20), (-118.70, 34.10), (-118.65, 34.05)
        ],
        'wind_speed': 13.0,
        'wind_direction': 90.0,
        'acres': 96949
    }
]

SIZE = 512  # Updated to 512x512 for large-scale wildfire mapping
validation_results = []

print(f"\n{'='*70}")
print("VALIDATING ON TEST FIRES")
print(f"{'='*70}\n")

for fire_idx, fire_event in enumerate(TEST_FIRES, 1):
    print(f"\n[{fire_idx}/{len(TEST_FIRES)}] {fire_event['name']}")
    print("-" * 70)
    
    try:
        # Fetch real data
        print("  Fetching real environmental data...")
        stack = build_real_input_stack(
            west=fire_event['bbox'][0],
            south=fire_event['bbox'][1],
            east=fire_event['bbox'][2],
            north=fire_event['bbox'][3],
            size=(SIZE, SIZE),
            dem_source='3dep',
            wind_speed=fire_event['wind_speed'],
            wind_direction_deg=fire_event['wind_direction']
        )
        
        # Run inference
        print("  Running inference...")
        with torch.no_grad():
            x = torch.from_numpy(stack[None, :, :, :]).float().to(DEVICE)
            logits = model(x)
            pred_prob = torch.sigmoid(logits).cpu().numpy()[0, 0, :, :]
        
        pred_binary = (pred_prob > 0.5).astype(np.float32)
        
        # Load actual fire extent from polygon
        print("  Loading actual fire extent...")
        y_coords = np.linspace(fire_event['bbox'][1], fire_event['bbox'][3], SIZE)
        x_coords = np.linspace(fire_event['bbox'][0], fire_event['bbox'][2], SIZE)
        
        fire_path = MplPath(fire_event['fire_points'])
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        actual_fire = fire_path.contains_points(points).reshape((SIZE, SIZE)).astype(np.float32)
        
        # Compute metrics
        print("  Computing metrics...")
        tp = (pred_binary * actual_fire).sum()
        fp = pred_binary.sum() - tp
        fn = actual_fire.sum() - tp
        tn = (SIZE * SIZE) - (tp + fp + fn)
        
        union = ((pred_binary + actual_fire) > 0).sum()
        intersection = tp
        
        iou = float(intersection / (union + 1e-6))
        dice = float(2 * intersection / (pred_binary.sum() + actual_fire.sum() + 1e-6))
        precision = float(tp / (tp + fp + 1e-6))
        recall = float(tp / (tp + fn + 1e-6))
        f1 = float(2 * (precision * recall) / (precision + recall + 1e-6))
        accuracy = float((tp + tn) / (SIZE * SIZE))
        
        # Store results
        result = {
            'name': fire_event['name'],
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'metrics': {
                'tp': float(tp),
                'fp': float(fp),
                'fn': float(fn),
                'tn': float(tn),
                'predicted_pixels': float(pred_binary.sum()),
                'actual_pixels': float(actual_fire.sum())
            }
        }
        validation_results.append(result)
        
        # Check if 80%+ threshold met
        meets_threshold = (iou >= 0.80 and dice >= 0.80 and f1 >= 0.80)
        threshold_str = "[OK] MEETS 80% TARGET" if meets_threshold else "[FAIL] Below 80% target"
        
        print(f"\n  Results:")
        print(f"    IoU: {iou:.4f} {threshold_str if iou >= 0.80 else ''}")
        print(f"    Dice: {dice:.4f} {threshold_str if dice >= 0.80 else ''}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1 Score: {f1:.4f}")
        print(f"    Accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        validation_results.append({
            'name': fire_event['name'],
            'error': str(e)
        })

# ============================================================================
# SUMMARY & REPORTING
# ============================================================================
print(f"\n{'='*70}")
print("VALIDATION SUMMARY")
print(f"{'='*70}\n")

# Compute averages (for successful validations)
successful = [r for r in validation_results if 'error' not in r]
if successful:
    avg_iou = np.mean([r['iou'] for r in successful])
    avg_dice = np.mean([r['dice'] for r in successful])
    avg_precision = np.mean([r['precision'] for r in successful])
    avg_recall = np.mean([r['recall'] for r in successful])
    avg_f1 = np.mean([r['f1_score'] for r in successful])
    avg_accuracy = np.mean([r['accuracy'] for r in successful])
    
    meets_80_iou = sum(1 for r in successful if r['iou'] >= 0.80)
    meets_80_dice = sum(1 for r in successful if r['dice'] >= 0.80)
    meets_80_all = sum(1 for r in successful if r['iou'] >= 0.80 and r['dice'] >= 0.80 and r['f1_score'] >= 0.80)
    
    print("ACROSS ALL TEST FIRES:")
    print(f"  Average IoU: {avg_iou:.4f}")
    print(f"  Average Dice: {avg_dice:.4f}")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Average Recall: {avg_recall:.4f}")
    print(f"  Average F1: {avg_f1:.4f}")
    print(f"  Average Accuracy: {avg_accuracy:.4f}")
    
    print(f"\nFIRES MEETING 80%+ THRESHOLD:")
    print(f"  IoU >= 0.80: {meets_80_iou}/{len(successful)} fires")
    print(f"  Dice >= 0.80: {meets_80_dice}/{len(successful)} fires")
    print(f"  All metrics >= 0.80: {meets_80_all}/{len(successful)} fires")
    
    # Overall verdict
    overall_success = avg_iou >= 0.80 and avg_dice >= 0.80 and avg_f1 >= 0.80
    verdict = "[OK] SUCCESS - MODEL MEETS 80% ACCURACY TARGET" if overall_success else "[WARNING] Partial success - some metrics below 80%"
    print(f"\n{'*'*70}")
    print(f"  {verdict}")
    print(f"{'*'*70}")

# Save results to JSON
results_file = ROOT / 'outputs' / 'validation_results_extended.json'
with open(results_file, 'w') as f:
    json.dump(validation_results, f, indent=2)
print(f"\nResults saved: {results_file.name}")

# Create comparison visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

if successful:
    fire_names = [r['name'] for r in successful]
    
    metrics_data = {
        'IoU': [r['iou'] for r in successful],
        'Dice': [r['dice'] for r in successful],
        'Precision': [r['precision'] for r in successful],
        'Recall': [r['recall'] for r in successful],
        'F1 Score': [r['f1_score'] for r in successful],
        'Accuracy': [r['accuracy'] for r in successful]
    }
    
    for idx, (metric_name, metric_values) in enumerate(metrics_data.items()):
        ax = axes.flat[idx]
        colors = ['green' if v >= 0.80 else 'orange' for v in metric_values]
        bars = ax.bar(range(len(fire_names)), metric_values, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0.80, color='red', linestyle='--', linewidth=2, label='80% Target')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(fire_names)))
        ax.set_xticklabels([name.replace(' ', '\n') for name in fire_names], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Add value labels on bars
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle(f'Multi-Fire Validation - Model Performance Across {len(successful)} Major Wildfires\n256x256 Resolution, Extended Training',
            fontsize=13, fontweight='bold')
plt.tight_layout()

viz_file = ROOT / 'outputs' / 'validation_metrics_extended.png'
plt.savefig(viz_file, dpi=150, bbox_inches='tight')
print(f"Visualization saved: {viz_file.name}")

print(f"\n{'='*70}\n")
