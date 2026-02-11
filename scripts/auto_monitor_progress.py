#!/usr/bin/env python3
"""
Autonomous monitoring script - checks progress and reports status.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import time
import json
from datetime import datetime

print("=" * 80)
print("AUTONOMOUS MONITORING SYSTEM")
print("=" * 80)

# Check what stage we're at
longterm_dataset = ROOT / "data" / "longterm_fires" / "longterm_dataset.npz"
longterm_model = ROOT / "outputs" / "checkpoints" / "unet_longterm_v1.pt"
metrics_file = ROOT / "outputs" / "longterm_training_metrics.json"

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking pipeline status...")

# Stage 1: Dataset generation
if not longterm_dataset.exists():
    print("\n📊 STAGE 1: Generating long-term dataset")
    print("  Status: IN PROGRESS")
    print("  Target: 200 samples, 150 timesteps")
    print("  Estimated time: 50-83 minutes")
    print("\n  Waiting for dataset generation to complete...")
else:
    file_size = longterm_dataset.stat().st_size / 1e6
    print(f"\n✅ STAGE 1: Dataset ready ({file_size:.1f} MB)")
    
    # Stage 2: Training
    if not longterm_model.exists():
        print("\n🔥 STAGE 2: Training on long-term data")
        print("  Status: READY TO START")
        print("  Will train for 40 epochs")
        print("  Estimated time: 2-3 hours")
    else:
        print(f"\n✅ STAGE 2: Model trained")
        
        # Stage 3: Validation
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            print(f"\n📈 Training Results:")
            print(f"  Best Validation IoU: {metrics['best_val_iou']:.4f}")
            print(f"  Training time: {metrics['training_time_minutes']:.1f} minutes")
            
        print(f"\n🎯 STAGE 3: MTBS Validation")
        print("  Status: READY")
        print("  Expected improvement: 0.6% → 30-60% IoU")

print("\n" + "=" * 80)
