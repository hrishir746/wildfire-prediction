#!/usr/bin/env python3
"""
GPU Verification Script
Run this in WSL2 to verify AMD GPU access
"""
import sys
import os

# UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("=" * 70)
print("GPU VERIFICATION")
print("=" * 70)
print()

# Check PyTorch
try:
    import torch
    print("[OK] PyTorch imported successfully")
    print(f"    Version: {torch.__version__}")
except Exception as e:
    print(f"[FAIL] Could not import PyTorch: {e}")
    sys.exit(1)

print()

# Check GPU availability
print("GPU Status:")
gpu_available = torch.cuda.is_available()
print(f"  Available: {gpu_available}")

if gpu_available:
    print(f"  Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  Device Count: {torch.cuda.device_count()}")
    
    # Memory info
    props = torch.cuda.get_device_properties(0)
    print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Try simple GPU operation
    print()
    print("Testing GPU operation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("  [OK] GPU computation successful")
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [FAIL] GPU operation failed: {e}")
else:
    print("  [WARNING] No GPU detected")
    print("  This script should be run in WSL2, not Windows")
    print("  Command: wsl python3 scripts/verify_gpu.py")

print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

if gpu_available:
    print()
    print("GPU is ready for training!")
    print()
    print("Next steps:")
    print("  1. Run integration test:")
    print("     python3 scripts/run_training_safe.py test_full_pipeline_512")
    print()
    print("  2. Start training:")
    print("     python3 scripts/run_training_safe.py train_with_dashboard")
else:
    print()
    print("No GPU found. Make sure you're running in WSL2.")
    print("In PowerShell: wsl python3 scripts/verify_gpu.py")
