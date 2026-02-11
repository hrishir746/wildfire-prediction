#!/usr/bin/env python
"""
GPU detection and setup script - comprehensive check for all available acceleration.
Tests CUDA, HIP, DirectML, and CPU fallback.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import subprocess
import torch

print("="*70)
print("GPU Detection & Acceleration Setup for Wildfire Prediction")
print("="*70)

# Check PyTorch basic info
print(f"\n1. PyTorch Version: {torch.__version__}")
print(f"   Build: {torch.version.build}")

# Check CUDA
print(f"\n2. CUDA Status:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
print(f"   CUDA Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")

# Check HIP/ROCm
print(f"\n3. HIP/ROCm Status:")
try:
    result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   ROCm Available: YES")
        print(result.stdout[:200])
    else:
        print("   ROCm Available: NO (rocm-smi not found)")
except Exception as e:
    print(f"   ROCm Check Failed: {e}")

# Try DirectML
print(f"\n4. DirectML Status:")
try:
    import torch_directml
    print("   DirectML Available: YES")
    device = torch_directml.device()
    print(f"   Device: {device}")
except ImportError:
    print("   DirectML Available: NO (not installed)")
except Exception as e:
    print(f"   DirectML Error: {e}")

# Final device recommendation
print(f"\n5. Recommended Device:")
device = None
if torch.cuda.is_available():
    device = 'cuda'
    print(f"   Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = 'mps'
    print(f"   Using Apple Metal Performance Shaders")
else:
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"   Using DirectML GPU")
    except:
        device = 'cpu'
        print(f"   Using CPU (no GPU detected)")

print(f"\n{'='*70}")
print(f"Recommended training device: {device}")
print(f"{'='*70}\n")

# Test inference on recommended device
if device and device != 'cpu':
    print("Testing inference on GPU...")
    from src.model import build_unet
    
    model = build_unet(in_channels=8, out_channels=1)
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        x = torch.randn(1, 8, 64, 64, device=device)
        y = model(x)
    
    print(f"Test inference SUCCESS!")
    print(f"  Input: {x.shape} on {x.device}")
    print(f"  Output: {y.shape} on {y.device}")
else:
    print("GPU acceleration not available - CPU training will be slow")
