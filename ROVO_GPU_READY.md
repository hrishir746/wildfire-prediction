# GPU Setup Complete - Rovo Dev Context

## Status: [OK] GPU Fully Functional

**Hardware & Driver Ready:**
- GPU: AMD Radeon RX 9070 XT (16GB VRAM)
- AMD Driver: 32.0.23017.1001 (Adrenalin 26.1.1)
- Windows 11 Pro 25H2

## What Was Done
1. Installed Python 3.12.10 (required for PyTorch wheels)
2. Created virtual environment: `.venv-gpu`
3. Installed ROCm SDK 7.2 (2.6 GB)
4. Installed PyTorch 2.9.1 + ROCm support
5. Installed all project dependencies (requirements.txt)

## Verified Working
```
GPU available: True
GPU count: 1
GPU name: AMD Radeon RX 9070 XT
GPU Memory Test: PASSED
PyTorch Version: 2.9.1+rocmsdk20260116
```

## How to Use

### Activate GPU Environment
```powershell
.\.venv-gpu\Scripts\Activate.ps1
```

### Run Training with GPU
```powershell
.\.venv-gpu\Scripts\Activate.ps1
python scripts/train_with_dashboard.py
```

### Quick GPU Test
```powershell
.\.venv-gpu\Scripts\Activate.ps1
python -c "import torch; print('GPU:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

## Environment Details
- Location: `C:\Users\Hrishi\Documents\wildfire-prediction\.venv-gpu`
- Python: 3.12.10
- PyTorch: 2.9.1+rocmsdk20260116
- Dependencies: Installed via requirements.txt
- Resolution: Scripts use 512×512 patches
- Batch Size: 2-4 (optimized for 16GB VRAM)

## Original Environment Still Available
- Default `python` command: Python 3.13 (CPU training)
- Use for data processing, non-GPU work

## Next Steps
- Run any training script with `.venv-gpu` activated
- GPU should provide 3-5× speedup vs CPU
- Monitor dashboard at http://localhost:8501 (when running train_with_dashboard.py)

---
**Created:** Feb 7, 2026
**Status:** Ready for production training
