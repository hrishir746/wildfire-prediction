# GPU Context for Rovo Dev - CRITICAL INFORMATION

## Hardware Setup Summary

**GPU**: AMD Radeon 9070 XT (16GB VRAM)
**NOT NVIDIA** - This is an AMD GPU using ROCm, NOT CUDA

## The Windows vs WSL2 Situation

### Windows Native Python (Current Issue)
- **Location**: `C:\Users\Hrishi\...` paths
- **GPU Access**: ❌ NONE - Cannot access AMD GPU
- **When you run**: `python scripts/test_full_pipeline_512.py`
  - Uses CPU only
  - Shows `Device: cpu`
  - Slow training (5-10+ minutes for integration test)
  - PyTorch detects `torch.cuda.is_available() = False`

### WSL2 Ubuntu 22.04 (HAS GPU ACCESS)
- **Location**: `/mnt/c/Users/Hrishi/...` paths (same files, accessed via WSL2)
- **GPU Access**: ✅ YES - Full AMD GPU access via ROCm 6.1
- **When you run**: `wsl python3 scripts/test_full_pipeline_512.py`
  - Uses AMD GPU
  - Shows `Device: cuda` (ROCm compatibility layer)
  - Fast training (2-3 minutes for integration test)
  - PyTorch detects `torch.cuda.is_available() = True`

## Why "cuda" for AMD GPU?

PyTorch with ROCm uses the **same API** as CUDA for compatibility:
- `torch.cuda.is_available()` → Returns `True` on ROCm
- `torch.cuda.get_device_name()` → Returns AMD GPU name
- `device = torch.device('cuda')` → Uses ROCm/HIP backend
- All `torch.cuda.*` functions work with ROCm

**This is intentional** - PyTorch abstracts the GPU backend so the same code works on NVIDIA (CUDA) or AMD (ROCm).

## Correct Commands for GPU Training

### ❌ WRONG (will use CPU):
```bash
python scripts/train_extended_gpu.py
python scripts/test_full_pipeline_512.py
streamlit run scripts/training_dashboard.py
```

### ✅ CORRECT (will use GPU):
```bash
wsl python3 scripts/train_extended_gpu.py
wsl python3 scripts/test_full_pipeline_512.py
wsl streamlit run scripts/training_dashboard.py
```

## Environment Variables (Already Set in WSL2)

These are configured in WSL2 environment:
```bash
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export HSA_OVERRIDE_GFX_VERSION=1100  # AMD GPU identifier
export HIP_VISIBLE_DEVICES=0           # Use first GPU
```

## Verification Commands

### Check if GPU is available in WSL2:
```bash
wsl python3 -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected Output:**
```
GPU Available: True
Device: AMD Radeon RX 9070 XT
```

### Check GPU memory in WSL2:
```bash
wsl rocm-smi
```

## Training Workflow (GPU-Enabled)

### 1. Quick Integration Test (2-3 minutes on GPU):
```bash
wsl python3 scripts/run_training_safe.py test_full_pipeline_512
```

### 2. Full Training with Dashboard:

**Terminal 1 (Training):**
```bash
wsl python3 scripts/run_training_safe.py train_with_dashboard
```

**Terminal 2 (Dashboard - can run on Windows):**
```bash
streamlit run scripts/training_dashboard.py
```

### 3. Extended Training (50+ epochs):
```bash
wsl python3 scripts/run_training_safe.py train_extended_gpu
```

## Memory Constraints

- **Batch Size**: 2-4 for 512×512 patches (16GB VRAM limit)
- **Resolution**: 512×512 = 15.36km × 15.36km coverage
- **Model Size**: 31M parameters (U-Net)

If OOM errors occur:
- Reduce batch size to 2
- Enable gradient checkpointing
- Use mixed precision (FP16)

## File Paths - IMPORTANT

The same files are accessible from both environments:

**Windows Path:**
```
C:\Users\Hrishi\Documents\wildfire-prediction\scripts\train_extended_gpu.py
```

**WSL2 Path (same file):**
```
/mnt/c/Users/Hrishi/Documents/wildfire-prediction/scripts/train_extended_gpu.py
```

When using `wsl` command from Windows PowerShell, it automatically handles path conversion.

## UTF-8 Encoding (Already Fixed)

All scripts have UTF-8 guards at the top:
```python
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
```

Plus, there's a safe wrapper:
```bash
wsl python3 scripts/run_training_safe.py <script_name>
```

## What the Integration Test Just Did (on CPU)

The user ran:
```bash
python scripts/test_full_pipeline_512.py
```

**What happened:**
- ✅ Test completed successfully
- ✅ Generated synthetic 512×512 samples
- ✅ Trained model for 2 epochs
- ✅ Validated on 3 samples
- ✅ Reported metrics
- ❌ BUT ran on **CPU** (slow, ~5-10 minutes)

**To run the same test on GPU (much faster):**
```bash
wsl python3 scripts/run_training_safe.py test_full_pipeline_512
```

## Key Takeaway for Rovo Dev

**ALWAYS use `wsl python3` for training/GPU tasks**

When the user asks to:
- "train the model"
- "run on GPU"
- "test with GPU"
- "start training"
- "evaluate on GPU"

**Use this pattern:**
```bash
wsl python3 scripts/run_training_safe.py <script_name>
```

**NOT:**
```bash
python scripts/<script_name>.py
```

## Dashboard Access

The Streamlit dashboard (`training_dashboard.py`) reads `outputs/training_metrics.json` which is written by the training script.

- Training script: MUST run in WSL2 (`wsl python3 scripts/train_with_dashboard.py`)
- Dashboard: CAN run on Windows (`streamlit run scripts/training_dashboard.py`)

Both access the same files since it's the same filesystem.

## ROCm vs CUDA - Technical Details

| Aspect | NVIDIA GPU | AMD GPU (This System) |
|--------|------------|----------------------|
| Hardware API | CUDA | ROCm/HIP |
| PyTorch Backend | torch.cuda | torch.cuda (ROCm) |
| tensor.cuda() | Uses CUDA | Uses ROCm/HIP |
| torch.cuda.is_available() | Checks CUDA | Checks ROCm |
| Drivers | NVIDIA drivers | ROCm 6.1 |
| Memory Check | nvidia-smi | rocm-smi |

The code is **identical** - PyTorch handles the backend automatically.

## Next Steps for Full Training

1. **Verify GPU in WSL2** (30 seconds):
   ```bash
   wsl python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Run GPU integration test** (2-3 minutes):
   ```bash
   wsl python3 scripts/run_training_safe.py test_full_pipeline_512
   ```

3. **Start full training with dashboard** (hours):
   ```bash
   # Terminal 1
   wsl python3 scripts/run_training_safe.py train_with_dashboard
   
   # Terminal 2
   streamlit run scripts/training_dashboard.py
   ```

4. **Monitor training**:
   - Check dashboard for loss/metrics curves
   - Watch GPU memory: `wsl watch -n 1 rocm-smi`
   - Check logs in `outputs/training_metrics.json`

## Common Mistakes to Avoid

1. ❌ Running `python` instead of `wsl python3`
   - Result: Uses CPU, very slow

2. ❌ Forgetting `scripts/` prefix
   - Wrong: `wsl python3 train_extended_gpu.py`
   - Right: `wsl python3 scripts/train_extended_gpu.py`

3. ❌ Assuming NVIDIA GPU commands work
   - Wrong: `nvidia-smi`
   - Right: `wsl rocm-smi`

4. ❌ Not using the safe wrapper
   - Risky: `wsl python3 scripts/train_extended_gpu.py`
   - Safe: `wsl python3 scripts/run_training_safe.py train_extended_gpu`

## Summary for Quick Reference

**Hardware:** AMD 9070 XT (16GB VRAM), ROCm 6.1, WSL2 Ubuntu 22.04

**Training Command Template:**
```bash
wsl python3 scripts/run_training_safe.py <script_name>
```

**Available Scripts:**
- `test_full_pipeline_512` - Integration test
- `train_extended_gpu` - Full training (50 epochs)
- `train_with_dashboard` - Training with live dashboard
- `train_gpu_optimized` - Optimized variant
- `train_fast_production` - Fast production training
- `quick_demo` - Quick demonstration

**GPU Check:**
```bash
wsl python3 -c "import torch; print(torch.cuda.is_available())"
```

**Always remember:** The user has an AMD GPU, not NVIDIA. But PyTorch uses the same `torch.cuda` API for both. All GPU operations must run in WSL2, not Windows Python.
