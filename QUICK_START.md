# Quick Start Guide - Wildfire Prediction System

## Getting Started in 5 Minutes

This guide will get you up and running with the wildfire prediction system quickly and safely.

---

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- For GPU training: WSL2 with ROCm (AMD) or CUDA (NVIDIA)

---

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Important: Using the Safe Wrapper

**To avoid UnicodeEncodeError on Windows**, always use the safe wrapper script:

```bash
# Instead of:
python scripts/train_model.py

# Use:
python scripts/run_training_safe.py train_model
```

The wrapper ensures proper UTF-8 encoding for Rovo Dev extension compatibility.

---

## Quick Demo (30 seconds)

Test that everything works:

```bash
python scripts/run_training_safe.py quick_demo
```

Expected output:
- System components check
- Rothermel simulator test
- Data augmentation test
- No Unicode errors

---

## Training Your First Model

### Option 1: Quick Training (5 minutes, CPU)

```bash
python scripts/run_training_safe.py train_model --epochs 5 --samples 200
```

### Option 2: GPU Training (WSL2 required)

```bash
wsl python3 scripts/run_training_safe.py train_extended_gpu
```

### Option 3: Training with Live Dashboard

**Terminal 1 - Start Training:**
```bash
python scripts/run_training_safe.py train_with_dashboard --epochs 50 --samples 1000
```

**Terminal 2 - Launch Dashboard:**
```bash
streamlit run scripts/training_dashboard.py
```

Then open: http://localhost:8501

---

## Available Scripts

List all available scripts:
```bash
python scripts/run_training_safe.py
```

### Training Scripts
- `train_model` - Basic training (synthetic data)
- `train_extended_gpu` - Extended GPU training (512x512)
- `train_gpu_optimized` - Optimized for 80%+ accuracy
- `train_fast_production` - Fast production training
- `train_with_dashboard` - Training with live monitoring
- `finetune_gpu` - Fine-tuning for specific regions

### Validation Scripts
- `validate_extended_model` - Multi-fire validation
- `validate_80_percent` - 80%+ accuracy validation
- `validate_camp_2018` - Camp Fire 2018 validation
- `wildfire_validation` - Real wildfire validation

### Evaluation Scripts
- `evaluate_model` - Evaluate on test set
- `evaluate_real_fires` - Real fire evaluation
- `quick_eval_gpu` - Quick GPU evaluation

### Demo/Test Scripts
- `quick_demo` - Quick system demo
- `comprehensive_demo` - Comprehensive demo
- `test_full_pipeline_512` - Integration test (512x512)

---

## Configuration Guide

### Patch Size: 512x512

All scripts now use **512x512 patches** by default for large-scale wildfire mapping.

**Benefits:**
- 4x more pixels than 256x256
- Better terrain detail
- Larger spatial context
- More accurate predictions

### Batch Size for GPU

**For 512x512 patches:**
- Batch size 2-4 (recommended)
- Do NOT exceed batch size 4 unless you have >16GB VRAM
- Out of memory? Reduce to batch size 2

**GPU Memory Usage:**
- Batch size 2: ~1 GB VRAM
- Batch size 4: ~2 GB VRAM
- Batch size 8: ~4 GB VRAM (may exceed limits)

### Training Time Estimates

**CPU (Windows native):**
- Quick demo: 10 seconds
- Integration test: 5-10 minutes
- Full training (50 epochs): 3-6 hours

**GPU (WSL2):**
- Quick demo: 5 seconds
- Integration test: 2-3 minutes
- Full training (50 epochs): 30-60 minutes

---

## Command Examples

### Basic Training
```bash
# Train for 20 epochs with 500 samples
python scripts/run_training_safe.py train_model --epochs 20 --samples 500
```

### GPU Training (WSL2)
```bash
# Extended training with 512x512 patches
wsl python3 scripts/run_training_safe.py train_extended_gpu

# Optimized training for 80%+ accuracy
wsl python3 scripts/run_training_safe.py train_gpu_optimized
```

### Validation
```bash
# Validate on multiple fire events
python scripts/run_training_safe.py validate_extended_model

# Validate on Camp Fire 2018
python scripts/run_training_safe.py validate_camp_2018
```

### Evaluation
```bash
# Evaluate model on test set
python scripts/run_training_safe.py evaluate_model

# Evaluate on real fire with MTBS data
python scripts/run_training_safe.py evaluate_real_fires --event camp_2018
```

### Integration Test
```bash
# Test entire pipeline with 512x512 patches
python scripts/run_training_safe.py test_full_pipeline_512
```

---

## Dashboard Monitoring

The live dashboard shows real-time training metrics:

**Features:**
- Training/validation loss curves
- IoU and Dice metrics
- Learning rate schedule
- GPU memory usage (if available)
- Validation sample predictions
- System information
- ETA for completion

**To use:**
1. Start training with `train_with_dashboard`
2. Launch dashboard: `streamlit run scripts/training_dashboard.py`
3. Open browser to http://localhost:8501
4. Watch metrics update in real-time

**GPU Memory Monitoring:**
- Allocated memory (currently used)
- Reserved memory (cached by PyTorch)
- Peak memory (maximum used)

---

## Troubleshooting

### UnicodeEncodeError

**Problem:**
```
UnicodeEncodeError: 'charmap' codec can't encode character...
```

**Solution:**
Always use the wrapper script:
```bash
python scripts/run_training_safe.py SCRIPT_NAME
```

**Why it happens:**
Rovo Dev extension spawns child processes with Windows-1252 encoding instead of UTF-8.

**More info:**
See `UNICODE_FIX_SOLUTION.md` for detailed explanation.

### Out of Memory (GPU)

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solution 1:** Reduce batch size
```bash
# Edit the script or use gradient accumulation
BATCH_SIZE = 2  # Instead of 4
```

**Solution 2:** Use smaller patch size (not recommended)
```bash
# Change SIZE to 256 instead of 512
```

**Solution 3:** Use CPU
```bash
# CPU has no memory limit (uses system RAM)
python scripts/run_training_safe.py train_model
```

### GPU Not Detected (WSL2)

**Problem:**
```
Device: cpu
```

**Solution:**
```bash
# Make sure you're running in WSL2, not Windows
wsl python3 scripts/SCRIPT_NAME

# Check GPU availability
wsl python3 -c "import torch; print(torch.cuda.is_available())"
```

**Expected:** `True`

**If False:**
- Check PyTorch has ROCm/CUDA support
- Verify GPU drivers installed in WSL2
- See `GPU_TRAINING_NEXT_STEPS.md`

### Slow Training

**Problem:** Training takes forever

**Causes:**
1. Running on CPU instead of GPU
2. Large patch size (512x512) on CPU
3. Too many samples

**Solutions:**
1. Use GPU in WSL2: `wsl python3 scripts/run_training_safe.py ...`
2. Reduce samples for quick test: `--samples 100`
3. Reduce epochs for testing: `--epochs 5`

### Dashboard Not Updating

**Problem:** Dashboard shows old metrics

**Solutions:**
1. Refresh browser (Ctrl+R)
2. Click "Rerun" button in dashboard
3. Check that training is actually running
4. Verify metrics file exists: `outputs/training_metrics.json`

---

## GPU Memory Considerations

### For 512x512 Patches

**Minimum VRAM:** 2 GB (batch size 2)
**Recommended VRAM:** 4 GB (batch size 4)
**Optimal VRAM:** 8+ GB (batch size 8+)

### Memory Usage by Component

- **Model:** ~100 MB
- **Single 512x512 sample:** ~32 MB
- **Batch of 4:** ~500 MB
- **Gradients + optimizer:** ~500 MB
- **Total (batch 4):** ~2 GB

### Monitoring GPU Memory

**In Dashboard:**
- Real-time memory usage displayed
- Allocated, reserved, and peak memory

**In Script:**
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

**Command line (WSL2):**
```bash
watch -n 1 rocm-smi  # AMD GPU
# or
watch -n 1 nvidia-smi  # NVIDIA GPU
```

---

## Next Steps

### After Training

1. **Evaluate your model:**
   ```bash
   python scripts/run_training_safe.py evaluate_model
   ```

2. **Validate on real fires:**
   ```bash
   python scripts/run_training_safe.py validate_extended_model
   ```

3. **Check validation metrics:**
   - IoU > 0.7: Good
   - IoU > 0.8: Excellent
   - Dice > 0.8: Production-ready

### For Production

1. **Train with more data:**
   ```bash
   wsl python3 scripts/run_training_safe.py train_extended_gpu
   ```

2. **Fine-tune for specific region:**
   ```bash
   wsl python3 scripts/run_training_safe.py finetune_gpu
   ```

3. **Validate thoroughly:**
   - Multiple fire events
   - Different terrain types
   - Various weather conditions

---

## File Locations

### Checkpoints
```
outputs/checkpoints/
  - unet_best.pt (best validation loss)
  - unet_final.pt (final epoch)
  - unet_epoch_N.pt (periodic saves)
```

### Metrics
```
outputs/
  - training_metrics.json (dashboard metrics)
  - training_curves.png (loss/IoU plots)
  - evaluation_summary.txt (evaluation results)
```

### Validation Samples
```
outputs/validation_samples/
  - sample_N_input.npy
  - sample_N_target.npy
```

---

## Documentation Index

### Quick References
- **This file (QUICK_START.md)** - Quick start guide
- **QUICK_FIX_GUIDE.md** - Unicode error fixes

### Detailed Guides
- **README.md** - Project overview
- **UNICODE_FIX_SOLUTION.md** - Unicode encoding fix
- **GPU_TRAINING_NEXT_STEPS.md** - GPU setup guide
- **UTF8_AND_512_UPGRADE_SUMMARY.md** - Recent upgrades

### Technical
- **SOURCES.md** - Scientific basis
- **DATA_README.md** - Data sources
- **PROJECT_SPECIFICATION.md** - Project spec

---

## Example Workflow

### Complete Training Pipeline

```bash
# 1. Quick test to verify setup
python scripts/run_training_safe.py quick_demo

# 2. Integration test (512x512 pipeline)
python scripts/run_training_safe.py test_full_pipeline_512

# 3. Start training with dashboard
# Terminal 1:
python scripts/run_training_safe.py train_with_dashboard --epochs 50

# Terminal 2:
streamlit run scripts/training_dashboard.py

# 4. After training completes, evaluate
python scripts/run_training_safe.py evaluate_model

# 5. Validate on real fires
python scripts/run_training_safe.py validate_extended_model

# 6. If needed, fine-tune for specific region
wsl python3 scripts/run_training_safe.py finetune_gpu
```

---

## Getting Help

### Documentation
- Read `UNICODE_FIX_SOLUTION.md` for encoding issues
- Read `GPU_TRAINING_NEXT_STEPS.md` for GPU setup
- Check `README.md` for project overview

### Common Issues
1. **Unicode errors** → Use wrapper script
2. **GPU not found** → Run in WSL2 with `wsl python3`
3. **Out of memory** → Reduce batch size
4. **Slow training** → Use GPU, reduce samples/epochs for testing

### Verification
```bash
# Check Python version
python --version  # Should be 3.8+

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Check GPU (WSL2)
wsl python3 -c "import torch; print(torch.cuda.is_available())"

# Check wrapper script
python scripts/run_training_safe.py
```

---

## Summary

**Key Points:**
1. Always use `python scripts/run_training_safe.py SCRIPT_NAME`
2. Use WSL2 for GPU training: `wsl python3 scripts/run_training_safe.py ...`
3. Default patch size is 512x512 (batch size 2-4)
4. Monitor training with dashboard: `streamlit run scripts/training_dashboard.py`
5. GPU memory tracked automatically

**Ready to start training!**

For more details, see:
- UNICODE_FIX_SOLUTION.md
- GPU_TRAINING_NEXT_STEPS.md
- README.md
