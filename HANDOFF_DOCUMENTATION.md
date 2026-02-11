# Wildfire Prediction Project - Complete Handoff Documentation

## Project Overview
AI-powered wildfire spread prediction using deep learning on AMD GPU hardware. The model predicts fire perimeter expansion using terrain, fuel, and weather data at 512×512 resolution (~236 km² coverage).

## Hardware & Environment
- **GPU**: AMD 9070 XT (16GB VRAM)
- **OS**: Windows 11 + WSL2 Ubuntu 22.04
- **ROCm**: 6.1
- **PyTorch**: 2.10.0+rocm6.1
- **Environment Variables** (permanently set):
  - `PYTHONIOENCODING=utf-8`
  - `PYTHONUTF8=1`
  - `HSA_OVERRIDE_GFX_VERSION=1100`
  - `HIP_VISIBLE_DEVICES=0`

## Model Architecture
- **Type**: U-Net (31M parameters)
- **Input**: 8 channels (elevation, slope, aspect, 4× fuel types, wind speed/direction)
- **Output**: 1 channel (fire spread probability)
- **Resolution**: 512×512 patches (15.36km × 15.36km at 30m/pixel)
- **Batch Size**: 2-4 (memory constrained)
- **Loss**: Combined Dice + BCE
- **Optimizer**: AdamW

## Data Sources
- USGS 3DEP: Digital Elevation Model (30m resolution)
- LANDFIRE: Fuel type classifications
- NOAA: Historical wind data
- MTBS: Fire perimeter ground truth

## What Has Been Done

### Phase 1: Resolution Upgrade (512×512 Patches)
Updated all training scripts from 256×256 to 512×512:

1. **train_extended_gpu.py**
   - RESOLUTION: 256 → 512
   - BATCH_SIZE: 8 → 4
   - Checkpoint: `unet_extended_512x512.pt`

2. **train_gpu_optimized.py**
   - RESOLUTION: 256 → 512
   - BATCH_SIZE: 16 → 4
   - Plot: `training_curves_extended_512.png`

3. **train_fast_production.py**
   - RES: 256 → 512
   - BATCH_SIZE: 8 → 2
   - Plot: `training_curves_fast_512.png`

4. **finetune_gpu.py**
   - SIZE: (64,64) → (512,512)
   - BATCH_SIZE: 4 → 2
   - Checkpoint: `unet_finetune_optimized_512.pt`

### Phase 2: UTF-8 Encoding System (Windows Compatibility)
Fixed Windows 'charmap' codec errors across all scripts:

**Solution Pattern Applied:**
```python
import sys
import os

# Force UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
```

**Files Modified with UTF-8 Fixes:**
- All training scripts (train_*.py, finetune_*.py)
- All validation scripts (validate_*.py)
- All evaluation scripts (evaluate_*.py)
- Quick demo scripts (quick_demo.py, run_full_demo.py, etc.)
- Agent interface (agent_app.py)

### Phase 3: Live Training Dashboard (Rovo Dev Contribution)

#### Created Files:

**1. training_dashboard.py (487 lines)**
- Real-time Streamlit dashboard
- Displays: Loss curves, IoU/Dice metrics, GPU stats, training ETA
- Checkpoint browser with download
- Auto-refreshes every 5 seconds
- Launch: `streamlit run scripts/training_dashboard.py`

**2. train_with_dashboard.py (389 lines)**
- Training script with DashboardLogger class
- Logs metrics to JSON every epoch
- Combined Dice+BCE loss
- Checkpoint saving with best model tracking
- Run: `python scripts/train_with_dashboard.py`

**3. run_training_safe.py (176 lines)**
- UTF-8 safe wrapper for all scripts
- Forces encoding BEFORE imports
- Router for all training/validation/test scripts
- Usage: `python scripts/run_training_safe.py <script_name>`
- Examples:
  - `python scripts/run_training_safe.py train_extended_gpu`
  - `python scripts/run_training_safe.py validate_camp_2018`
  - `python scripts/run_training_safe.py test_full_pipeline_512`

**4. test_full_pipeline_512.py (376 lines)**
- End-to-end integration test
- Generates 10 synthetic 512×512 samples
- Trains for 2 epochs
- Validates on 3 samples
- Reports: IoU, Dice, pixel accuracy, loss
- Run: `python scripts/run_training_safe.py test_full_pipeline_512`

### Phase 4: Development Tools Setup

**Git Installation:**
- Installed Git for Windows
- Added to PATH: `C:\Program Files\Git\cmd`
- Configured for Rovo Dev integration

**Rovo Dev Setup:**
- Atlassian's AI coding assistant (unlimited credits)
- Installed as VS Code extension
- Cleared cache to fix agent termination errors
- Configured with UTF-8 encoding
- Successfully integrated with project

## Current Project Status

### Completed ✓
- [x] 512×512 patch support across all scripts
- [x] UTF-8 encoding system-wide
- [x] Live training dashboard with Streamlit
- [x] Safe wrapper for UTF-8 compliance
- [x] Integration test for 512×512 pipeline
- [x] Rovo Dev setup and integration
- [x] Git installation and configuration
- [x] All training scripts updated for new resolution

### In Progress 🔄
- [ ] GPU memory monitoring in dashboard (planned)
- [ ] Large-scale training data collection

### Pending Tasks 📋
1. **Add GPU Memory Monitoring to Dashboard**
   - Modify `training_dashboard.py` to display:
     - `torch.cuda.memory_allocated()`
     - `torch.cuda.memory_reserved()`
     - `torch.cuda.max_memory_allocated()`
   - Update `DashboardLogger` in `train_with_dashboard.py` to log memory

2. **Run Integration Test**
   ```bash
   python scripts/run_training_safe.py test_full_pipeline_512
   ```
   - Verify no Unicode errors
   - Check GPU utilization
   - Confirm successful training/validation

3. **Start Large-Scale Training**
   ```bash
   # Terminal 1: Start training
   python scripts/run_training_safe.py train_extended_gpu
   
   # Terminal 2: Launch dashboard
   streamlit run scripts/training_dashboard.py
   ```

4. **Multi-Fire Validation**
   - Run: `python scripts/run_training_safe.py validate_extended_model`
   - Target: 80%+ IoU/Dice scores
   - Tune classification thresholds if needed

5. **Create Quick Start Guide**
   - Document dashboard usage
   - List all available commands
   - Add troubleshooting section

## Key Commands Reference

### Training
```bash
# Safe wrapper (recommended - handles UTF-8)
python scripts/run_training_safe.py train_extended_gpu
python scripts/run_training_safe.py train_gpu_optimized
python scripts/run_training_safe.py train_fast_production
python scripts/run_training_safe.py finetune_gpu

# With dashboard logging
python scripts/train_with_dashboard.py
```

### Dashboard
```bash
# Launch live monitoring dashboard
streamlit run scripts/training_dashboard.py
```

### Validation
```bash
python scripts/run_training_safe.py validate_camp_2018
python scripts/run_training_safe.py validate_80_percent
python scripts/run_training_safe.py evaluate_real_fires
```

### Testing
```bash
# Integration test (512×512 pipeline)
python scripts/run_training_safe.py test_full_pipeline_512

# Quick demo
python scripts/run_training_safe.py quick_demo
```

### Data Fetching
```bash
python scripts/fetch_real_data.py
python scripts/fetch_mtbs_data.py
python scripts/fetch_mtbs_mirrors.py
```

## File Structure

### Key Directories
- **scripts/**: All training, validation, and utility scripts
- **src/**: Core model and data pipeline code
- **outputs/**: Training results, checkpoints, evaluations
- **config/**: Configuration files (default.yaml)

### Important Files
- **training_dashboard.py**: Real-time monitoring UI
- **train_with_dashboard.py**: Training with dashboard integration
- **run_training_safe.py**: UTF-8 safe script wrapper
- **test_full_pipeline_512.py**: End-to-end integration test
- **train_extended_gpu.py**: Main training script (512×512)
- **train_gpu_optimized.py**: Optimized training variant
- **train_fast_production.py**: Fast production training
- **finetune_gpu.py**: Fine-tuning script

---

## Feb 2026 FIRMS Experiments (Spatial Fire Spread, 256×256)

### Summary
We ran a focused set of experiments on FIRMS VIIRS fire progression pairs (2020-2023). The key finding: data sparsity is the hard limit. Most samples have 2-3 fire pixels, which makes spatial prediction nearly unlearnable. Best IoU achieved: **7.55%**.

### Best Result (Current Ceiling with FIRMS)
- **Training script**: `scripts/train_large_fires_only.py`
- **Result**: IoU 7.55% (val)
- **Dataset**: filtered to fires with >= 5 fire pixels (55 samples)
- **Checkpoint**: `outputs/checkpoints/unet_large_fires.pt`
- **Progress log**: `outputs/training_large_fires_progress.json`
- **Metrics**: `outputs/large_fires_training_metrics.json`

### Baseline and Failed Attempts
- **Baseline (real FIRMS)**: 6.62% IoU (300 samples)
- **Focal Loss** (`scripts/train_firms_focal_loss.py`): 6.17% IoU
- **Augmentation** (`scripts/train_firms_augmented.py`): 6.21% IoU (stopped)
- **Larger U-Net** (`scripts/train_large_model.py`): 5.23% IoU (overfit)

### Root Cause: Data Sparsity
- Median fire pixels per image: **3**
- 82% of samples have < 5 fire pixels (noise)
- Class imbalance ratio: ~1:16,847
- This caps IoU at ~7-8% for FIRMS point detections

### Documentation Added
- `COMPLETE_ANALYSIS.md`: Full failure analysis and conclusions
- `FOCAL_LOSS_STRATEGY.md`: Focal loss reasoning + results
- `MDPI_ANALYSIS.md`: Paper summary showing 86.73% IoU requires dense fire imagery

### Dashboard Note
`scripts/fu_unet_dashboard.py` currently points at focal loss metrics:
- METRICS_FILE: `outputs/firms_focal_training_metrics.json`
- LIVE_PROGRESS_FILE: `outputs/training_focal_progress.json`

Update these if monitoring large-fires runs.

---

## Recommendations for Rovo Dev

### If goal is >= 30% accuracy
**Must change task or data**:
1. **Dense fire imagery** (drone or high-res satellite) + segmentation model
   - Expected IoU: 60-80%
2. **Temporal forecasting (Prophet)** using FRP time series
   - Expected MAPE: 30-50%
3. **Hybrid**: Prophet for when + spatial model for where

### If staying on FIRMS point detections
- 7-8% IoU is the realistic ceiling
- Further architecture changes (Swin, attention) likely yield small gains only


### Backup Files
All `.unicode_backup` files are safety backups created before UTF-8 fixes. Can be deleted once verified.

## Known Issues & Solutions

### Issue 1: UnicodeEncodeError on Windows
**Error**: `'charmap' codec can't encode characters`

**Solution**: Always use `run_training_safe.py` wrapper:
```bash
python scripts/run_training_safe.py <script_name>
```

### Issue 2: Out of Memory (OOM)
**Error**: CUDA out of memory during training

**Solutions**:
- Reduce batch size (currently 2-4 for 512×512)
- Enable gradient checkpointing
- Use mixed precision (FP16)
- Clear cache: `torch.cuda.empty_cache()`

### Issue 3: Dashboard Not Updating
**Solution**: 
- Ensure training script is logging to `outputs/training_metrics.json`
- Check dashboard refresh interval (5 seconds default)
- Verify Streamlit is running on correct port

## User Preferences & Notes

### Training Preferences
- Prefers GPU-accelerated training on AMD hardware
- Wants real-time monitoring with visuals
- Prioritizes large-scale (512×512) coverage for forest mapping
- Target accuracy: 80%+ IoU/Dice on validation set

### Development Style
- Uses Rovo Dev for rapid development (unlimited credits)
- Prefers comprehensive prompts with full context
- Values detailed documentation for handoffs
- Appreciates safety backups before major changes

### Communication Style
- Direct, concise responses
- Wants actionable next steps
- Prefers copy-pastable commands
- Values complete context for new sessions

## Next Steps for New Session

1. **Verify Current State**
   ```bash
   # Check if training has been run
   ls -lh outputs/checkpoints/
   ls -lh outputs/training_metrics.json
   ```

2. **Run Integration Test First**
   ```bash
   python scripts/run_training_safe.py test_full_pipeline_512
   ```

3. **Add GPU Memory Monitoring**
   - Edit `training_dashboard.py` (add memory metrics display)
   - Edit `train_with_dashboard.py` (log memory per epoch)

4. **Start Full Training Pipeline**
   ```bash
   # Terminal 1
   python scripts/run_training_safe.py train_extended_gpu
   
   # Terminal 2
   streamlit run scripts/training_dashboard.py
   ```

5. **Monitor and Validate**
   - Watch dashboard for GPU utilization, loss convergence
   - Run validation after training completes
   - Tune thresholds if needed to hit 80%+ accuracy

## Additional Context

### Why 512×512 Resolution?
- Original 256×256 = ~59 km² coverage (insufficient for large wildfires)
- Upgraded 512×512 = ~236 km² coverage (4× area, better for forest-scale predictions)
- Trade-off: Reduced batch size due to memory constraints

### Why UTF-8 Safe Wrapper?
- Windows PowerShell defaults to 'charmap' encoding
- Unicode characters in training logs cause crashes
- `run_training_safe.py` forces UTF-8 BEFORE any imports
- All scripts now have UTF-8 guards at the top

### Why Streamlit Dashboard?
- Real-time training visibility (requested by user)
- No need to check logs manually
- GPU monitoring, ETA, checkpoint browser
- Auto-refreshes every 5 seconds
- Easy to launch and use

## Contact & Continuation

This document contains **everything** done by both the GitHub Copilot agent and Rovo Dev. It's designed to be copy-pasted into a new chat session for seamless continuation without losing context.

**Key Principle**: The project is ready for large-scale training. All infrastructure (512×512 support, UTF-8 fixes, dashboard) is in place. Next session should focus on:
1. Integration testing
2. GPU memory monitoring
3. Full training run
4. Multi-fire validation
5. Threshold tuning for 80%+ accuracy

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-07  
**Created By**: GitHub Copilot (Claude Sonnet 4.5)  
**Project**: Wildfire Prediction with Deep Learning
