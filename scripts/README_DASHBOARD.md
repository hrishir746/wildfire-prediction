# Live Training Dashboard Guide

## Overview

The wildfire prediction project now includes a **live training dashboard** for real-time monitoring of model training with the following features:

- 📊 **Real-time metrics**: Loss, IoU, Dice coefficient, learning rate
- 📈 **Interactive plots**: Training curves with auto-refresh
- 🔮 **Prediction visualization**: View model predictions on validation samples
- 💾 **Checkpoint management**: Browse and inspect saved checkpoints
- 🖥️ **System monitoring**: GPU info and training progress

## Quick Start

### 1. Start Training with Dashboard Integration

Open **Terminal 1** and run:

```bash
# Recommended: GPU-optimized training with dashboard
python scripts/train_with_dashboard.py --epochs 50 --samples 1000

# Options:
# --epochs 50           Number of training epochs
# --samples 1000        Number of synthetic samples
# --grid-size 512       Grid resolution (512x512)
# --batch-size 4        Batch size (adjust for GPU memory)
# --lr 5e-4             Learning rate
# --val-split 0.2       Validation split ratio (20%)
# --save-samples 5      Number of validation samples to save for viz
```

### 2. Launch Dashboard

Open **Terminal 2** and run:

```bash
streamlit run scripts/training_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 3. Monitor Training

The dashboard provides 4 views:

#### **Overview** 📊
- Current training status and progress
- Latest metrics (loss, IoU, Dice)
- Real-time training curves
- System information (GPU, device)

#### **Training Curves** 📈
- Detailed plots of all metrics over time
- Export training data to CSV
- Interactive zoom and pan

#### **Predictions** 🔮
- Select any checkpoint
- Visualize model predictions vs ground truth
- View all 8 input channels
- Per-sample metrics (IoU, Dice, MAE)

#### **Checkpoints** 💾
- Browse all saved checkpoints
- View checkpoint metadata (epoch, loss, size)
- Inspect checkpoint details

## Dashboard Features

### Auto-Refresh
- Enable "Auto-refresh (5s)" in the sidebar to automatically update metrics
- Manual refresh button available

### Metrics Tracked
- **Train Loss**: Combined Dice + BCE loss on training set
- **Val Loss**: Validation loss
- **Val IoU**: Intersection over Union (target: 80%+)
- **Val Dice**: Dice coefficient (target: 80%+)
- **Learning Rate**: Current learning rate (with scheduler)

### Visualization
- **Input Channels**: Slope, Aspect (sin/cos), Fuel, Wind (speed, sin/cos), Initial Fire
- **Predictions**: Probability maps from model
- **Ground Truth**: Target fire spread from simulator
- **Error Maps**: Absolute difference visualization

## Files and Directories

```
outputs/
├── training_metrics.json          # Real-time metrics log
├── validation_samples/            # Validation samples for viz
│   ├── sample_0_input.npy
│   ├── sample_0_target.npy
│   └── ...
└── checkpoints/                   # Model checkpoints
    ├── unet_best.pt              # Best validation loss
    ├── unet_final.pt             # Final epoch
    ├── unet_epoch_10.pt          # Periodic checkpoints
    └── ...
```

## UTF-8 Encoding (Windows)

Both training and dashboard scripts automatically configure UTF-8 encoding for Windows:

```python
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
```

No manual terminal configuration needed!

## Alternative Training Scripts

You can also use existing training scripts, but they won't log to the dashboard:

```bash
# Standard training (no dashboard logging)
python scripts/train_model.py --epochs 50 --samples 1000

# GPU-optimized (no dashboard logging)
python scripts/train_gpu_optimized.py

# Extended GPU training (no dashboard logging)
python scripts/train_extended_gpu.py
```

To enable dashboard monitoring for these scripts, they would need to be modified to use the `DashboardLogger` class.

## Troubleshooting

### Dashboard shows "No training metrics found"
- Make sure you're running `train_with_dashboard.py`, not other training scripts
- Check that `outputs/training_metrics.json` exists

### Predictions not showing
- Ensure validation samples are saved (check `outputs/validation_samples/`)
- Verify checkpoint exists and is loadable

### Dashboard not refreshing
- Enable "Auto-refresh" in sidebar
- Click "Refresh Now" button manually
- Check that training script is still running

### Encoding errors on Windows
- Both scripts have UTF-8 configured automatically
- If issues persist, set environment variable: `set PYTHONIOENCODING=utf-8`

### GPU out of memory
- Reduce `--batch-size` (try 2 or 1)
- Reduce `--grid-size` (try 256)
- Close other GPU applications

## Example Workflow

```bash
# Terminal 1: Start training
python scripts/train_with_dashboard.py --epochs 100 --samples 2000 --batch-size 4

# Terminal 2: Launch dashboard
streamlit run scripts/training_dashboard.py

# Browser: Open http://localhost:8501
# - Watch metrics in Overview tab
# - Monitor curves in Training Curves tab
# - Inspect predictions in Predictions tab

# After training completes:
# - Review final metrics
# - Export training data
# - Visualize predictions from best checkpoint
```

## Advanced Usage

### Custom Grid Size
```bash
# 256x256 for faster training
python scripts/train_with_dashboard.py --grid-size 256 --epochs 30

# 1024x1024 for high resolution (requires more GPU memory)
python scripts/train_with_dashboard.py --grid-size 1024 --batch-size 1
```

### Learning Rate Tuning
```bash
# Higher learning rate for faster convergence
python scripts/train_with_dashboard.py --lr 1e-3

# Lower learning rate for fine-tuning
python scripts/train_with_dashboard.py --lr 1e-5
```

### More Validation Samples
```bash
# Save 10 samples for visualization
python scripts/train_with_dashboard.py --save-samples 10
```

## Integration with Existing Tools

The dashboard complements existing tools:

- **Evaluate Model**: `python scripts/evaluate_model.py`
- **Real Fire Validation**: `python scripts/validate_camp_2018.py`
- **Agent Interface**: `streamlit run scripts/agent_app.py`
- **Quick Demo**: `python scripts/quick_demo.py`

## Next Steps

After training with the dashboard:

1. **Validate on Real Fires**: 
   ```bash
   python scripts/validate_camp_2018.py --checkpoint outputs/checkpoints/unet_best.pt
   ```

2. **Run Full Evaluation**:
   ```bash
   python scripts/evaluate_model.py --checkpoint outputs/checkpoints/unet_best.pt
   ```

3. **Test on New Areas**:
   ```bash
   python scripts/fetch_real_data.py --west -122.5 --south 37.5 --east -122 --north 38
   python scripts/visualize_predictions.py --checkpoint outputs/checkpoints/unet_best.pt
   ```

4. **Interactive Agent**:
   ```bash
   streamlit run scripts/agent_app.py
   ```

## Performance Tips

- **GPU Training**: Always use GPU for 512x512 grids (20-50x faster)
- **Batch Size**: Larger = faster, but requires more memory (start with 4)
- **Samples**: More = better accuracy, but longer training (1000+ recommended)
- **Epochs**: 50-100 for good convergence, 100+ for 80%+ accuracy

## Target Metrics

For **80%+ accuracy** on large-scale validation:

- **Validation IoU**: ≥ 0.80
- **Validation Dice**: ≥ 0.80
- **Training Loss**: < 0.3 (combined Dice+BCE)

Monitor these in the dashboard's Overview tab!
