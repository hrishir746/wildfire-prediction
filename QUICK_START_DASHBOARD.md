# 🚀 Quick Start: Live Training Dashboard

## One-Minute Setup

### 1. Start Training
```bash
python scripts/train_with_dashboard.py --epochs 50 --samples 1000
```

### 2. Launch Dashboard (new terminal)
```bash
streamlit run scripts/training_dashboard.py
```

### 3. Open Browser
```
http://localhost:8501
```

## That's it! 🎉

## Dashboard Views

- **📊 Overview**: Real-time status, latest metrics, training curves
- **📈 Training Curves**: Detailed plots, export to CSV
- **🔮 Predictions**: Visualize model outputs vs ground truth
- **💾 Checkpoints**: Browse and inspect saved models

## Key Metrics to Watch

| Metric | Target | Location |
|--------|--------|----------|
| **IoU** | ≥ 80% | Overview tab |
| **Dice** | ≥ 80% | Overview tab |
| **Loss** | < 0.3 | Training Curves tab |

## Command Options

```bash
# Quick test (3 min)
python scripts/train_with_dashboard.py --epochs 10 --samples 100 --grid-size 128

# Production (30-60 min on GPU)
python scripts/train_with_dashboard.py --epochs 50 --samples 1000 --grid-size 512

# High accuracy (2-4 hours on GPU)
python scripts/train_with_dashboard.py --epochs 100 --samples 2000 --grid-size 512
```

## Troubleshooting

**Dashboard shows "No metrics"?**
- Use `train_with_dashboard.py`, not other training scripts

**Want auto-refresh?**
- Enable "Auto-refresh (5s)" in dashboard sidebar

**GPU out of memory?**
- Add `--batch-size 2` or `--grid-size 256`

## Full Documentation

See `scripts/README_DASHBOARD.md` for complete guide.

---

**Happy training! 🔥📊**
