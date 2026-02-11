# Track 2: FU-NetCastV2 Implementation Plan
## 24-Hour Fire Spread Prediction with Satellite Data

**Timeline:** Feb 8-10 (3 days), overnight GPU training
**Target:** 80%+ accuracy on real fires
**Status:** Ready to execute

---

## Phase 1: Data Preparation (Feb 8, 6 hours)

### Task 1.1: Extract 24-hour Fire Perimeter Pairs from MTBS
```python
# File: scripts/prepare_fire_pairs.py
# Objective: Create dataset of (Day N, Day N+1) fire perimeters

Input:  MTBS fire perimeter database
Output: JSON with fire pairs
  {
    "fire_id": "camp_2018",
    "date": "2018-11-08",
    "perimeter_start": ".../mtbs_perimeters/camp_2018_perim_20181108.geojson",
    "perimeter_24h": ".../mtbs_perimeters/camp_2018_perim_20181109.geojson",
    "start_acres": 1500,
    "acres_24h": 24000,
    "acres_burned": 22500
  }

Target: 50-100 consecutive fire pairs (realistic given MTBS data availability)
```

### Task 1.2: Download Landsat-8 Satellite Data
```python
# File: scripts/fetch_landsat_data.py
# Objective: Download Landsat-8 bands for each fire location/date

Bands needed:
  - Band 2 (Blue) - water/fire detection
  - Band 3 (Green) - vegetation
  - Band 4 (Red) - vegetation
  - Band 5 (Near-IR) - vegetation health
  - Band 6 (SWIR-1) - active fire
  - Band 7 (SWIR-2) - active fire
  + NDVI (Normalized Difference Vegetation Index)

Source: Google Cloud Storage (gs://landsat-public-data)
Resolution: 30m (same as elevation model)
Download: 2-3 hours (parallel downloads)
```

### Task 1.3: Prepare Weather Timeseries
```python
# File: scripts/fetch_weather_timeseries.py
# Objective: Aggregate 24-hour weather for each fire prediction

Hourly data (24 hours):
  - Wind speed (m/s)
  - Wind direction (degrees)
  - Temperature (°C)
  - Humidity (%)
  - Precipitation (mm)
  - Solar radiation (W/m²)

Source: NOAA or Visual Crossing Weather API
Time period: Fire date ± 1 day
Resolution: Hourly (24 data points)
```

### Task 1.4: Align All Data to Common Grid
```python
# File: scripts/align_training_data.py
# Objective: Create training samples (input, target) tuples

Input tensor (Time T):
  - Fire perimeter (T) - 512×512 binary
  - Elevation - 512×512 uint16
  - Slope - 512×512 uint8
  - Aspect (sin, cos) - 512×512×2 float32
  - Landsat bands 2-7 - 512×512×6 uint16 (reflectance 0-10000)
  - NDVI - 512×512 float32 (-1 to +1)
  - Weather timeseries - 24×6 float32 (24 hours × 6 metrics)
  
Output tensor (Time T+24h):
  - Fire perimeter (T+24h) - 512×512 binary (ground truth)

Final shape: (N_fires, 31, 512, 512) where 31 = 1+1+1+2+6+1+19
  (Actually: 1+1+1+2+6+1 + 24×6 flattened = 1+1+1+2+6+1+144 = 156 channels if we include all weather per pixel)
  
Simplified: Append weather timeseries separately, use sequence model or time-aggregated features
```

---

## Phase 2: Model Development (Feb 9, 12-18 hours, 8-10 hours GPU)

### Task 2.1: Adapt U-Net for Multi-Modal Input
```python
# File: src/model/fu_unet.py
# Based on: Khennou & Akhloufi (2023) FU-NetCastV2

Input channels: 19
├─ Fire perimeter (T) - 1
├─ Elevation - 1
├─ Slope - 1
├─ Aspect (sin, cos) - 2
├─ Landsat bands (2-7) - 6
├─ NDVI - 1
├─ Weather features (aggregated) - 6
  └─ Wind speed
  └─ Wind direction (sin, cos)
  └─ Temperature
  └─ Humidity

Architecture: Standard U-Net
├─ Encoder: 5 levels (down 32→64→128→256→512)
├─ Decoder: 5 levels (up 512→256→128→64→32)
└─ Skip connections preserved

Output: Fire probability map (512×512×1)

Total parameters: ~35M (slightly more than current model)
Training time: 8-10 hours (overnight)
```

### Task 2.2: Implement Time-Aware Weather Features
```python
# Option A: Weather as static input
# Aggregate 24-hour weather into summary features
#   - Mean wind speed / direction
#   - Mean temperature / humidity
#   - Max wind gust
# Add as 6 channel input (broadcast across 512×512)

# Option B: Weather as temporal context (if time = allow)
# Use LSTM/GRU to process 24-hour timeseries
# Feed sequence encoder output to U-Net decoder

# Recommended: Option A (simpler, proven in FU-NetCastV2)
```

### Task 2.3: Loss Function & Training
```python
# File: scripts/train_fu_unet.py

Loss: Dice + BCE (weighted combination)
  loss = 0.6 * dice_loss + 0.4 * bce_loss

Optimizer: AdamW
  lr = 0.001
  weight_decay = 1e-4

Learning rate schedule: ReduceLROnPlateau
  - Monitor: validation Dice
  - Patience: 5 epochs
  - Factor: 0.5

Early stopping:
  - Monitor: validation Dice
  - Patience: 10 epochs
  - Restore best weights

Default batch size: 4 (GPU memory 16GB)
Default epochs: 100 (early stopping expected ~epoch 30-50)

Data split: 80% train, 10% val, 10% test
```

### Task 2.4: Training Run
```bash
# Expected timeline
Time    Event
----    -----
0:00    Start training (8 GB VRAM used)
2:00    Epoch ~20-30 (validation first improvement plateau)
4:00    Epoch ~40-50 (early stopping likely triggered)
6:00    Training complete
6-10:   Checkpoint saving, validation on holdout set

Total wall time: 6-10 hours (overnight friendly)
Best validation Dice expected: 75-85%
```

---

## Phase 3: Evaluation (Feb 10, 4 hours)

### Task 3.1: Final Validation on Holdout Fires
```python
# File: scripts/validate_fu_unet.py

Test on independent fires (not in training set):
  - 10-20% of MTBS fires held back
  - Multiple fires from different regions/years
  - Calculate:
    ├─ IoU (Intersection over Union)
    ├─ Dice coefficient
    ├─ Precision/Recall
    ├─ F1-score
    └─ Per-fire error analysis

Expected results:
  - Conservative: 60-70% accuracy
  - Realistic: 75-85% accuracy
  - Best case: 85-95% accuracy (if FU-NetCastV2 approach is sound)
```

### Task 3.2: Compare to Benchmarks
```python
# Generate comparison table:

| Baseline | Accuracy | Notes |
|----------|----------|-------|
| Synthetic U-Net | 91% | ✅ Synthetic, ❌ Fails on real (1%) |
| Constant prediction | ~5% | All pixels fire (baseline) |
| Rothermel model | ~40% | Physics-based, no ML |
| FU-NetCastV2 (paper) | 94.6% | Peer-reviewed (2023) |
| **Our FU-NetCastV2** | **75-95%** | (TBD - expected range) |

Success threshold: >80% on our real fire test set
```

### Task 3.3: Visualization & Analysis
```python
# File: scripts/visualize_fu_unet_results.py

For top 5 predictions:
  1. Input: (Elevation, Fire(T), Satellite NDVI) - RGB composite
  2. Ground truth: Fire(T+24h) - green outline
  3. Prediction: Model output - red heatmap
  4. Error: FN (missed), FP (overpredict) - separate colors
  5. Metrics: Dice, IoU, intersection/union visualization

Generate publication-quality figures
```

---

## Implementation Milestones

| Date | Time | Task | Status |
|------|------|------|--------|
| Feb 8 | 6h | Phase 1: Data prep (fire pairs, Landsat, weather) | ⏳ Ready |
| Feb 9 | 12h | Phase 2: Model dev + overnight training | ⏳ Ready |
| Feb 10 | 4h | Phase 3: Validation + visualizations | ⏳ Ready |
| Feb 11 | 4h | Final presentation + polishing | ⏳ Ready |

---

## Success Criteria

✅ **Minimum (70%):** 
- Data preparation complete
- Model trains without errors
- Accuracy >60% on real fires

✅ **Target (85%):**
- All phases complete
- Accuracy 75-85% on holdout fires
- Visualizations generated
- Comparison to benchmarks

✅ **Excellent (95%):**
- Accuracy 85%+ achieved
- Publication-quality figures
- Statistical significance testing
- Error analysis documented

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|-----------|
| MTBS fire pair data insufficient | Medium | Have 50-100 pairs target; fallback to 30 pairs if needed |
| Landsat data gaps (clouds) | High | Download multiple scenes per date, interpolate |
| Weather data missing | Low | Multiple weather APIs available as fallback |
| GPU memory OOM | Low | Reduce batch to 2, or use gradient accumulation |
| Training too slow | Low | 10 hours GPU time is feasible overnight |
| Model underfits | Low | U-Net proven architecture; likely to converge |

---

## Rapid Start Commands

Once you're ready to start Feb 8:

```bash
# 1. Data preparation
python scripts/prepare_fire_pairs.py
python scripts/fetch_landsat_data.py  # Parallel download, ~2-3h
python scripts/fetch_weather_timeseries.py
python scripts/align_training_data.py

# 2. Training (overnight)
python scripts/train_fu_unet.py --epochs 100 --batch_size 4

# 3. Validation + viz
python scripts/validate_fu_unet.py
python scripts/visualize_fu_unet_results.py

# 4. Report
cat outputs/fu_unet_validation_results.txt
```

---

## Why This Will Work

1. **Proven architecture:** FU-NetCastV2 achieved 94.6% in peer review
2. **Real training data:** Using actual fire perimeter pairs (not synthetic)
3. **Realistic prediction scale:** 24-hour horizon (not 150 synthetic steps)
4. **Satellite integration:** Spectral data addresses domain gap
5. **GPU ready:** AMD Radeon RX 9070 XT verified functional
6. **Tight timeline:** 3 days is achievable with focused execution

---

## Status: Ready to Execute

All prerequisites confirmed.  
All code scaffolds prepared.  
All data sources identified.  
Timeline feasible with overnight GPU training.  
Expected outcome: 80-95% accuracy on real fires.

**Next step:** Begin Phase 1 on Feb 8. Report daily progress to stay on track.

---

**Prepared: February 7, 2026**  
**Execution window: Feb 8-10, 2026**  
**Presentation: Feb 11, 2026**
