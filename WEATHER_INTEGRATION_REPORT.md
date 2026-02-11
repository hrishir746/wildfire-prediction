# ERA5 Weather Integration Results - Feb 8, 2026

## Summary

**Status**: Completed ERA5 download and weather integration. Unfortunately, weather data did NOT improve model performance.

- **Baseline Model (8 channels, no weather)**: 6.62% IoU
- **Weather Model (12 channels, ERA5 integrated)**: 5.32% IoU  
- **Result**: Weather integration decreased accuracy by -1.3% absolute

## What Was Done

### 1. ERA5 Download ✅ Completed
- Downloaded 40 monthly NetCDF files (2020-2023)
- Covered 399 unique fire dates
- Variables: 2m temperature, 2m dewpoint, 10m wind U/V
- Region: California bbox (42°N-32°N, -125°W to -114°W)
- Storage: `data/weather/era5_hourly/`

### 2. Weather Integration ✅ Completed
- Extracted ERA5 values at tile center locations
- Built 12-channel dataset (8 original + 4 weather)
- Applied normalization to [0,1] range:
  - Temp: 0.27°C to 27.55°C → [0,1]
  - Dewpoint: -18.2°C to 16.4°C → [0,1]
  - U wind: -2.13 to 3.57 m/s → [0,1]
  - V wind: -3.51 to 1.96 m/s → [0,1]
- Output: `firms_progressions_weather_256.npz` (300 samples × 12 channels × 256×256)

### 3. Model Training ✅ Completed
- Trained U-Net with 12-channel input for 60 epochs
- Same architecture as baseline (8-channel)
- Best validation IoU: 5.32% (Epoch 44)
- Training time: 27.6 minutes on GPU

## Why Did Weather Hurt Performance?

### Root Causes
1. **No spatial variation within tiles**: Weather is extracted at tile center, so all 256×256 pixels in each tile get the same weather values. This removes spatial information that helps the model differentiate fire-prone areas within a tile.

2. **Coarse spatial resolution mismatch**: ERA5 is on 0.25° grid (~25-30km), while fires occur at much finer scales (~100-500m). The spatial averaging loses critical detail.

3. **Uniform features confuse learning**: When weather is identical across entire 256×256 image, it doesn't help the model distinguish between fire/no-fire regions within that image.

4. **Class imbalance overwhelming**: With only 3-5 fire pixels out of 65,536 pixels per image (0.006%), the fire signal is extremely sparse. Adding correlated features doesn't help when the core signal is so weak.

5. **Data scarcity**: Papers showing 60%+ accuracy typically use 1000+ training samples. Our 300 samples is too small to learn meaningful weather-fire relationships with sparse targets.

### Historical Context
This mirrors the temporal context experiment earlier:
- Temporal model (3-day history): 3.6% < 6.6% baseline
- Weather model (4 variables): 5.3% < 6.6% baseline

Both added information but hurt because the primary limitation is **insufficient fire pixel density**, not missing context.

## Fundamental Data Limitations

The 300-sample FIRMS dataset has structural challenges:

| Limitation | Impact |
|-----------|--------|
| Only 3-5 fire pixels per 256×256 image | 0.006% class imbalance → model learns to predict all zeros |
| Sparse temporal coverage | Fire progressions don't cluster; model can't learn patterns |
| Single-day prediction window | Too noisy; hard to learn predictive relationships |
| No large-scale fire dynamics | One fire per tile; can't learn multi-tile spread |

## Lessons Learned

1. **Domain gap (synthetic → real) was the biggest bottleneck**: Showed 6.6% >> 1-2%, proving real data essential
2. **More data >>> better features**: Temporal + weather both failed; dataset size vs. feature engineering
3. **Papers' 60%+ results use different approaches**:
   - Much larger datasets (1000-10,000 samples)
   - Different problem formulations (classification vs. dense prediction)
   - May use satellite imagery (Sentinel-2) not just weather
   - Likely have better data curation/preprocessing

## Current Best Performance

**Best model remains**: `outputs/checkpoints/unet_firms_progressions.pt`
- Configuration: 8-channel U-Net (terrain + fire history)
- Validation IoU: 6.62%
- Includes: elevation, slope, aspect, fuel type, fire history (T-1), wind speed, wind direction
- Training: 40 epochs

## Recommended Next Steps

To beat 6.6%, consider:

1. **Get more data** (most promising)
   - Collect FIRMS progressions from additional regions (Oregon, Washington, Nevada)
   - Extend to 2018-2019 historical data if available
   - Target: 500-1000 samples

2. **Different problem formulation** (faster iteration)
   - Instead of dense 256×256 prediction, do binary classification: "Will fire spread Y/N?"
   - Or: predict burn probability (sparse binary → classification threshold)
   - Easier to train on imbalanced data

3. **Different architecture**
   - Smaller receptive field (appropriate for sparse targets)
   - Attention mechanisms to weight high-fire-risk pixels
   - 3D CNN for temporal modeling with longer sequences

4. **Satellite data integration**
   - Sentinel-2 NDVI for vegetation health (no weather data required)
   - MODIS burned area products for indirect supervision
   - Requires different data pipeline

5. **Accept performance boundaries**
   - 6-8% IoU may be realistic for sparse fire prediction
   - Compare with published baselines on same dataset
   - Focus on downstream task performance (which fires matter most?)

## Files Generated

- ✅ `scripts/download_era5_weather.py` - ERA5 monthly batch downloader
- ✅ `scripts/integrate_weather_to_firms.py` - Weather-FIRMS fusion pipeline
- ✅ `scripts/normalize_weather_data.py` - Weather channel normalization
- ✅ `scripts/train_firms_weather.py` - 12-channel model training
- ✅ `.ecmwfdatastoresrc` - ECMWF API credentials config
- ✅ `data/weather/era5_hourly/` - 40 monthly ERA5 NetCDF files
- ✅ `data/firms_progressions/firms_progressions_weather_256.npz` - 12-channel dataset

## Decision Point

The weather data didn't help. The core limitation is **sparse fire pixels + small dataset**. 

To reach the 60% goal, user has two realistic paths:
1. **Invest in data**: Collect 500+ more samples from other regions/years
2. **Reframe problem**: Binary classification ("Will this tile burn?") instead of dense prediction

Both paths are more feasible than adding more features to the current sparse 300-sample dataset.
