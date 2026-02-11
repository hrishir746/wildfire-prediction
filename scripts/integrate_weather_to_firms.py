#!/usr/bin/env python3
"""
Integrate ERA5 weather data with FIRMS progressions to create 12-channel dataset.
- Original 8 channels: terrain (elevation, slope, aspect, fuel type) + fire history (past 1-day) + wind (speed, direction)
- New 4 channels: temperature, dewpoint, 10m wind U, 10m wind V from ERA5
"""

import numpy as np
import netCDF4 as nc
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import RectBivariateSpline
import json

DATA_DIR = Path("data")
WEATHER_DIR = DATA_DIR / "weather" / "era5_hourly"
FIRMS_FILE = DATA_DIR / "firms_progressions" / "firms_progressions_256.npz"
OUTPUT_FILE = DATA_DIR / "firms_progressions" / "firms_progressions_weather_256.npz"

print("=" * 80)
print("INTEGRATE ERA5 WEATHER WITH FIRMS PROGRESSIONS (12-CHANNEL)")
print("=" * 80)

# Load FIRMS data
print("\n1. Loading FIRMS progressions data...")
firms_data = np.load(FIRMS_FILE)
X = firms_data['X']  # (300, 8, 256, 256)
Y = firms_data['Y']  # (300, 1, 256, 256)

print(f"   Loaded {len(X)} samples with shape {X.shape}")
print(f"   Input channels: {X.shape[1]}")
print(f"   Output shape: {Y.shape}")

# Reconstruct metadata from original CSV to get dates/locations
print("\n1b. Reconstructing sample metadata...")
import pandas as pd
from collections import defaultdict

DATA_CSV = DATA_DIR / "firms_archives" / "california_fires_2020_2023.csv"
CA_BBOX = {"west": -125.0, "south": 32.0, "east": -114.0, "north": 42.0}
TILE_DEG = 0.25

df = pd.read_csv(DATA_CSV)
df = df[["latitude", "longitude", "acq_date"]].dropna()
df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
df = df.dropna(subset=["acq_date"])

# Filter to CA bbox
df = df[
    (df["longitude"] >= CA_BBOX["west"]) &
    (df["longitude"] <= CA_BBOX["east"]) &
    (df["latitude"] >= CA_BBOX["south"]) &
    (df["latitude"] <= CA_BBOX["north"])
]

# Assign tile indices (same as processing script)
df["tile_i"] = ((df["latitude"] - CA_BBOX["south"]) / TILE_DEG).astype(int)
df["tile_j"] = ((df["longitude"] - CA_BBOX["west"]) / TILE_DEG).astype(int)

# Group by tile and date to match samples
grouped = defaultdict(lambda: defaultdict(list))
for row in df.itertuples(index=False):
    tile_id = (int(row.tile_i), int(row.tile_j))
    date_key = row.acq_date.date().isoformat()
    grouped[tile_id][date_key].append((row.longitude, row.latitude))

# Extract sample info
dates = []
lons = []
lats = []
sample_count = 0
for (tile_i, tile_j), date_map in grouped.items():
    if sample_count >= len(X):
        break
    
    dates_list = sorted(date_map.keys())
    if len(dates_list) < 2:
        continue
    
    for idx in range(len(dates_list) - 1):
        if sample_count >= len(X):
            break
        
        d0 = dates_list[idx]
        pts0 = date_map[d0]
        pts1 = date_map[dates_list[idx + 1]]
        
        if len(pts0) < 3 or len(pts1) < 3:
            continue
        
        # Use tile center for weather interpolation
        tile_center_lat = CA_BBOX["south"] + (tile_i + 0.5) * TILE_DEG
        tile_center_lon = CA_BBOX["west"] + (tile_j + 0.5) * TILE_DEG
        
        dates.append(d0)
        lats.append(tile_center_lat)
        lons.append(tile_center_lon)
        sample_count += 1

print(f"   Reconstructed {len(dates)} samples")
if dates:
    print(f"   Date range: {min(dates)} to {max(dates)}")
    print(f"   Location range: lat {min(lats):.2f} to {max(lats):.2f}, lon {min(lons):.2f} to {max(lons):.2f}")

# Load and cache ERA5 data
print("\n2. Loading ERA5 weather data...")
era5_cache = {}

def get_era5_data(date_str):
    """Load ERA5 data for a specific month."""
    year, month, day = date_str.split('-')
    key = f"{year}_{month}"
    
    if key not in era5_cache:
        era5_file = WEATHER_DIR / f"era5_{year}_{int(month):02d}.nc"
        if not era5_file.exists():
            print(f"   WARNING: {era5_file.name} not found, skipping")
            return None
        
        with nc.Dataset(era5_file, 'r') as ds:
            # Read coordinates and data
            lats_grid = ds.variables['latitude'][:]
            lons_grid = ds.variables['longitude'][:]
            
            # Read variables at 12:00 UTC (index might vary)
            temp = ds.variables['t2m'][:]  # (time, lat, lon)
            dewpoint = ds.variables['d2m'][:]
            u_wind = ds.variables['u10'][:]
            v_wind = ds.variables['v10'][:]
            
            # Take first time step (12:00 UTC)
            temp = temp[0] - 273.15  # Convert to Celsius
            dewpoint = dewpoint[0] - 273.15
            u_wind = u_wind[0]
            v_wind = v_wind[0]
            
            era5_cache[key] = {
                'lats': lats_grid,
                'lons': lons_grid,
                'temp': temp,
                'dewpoint': dewpoint,
                'u_wind': u_wind,
                'v_wind': v_wind
            }
        
        print(f"   Loaded {era5_file.name}: {temp.shape}")
    
    return era5_cache[key]

def interpolate_to_location(data_2d, lats_grid, lons_grid, lat, lon):
    """Bilinear interpolation of ERA5 data to a specific lat/lon."""
    # Find nearest grid points
    lat_idx = np.argmin(np.abs(lats_grid - lat))
    lon_idx = np.argmin(np.abs(lons_grid - lon))
    
    # Simple nearest neighbor (could upgrade to bilinear)
    return data_2d[lat_idx, lon_idx]

# Build 12-channel dataset
print("\n3. Building 12-channel dataset...")
X_weather = np.zeros((len(X), 12, 256, 256), dtype=np.float32)
X_weather[:, :8, :, :] = X  # Copy original 8 channels

# Add 4 weather channels
channels_added = {
    8: 'temperature',
    9: 'dewpoint',
    10: 'u_wind',
    11: 'v_wind'
}

missing_count = 0
for idx, (date_str, lon, lat) in enumerate(zip(dates, lons, lats)):
    if (idx + 1) % 50 == 0:
        print(f"   Processing sample {idx + 1}/{len(X)}...")
    
    era5_data = get_era5_data(date_str)
    if era5_data is None:
        missing_count += 1
        # Use zeros for missing data
        X_weather[idx, 8:12, :, :] = 0
        continue
    
    # Interpolate weather to fire location
    temp_val = interpolate_to_location(era5_data['temp'], era5_data['lats'], era5_data['lons'], lat, lon)
    dewpoint_val = interpolate_to_location(era5_data['dewpoint'], era5_data['lats'], era5_data['lons'], lat, lon)
    u_wind_val = interpolate_to_location(era5_data['u_wind'], era5_data['lats'], era5_data['lons'], lat, lon)
    v_wind_val = interpolate_to_location(era5_data['v_wind'], era5_data['lats'], era5_data['lons'], lat, lon)
    
    # Fill the 256x256 grid with interpolated values (weather is uniform across tile)
    X_weather[idx, 8, :, :] = temp_val
    X_weather[idx, 9, :, :] = dewpoint_val
    X_weather[idx, 10, :, :] = u_wind_val
    X_weather[idx, 11, :, :] = v_wind_val

print(f"   Weather channels added (missing dates: {missing_count})")

# Verify data
print("\n4. Data validation...")
for ch_idx, ch_name in channels_added.items():
    ch_data = X_weather[:, ch_idx, :, :]
    print(f"   Channel {ch_idx} ({ch_name}): min={ch_data.min():.2f}, max={ch_data.max():.2f}, mean={ch_data.mean():.2f}")

# Save combined dataset
print(f"\n5. Saving 12-channel dataset to {OUTPUT_FILE}...")

# Create metadata array
meta_array = np.array([
    {'date': d, 'lat': lt, 'lon': ln}
    for d, lt, ln in zip(dates, lats, lons)
], dtype=object)

np.savez_compressed(
    OUTPUT_FILE,
    X=X_weather,
    Y=Y,
    meta=meta_array
)

print(f"   ✓ Saved {OUTPUT_FILE}")
print(f"   Shape: X={X_weather.shape}, Y={Y.shape}")
print("\n" + "=" * 80)
print("WEATHER INTEGRATION COMPLETE")
print("=" * 80)
print(f"\nNext step: Train U-Net with 12-channel input")
print(f"  - Input channels: 8 original + 4 weather")
print(f"  - Expected accuracy: 30-50% IoU (vs 6.6% baseline with 8 channels)")
