#!/usr/bin/env python3
"""
Normalize and rebuild the 12-channel weather-integrated dataset.
Scales weather channels to [0, 1] and combines with FIRMS input.
"""

import numpy as np
import netCDF4 as nc
from pathlib import Path
from scipy.interpolate import RectBivariateSpline

DATA_DIR = Path("data")
FIRMS_FILE = DATA_DIR / "firms_progressions" / "firms_progressions_256.npz"
OUTPUT_FILE = DATA_DIR / "firms_progressions" / "firms_progressions_weather_256.npz"
WEATHER_DIR = DATA_DIR / "weather" / "era5_hourly"

print("=" * 80)
print("NORMALIZE WEATHER DATA AND INTEGRATE WITH FIRMS")
print("=" * 80)

# Load original FIRMS data (already normalized)
print("\n1. Loading FIRMS data...")
firms_data = np.load(FIRMS_FILE)
X = firms_data['X']  # (300, 8, 256, 256) - already in [0,1]
Y = firms_data['Y']
print(f"   Loaded {len(X)} samples: {X.shape}")

# First pass: extract raw weather values to compute statistics
print("\n2. Building dataset with raw weather values...")
import pandas as pd
from collections import defaultdict

DATA_CSV = DATA_DIR / "firms_archives" / "california_fires_2020_2023.csv"
CA_BBOX = {"west": -125.0, "south": 32.0, "east": -114.0, "north": 42.0}
TILE_DEG = 0.25

df = pd.read_csv(DATA_CSV)
df = df[["latitude", "longitude", "acq_date"]].dropna()
df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
df = df.dropna(subset=["acq_date"])

df = df[
    (df["longitude"] >= CA_BBOX["west"]) &
    (df["longitude"] <= CA_BBOX["east"]) &
    (df["latitude"] >= CA_BBOX["south"]) &
    (df["latitude"] <= CA_BBOX["north"])
]

df["tile_i"] = ((df["latitude"] - CA_BBOX["south"]) / TILE_DEG).astype(int)
df["tile_j"] = ((df["longitude"] - CA_BBOX["west"]) / TILE_DEG).astype(int)

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
        
        tile_center_lat = CA_BBOX["south"] + (tile_i + 0.5) * TILE_DEG
        tile_center_lon = CA_BBOX["west"] + (tile_j + 0.5) * TILE_DEG
        
        dates.append(d0)
        lats.append(tile_center_lat)
        lons.append(tile_center_lon)
        sample_count += 1

print(f"   Reconstructed {len(dates)} samples (date range: {min(dates)} to {max(dates)})")

# Load ERA5 data and build 12-channel dataset
print("\n3. Loading ERA5 data and building 12-channel dataset...")
X_weather = np.zeros((len(X), 12, 256, 256), dtype=np.float32)
X_weather[:, :8, :, :] = X  # Copy original 8 normalized channels

era5_cache = {}
weather_values = {8: [], 9: [], 10: [], 11: []}  # For normalization statistics

def get_era5_data(date_str):
    """Load ERA5 data for a specific month."""
    year, month, day = date_str.split('-')
    key = f"{year}_{month}"
    
    if key not in era5_cache:
        era5_file = WEATHER_DIR / f"era5_{year}_{int(month):02d}.nc"
        if not era5_file.exists():
            return None
        
        with nc.Dataset(era5_file, 'r') as ds:
            lats_grid = ds.variables['latitude'][:]
            lons_grid = ds.variables['longitude'][:]
            
            temp = ds.variables['t2m'][0] - 273.15  # Celsius
            dewpoint = ds.variables['d2m'][0] - 273.15
            u_wind = ds.variables['u10'][0]
            v_wind = ds.variables['v10'][0]
            
            era5_cache[key] = {
                'lats': lats_grid,
                'lons': lons_grid,
                'temp': temp,
                'dewpoint': dewpoint,
                'u_wind': u_wind,
                'v_wind': v_wind
            }
        
        print(f"   Loaded {era5_file.name}")
    
    return era5_cache[key]

def interpolate_to_location(data_2d, lats_grid, lons_grid, lat, lon):
    """Nearest neighbor interpolation."""
    lat_idx = np.argmin(np.abs(lats_grid - lat))
    lon_idx = np.argmin(np.abs(lons_grid - lon))
    return data_2d[lat_idx, lon_idx]

# Build 12-channel data and compute statistics
for idx, (date_str, lon, lat) in enumerate(zip(dates, lons, lats)):
    if (idx + 1) % 100 == 0:
        print(f"   Processing sample {idx + 1}/{len(X)}...")
    
    era5_data = get_era5_data(date_str)
    if era5_data is None:
        X_weather[idx, 8:12, :, :] = 0
        continue
    
    temp_val = interpolate_to_location(era5_data['temp'], era5_data['lats'], era5_data['lons'], lat, lon)
    dewpoint_val = interpolate_to_location(era5_data['dewpoint'], era5_data['lats'], era5_data['lons'], lat, lon)
    u_wind_val = interpolate_to_location(era5_data['u_wind'], era5_data['lats'], era5_data['lons'], lat, lon)
    v_wind_val = interpolate_to_location(era5_data['v_wind'], era5_data['lats'], era5_data['lons'], lat, lon)
    
    X_weather[idx, 8, :, :] = temp_val
    X_weather[idx, 9, :, :] = dewpoint_val
    X_weather[idx, 10, :, :] = u_wind_val
    X_weather[idx, 11, :, :] = v_wind_val
    
    # Collect values for statistics
    weather_values[8].append(temp_val)
    weather_values[9].append(dewpoint_val)
    weather_values[10].append(u_wind_val)
    weather_values[11].append(v_wind_val)

# Compute normalization statistics
print("\n4. Computing normalization statistics...")
stats = {}
channel_names = {8: 'temperature (°C)', 9: 'dewpoint (°C)', 10: 'u_wind (m/s)', 11: 'v_wind (m/s)'}
for ch_idx in [8, 9, 10, 11]:
    vals = np.array(weather_values[ch_idx])
    ch_min = vals.min()
    ch_max = vals.max()
    stats[ch_idx] = {'min': ch_min, 'max': ch_max}
    print(f"   Channel {ch_idx} ({channel_names[ch_idx]}): {ch_min:.2f} to {ch_max:.2f}")

# Normalize weather channels to [0, 1]
print("\n5. Normalizing weather channels to [0, 1]...")
for ch_idx in [8, 9, 10, 11]:
    ch_min = stats[ch_idx]['min']
    ch_max = stats[ch_idx]['max']
    ch_range = ch_max - ch_min
    if ch_range > 0:
        X_weather[:, ch_idx, :, :] = (X_weather[:, ch_idx, :, :] - ch_min) / ch_range
    else:
        X_weather[:, ch_idx, :, :] = 0.5  # Middle value if no range
    
    print(f"   Channel {ch_idx}: normalized to [0, 1]")

# Verify normalized data
print("\n6. Data validation...")
for ch_idx in [8, 9, 10, 11]:
    ch_data = X_weather[:, ch_idx, :, :]
    print(f"   Channel {ch_idx}: min={ch_data.min():.3f}, max={ch_data.max():.3f}, mean={ch_data.mean():.3f}")

# Save normalized dataset
print(f"\n7. Saving normalized 12-channel dataset...")
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
print(f"   Normalization stats: {stats}")

print("\n" + "=" * 80)
print("NORMALIZATION COMPLETE")
print("=" * 80)
print("\nAll 12 channels now normalized to [0, 1]:")
print("  Channels 0-7: Original terrain/fire data (already normalized)")
print("  Channels 8-11: ERA5 weather (now normalized)")
print("\nReady to train 12-channel model.")
