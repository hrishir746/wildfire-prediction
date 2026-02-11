#!/usr/bin/env python3
"""
Process FIRMS VIIRS detections into temporal fire progression training data.

TEMPORAL VERSION: Uses 3-day history as input.

Input: data/firms_archives/california_fires_2020_2023.csv
Output: data/firms_progressions/firms_progressions_temporal_*.npz

Builds samples where input is days T-2, T-1, T fire masks (plus terrain/fuel/wind),
and target is day T+1 fire mask. This provides temporal context for better prediction.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.real_data import build_real_input_stack

DATA_CSV = ROOT / "data" / "firms_archives" / "california_fires_2020_2023.csv"
OUTPUT_DIR = ROOT / "data" / "firms_progressions"
CACHE_DIR = OUTPUT_DIR / "tile_cache"

CA_BBOX = {
    "west": -125.0,
    "south": 32.0,
    "east": -114.0,
    "north": 42.0,
}


def _tile_bounds(tile_i: int, tile_j: int, tile_deg: float) -> tuple[float, float, float, float]:
    west = CA_BBOX["west"] + tile_j * tile_deg
    east = west + tile_deg
    south = CA_BBOX["south"] + tile_i * tile_deg
    north = south + tile_deg
    return west, south, east, north


def _rasterize_points(points: np.ndarray, west: float, south: float, east: float, north: float, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    if len(points) == 0:
        return mask

    lon = points[:, 0]
    lat = points[:, 1]
    # Normalize to 0..1
    x_norm = (lon - west) / (east - west)
    y_norm = (lat - south) / (north - south)
    # Convert to pixel indices
    x = np.clip((x_norm * (size - 1)).astype(int), 0, size - 1)
    y = np.clip((y_norm * (size - 1)).astype(int), 0, size - 1)
    # Flip y so north is at the top
    row = (size - 1) - y
    col = x
    mask[row, col] = 1.0
    return mask


def _load_or_create_terrain(tile_i: int, tile_j: int, tile_deg: float, size: int) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"tile_{tile_i}_{tile_j}_{size}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        return data["stack"]

    west, south, east, north = _tile_bounds(tile_i, tile_j, tile_deg)

    # Build terrain stack with empty initial fire (channels will be overwritten per sample).
    stack = build_real_input_stack(
        west=west,
        south=south,
        east=east,
        north=north,
        size=(size, size),
        dem_source="3dep",
        wind_speed=5.0,
        wind_direction_deg=270.0,
        initial_fire=np.zeros((size, size), dtype=np.float32),
    ).astype(np.float32)

    np.savez_compressed(cache_file, stack=stack)
    return stack


def main():
    if not DATA_CSV.exists():
        print("FIRMS CA data not found.")
        print(f"Expected: {DATA_CSV}")
        print("Run: python scripts/download_firms_archives.py")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_size = 256
    tile_deg = 1.0  # Larger tiles to capture more fire points (4x area)
    max_samples = 200  # Fewer samples but richer data
    min_points_per_day = 10  # Need substantial fire per sample
    history_days = 3  # Use 3 consecutive days as input

    print("=" * 80)
    print("PROCESS FIRMS PROGRESSIONS - TEMPORAL VERSION")
    print("=" * 80)
    print(f"Input CSV: {DATA_CSV}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Tile size: {tile_deg} deg")
    print(f"Max samples: {max_samples}")
    print(f"Min points/day: {min_points_per_day}")
    print(f"History days: {history_days}")

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

    print(f"Detections after CA filter: {len(df):,}")

    # Assign tile indices
    df["tile_i"] = ((df["latitude"] - CA_BBOX["south"]) / tile_deg).astype(int)
    df["tile_j"] = ((df["longitude"] - CA_BBOX["west"]) / tile_deg).astype(int)

    # Group points by tile and date
    grouped = defaultdict(lambda: defaultdict(list))
    for row in df.itertuples(index=False):
        tile_id = (int(row.tile_i), int(row.tile_j))
        date_key = row.acq_date.date().isoformat()
        grouped[tile_id][date_key].append((row.longitude, row.latitude))

    print(f"Tiles with detections: {len(grouped):,}")

    X_list = []
    Y_list = []
    meta = []

    sample_count = 0
    for (tile_i, tile_j), date_map in grouped.items():
        if sample_count >= max_samples:
            break

        dates = sorted(date_map.keys())
        # Need at least history_days + 1 days for temporal sequence
        if len(dates) < history_days + 1:
            continue

        # Precompute terrain stack for this tile
        try:
            terrain_stack = _load_or_create_terrain(tile_i, tile_j, tile_deg, grid_size)
        except Exception as exc:
            print(f"Tile {tile_i},{tile_j} terrain failed: {str(exc)[:80]}")
            continue

        # Create temporal sequences: days[i-2], days[i-1], days[i] -> days[i+1]
        for idx in range(history_days, len(dates)):
            if sample_count >= max_samples:
                break

            # Get history_days consecutive days plus the target day
            sequence_dates = dates[idx - history_days:idx]
            target_date = dates[idx]

            # Check all days have enough points
            valid_sequence = True
            for d in sequence_dates + [target_date]:
                if len(date_map[d]) < min_points_per_day:
                    valid_sequence = False
                    break

            if not valid_sequence:
                continue

            west, south, east, north = _tile_bounds(tile_i, tile_j, tile_deg)

            # Rasterize fire masks for history days
            fire_history = []
            for d in sequence_dates:
                pts = date_map[d]
                mask = _rasterize_points(np.array(pts), west, south, east, north, grid_size)
                fire_history.append(mask)

            # Rasterize target
            target_pts = date_map[target_date]
            mask_target = _rasterize_points(np.array(target_pts), west, south, east, north, grid_size)

            # Skip if target is empty
            if mask_target.sum() == 0:
                continue

            # Build input: terrain (7 channels) + fire history (3 channels) = 10 channels
            # terrain_stack is (8, H, W) where channel 7 was placeholder fire
            # Take first 7 channels (terrain features), then add 3 fire history channels
            x = np.concatenate([
                terrain_stack[:7],  # Terrain features (slope, aspect, fuel, wind)
                np.stack(fire_history, axis=0)  # Fire history (3 days)
            ], axis=0).astype(np.float32)

            y = mask_target[np.newaxis, ...].astype(np.float32)

            X_list.append(x)
            Y_list.append(y)
            meta.append({
                "tile_i": tile_i,
                "tile_j": tile_j,
                "date_history": sequence_dates,
                "date_target": target_date,
                "points_history": [int(len(date_map[d])) for d in sequence_dates],
                "points_target": int(len(target_pts)),
            })
            sample_count += 1

        if sample_count > 0 and sample_count % 50 == 0:
            print(f"Built {sample_count} samples...")

    if sample_count == 0:
        print("No training samples created. Check filters or tile size.")
        sys.exit(1)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)

    output_file = OUTPUT_DIR / f"firms_progressions_temporal_{grid_size}.npz"
    np.savez_compressed(output_file, X=X, Y=Y)

    meta_file = OUTPUT_DIR / f"firms_progressions_temporal_{grid_size}_meta.json"
    import json
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "grid_size": grid_size,
            "tile_deg": tile_deg,
            "max_samples": max_samples,
            "min_points_per_day": min_points_per_day,
            "history_days": history_days,
            "samples": sample_count,
            "input_channels": X.shape[1],
            "meta_samples": meta[:100],
        }, f, indent=2, default=str)

    print("=" * 80)
    print("DONE")
    print(f"Samples created: {sample_count}")
    print(f"Input shape: {X.shape} ({X.shape[1]} channels)")
    print(f"Target shape: {Y.shape}")
    print(f"Dataset: {output_file}")
    print(f"Meta: {meta_file}")


if __name__ == "__main__":
    main()
