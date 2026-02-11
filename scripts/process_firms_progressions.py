#!/usr/bin/env python3
"""
Process FIRMS VIIRS detections into day-to-day fire progression training data.

Input: data/firms_archives/california_fires_2020_2023.csv
Output: data/firms_progressions/firms_progressions_*.npz

This builds samples where input is the day T fire mask (plus terrain/fuel/wind),
and target is day T+1 fire mask. This is a realistic 24h prediction task.
"""
import sys
from pathlib import Path
from datetime import datetime
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

    # Build terrain stack with empty initial fire (channel 7 will be overwritten per sample).
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
    tile_deg = 0.25
    max_pairs = 300
    min_points_per_day = 3

    print("=" * 80)
    print("PROCESS FIRMS PROGRESSIONS")
    print("=" * 80)
    print(f"Input CSV: {DATA_CSV}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Tile size: {tile_deg} deg")
    print(f"Max pairs: {max_pairs}")
    print(f"Min points/day: {min_points_per_day}")

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

    pair_count = 0
    for (tile_i, tile_j), date_map in grouped.items():
        if pair_count >= max_pairs:
            break

        dates = sorted(date_map.keys())
        if len(dates) < 2:
            continue

        # Precompute terrain stack for this tile
        try:
            terrain_stack = _load_or_create_terrain(tile_i, tile_j, tile_deg, grid_size)
        except Exception as exc:
            print(f"Tile {tile_i},{tile_j} terrain failed: {str(exc)[:80]}")
            continue

        for idx in range(len(dates) - 1):
            if pair_count >= max_pairs:
                break

            d0 = dates[idx]
            d1 = dates[idx + 1]
            pts0 = date_map[d0]
            pts1 = date_map[d1]

            if len(pts0) < min_points_per_day or len(pts1) < min_points_per_day:
                continue

            west, south, east, north = _tile_bounds(tile_i, tile_j, tile_deg)
            mask_t = _rasterize_points(np.array(pts0), west, south, east, north, grid_size)
            mask_t1 = _rasterize_points(np.array(pts1), west, south, east, north, grid_size)

            # Skip if target is empty
            if mask_t1.sum() == 0:
                continue

            # Build input: copy terrain and set initial fire channel
            x = terrain_stack.copy()
            x[7] = mask_t
            y = mask_t1[np.newaxis, ...]

            X_list.append(x.astype(np.float32))
            Y_list.append(y.astype(np.float32))
            meta.append({
                "tile_i": tile_i,
                "tile_j": tile_j,
                "date_t": d0,
                "date_t1": d1,
                "points_t": int(len(pts0)),
                "points_t1": int(len(pts1)),
            })
            pair_count += 1

        if pair_count > 0 and pair_count % 50 == 0:
            print(f"Built {pair_count} pairs...")

    if pair_count == 0:
        print("No training pairs created. Check filters or tile size.")
        sys.exit(1)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)

    output_file = OUTPUT_DIR / f"firms_progressions_{grid_size}.npz"
    np.savez_compressed(output_file, X=X, Y=Y)

    meta_file = OUTPUT_DIR / f"firms_progressions_{grid_size}_meta.json"
    import json
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "grid_size": grid_size,
            "tile_deg": tile_deg,
            "max_pairs": max_pairs,
            "min_points_per_day": min_points_per_day,
            "pairs": pair_count,
            "meta_samples": meta,
        }, f, indent=2)

    print("=" * 80)
    print("DONE")
    print(f"Pairs created: {pair_count}")
    print(f"Dataset: {output_file}")
    print(f"Meta: {meta_file}")


if __name__ == "__main__":
    main()
