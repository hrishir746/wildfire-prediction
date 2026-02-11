#!/usr/bin/env python3
"""
Download ERA5 hourly weather data for FIRMS fire dates.
Uses the ECMWF datastores client with a personal access token.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
from datetime import datetime
from collections import defaultdict

try:
    from ecmwf.datastores import Client
except ImportError:
    print("Missing dependency: ecmwf-datastores-client")
    print("Install with: pip install ecmwf-datastores-client")
    sys.exit(1)

print("=" * 80)
print("DOWNLOAD ERA5 WEATHER DATA FOR FIRMS FIRES")
print("=" * 80)

# Load FIRMS metadata to get fire dates/locations
FIRMS_META = ROOT / "data" / "firms_progressions" / "firms_progressions_256_meta.json"
OUTPUT_DIR = ROOT / "data" / "weather" / "era5_hourly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "reanalysis-era5-single-levels"
TIME_UTC = "12:00"
AREA = [42, -125, 32, -114]  # North, West, South, East (California bbox)
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

if not FIRMS_META.exists():
    print("FIRMS metadata not found. Run process_firms_progressions.py first.")
    sys.exit(1)

with open(FIRMS_META) as f:
    meta = json.load(f)

print(f"Loaded {meta['pairs']} FIRMS fire pairs")
print("\nVariables:")
for var in VARIABLES:
    print(f"  - {var}")

print("\nSetup required:")
print("Create %USERPROFILE%\\.ecmwfdatastoresrc with:")
print("  url: https://cds.climate.copernicus.eu/api")
print("  key: <your-personal-access-token>")

# Extract unique dates from full metadata
dates_needed = set()
for sample in meta["meta_samples"]:
    dates_needed.add(sample["date_t"])
    dates_needed.add(sample["date_t1"])

dates_needed = sorted(dates_needed)
print(f"\nUnique dates needed: {len(dates_needed)}")

groups = defaultdict(set)
for date_str in dates_needed:
    dt = datetime.fromisoformat(date_str)
    groups[(dt.year, dt.month)].add(dt.day)

client = Client()
try:
    client.check_authentication()
    print("\n✓ API authentication OK")
except Exception as exc:
    print(f"\n❌ API authentication failed: {exc}")
    print("Make sure %USERPROFILE%\\.ecmwfdatastoresrc is set up.")
    sys.exit(1)

index = {
    "dataset": DATASET,
    "variables": VARIABLES,
    "time_utc": TIME_UTC,
    "area": AREA,
    "files": [],
}

print("\n" + "=" * 80)
print("DOWNLOADING ERA5 (monthly batches)")
print("=" * 80)

for (year, month), days in sorted(groups.items()):
    target = OUTPUT_DIR / f"era5_{year}_{month:02d}.nc"
    if target.exists():
        print(f"Skipping existing: {target.name}")
        index["files"].append({"file": target.name, "year": year, "month": month, "days": sorted(days)})
        continue

    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in sorted(days)],
        "time": [TIME_UTC],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    print(f"Downloading {year}-{month:02d} ({len(days)} days)...")
    client.retrieve(DATASET, request, target=str(target))

    index["files"].append({"file": target.name, "year": year, "month": month, "days": sorted(days)})

index_file = OUTPUT_DIR / "era5_download_index.json"
with open(index_file, "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2)

print("\n✓ ERA5 download complete")
print(f"Index: {index_file}")
