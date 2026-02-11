#!/usr/bin/env python3
"""
Download FIRMS archive files directly.
These are pre-packaged yearly files - much faster!
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests
from tqdm import tqdm

print("=" * 80)
print("DOWNLOADING FIRMS ARCHIVE FILES")
print("=" * 80)

OUTPUT_DIR = ROOT / "data" / "firms_archives"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FIRMS archive URLs (publicly available, no key needed for bulk downloads)
# These are global files - we'll filter for CA after download

ARCHIVE_URLS = {
    2023: "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/2023/viirs-snpp_2023_United_States.csv",
    2022: "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/2022/viirs-snpp_2022_United_States.csv",
    2021: "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/2021/viirs-snpp_2021_United_States.csv",
    2020: "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/2020/viirs-snpp_2020_United_States.csv",
}

print("\nDownloading USA fire data (will filter for CA)")
print("Files are ~200-400 MB each (compressed)")
print("Total: ~1-2 GB\n")

for year, url in ARCHIVE_URLS.items():
    print(f"[{year}] Downloading...")
    output_file = OUTPUT_DIR / f"viirs_usa_{year}.csv"
    
    if output_file.exists():
        print(f"  ✓ Already exists: {output_file}")
        continue
    
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=f"  {year}"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        print(f"  ✓ Saved: {output_file}")
        print(f"  ✓ Size: {output_file.stat().st_size / 1e6:.1f} MB\n")
        
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}\n")

print("=" * 80)
print("FILTERING FOR CALIFORNIA")
print("=" * 80)

import pandas as pd

CA_BBOX = {'west': -125.0, 'south': 32.0, 'east': -114.0, 'north': 42.0}
all_ca_fires = []

for year in [2020, 2021, 2022, 2023]:
    file = OUTPUT_DIR / f"viirs_usa_{year}.csv"
    if not file.exists():
        continue
    
    print(f"\n[{year}] Filtering...")
    df = pd.read_csv(file)
    
    ca = df[
        (df['longitude'] >= CA_BBOX['west']) &
        (df['longitude'] <= CA_BBOX['east']) &
        (df['latitude'] >= CA_BBOX['south']) &
        (df['latitude'] <= CA_BBOX['north'])
    ]
    
    print(f"  USA: {len(df):,} detections")
    print(f"  CA:  {len(ca):,} detections")
    
    all_ca_fires.append(ca)

if all_ca_fires:
    combined = pd.concat(all_ca_fires, ignore_index=True)
    output = OUTPUT_DIR / "california_fires_2020_2023.csv"
    combined.to_csv(output, index=False)
    
    print(f"\n{'='*80}")
    print(f"SUCCESS!")
    print(f"{'='*80}")
    print(f"\n✓ Total CA detections: {len(combined):,}")
    print(f"✓ Saved: {output}")
    print(f"✓ Size: {output.stat().st_size / 1e6:.1f} MB")
    print(f"\n📊 Date range: {combined['acq_date'].min()} to {combined['acq_date'].max()}")
    print(f"📊 Unique days: {combined['acq_date'].nunique()}")
    
    print(f"\n🎯 NEXT: Process into day-by-day progressions")
    
else:
    print("\n❌ No data downloaded")
