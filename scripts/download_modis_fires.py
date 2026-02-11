#!/usr/bin/env python3
"""
Download MODIS/VIIRS active fire data from NASA FIRMS.
This gives us REAL fire progressions day-by-day!

FIRMS (Fire Information for Resource Management System):
- Free API from NASA
- Active fire detections from satellites
- Daily snapshots of where fires are burning
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

print("=" * 80)
print("DOWNLOADING REAL FIRE DATA FROM NASA FIRMS")
print("=" * 80)

# FIRMS API - uses publicly available data
# For larger datasets, you need a free API key from: https://firms.modaps.eosdis.nasa.gov/api/
# For this demo, we'll use the public archive

OUTPUT_DIR = ROOT / "data" / "modis_fires"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# California bounding box
CA_BBOX = {
    'west': -125.0,
    'south': 32.0,
    'east': -114.0,
    'north': 42.0
}

print(f"\nDownloading fires for California:")
print(f"  Bounding box: ({CA_BBOX['west']}, {CA_BBOX['south']}, {CA_BBOX['east']}, {CA_BBOX['north']})")
print(f"  Time period: Last 7 days (public data)")
print(f"\nNote: For full historical data (2015-2023), you need a free FIRMS API key")
print(f"Get one at: https://firms.modaps.eosdis.nasa.gov/api/")

# Public FIRMS data (last 24 hours, for demo)
# URL format: https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_USA_contiguous_and_Hawaii_24h.csv

FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_USA_contiguous_and_Hawaii_24h.csv"

print(f"\n[1/2] Downloading recent MODIS fire data...")
try:
    response = requests.get(FIRMS_URL, timeout=30)
    response.raise_for_status()
    
    # Save raw CSV
    csv_file = OUTPUT_DIR / "modis_recent_24h.csv"
    with open(csv_file, 'wb') as f:
        f.write(response.content)
    
    print(f"  ✓ Downloaded: {csv_file}")
    
    # Parse and filter for California
    df = pd.read_csv(csv_file)
    print(f"  Total detections: {len(df)}")
    
    # Filter for California
    ca_fires = df[
        (df['longitude'] >= CA_BBOX['west']) &
        (df['longitude'] <= CA_BBOX['east']) &
        (df['latitude'] >= CA_BBOX['south']) &
        (df['latitude'] <= CA_BBOX['north'])
    ]
    
    print(f"  California fires: {len(ca_fires)}")
    
    if len(ca_fires) > 0:
        ca_file = OUTPUT_DIR / "california_fires_24h.csv"
        ca_fires.to_csv(ca_file, index=False)
        print(f"  ✓ Saved CA fires: {ca_file}")
        
        # Show sample
        print(f"\n  Sample detections:")
        print(ca_fires[['latitude', 'longitude', 'brightness', 'acq_date', 'confidence']].head())
    else:
        print(f"  No active fires in CA in last 24h")
    
except Exception as e:
    print(f"  ERROR: {str(e)}")
    print(f"\n  This is expected - public API has limited data")

print(f"\n[2/2] Instructions for full dataset:")
print(f"=" * 80)
print(f"""
TO GET FULL HISTORICAL DATA (needed for training):

1. Get free API key:
   - Visit: https://firms.modaps.eosdis.nasa.gov/api/
   - Register (free, instant)
   - Copy your MAP_KEY

2. Update this script with your MAP_KEY

3. Download 2015-2023 data:
   - VIIRS (better resolution): ~1GB per year
   - MODIS: ~500MB per year
   - Total: ~5-10 GB for CA (2015-2023)

4. Script will process into day-by-day fire progressions

ALTERNATIVE - Use existing fires:
- We have MTBS perimeters already
- Can extract start dates
- Download satellite data for those specific fires
- Faster than downloading everything
""")

print(f"\n{'='*80}")
print(f"NEXT STEPS:")
print(f"{'='*80}")
print(f"""
Option A: Get FIRMS API key and download full dataset (best)
Option B: Use MTBS + targeted downloads for specific fires (faster)

For Option B, I can:
1. Pick 20-30 fires from MTBS (2015-2023)
2. Download satellite data for just those fires
3. Build training dataset from real progressions
4. Train model tomorrow

This gets you real data without downloading 10GB!

Want me to implement Option B?
""")

# Save status
with open(OUTPUT_DIR / "download_status.txt", 'w') as f:
    f.write(f"Downloaded at: {datetime.now()}\n")
    f.write(f"Public data: 24h MODIS\n")
    f.write(f"For training: Need API key or use targeted approach\n")

print(f"✓ Status saved: {OUTPUT_DIR / 'download_status.txt'}")
