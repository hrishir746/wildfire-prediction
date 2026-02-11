#!/usr/bin/env python3
"""
Download full FIRMS historical data with API key.
This gets ALL California fires 2015-2023!
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests
import pandas as pd
from datetime import datetime
import time

print("=" * 80)
print("FULL FIRMS DATA DOWNLOAD")
print("=" * 80)

# GET YOUR API KEY:
# 1. Go to: https://firms.modaps.eosdis.nasa.gov/api/
# 2. Click "Request Key"
# 3. Fill form (instant approval)
# 4. Copy MAP_KEY from email
# 5. Paste below:

MAP_KEY = "7448b20b4322b4d7fb13a222d5b7436d"  # Your FIRMS API key

if MAP_KEY == "YOUR_API_KEY_HERE":
    print("\n❌ ERROR: You need to add your FIRMS API key!")
    print("\nSTEPS:")
    print("1. Go to: https://firms.modaps.eosdis.nasa.gov/api/")
    print("2. Click 'Request Key' (free, instant)")
    print("3. Check your email for MAP_KEY")
    print("4. Edit this script and paste key on line 22")
    print("5. Run again")
    sys.exit(1)

OUTPUT_DIR = ROOT / "data" / "firms_historical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# California bounding box
CA_BBOX = "-125.0,32.0,-114.0,42.0"  # west,south,east,north

# Download by year (VIIRS - better resolution)
YEARS = [2020, 2021, 2022, 2023]  # Start with recent years

print(f"\nDownloading VIIRS data for California")
print(f"Years: {YEARS}")
print(f"Bbox: {CA_BBOX}")
print(f"\nThis will download ~2-4 GB total")
print(f"Each year takes ~2-5 minutes on fast internet\n")

all_data = []

for year in YEARS:
    print(f"\n[{year}] Downloading...")
    
    # Try different satellites and endpoints
    # VIIRS has multiple satellites - try SNPP (older, more data)
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/{CA_BBOX}/1/{year}-01-01/{year}-12-31"
    
    try:
        response = requests.get(url, timeout=300)  # 5 min timeout
        response.raise_for_status()
        
        # Save raw
        output_file = OUTPUT_DIR / f"viirs_ca_{year}.csv"
        with open(output_file, 'w') as f:
            f.write(response.text)
        
        # Parse
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        print(f"  ✓ Downloaded {len(df):,} detections")
        print(f"  ✓ Saved: {output_file}")
        print(f"  ✓ Size: {output_file.stat().st_size / 1e6:.1f} MB")
        
        all_data.append(df)
        
        time.sleep(1)  # Be nice to API
        
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")

if all_data:
    # Combine all years
    combined = pd.concat(all_data, ignore_index=True)
    combined_file = OUTPUT_DIR / "viirs_ca_2020_2023.csv"
    combined.to_csv(combined_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"\n✓ Total detections: {len(combined):,}")
    print(f"✓ Combined file: {combined_file}")
    print(f"✓ Size: {combined_file.stat().st_size / 1e6:.1f} MB")
    
    # Show stats
    print(f"\n📊 Data Summary:")
    print(f"  Unique dates: {combined['acq_date'].nunique()}")
    print(f"  Date range: {combined['acq_date'].min()} to {combined['acq_date'].max()}")
    print(f"  High confidence detections: {len(combined[combined['confidence'] == 'h']):,}")
    
    print(f"\n🎯 NEXT: Process into fire progressions")
    print(f"  Run: python scripts/process_firms_into_progressions.py")
    
else:
    print(f"\n❌ No data downloaded. Check your API key!")
