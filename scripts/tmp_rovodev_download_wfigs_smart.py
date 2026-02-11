#!/usr/bin/env python3
"""
Smart WFIGS downloader - use your Excel file to find fires, then get perimeters
"""
import pandas as pd
import requests
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "wfigs_perimeters"
OUTPUT_DIR.mkdir(exist_ok=True)

API_URL = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query"

print("="*80)
print("SMART WFIGS DOWNLOAD - Using Excel file")
print("="*80)

# Load the Excel file you already have
print("\nLoading your WFIGS Excel file...")
df = pd.read_excel('WFIGS_Interagency_Perimeters_6828509647405635914.xlsx')
print(f"Total records: {len(df):,}")

# Filter California 2020-2023
df['date'] = pd.to_datetime(df['attr_FireDiscoveryDateTime'], errors='coerce')
ca_recent = df[
    (df['attr_POOState'] == 'US-CA') &
    (df['date'] >= '2020-01-01') &
    (df['date'] < '2024-01-01')
]

print(f"CA 2020-2023 records: {len(ca_recent):,}")

# Group by fire name to find fires with multiple perimeters
fire_counts = ca_recent.groupby('attr_IncidentName').size().sort_values(ascending=False)
fires_with_multiple = fire_counts[fire_counts >= 2]

print(f"\nFires with 2+ perimeters: {len(fires_with_multiple)}")
print(f"\nTop 20 fires by perimeter count:")
for name, count in fires_with_multiple.head(20).items():
    print(f"  {name[:40]:40} - {count} perimeters")

# Download top 50 fires with most perimeters
TOP_N = 50
print(f"\n{'='*80}")
print(f"Downloading geometry for top {TOP_N} fires...")
print(f"{'='*80}")

downloaded_fires = []

for fire_name in tqdm(list(fires_with_multiple.head(TOP_N).index), desc="Downloading"):
    # Query API for this fire's perimeters
    params = {
        'where': f"attr_IncidentName='{fire_name}' AND attr_POOState='US-CA' AND attr_FireDiscoveryDateTime >= timestamp '2020-01-01 00:00:00'",
        'outFields': 'attr_IncidentName,attr_FireDiscoveryDateTime,poly_PolygonDateTime,poly_GISAcres,attr_IrwinID',
        'f': 'json',
        'returnGeometry': 'true',
        'orderByFields': 'poly_PolygonDateTime',
        'resultRecordCount': 100
    }
    
    try:
        response = requests.get(API_URL, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            
            if len(features) >= 2:
                # Parse dates
                for feat in features:
                    poly_date = feat['attributes'].get('poly_PolygonDateTime')
                    if poly_date:
                        feat['attributes']['poly_date_parsed'] = datetime.fromtimestamp(poly_date/1000).isoformat()
                
                downloaded_fires.append({
                    'name': fire_name,
                    'num_perimeters': len(features),
                    'perimeters': features
                })
                
                tqdm.write(f"✓ {fire_name}: {len(features)} perimeters")
        
        time.sleep(0.2)  # Rate limiting
        
    except Exception as e:
        tqdm.write(f"✗ {fire_name}: {str(e)[:50]}")
        continue

# Save
print(f"\n{'='*80}")
print("Saving data...")
print(f"{'='*80}")

output_file = OUTPUT_DIR / "wfigs_top_fires_with_geometry.json"
with open(output_file, 'w') as f:
    json.dump({
        'metadata': {
            'download_date': datetime.now().isoformat(),
            'num_fires': len(downloaded_fires),
            'total_perimeters': sum(f['num_perimeters'] for f in downloaded_fires),
            'date_range': '2020-2023',
            'state': 'California',
            'source': 'WFIGS API + Excel file'
        },
        'fires': downloaded_fires
    }, f, indent=2)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1e6:.1f} MB")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"\nDownloaded: {len(downloaded_fires)} fires")
print(f"Total perimeters: {sum(f['num_perimeters'] for f in downloaded_fires)}")

# Calculate potential training samples
total_pairs = sum(f['num_perimeters'] - 1 for f in downloaded_fires if f['num_perimeters'] > 1)
print(f"Potential T→T+1 pairs: {total_pairs}")

print(f"\nTop fires:")
for fire in sorted(downloaded_fires, key=lambda x: x['num_perimeters'], reverse=True)[:10]:
    print(f"  {fire['name'][:40]:40} - {fire['num_perimeters']} perimeters")

print(f"\n🚀 Ready for rasterization and training!")
print("="*80)
