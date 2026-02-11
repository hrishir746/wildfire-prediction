#!/usr/bin/env python3
"""
Download WFIGS fire perimeter progressions for California 2020-2023.
This is the data that will get us to 80% IoU!
"""
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
print("DOWNLOADING WFIGS PERIMETER PROGRESSIONS")
print("="*80)

# Step 1: Get list of unique fires in CA 2020-2023
print("\nStep 1: Finding California fires (2020-2023)...")

params = {
    'where': "attr_POOState='US-CA' AND attr_FireDiscoveryDateTime >= timestamp '2020-01-01 00:00:00' AND attr_FireDiscoveryDateTime < timestamp '2024-01-01 00:00:00'",
    'outFields': 'attr_IncidentName,attr_IrwinID,attr_FireDiscoveryDateTime',
    'f': 'json',
    'returnGeometry': 'false',
    'returnDistinctValues': 'true',
    'resultRecordCount': 1000
}

response = requests.get(API_URL, params=params, timeout=60)
if response.status_code != 200:
    print(f"❌ Failed to query API: {response.status_code}")
    exit(1)

data = response.json()
fires = data.get('features', [])
print(f"✓ Found {len(fires)} unique California fires")

# Step 2: For each fire, get ALL perimeters
print("\nStep 2: Downloading perimeters for each fire...")

all_perimeters = []
fires_with_progressions = []

for fire in tqdm(fires, desc="Processing fires"):
    attrs = fire['attributes']
    fire_name = attrs.get('attr_IncidentName', 'Unknown')
    irwin_id = attrs.get('attr_IrwinID')
    
    if not fire_name or fire_name == 'Unknown':
        continue
    
    # Get all perimeters for this fire
    params2 = {
        'where': f"attr_IncidentName='{fire_name}' AND attr_FireDiscoveryDateTime >= timestamp '2020-01-01 00:00:00'",
        'outFields': '*',
        'f': 'json',
        'returnGeometry': 'true',
        'orderByFields': 'poly_PolygonDateTime',
        'resultRecordCount': 100
    }
    
    try:
        response2 = requests.get(API_URL, params=params2, timeout=30)
        if response2.status_code == 200:
            data2 = response2.json()
            perimeters = data2.get('features', [])
            
            if len(perimeters) >= 2:  # Need at least 2 perimeters for progression
                fires_with_progressions.append({
                    'name': fire_name,
                    'irwin_id': irwin_id,
                    'num_perimeters': len(perimeters),
                    'perimeters': perimeters
                })
                all_perimeters.extend(perimeters)
        
        time.sleep(0.1)  # Rate limiting
        
    except Exception as e:
        tqdm.write(f"Error fetching {fire_name}: {str(e)[:50]}")
        continue

print(f"\n✓ Found {len(fires_with_progressions)} fires with multiple perimeters!")
print(f"✓ Total perimeters: {len(all_perimeters)}")

# Step 3: Save data
print("\nStep 3: Saving data...")

output_file = OUTPUT_DIR / "california_fire_progressions_2020_2023.json"
with open(output_file, 'w') as f:
    json.dump({
        'metadata': {
            'download_date': datetime.now().isoformat(),
            'num_fires': len(fires_with_progressions),
            'num_perimeters': len(all_perimeters),
            'date_range': '2020-2023',
            'state': 'California'
        },
        'fires': fires_with_progressions
    }, f, indent=2)

print(f"✓ Saved to: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1e6:.1f} MB")

# Summary
print(f"\n{'='*80}")
print("DOWNLOAD SUMMARY")
print(f"{'='*80}")

print(f"\nFires with progressions: {len(fires_with_progressions)}")
print(f"Total perimeters: {len(all_perimeters)}")

# Show top fires by number of perimeters
fires_with_progressions.sort(key=lambda x: x['num_perimeters'], reverse=True)
print(f"\nTop 10 fires by number of perimeter updates:")
for i, fire in enumerate(fires_with_progressions[:10]):
    print(f"  {i+1}. {fire['name'][:40]:40} - {fire['num_perimeters']} perimeters")

print(f"\n{'='*80}")
print("NEXT STEPS")
print(f"{'='*80}")
print("\n1. Process perimeters into raster grids (256x256)")
print("2. Create day-to-day progression pairs (T → T+1)")
print("3. Extract terrain/weather for each location")
print("4. Train U-Net on real perimeter progressions")
print("5. Target: 70-80% IoU!")
print("="*80)
