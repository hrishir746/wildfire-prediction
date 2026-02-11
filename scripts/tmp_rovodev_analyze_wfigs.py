#!/usr/bin/env python3
"""
Analyze WFIGS perimeter data - this could be the key to 80% IoU!
"""
import pandas as pd
from datetime import datetime

print("="*80)
print("ANALYZING WFIGS PERIMETER DATA")
print("="*80)

df = pd.read_excel('WFIGS_Interagency_Perimeters_6828509647405635914.xlsx')

print(f"\nTotal perimeters: {len(df):,}")
print(f"\nColumns ({len(df.columns)}):")
for col in df.columns[:20]:
    print(f"  - {col}")

# Check for geometry
has_geom = False
geom_col = None
for col in ['SHAPE', 'geometry', 'poly_Geometry', 'Shape']:
    if col in df.columns:
        has_geom = True
        geom_col = col
        break

print(f"\nGeometry column: {geom_col if has_geom else 'NOT FOUND'}")

# Parse dates
df['date'] = pd.to_datetime(df['attr_FireDiscoveryDateTime'], errors='coerce')
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")

# Filter California
state_col = None
for col in ['attr_POOState', 'poly_State', 'poly_POOState']:
    if col in df.columns:
        state_col = col
        break

if state_col:
    ca_fires = df[df[state_col] == 'US-CA']
    print(f"\nCalifornia fires: {len(ca_fires):,}")
else:
    print("\nCannot filter by state (no state column found)")
    ca_fires = df

# Check 2020-2023 range
recent = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2023-12-31')]
print(f"2020-2023 fires: {len(recent):,}")

if state_col:
    ca_recent = recent[recent[state_col] == 'US-CA']
    print(f"California 2020-2023: {len(ca_recent):,}")

# Sample fires
print(f"\n{'='*80}")
print("SAMPLE FIRES")
print(f"{'='*80}")

sample = ca_recent.head(10) if len(ca_recent) > 0 else recent.head(10)
for idx, row in sample.iterrows():
    name = row.get('attr_IncidentName', row.get('poly_IncidentName', 'Unknown'))
    date = row.get('attr_FireDiscoveryDateTime', 'Unknown')
    acres = row.get('poly_GISAcres', row.get('attr_GISAcres', 0))
    print(f"  {name[:30]:30} | {str(date)[:10]} | {acres:>8.0f} acres")

print(f"\n{'='*80}")
print("POTENTIAL FOR TRAINING")
print(f"{'='*80}")

if has_geom:
    print("\n✅ HAS GEOMETRY DATA!")
    print("   This means we can extract actual fire perimeter polygons!")
    print("   → This is EXACTLY what FU-NetCastV2 used to get 94% accuracy!")
    print("\n🚀 WITH THIS DATA, 80% IoU IS POSSIBLE!")
else:
    print("\n⚠️  No geometry column found")
    print("   Need to check if geometry is in a different format")

print(f"\nNext steps to hit 80% IoU:")
print("  1. Extract fire perimeter polygons from geometry")
print("  2. Find fires with multiple perimeters (day-to-day progression)")
print("  3. Rasterize perimeters to 256x256 grids")
print("  4. Train U-Net on real perimeter progressions (not FIRMS points!)")
print("="*80)
