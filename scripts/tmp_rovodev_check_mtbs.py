#!/usr/bin/env python3
"""
Check MTBS shapefile - do we have the geometry for 80% IoU?
"""
import geopandas as gpd
import pandas as pd

print("="*80)
print("CHECKING MTBS PERIMETER DATA")
print("="*80)

# Load MTBS
mtbs = gpd.read_file('data/mtbs_perimeters/mtbs_perims_DD.shp')

print(f"\nTotal fires in MTBS: {len(mtbs)}")
print(f"Columns: {list(mtbs.columns)}")

# California fires
ca_mtbs = mtbs[mtbs['State'] == 'CA']
print(f"\nCalifornia fires: {len(ca_mtbs)}")

# 2020-2023
ca_mtbs['Ig_Date'] = pd.to_datetime(ca_mtbs['Ig_Date'])
ca_recent = ca_mtbs[(ca_mtbs['Ig_Date'] >= '2020-01-01') & (ca_mtbs['Ig_Date'] <= '2023-12-31')]
print(f"California 2020-2023: {len(ca_recent)}")

print(f"\nSample fires:")
for idx, row in ca_recent.head(10).iterrows():
    name = row['Fire_Name']
    date = row['Ig_Date']
    acres = row['BurnBndAc']
    print(f"  {name[:30]:30} | {str(date)[:10]} | {acres:>8.0f} acres")

print(f"\n{'='*80}")
print("GEOMETRY CHECK")
print(f"{'='*80}")

print(f"\n✅ Has geometry: {mtbs.geometry.name}")
print(f"   Geometry types: {mtbs.geometry.type.unique()}")
print(f"   CRS: {mtbs.crs}")

print(f"\n{'='*80}")
print("THE PROBLEM WITH MTBS")
print(f"{'='*80}")

print("\nMTBS only has FINAL fire perimeters (end of fire)")
print("We need DAILY progressions (day 1, day 2, day 3...)")
print("\nMTBS shows:")
print("  - Ignition date")
print("  - Final perimeter (after entire fire)")
print("\nWhat we need for 80% IoU:")
print("  - Day 1 perimeter")
print("  - Day 2 perimeter")
print("  - Day 3 perimeter")
print("  - ... (daily progressions)")

print(f"\n{'='*80}")
print("NEXT STEPS")
print(f"{'='*80}")

print("\nOption 1: Use WFIGS data")
print("  - Need to download actual shapefile (not Excel)")
print("  - WFIGS has daily perimeter updates")
print("  - This is what FU-NetCastV2 used")

print("\nOption 2: Use FIRMS + MTBS")
print("  - FIRMS for daily progression (sparse)")
print("  - MTBS for final validation")
print("  - Hybrid approach")

print("\nOption 3: Stick with current 38.70%")
print("  - Already impressive for science fair")
print("  - Focus on presentation")

print("="*80)
