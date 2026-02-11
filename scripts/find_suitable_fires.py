#!/usr/bin/env python3
"""
Find fires in MTBS database that match our model's capabilities:
- Small to medium size (100-10,000 acres)
- Short duration (ideally 1-7 days)
- California fires
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import pandas as pd

MTBS_SHAPEFILE = ROOT / "data" / "mtbs_perimeters" / "mtbs_perims_DD.shp"

print("=" * 80)
print("FINDING SUITABLE VALIDATION FIRES")
print("=" * 80)

# Load MTBS
mtbs = gpd.read_file(MTBS_SHAPEFILE)
print(f"\nTotal fires in database: {len(mtbs)}")

# Filter for California fires
ca_fires = mtbs[
    (mtbs.geometry.centroid.x < -114) &  # West of Nevada
    (mtbs.geometry.centroid.x > -125) &  # East of Pacific
    (mtbs.geometry.centroid.y > 32) &    # North of Mexico
    (mtbs.geometry.centroid.y < 42)      # South of Oregon
]
print(f"California fires: {len(ca_fires)}")

# Filter for 2015-2023 (recent, good data)
ca_fires = ca_fires[(ca_fires['Ig_Date'].dt.year >= 2015) & 
                    (ca_fires['Ig_Date'].dt.year <= 2023)]
print(f"Recent CA fires (2015-2023): {len(ca_fires)}")

# Filter by size (100-10,000 acres)
suitable_fires = ca_fires[
    (ca_fires['BurnBndAc'] >= 100) & 
    (ca_fires['BurnBndAc'] <= 10000)
]
print(f"Size 100-10k acres: {len(suitable_fires)}")

# Sort by size
suitable_fires = suitable_fires.sort_values('BurnBndAc', ascending=False)

print("\n" + "=" * 80)
print("TOP 20 CANDIDATE FIRES FOR VALIDATION")
print("=" * 80)
print("\nThese fires are better matched to our model's 150-timestep (~6 day) capability:\n")

for idx, (_, fire) in enumerate(suitable_fires.head(20).iterrows(), 1):
    print(f"{idx:2d}. {fire['Incid_Name'][:30]:30s} | "
          f"{fire['BurnBndAc']:7,.0f} acres | "
          f"{fire['Ig_Date'].strftime('%Y-%m-%d')} | "
          f"({fire.geometry.centroid.y:.2f}, {fire.geometry.centroid.x:.2f})")

# Save to file for easy reference
output = ROOT / "outputs" / "suitable_validation_fires.csv"
suitable_fires[['Incid_Name', 'Ig_Date', 'BurnBndAc', 'BurnBndLat', 'BurnBndLon']].head(20).to_csv(
    output, index=False
)
print(f"\n✓ Full list saved: {output}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("\nPick a fire from this list and validate your model on it!")
print("Smaller fires (1,000-5,000 acres) are most likely to give good results.")
print("\nExample: Run validation on fire #1-5 to see if we get 20-40% IoU instead of 1.2%")
