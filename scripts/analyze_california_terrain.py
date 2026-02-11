#!/usr/bin/env python3
"""
Analyze real California wildfire terrain to extract statistical distributions.
This will guide our realistic synthetic data generation.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from src.data.real_data import build_real_input_stack

print("=" * 70)
print("CALIFORNIA WILDFIRE TERRAIN ANALYSIS")
print("=" * 70)

# Sample multiple CA wildfire-prone regions
regions = {
    'Butte County (Camp Fire)': (-121.3, 38.3, -120.8, 38.8),
    'Sonoma County (Tubbs Fire)': (-122.8, 38.3, -122.3, 38.6),
    'Santa Barbara (Thomas Fire)': (-119.8, 34.3, -119.3, 34.6),
    'Paradise Area': (-121.7, 39.6, -121.5, 39.8),
}

stats = {
    'slope': [],
    'aspect': [],
    'elevation': [],
    'fuel': []
}

print(f"\nFetching terrain data from {len(regions)} CA regions...")

for name, (w, s, e, n) in regions.items():
    print(f"\n[{name}]")
    print(f"  Bbox: ({w}, {s}, {e}, {n})")
    
    try:
        stack = build_real_input_stack(
            west=w, south=s, east=e, north=n,
            size=(256, 256),
            dem_source='3dep',
            wind_speed=20.0,
            wind_direction_deg=270.0
        )
        
        # Extract channels (slope, aspect_sin, aspect_cos, fuel, ...)
        # Channel 0: slope (normalized 0-1, actual 0-45 degrees)
        slope_deg = stack[0] * 45.0
        
        # Channel 1-2: aspect (sin, cos) - reconstruct aspect
        aspect_rad = np.arctan2(stack[1], stack[2])
        aspect_deg = (np.degrees(aspect_rad) + 360) % 360
        
        # Channel 3: fuel density
        fuel = stack[3]
        
        # Channel 7: initial fire (ignore, should be mostly zeros)
        
        # For elevation, we need to extract it differently
        # The stack doesn't directly contain elevation, but we can approximate from slope
        
        stats['slope'].extend(slope_deg.flatten())
        stats['aspect'].extend(aspect_deg.flatten())
        stats['fuel'].extend(fuel.flatten())
        
        print(f"  ✓ Slope: {slope_deg.mean():.1f}° (σ={slope_deg.std():.1f}°)")
        print(f"  ✓ Aspect: {aspect_deg.mean():.1f}° (σ={aspect_deg.std():.1f}°)")
        print(f"  ✓ Fuel: {fuel.mean():.3f} (σ={fuel.std():.3f})")
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:60]}")

# Convert to arrays
for key in stats:
    stats[key] = np.array(stats[key])

print("\n" + "=" * 70)
print("CALIFORNIA TERRAIN STATISTICS SUMMARY")
print("=" * 70)

if len(stats['slope']) > 0:
    print(f"\nSlope Distribution:")
    print(f"  Mean: {stats['slope'].mean():.2f}°")
    print(f"  Std:  {stats['slope'].std():.2f}°")
    print(f"  Min:  {stats['slope'].min():.2f}°")
    print(f"  Max:  {stats['slope'].max():.2f}°")
    print(f"  P50:  {np.percentile(stats['slope'], 50):.2f}°")
    print(f"  P90:  {np.percentile(stats['slope'], 90):.2f}°")
    
    print(f"\nAspect Distribution:")
    print(f"  Mean: {stats['aspect'].mean():.2f}°")
    print(f"  Std:  {stats['aspect'].std():.2f}°")
    
    print(f"\nFuel Density Distribution:")
    print(f"  Mean: {stats['fuel'].mean():.3f}")
    print(f"  Std:  {stats['fuel'].std():.3f}")
    print(f"  Min:  {stats['fuel'].min():.3f}")
    print(f"  Max:  {stats['fuel'].max():.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(stats['slope'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Slope (degrees)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('California Terrain Slope Distribution')
    axes[0, 0].axvline(stats['slope'].mean(), color='red', linestyle='--', label=f'Mean: {stats["slope"].mean():.1f}°')
    axes[0, 0].legend()
    
    axes[0, 1].hist(stats['aspect'], bins=36, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Aspect (degrees)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('California Terrain Aspect Distribution')
    
    axes[1, 0].hist(stats['fuel'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Fuel Density')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('California Fuel Density Distribution')
    axes[1, 0].axvline(stats['fuel'].mean(), color='red', linestyle='--', label=f'Mean: {stats["fuel"].mean():.2f}')
    axes[1, 0].legend()
    
    # Wind speed distribution (typical CA wildfire conditions)
    ca_wind_speeds = [15, 20, 25, 30, 35, 40, 45]  # mph
    axes[1, 1].hist(ca_wind_speeds, bins=10, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Wind Speed (mph)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Typical CA Wildfire Wind Speeds')
    
    plt.tight_layout()
    output_path = ROOT / 'outputs' / 'california_terrain_stats.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")
    
    # Save statistics to file
    stats_output = ROOT / 'outputs' / 'california_terrain_stats.npz'
    np.savez(stats_output, **stats)
    print(f"✓ Statistics saved: {stats_output}")
    
else:
    print("\n⚠ No data collected. Check internet connection and API endpoints.")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("\nNext steps:")
print("1. Use these statistics to update synthetic terrain generator")
print("2. Match slope, aspect, and fuel distributions")
print("3. Generate realistic CA wildfire training data")
