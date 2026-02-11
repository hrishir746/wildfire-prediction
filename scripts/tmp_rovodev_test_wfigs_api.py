#!/usr/bin/env python3
"""
Test WFIGS API - can we get daily fire perimeters?
This could be the path to 80% IoU!
"""
import requests
import json
from datetime import datetime

API_URL = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query"

print("="*80)
print("TESTING WFIGS API")
print("="*80)

# Test query: Get California fires from 2020
params = {
    'where': "attr_POOState='US-CA' AND attr_FireDiscoveryDateTime >= timestamp '2020-01-01 00:00:00' AND attr_FireDiscoveryDateTime < timestamp '2021-01-01 00:00:00'",
    'outFields': 'attr_IncidentName,attr_FireDiscoveryDateTime,poly_PolygonDateTime,poly_GISAcres,attr_IrwinID',
    'f': 'json',
    'returnGeometry': 'true',
    'resultRecordCount': 10
}

print("\nQuerying API for CA 2020 fires (first 10)...")
response = requests.get(API_URL, params=params, timeout=30)

if response.status_code == 200:
    data = response.json()
    
    if 'features' in data:
        features = data['features']
        print(f"\n✅ Got {len(features)} fire perimeters!")
        
        print(f"\nSample fires:")
        for feat in features[:5]:
            attrs = feat['attributes']
            geom = feat.get('geometry', {})
            
            name = attrs.get('attr_IncidentName', 'Unknown')
            disc_date = attrs.get('attr_FireDiscoveryDateTime')
            poly_date = attrs.get('poly_PolygonDateTime')
            acres = attrs.get('poly_GISAcres', 0)
            
            # Convert timestamps
            if disc_date:
                disc_date = datetime.fromtimestamp(disc_date/1000).strftime('%Y-%m-%d')
            if poly_date:
                poly_date = datetime.fromtimestamp(poly_date/1000).strftime('%Y-%m-%d')
            
            has_geom = 'rings' in geom or 'paths' in geom
            
            acres_str = f"{acres:>6.0f}" if acres else "   N/A"
            print(f"  {name[:30]:30} | Disc: {disc_date} | Poly: {poly_date} | {acres_str} ac | Geom: {has_geom}")
        
        # Check if we have geometry
        if features and 'geometry' in features[0]:
            print(f"\n✅ API RETURNS GEOMETRY!")
            print(f"   Geometry type: {features[0]['geometry'].get('type', 'polygon rings')}")
            
            # Check for multiple perimeters of same fire
            print(f"\n{'='*80}")
            print("CHECKING FOR DAILY PROGRESSIONS")
            print(f"{'='*80}")
            
            # Query for fires with multiple perimeters
            params2 = {
                'where': "attr_IncidentName='CREEK' AND attr_FireDiscoveryDateTime >= timestamp '2020-01-01 00:00:00'",
                'outFields': 'attr_IncidentName,poly_PolygonDateTime,poly_GISAcres',
                'f': 'json',
                'returnGeometry': 'false',
                'orderByFields': 'poly_PolygonDateTime',
                'resultRecordCount': 50
            }
            
            print("\nSearching for 'CREEK' fire perimeters...")
            response2 = requests.get(API_URL, params=params2, timeout=30)
            
            if response2.status_code == 200:
                data2 = response2.json()
                creek_perims = data2.get('features', [])
                
                if len(creek_perims) > 1:
                    print(f"✅ Found {len(creek_perims)} perimeters for CREEK fire!")
                    print(f"\nProgression:")
                    for i, feat in enumerate(creek_perims[:10]):
                        poly_date = feat['attributes'].get('poly_PolygonDateTime')
                        acres = feat['attributes'].get('poly_GISAcres', 0)
                        if poly_date:
                            poly_date = datetime.fromtimestamp(poly_date/1000).strftime('%Y-%m-%d %H:%M')
                        print(f"  {i+1}. {poly_date} - {acres:>8.0f} acres")
                    
                    print(f"\n🚀 THIS IS IT! We can build day-to-day progressions!")
                else:
                    print(f"⚠️  Only found {len(creek_perims)} perimeter(s)")
        else:
            print(f"\n❌ No geometry in response")
    else:
        print(f"\n❌ No features in response: {data}")
else:
    print(f"\n❌ API request failed: {response.status_code}")
    print(response.text[:500])

print(f"\n{'='*80}")
print("BOTTOM LINE")
print(f"{'='*80}")

print("\nIf WFIGS API returns:")
print("  ✅ Geometry data (polygons)")
print("  ✅ Multiple perimeters per fire (daily updates)")
print("  ✅ California fires 2020-2023")
print("\nThen:")
print("  🚀 We can build TRUE daily progression dataset")
print("  🚀 Train on real perimeter changes (not FIRMS points)")
print("  🚀 Hit 70-80% IoU (like FU-NetCastV2!)")
print("="*80)
