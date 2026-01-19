# export_shellline_line.py
import json
import geopandas as gpd
from shapely.geometry import LineString

# Paths
SUMMARY_FILE = "OUTPUT/test_2016/summary_20260115_162843.json"
TRANSECT_FILE = "INPUT/shorelineTransPais.json"
OUTPUT_FILE = "OUTPUT/test_2016/shellline.geojson"

# Load data
with open(SUMMARY_FILE) as f:
    summary = json.load(f)

transects = gpd.read_file(TRANSECT_FILE)
transect_lookup = {str(row.TransectID): row.geometry for _, row in transects.iterrows()}

# Collect shell line points with their TransectID for sorting
points_with_order = []

for t in summary["transects"]:
    tid = t["transect_id"]
    
    for trans in t["transitions"]:
        if trans["type"] != "dry_wet_derivative":
            continue
        
        if tid not in transect_lookup:
            continue
        
        line = transect_lookup[tid]
        distance = min(trans["distance"], line.length)
        point = line.interpolate(distance)
        
        points_with_order.append({
            "transect_id": int(tid),
            "point": point,
            "confidence": trans["confidence"]
        })

# Sort by TransectID
points_with_order.sort(key=lambda x: x["transect_id"])

# Create LineString from sorted points
coords = [p["point"] for p in points_with_order]

if len(coords) < 2:
    print("Error: Need at least 2 points to create a line")
else:
    shoreline = LineString(coords)
    
    # Calculate average confidence
    avg_confidence = sum(p["confidence"] for p in points_with_order) / len(points_with_order)
    
    gdf = gpd.GeoDataFrame([{
        "geometry": shoreline,
        "num_points": len(coords),
        "avg_confidence": avg_confidence
    }], crs=transects.crs)
    
    gdf.to_file(OUTPUT_FILE, driver="GeoJSON")
    print(f"Exported shoreline with {len(coords)} vertices to {OUTPUT_FILE}")