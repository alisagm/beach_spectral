# survey_all_imagery.py
from pathlib import Path
import rasterio

imagery_root = Path("PAIS_shorelines/imagery")

# Find all rasters recursively
tifs = list(imagery_root.rglob("*.tif")) + list(imagery_root.rglob("*.tiff"))
jp2s = list(imagery_root.rglob("*.jp2"))
sids = list(imagery_root.rglob("*.sid"))

print(f"Found: {len(tifs)} TIF, {len(jp2s)} JP2, {len(sids)} SID")
print()

# Check band counts and CRS for each
summary = {"4-band": [], "3-band": [], "other": [], "error": []}

for path in tifs + jp2s:
    try:
        with rasterio.open(path) as src:
            bands = src.count
            crs = src.crs
            
            # Categorize
            if bands == 4:
                summary["4-band"].append((path, crs))
            elif bands == 3:
                summary["3-band"].append((path, crs))
            else:
                summary["other"].append((path, bands, crs))
    except Exception as e:
        summary["error"].append((path, str(e)))

print(f"4-band (NIR available): {len(summary['4-band'])}")
print(f"3-band (RGB only):      {len(summary['3-band'])}")
print(f"Other band counts:      {len(summary['other'])}")
print(f"Errors:                 {len(summary['error'])}")
print()

# Show CRS distribution for 4-band
if summary["4-band"]:
    crs_counts = {}
    for path, crs in summary["4-band"]:
        crs_str = str(crs)
        crs_counts[crs_str] = crs_counts.get(crs_str, 0) + 1
    print("CRS distribution (4-band):")
    for crs, count in sorted(crs_counts.items(), key=lambda x: -x[1]):
        print(f"  {crs}: {count}")
print()

# Show which date folders have 4-band imagery
if summary["4-band"]:
    date_folders = set()
    for path, _ in summary["4-band"]:
        # Get parent folder (date folder)
        date_folders.add(path.parent)
    print(f"Date folders with 4-band imagery: {len(date_folders)}")
    for folder in sorted(date_folders):
        count = sum(1 for p, _ in summary["4-band"] if p.parent == folder)
        print(f"  {folder.relative_to(imagery_root)}: {count} files")