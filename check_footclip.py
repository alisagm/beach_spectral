from spectral_classifier.footprint_clip import get_raster_valid_footprint
from pathlib import Path

result = get_raster_valid_footprint(Path("PAIS_shorelines/imagery/19950102/2697052a.tif"))
if result:
    footprint, crs = result
    print(f"Footprint type: {footprint.geom_type}")
    print(f"Area: {footprint.area/1e6:.2f} km²")
    print(f"CRS: {crs}")