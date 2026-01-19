# test_format_support.py
import rasterio

# Just try opening files directly
test_files = {
    "jp2": "PAIS_shorelines/imagery/20080429/l2697_05_2_cir_29042008.jp2", 
    "sid": "PAIS_shorelines/imagery/19950102/2697052a.sid",  
}

for fmt, path in test_files.items():
    try:
        with rasterio.open(path) as src:
            print(f"{fmt}: ✓ Works — {src.count} bands, {src.crs}")
    except Exception as e:
        print(f"{fmt}: ✗ Failed — {e}")