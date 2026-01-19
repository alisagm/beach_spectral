import rasterio
import geopandas as gpd

tif_path = "PAIS_shorelines/imagery/2016/20160122/top15-nc-cir-50cm_2697052_20160122.tif"
transects = gpd.read_file("INPUT/shorelineTransPais.json")

print(f"Transect CRS: {transects.crs}")

with rasterio.open(tif_path) as src:
    print(f"Bands: {src.count}")
    print(f"Band dtypes: {src.dtypes}")
    print(f"Band descriptions: {src.descriptions}")
    print(f"Colorinterp: {[src.colorinterp[i] for i in range(src.count)]}")
    print(f"CRS: {src.crs}")
    print(f"Size: {src.width} x {src.height}")
    print(f"Bounds: {src.bounds}")