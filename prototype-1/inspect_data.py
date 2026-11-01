import geopandas as gpd
import rasterio

raster_path = "prototype-data/source/lychee_orthophoto.tif"
gpkg_path = "prototype-data/source/leetree.gpkg"

print("\n--- RASTER ---")
with rasterio.open(raster_path) as src:
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("Bounds:", src.bounds)

print("\n--- POLYGONS ---")
gdf = gpd.read_file(gpkg_path,layer="tree_crown")

print("CRS:", gdf.crs)
print("Number of features:", len(gdf))
print("Geometry types:")
print(gdf.geometry.geom_type.value_counts())
print("Bounds:", gdf.total_bounds)