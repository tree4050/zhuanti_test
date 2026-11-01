import rasterio

raster_path = "prototype-data/source/lychee_orthophoto.tif"

with rasterio.open(raster_path) as src:
    print("Number of bands:", src.count)
    print("Data types:", src.dtypes)
    print("Color interpretation:", src.colorinterp)
    print("Descriptions:", src.descriptions)

    print("\n--- BAND METADATA ---")
    for i in range(1, src.count + 1):
        print(f"\nBand {i}")
        print("Description:", src.descriptions[i - 1])
        print("Tags:", src.tags(i))