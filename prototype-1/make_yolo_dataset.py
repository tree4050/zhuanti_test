from pathlib import Path
import random

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds
from shapely.geometry import box
from PIL import Image
from shapely import make_valid

# -----------------------------
# SETTINGS
# -----------------------------
RASTER_PATH = Path("prototype-data/source/lychee_orthophoto.tif")
GPKG_PATH = Path("prototype-data/source/leetree.gpkg")
LAYER_NAME = "tree_crown"

OUTPUT_DIR = Path("prototype-data/yolo_dataset")

TILE_SIZE = 640
TRAIN_RATIO = 0.80
RANDOM_SEED = 42

# TIFF bands:
# 1 = Red
# 2 = Green
# 3 = Blue
RGB_BANDS = [1, 2, 3]


# -----------------------------
# OUTPUT FOLDERS
# -----------------------------
for split in ["train", "val"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


# -----------------------------
# LOAD TREE POLYGONS
# -----------------------------
gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)

print(f"Loaded {len(gdf)} tree crowns.")

if len(gdf) == 0:
    raise RuntimeError("No polygons found.")

# -----------------------------
# REPAIR INVALID POLYGONS
# -----------------------------
invalid_mask = ~gdf.geometry.is_valid

print("Invalid polygons found:", invalid_mask.sum())

for idx in gdf[invalid_mask].index:
    geom = make_valid(gdf.loc[idx, "geometry"])

    # Keep one polygon per tree feature
    if geom.geom_type == "Polygon":
        repaired = geom

    elif geom.geom_type == "MultiPolygon":
        repaired = max(
            geom.geoms,
            key=lambda g: g.area
        )

    elif geom.geom_type == "GeometryCollection":
        polygons = []

        for part in geom.geoms:
            if part.geom_type == "Polygon":
                polygons.append(part)
            elif part.geom_type == "MultiPolygon":
                polygons.extend(part.geoms)

        repaired = (
            max(polygons, key=lambda g: g.area)
            if polygons
            else None
        )

    else:
        repaired = None

    gdf.loc[idx, "geometry"] = repaired

    print(
    gdf.loc[invalid_mask].geometry.is_valid_reason()
)

# Remove anything that could not be repaired as a polygon
gdf = gdf[
    gdf.geometry.notna()
    & ~gdf.geometry.is_empty
].copy()

print("Crowns after geometry repair:", len(gdf))
print("Still invalid:", (~gdf.geometry.is_valid).sum())

# -----------------------------
# OPEN RASTER
# -----------------------------
with rasterio.open(RASTER_PATH) as src:

    if gdf.crs != src.crs:
        print("Reprojecting polygons to raster CRS...")
        gdf = gdf.to_crs(src.crs)

    print("Raster size:", src.width, "x", src.height)
    print("Raster CRS:", src.crs)

    # -------------------------------------------------
    # FIND GLOBAL RGB DISPLAY RANGE
    # -------------------------------------------------
    # Read a smaller copy for calculating brightness.
    sample_width = min(1000, src.width)
    sample_height = max(
        1,
        int(src.height * sample_width / src.width)
    )

    sample = src.read(
        RGB_BANDS,
        out_shape=(3, sample_height, sample_width),
        resampling=Resampling.bilinear
    ).astype(np.float32)

    low_values = []
    high_values = []

    for band in sample:
        valid = band[
            np.isfinite(band) &
            (band != 0)
        ]

        if len(valid) == 0:
            low_values.append(0)
            high_values.append(1)
        else:
            low_values.append(np.percentile(valid, 2))
            high_values.append(np.percentile(valid, 98))

    print("RGB scaling:")
    print("Low:", low_values)
    print("High:", high_values)

    # -------------------------------------------------
    # FIND TILES CONTAINING LABELED CROWNS
    # -------------------------------------------------
    eligible_tiles = []

    tile_number = 0

    for row_off in range(0, src.height, TILE_SIZE):
        for col_off in range(0, src.width, TILE_SIZE):

            tile_width = min(TILE_SIZE, src.width - col_off)
            tile_height = min(TILE_SIZE, src.height - row_off)

            window = Window(
                col_off,
                row_off,
                tile_width,
                tile_height
            )

            left, bottom, right, top = bounds(
                window,
                src.transform
            )

            tile_geometry = box(left, bottom, right, top)

            intersecting = gdf[
                gdf.geometry.intersects(tile_geometry)
            ]

            if len(intersecting) == 0:
                continue

            eligible_tiles.append(
                (
                    tile_number,
                    window,
                    tile_geometry,
                    intersecting.index.tolist()
                )
            )

            tile_number += 1

    print(
        f"\nFound {len(eligible_tiles)} tiles "
        "containing labeled trees."
    )

    # -------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # -------------------------------------------------
    random.seed(RANDOM_SEED)
    random.shuffle(eligible_tiles)

    split_index = int(
        len(eligible_tiles) * TRAIN_RATIO
    )

    train_tiles = eligible_tiles[:split_index]
    val_tiles = eligible_tiles[split_index:]

    print("Train tiles:", len(train_tiles))
    print("Validation tiles:", len(val_tiles))

    # -------------------------------------------------
    # TILE WRITER
    # -------------------------------------------------
    def process_tiles(tile_list, split):

        crown_count = 0

        for tile_id, window, tile_geometry, polygon_indices in tile_list:

            col_off = int(window.col_off)
            row_off = int(window.row_off)

            tile_width = int(window.width)
            tile_height = int(window.height)

            # -----------------------------------------
            # READ RGB
            # -----------------------------------------
            raw = src.read(
                RGB_BANDS,
                window=window
            ).astype(np.float32)

            # Pad edge tiles to exactly 640x640
            padded = np.zeros(
                (3, TILE_SIZE, TILE_SIZE),
                dtype=np.float32
            )

            padded[
                :,
                :tile_height,
                :tile_width
            ] = raw

            # -----------------------------------------
            # FLOAT32 -> UINT8 RGB
            # -----------------------------------------
            rgb = np.zeros(
                (TILE_SIZE, TILE_SIZE, 3),
                dtype=np.uint8
            )

            for b in range(3):

                low = low_values[b]
                high = high_values[b]

                scaled = (
                    (padded[b] - low)
                    / (high - low + 1e-8)
                )

                scaled = np.clip(
                    scaled,
                    0,
                    1
                )

                rgb[:, :, b] = (
                    scaled * 255
                ).astype(np.uint8)

            # -----------------------------------------
            # LABEL POLYGONS
            # -----------------------------------------
            label_lines = []

            tile_transform = src.window_transform(window)
            inverse_transform = ~tile_transform

            for idx in polygon_indices:

                original_geometry = gdf.loc[idx].geometry

                clipped = original_geometry.intersection(
                    tile_geometry
                )

                if clipped.is_empty:
                    continue

                # If clipping somehow creates several pieces,
                # use the largest one.
                if clipped.geom_type == "MultiPolygon":
                    clipped = max(
                        clipped.geoms,
                        key=lambda p: p.area
                    )

                if clipped.geom_type != "Polygon":
                    continue

                coords = list(
                    clipped.exterior.coords
                )

                # Remove duplicated closing coordinate
                if coords[0] == coords[-1]:
                    coords = coords[:-1]

                if len(coords) < 3:
                    continue

                yolo_points = []

                for x, y in coords:

                    pixel_x, pixel_y = inverse_transform * (x, y)

                    norm_x = pixel_x / TILE_SIZE
                    norm_y = pixel_y / TILE_SIZE

                    norm_x = float(
                        np.clip(norm_x, 0, 1)
                    )
                    norm_y = float(
                        np.clip(norm_y, 0, 1)
                    )

                    yolo_points.extend(
                        [norm_x, norm_y]
                    )

                if len(yolo_points) < 6:
                    continue

                # Class 0 = tree
                line = "0 " + " ".join(
                    f"{value:.6f}"
                    for value in yolo_points
                )

                label_lines.append(line)
                crown_count += 1

            if not label_lines:
                continue

            filename = (
                f"tile_r{row_off}_c{col_off}"
            )

            image_path = (
                OUTPUT_DIR
                / "images"
                / split
                / f"{filename}.png"
            )

            label_path = (
                OUTPUT_DIR
                / "labels"
                / split
                / f"{filename}.txt"
            )

            Image.fromarray(rgb).save(
                image_path
            )

            label_path.write_text(
                "\n".join(label_lines),
                encoding="utf-8"
            )

        return crown_count


    train_crowns = process_tiles(
        train_tiles,
        "train"
    )

    val_crowns = process_tiles(
        val_tiles,
        "val"
    )

print("\n--- FINISHED ---")
print("Training crown labels:", train_crowns)
print("Validation crown labels:", val_crowns)
print("Total crown labels:", train_crowns + val_crowns)
print("Dataset saved to:", OUTPUT_DIR)