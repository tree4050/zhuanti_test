# PROTOTYPE 1 — Tree Canopy Segmentation (YOLO)

This folder contains code for preparing RGB tree-canopy segmentation tiles from a multispectral lychee orthophoto and annotated crown polygons, inspecting inputs and labels, and running inference/tree counting with an existing trained YOLO segmentation model.

IMPORTANT NOTES:
- qgis file used did not follow the required drawn 1/5th number of trees clustered together in one area of the tiff file photo
- this prototype was created prior to reading the reference file sent in the LINE group chat (10/25/26)

## What is included

- `inspect_bands.py`: prints raster band metadata.
- `inspect_data.py`: inspects the raster and tree-crown annotation layer.
- `make_yolo_dataset.py`: produces 640×640 RGB image tiles and YOLO polygon segmentation labels, with an 80/20 tile train/validation split.
- `check_yolo_labels.py`: draws saved YOLO polygons onto selected image tiles for visual review.
- `count_trees.py`: loads an existing model checkpoint and predicts masks on validation tiles, reporting mask counts.

## Required local inputs (not committed)

Run these scripts from **this folder** as the working directory. Supply:

- `prototype-data/source/lychee_orthophoto.tif`
- `prototype-data/source/leetree.gpkg` with a layer named `tree_crown`

The raster is assumed to have RGB in bands 1, 2, 3. The data builder creates `prototype-data/yolo_dataset/images/{train,val}` and `labels/{train,val}`. The generated images and labels are excluded from Git to avoid committing large datasets.

## Environment

Create and activate a Python virtual environment, then install packages:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Package versions are not pinned because the original environment versions were not provided. GIS dependencies (GeoPandas/Rasterio) may require a compatible Python version.

## Typical commands

```powershell
python inspect_bands.py
python inspect_data.py
python make_yolo_dataset.py
python check_yolo_labels.py
```

After training a YOLO segmentation model separately, adjust the weights path in `count_trees.py` (currently `runs/segment/train-2/weights/best.pt`) and run:

```powershell
python count_trees.py
```

## Important limitations / work still needed

- **The training script or precise training command is NOT included in the supplied source files.** This repository alone does not reproduce training or provide model weights.
- Source files have been copied unchanged; they have **not** been executed or tested against the original UAV imagery/GeoPackage.
- `make_yolo_dataset.py` has a potentially problematic geometry-validation print inside its repair loop. Test this with your GeoPandas/Shapely environment before relying on a complete run.
- The tile-level random split may put nearby imagery from the same orthophoto in both training and validation sets; evaluate spatial leakage before reporting generalization.
- The scripts currently assume a specific local layout and sample dataset. Adjust paths/configuration for other data.

Never upload confidential location data or private imagery without permission.
