from pathlib import Path
import cv2
import numpy as np

# Change "train" to "val" if you want to inspect validation tiles
SPLIT = "train"

IMAGE_DIR = Path(f"prototype-data/yolo_dataset/images/{SPLIT}")
LABEL_DIR = Path(f"prototype-data/yolo_dataset/labels/{SPLIT}")
OUTPUT_DIR = Path(f"prototype-data/yolo_dataset/preview/{SPLIT}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only preview first 10 images
image_files = list(IMAGE_DIR.glob("*.png"))[:10]

for image_path in image_files:
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path)
        continue

    h, w = image.shape[:2]

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        print("Missing label:", label_path)
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split()

        class_id = int(values[0])
        coords = list(map(float, values[1:]))

        points = []

        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)

            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(str(output_path), image)

    print("Saved:", output_path)

print("\nFinished making previews.")