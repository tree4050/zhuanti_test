from ultralytics import YOLO

model = YOLO("runs/segment/train-2/weights/best.pt")

results = model.predict(
    source="prototype-data/yolo_dataset/images/val",
    conf=0.10,
    save=True
)

print("\n--- TREE COUNTS ---")

for result in results:

    if result.masks is not None:
        count = len(result.masks.data)
    else:
        count = 0

    print(f"{result.path}: {count} trees") 