import os
from ultralytics import YOLO

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS  = os.path.join(BASE_DIR, "yolov8n.pt")
YAML     = os.path.join(BASE_DIR, "dataset.yaml")

model = YOLO(WEIGHTS)
model.train(
    data=YAML,
    epochs=20,
    imgsz=640,
    batch=8,
    lr0=0.0001,
    lrf=0.00001,
    workers=0,
    name="phone_finetune_v2",
    pretrained=True,
    patience=15,
    save=True,
    freeze=10,
    project=os.path.join(BASE_DIR, "runs"),
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.1,
    degrees=10.0,
    translate=0.2,
    scale=0.5,
    shear=5.0,
)

print("\nDone! Best weights saved to:")
print(os.path.join(BASE_DIR, "runs", "phone_finetune", "weights", "best.pt"))