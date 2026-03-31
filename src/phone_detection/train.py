from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="src/phone_detection/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="phone_detection_run",
    project="runs/phone_detection",
    patience=10
)

print("Training has been completed")