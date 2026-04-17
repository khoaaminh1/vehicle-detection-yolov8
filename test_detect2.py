import torch
from PIL import Image
from ultralytics import YOLO

model = YOLO("vehicle-detection-yolov8/models/best_yolov8n.pt")
# Create a dummy image
img = Image.new("RGB", (640, 640), color="white")
results = model(img)
result = results[0]
if result.boxes is not None:
    print(type(result.boxes))
    try:
        mask = torch.tensor([True] * len(result.boxes))
        result.boxes = result.boxes[mask]
        print("Assignment success")
    except Exception as e:
        print("Assignment failed:", type(e), e)
