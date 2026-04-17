import sys
import os
sys.path.insert(0, os.path.abspath("."))
from PIL import Image
from ultralytics import YOLO
from app.services import detect_vehicles

model = YOLO("../models/best_yolov8n.pt")
img = Image.open("../assets/results_yolov8s.png")
result_img_str, detections = detect_vehicles(model, img)
print("Detections count:", len(detections))
