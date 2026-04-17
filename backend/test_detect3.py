from PIL import Image
from ultralytics import YOLO
import app.services

model = YOLO("../models/best_yolov8n.pt")
# Try with an actual image
img = Image.open("../assets/image1.jpg") # assume there's one? I don't know if assets has images. Let's list assets
