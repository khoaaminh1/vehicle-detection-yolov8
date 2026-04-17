from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import torch
from pathlib import Path
from PIL import Image, ImageDraw
import os
import base64
import io
from functools import lru_cache

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([DetectionModel])

# --- Model Loading ---

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

@lru_cache(maxsize=2)
def load_yolo_model(model_name: str):
    """
    Loads a YOLO model from the 'models' directory.
    Uses lru_cache to keep recently used models in memory.
    """
    model_path = MODELS_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = YOLO(str(model_path))
    return model

def get_model(model_name: str):
    """Dependency to get the selected model."""
    try:
        return load_yolo_model(model_name)
    except FileNotFoundError as e:
        # This will be caught by FastAPI and returned as a 500 error
        raise e
    except Exception as e:
        # Expose the real exception
        raise e


# --- Detection Logic ---

def detect_vehicles(model: YOLO, image: Image.Image):
    """
    Runs vehicle detection on an image and returns the result image and data.
    """
    # Perform detection with a slightly stricter confidence
    results = model(image, conf=0.40, iou=0.5)
    result = results[0]

    detected_objects = []
    
    # Prepare manual plotting to sync perfectly with filtered objects
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    image_area = image.width * image.height
    
    VALID_VEHICLE_CLASSES = {'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'bicycle'}
    
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            class_name = model.names[cls_id].lower()

            if class_name not in VALID_VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 0.0)
            aspect_ratio = height / width
            area = width * height
            
            # Simple threshold for confidence
            # Very small tiny confidence often leads to wild misclassification (e.g. serum bottle -> car or bus)
            if conf < 0.45:
                continue

            # Domain heuristics:
            if aspect_ratio > 1.3 and class_name in ['car', 'bus', 'truck']:
                # Very tall object claimed to be a car/bus is likely false positive
                if conf < 0.70:
                    continue

            # Keep valid objects
            detected_objects.append({
                "class_name": class_name,
                "confidence": conf,
                "xyxy": [x1, y1, x2, y2]
            })

        # Draw cleanly-styled bounding boxes for remaining VALID objects
        for obj in detected_objects:
            x1, y1, x2, y2 = obj.pop('xyxy') # Remove from output to avoid Pydantic schema issues
            class_name = obj['class_name']
            conf = obj['confidence']
            
            box_color = "#3B82F6" # Professional blue
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            
            label = f"{class_name} {conf:.2f}"
            
            try:
                # Get text size for the background block
                bbox = draw.textbbox((0, 0), label)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                text_width, text_height = 80, 15  # Fallback for old PIL
            
            label_bg_x2 = x1 + text_width + 6
            label_bg_y2 = max(0, y1)
            label_bg_y1 = label_bg_y2 - text_height - 6
            
            # Draw semi-transparent or solid background for the text
            draw.rectangle([x1, label_bg_y1, label_bg_x2, label_bg_y2], fill=box_color)
            draw.text((x1 + 3, label_bg_y1 + 3), label, fill="#FFFFFF")

    # Convert synchronized result image to base64
    if len(detected_objects) == 0:
        # If no objects survived the filters, send back the absolute original untouched image
        img_to_encode = image
    else:
        img_to_encode = result_image
        
    buffered = io.BytesIO()
    img_to_encode.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    return img_str, detected_objects
