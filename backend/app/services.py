from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import torch
from pathlib import Path
from PIL import Image
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
    # Perform detection
    results = model(image)
    result = results[0]

    # Plot the result on a new image
    result_image = Image.fromarray(result.plot()[:,:,::-1]) # Convert BGR to RGB

    # Convert result image to base64
    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Extract detection data
    detected_objects = []
    for box in result.boxes:
        class_name = model.names[int(box.cls)]
        confidence = float(box.conf)
        detected_objects.append({
            "class_name": class_name,
            "confidence": f"{confidence:.2%}"
        })
        
    return img_str, detected_objects
