from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
from typing import List
from pathlib import Path

from .services import detect_vehicles, get_model
from .models import DetectionResult
from .db import save_detection

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

app = FastAPI(title="Vehicle Detection API")

# --- CORS Middleware ---
# Allow all origins for development purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    # Pre-load models on startup if needed, or handle it on-demand
    # For simplicity, we'll load them on first request via Depends
    pass

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vehicle Detection API"}

@app.post("/detect/", response_model=DetectionResult)
async def run_detection(
    model_name: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image content
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        model = get_model(model_name)

        # Perform detection
        result_image_bytes, detected_objects = detect_vehicles(model, image)

        await save_detection({
            "filename": file.filename,
            "model_name": model_name,
            "detections": detected_objects,
        })

        return {
            "image_b64": result_image_bytes,
            "detections": detected_objects
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during detection: {str(e)}")

@app.get("/models/", response_model=List[str])
def get_available_models():
    """Returns a list of available .pt model files."""
    if not MODELS_DIR.exists():
        return []
    return [f.name for f in MODELS_DIR.iterdir() if f.is_file() and f.suffix == '.pt']
