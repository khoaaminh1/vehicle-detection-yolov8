from pydantic import BaseModel
from typing import List, Dict

class Detection(BaseModel):
    class_name: str
    confidence: float

class DetectionResult(BaseModel):
    image_b64: str
    detections: List[Detection]
