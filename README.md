# Vehicle Detection with YOLOv8

## Overview
This project builds a vehicle detection system using YOLOv8 on a filtered subset of the COCO dataset.

The detector focuses on 4 vehicle classes:
- car
- motorcycle
- bus
- truck

## Dataset
The dataset was created by filtering the COCO 2017 detection dataset and keeping only 4 vehicle-related classes.
The final dataset was converted into YOLO format for training.

## Models
Two YOLOv8 variants were tested:
- YOLOv8n
- YOLOv8s

## Training Setup
- Platform: Google Colab
- GPU: NVIDIA T4
- Image size: 640
- Dataset format: YOLO detection format

## Project Structure
- `models/`: trained weights
- `assets/`: training result plots
- `configs/`: dataset config files

## Full-stack Demo (Frontend + Backend)

This demo adds a FastAPI backend and a React frontend.

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Optional MongoDB (store detections):

```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB="vehicle_detection"
export MONGODB_COLLECTION="detections"
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open: `http://localhost:5173`

## Future Improvements
- Build a Streamlit demo
- Compare inference speed
- Improve small-object detection