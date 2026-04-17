import urllib.request
import io
from PIL import Image
from backend.app.services import detect_vehicles, get_model
import traceback

model = get_model('best_yolov8n.pt')
# Create a dummy image
img = Image.new('RGB', (640, 640), color='white')
try:
    detect_vehicles(model, img)
    print("Success")
except Exception as e:
    traceback.print_exc()
