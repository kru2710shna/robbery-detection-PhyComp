# File: setup_models.py

import os
from pathlib import Path
import urllib.request

def download_yolov8s_model():
    model_dir = Path("../models/yolo_weights/")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "yolov8s.pt"

    if not model_path.exists():
        print("⬇️ Downloading yolov8s.pt model weights...")
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"
        urllib.request.urlretrieve(url, model_path)
        print("✅ Download complete:", model_path)
    else:
        print("✅ yolov8m.pt already exists.")

if __name__ == "__main__":
    download_yolov8s_model()
