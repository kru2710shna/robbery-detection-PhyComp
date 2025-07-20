# File: main.py

from src.detectors.person_detector import detect_people_and_object as dpao
from pathlib import Path as ph

def main():
    # Define paths
    model_path = "models/yolo_weights/yolov8s.pt"
    video_path = "data/raw_training/Robbery111_x264.mp4"

    # Run detection
    dpao(
        video_path=video_path,
        model_path=model_path,
        show=True # Set to False to disable video display
    )

if __name__ == "__main__":
    main()
