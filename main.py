# File: main.py

from src.detectors.person_detector import detect_people_and_object as dpao
from pathlib import Path as ph
from src.detectors.aggressive_movement import load_pose_model

def main():
    # Define paths
    model_path = "models/yolo_weights/yolov8s.pt"
    robbery_video_path = "data/raw_training/Robbery150_x264.mp4"
    # normal_video_path = 'data/raw_training/Normal_Videos137_x264.mp4'
    
    
    pose_model_path = "models/yolo_weights/yolov8s-pose.pt"
    pose_model = load_pose_model(pose_model_path)

    # Run detection
    dpao(
        video_path=robbery_video_path,
        model_path=model_path,
        pose_model=pose_model,
        show=True # Set to False to disable video display
    )

if __name__ == "__main__":
    main()