# File: src/detectors/weapon_detector.py

import cv2
from ultralytics import YOLO

# Set confidence threshold for weapon detection
CONFIDENCE_THRESHOLD = 0.5

# Load pretrained Roboflow-exported YOLOv8 weapon detection model
weapon_model = YOLO("models/yolo_weights/weapon_yolov8.pt")


def detect_weapons(frame):
    results = weapon_model(frame, conf=0.5)
    detections = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = weapon_model.names[cls]

        if label in ["knife", "pistol", "long_weapon"]:
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": box.xyxy[0].cpu().numpy().astype(int),
            })

    return detections


def draw_weapon_boxes(frame, detections):
    """
    Draw bounding boxes for weapon detections on the frame.

    Args:
        frame (np.ndarray): Image to draw on.
        detections (list): Output from detect_weapons().
    
    Returns:
        frame (np.ndarray): Annotated image.
    """
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red box
        # Draw label
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

    return frame