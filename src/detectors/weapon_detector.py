import cv2
from ultralytics import YOLO

# Threshold to filter out low-confidence detections
CONFIDENCE_THRESHOLD = 0.8

# Load the YOLOv8 weapon detection model
weapon_model = YOLO("/opt/homebrew/runs/detect/weapon_yolov8s_mac2/weights/best.pt")


def detect_weapons(frame):
    results = weapon_model(frame, conf=CONFIDENCE_THRESHOLD)
    detections = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = weapon_model.names[cls]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)

        if label in ["knife", "pistol", "long_weapon"]:
            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": xyxy,
                }
            )
    return detections


def draw_weapon_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return frame
