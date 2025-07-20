
# src/detectors/person_detector.py

import cv2
import sys
import os
from ultralytics import YOLO
import numpy as np
from src.utils.save_video import init_video_writer
from src.detectors.aggressive_movement import making_skeleton
from src.utils.draw_utils import draw_box_with_label


BOX_THICKNESS = 2
FONT_SCALE = 0.6
TEXT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX




def detect_people_and_object(
    model_path: str, video_path: str, pose_model: YOLO, show: bool = True
):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    os.makedirs("outputs/annotated_videos", exist_ok=True)
    output_path = "outputs/annotated_videos/robbery_output.mp4"
    out = init_video_writer(cap, output_path)

    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        sys.exit(1)

    NEARBY_CLASSES = {
        0: "person",
        24: "backpack",
        26: "handbag",
        28: "suitcase",
        67: "phone",
        63: "laptop",
        43: "knife",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        56: "chair",
    }
    NEARBY_THRESHOLD = 40
    LOITERING_THRESHOLD_FRAMES = 300
    STATIONARY_DIST_THRESHOLD = 20

    next_person_id = 0
    person_tracker = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        time_str = f"{int(timestamp_sec // 60):02}:{int(timestamp_sec % 60):02}"

        results = model(frame)[0]
        shape_str = str(results.orig_shape)
        inference_time = results.speed["inference"]

        persons, nearby_objects = [], []

        for box in results.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if class_id == 0:
                persons.append({"bbox": (x1, y1, x2, y2), "center": (cx, cy)})
            elif class_id in NEARBY_CLASSES:
                nearby_objects.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "center": (cx, cy),
                        "label": NEARBY_CLASSES[class_id],
                    }
                )

        tracked_ids = {}
        for person in persons:
            cx, cy = person["center"]
            matched = False
            for pid, info in person_tracker.items():
                prev_cx, prev_cy = info["center"]
                dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                if dist < STATIONARY_DIST_THRESHOLD:
                    person_tracker[pid]["center"] = (cx, cy)
                    person_tracker[pid]["frames"] += 1
                    tracked_ids[id(person)] = (pid, person_tracker[pid]["frames"])
                    matched = True
                    break
            if not matched:
                person_tracker[next_person_id] = {"center": (cx, cy), "frames": 1}
                tracked_ids[id(person)] = (next_person_id, 1)
                next_person_id += 1

        # --- Draw persons ---
        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            pid, frame_count = tracked_ids.get(id(person), (-1, 0))
            is_loitering = frame_count >= LOITERING_THRESHOLD_FRAMES
            label = f"{'Loitering' if is_loitering else 'Person'} #{pid}"
            color = (0, 255, 255) if is_loitering else (0, 255, 0)

            draw_box_with_label(frame, x1, y1, x2, y2, label, color)

            if is_loitering:
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    frame = making_skeleton(frame, pose_model, crop, (x1, y1))

        # --- Draw objects ---
        for obj in nearby_objects:
            cx, cy = obj["center"]
            label = obj["label"]
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                label,
                (cx + 10, cy),
                FONT,
                FONT_SCALE,
                (0, 0, 255),
                TEXT_THICKNESS,
            )

            for person in persons:
                px, py = person["center"]
                distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if distance < NEARBY_THRESHOLD:
                    cv2.line(frame, (px, py), (cx, cy), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f"Near {label}",
                        (px, py - 10),
                        FONT,
                        FONT_SCALE,
                        (255, 0, 0),
                        TEXT_THICKNESS,
                    )

        # --- Status ---
        sys.stdout.write(
            f"\r⏱️ {time_str} | 👤 Persons: {len(persons)} | 🎒 Objects: {len(nearby_objects)} | ⚡ {inference_time:.1f}ms | Shape: {shape_str}"
        )
        sys.stdout.flush()

        if out:
            out.write(frame)
        if show:
            cv2.imshow("Robbery Detection Viewer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
