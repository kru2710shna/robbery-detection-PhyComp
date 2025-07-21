import cv2
import sys
import os
from ultralytics import YOLO
import numpy as np
import time
from collections import defaultdict

from src.utils.save_video import init_video_writer
from src.utils.draw_utils import draw_box_with_label, draw_styled_label
from src.detectors.aggressive_movement import (
    making_skeleton,
    detect_aggressive_arm_motion,
    detect_both_hands_aggression,
    detect_side_or_back_aggression,
    detect_aggression_low_body_motion,
    too_far_for_skeleton,
    track_object_disappearance,
    is_too_close,
)

# Constants
BOX_THICKNESS = 2
FONT_SCALE = 0.6
TEXT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

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
NEARBY_THRESHOLD = 40  # pixels
LOITERING_THRESHOLD_FRAMES = 280
STATIONARY_DIST_THRESHOLD = 20  # pixels


STATE_COLOR_MAP = {
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "blinking": (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (0, 0, 100),
}


def detect_people_and_object(
    model_path: str, video_path: str, pose_model: YOLO, show: bool = True
):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        sys.exit(1)

    os.makedirs("outputs/annotated_videos", exist_ok=True)
    output_path = "outputs/annotated_videos/robbery_output.mp4"
    out = init_video_writer(cap, output_path)

    person_tracker = {}
    next_person_id = 0
    
    frame_count = 0
   

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        time_str = f"{int(timestamp_sec // 60):02}:{int(timestamp_sec % 60):02}"
        results = model(frame)[0]
        shape_str = str(results.orig_shape)
        inference_time = results.speed["inference"]

        persons, nearby_objects = parse_detections(results)
        tracked_ids, next_person_id = track_persons(
            persons, person_tracker, next_person_id
        )

        annotate_persons(frame, persons, object_tracker, tracked_ids, pose_model, person_tracker, frame_count)
        annotate_objects(frame, persons, nearby_objects, frame_count, object_tracker, person_tracker)

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


def parse_detections(results):
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
    return persons, nearby_objects


def track_persons(persons, tracker, next_id):
    tracked_ids = {}
    for person in persons:
        cx, cy = person["center"]
        matched = False
        for pid, info in tracker.items():
            prev_cx, prev_cy = info["center"]
            dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
            if dist < STATIONARY_DIST_THRESHOLD:
                tracker[pid]["center"] = (cx, cy)
                tracker[pid]["frames"] += 1
                tracked_ids[id(person)] = (pid, tracker[pid]["frames"])
                matched = True
                break
        if not matched:
            tracker[next_id] = {
                "center": (cx, cy),
                "frames": 1,
                "state": "green",
                "aggression_frames": 0,
                "weapon_detected": False,
                "victim_candidate": False,
            }
            tracked_ids[id(person)] = (next_id, 1)
            next_id += 1
    return tracked_ids, next_id


kp_history = {}
object_tracker = {}
alerts = defaultdict(list)


def annotate_persons(frame, persons, object_tracker, tracked_ids, pose_model, person_tracker, frame_count):
    for person in persons:
        x1, y1, x2, y2 = person["bbox"]
        pid, loitering_frames = tracked_ids.get(id(person), (-1, 0))
        is_loitering = loitering_frames >= LOITERING_THRESHOLD_FRAMES

        # ✅ Add this at the top of your file if not already imported

        # Set default to green initially, will update below if available
        color = STATE_COLOR_MAP.get("green")
        label = f"Person #{pid}"
        top_labels = []

        if is_loitering:
            crop = frame[y1:y2, x1:x2]
            if too_far_for_skeleton((x1, y1, x2, y2)):
                top_labels.append(" Far View – Pose Unavailable")
                color = (255, 165, 0)  # Orange
                if pid in person_tracker and person_tracker[pid]["state"] == "green":
                    person_tracker[pid]["state"] = (
                        "yellow"  # Escalate if loitering detected too
                    )
            else:
                # ✅ Proceed with pose estimation if bounding box is large enough
                frame, keypoints = making_skeleton(
                    frame, pose_model, crop, (x1, y1), return_keypoints=True
                )

                if keypoints is not None:
                    if pid not in kp_history:
                        kp_history[pid] = []
                    kp_history[pid].append(keypoints)
                    if len(kp_history[pid]) > 2:
                        kp_history[pid] = kp_history[pid][-2:]

                # Check all types of aggression
                detected = False

                # ✅ Priority-based detection
                if detect_both_hands_aggression(kp_history, pid):
                    top_labels.append("🟥  Both-Hands Aggression 🟥  ")
                    color = (0, 0, 255)
                    detected = True
                elif detect_side_or_back_aggression(kp_history, pid):
                    top_labels.append("🟥  Side/Back Aggression 🟥  ")
                    color = (0, 0, 255)
                    detected = True
                elif detect_aggression_low_body_motion(kp_history, pid):
                    top_labels.append("🟥  Low Motion Aggression 🟥 ")
                    color = (0, 0, 255)
                    detected = True
                elif detect_aggressive_arm_motion(kp_history, pid):
                    top_labels.append("🟥  Arm Aggression 🟥 ")
                    color = (0, 0, 255)
                    detected = True

                # ✅ Fallback label if none of the above triggered
                if not detected:
                    top_labels.append("Loitering")
                    color = (0, 255, 255)

                for other in persons:
                    if other == person:
                        continue
                    if is_too_close(person, other):
                        top_labels.append("⚠️ Close Contact – Suspect")
                        break

                # ✅ Add FSM state updates based on label
                if pid in person_tracker:
                    if any("Aggression" in lbl for lbl in top_labels):
                        person_tracker[pid]["state"] = "red"
                        person_tracker[pid]["aggression_frames"] += 1
                    elif (
                        "Loitering" in top_labels
                        and person_tracker[pid]["state"] == "green"
                    ):
                        person_tracker[pid]["state"] = "yellow"
                    elif "Loitering" not in top_labels:
                        person_tracker[pid]["state"] = "green"

        # Draw box and all non-overlapping labels
        if pid in person_tracker:
            color = STATE_COLOR_MAP.get(person_tracker[pid]["state"], (0, 255, 0))

        draw_box_with_label(frame, x1, y1, x2, y2, label, color)

        label_y = max(22, y1)  # Prevent drawing above frame
        for motion_label in top_labels:
            draw_styled_label(frame, x1, label_y, motion_label, color)
            label_y -= 22  # Move up for next label to avoid overlap
    disappeared_objects = track_object_disappearance(object_tracker, frame_count)

    for obj_id, info in disappeared_objects:
        pid = info.get("last_seen_by_pid")
        if pid in person_tracker:
            person_tracker[pid]["state"] = "red"
            person_tracker[pid]["reason"] = "Object Stolen"
            if pid not in alerts:
                alerts[pid] = []
            alerts[pid].append("🟥 Object Taken 🟥")


def annotate_objects(
    frame, persons, nearby_objects, frame_count, object_tracker, person_tracker ):
    for obj in nearby_objects:
        cx, cy = obj["center"]
        label = obj["label"]

        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            frame, label, (cx + 10, cy), FONT, FONT_SCALE, (0, 0, 255), TEXT_THICKNESS
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

                # ✅ Update object_tracker here
                pid = [k for k, v in person_tracker.items() if v["center"] == (px, py)]
                pid = pid[0] if pid else -1  # Defensive fallback
                if label not in object_tracker:
                    object_tracker[label] = {
                        "center": (cx, cy),
                        "frames": 0,
                        "last_seen": frame_count,
                        "last_seen_by_pid": pid,
                    }
                else:
                    object_tracker[label]["center"] = (cx, cy)
                    object_tracker[label]["last_seen"] = frame_count
                    object_tracker[label]["last_seen_by_pid"] = pid
