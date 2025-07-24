# src/utils/draw_utils.py

import cv2
import numpy as np
import torch


# Draw a solid bounding box with label (no transparency)
def draw_box_with_label(frame, x1, y1, x2, y2, label, color, thickness=2):
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Calculate label background dimensions
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_width = label_size[0] + 10
    label_height = label_size[1] + 10
    y_offset = y1 - 25 if y1 - 25 > 0 else y1 + 25

    # Draw filled background for label
    cv2.rectangle(
        frame, (x1, y_offset), (x1 + label_width, y_offset + label_height), color, -1
    )

    # Put label text
    cv2.putText(
        frame,
        label,
        (x1 + 5, y_offset + label_height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )


# Draw keypoints and skeleton
def draw_skeleton_with_lines(frame, keypoints, top_left, threshold=0.3):
    COCO_PAIRS = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),  # Arms
        (5, 6),
        (11, 12),
        (5, 11),
        (6, 12),  # Torso
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),  # Legs
    ]

    for kp in keypoints:
        if not isinstance(kp[0], torch.Tensor):
            continue
        if kp[0].numel() == 0 or kp[0].dim() != 2 or kp[0].shape[1] != 3:
            continue

        for idx, (x, y, c) in enumerate(kp[0]):
            if c > threshold:
                xg, yg = int(x.item()) + top_left[0], int(y.item()) + top_left[1]
                cv2.circle(frame, (xg, yg), 4, (0, 0, 255), -1)

        for i, j in COCO_PAIRS:
            if kp[0][i][2] > threshold and kp[0][j][2] > threshold:
                pt1 = (
                    int(kp[0][i][0].item()) + top_left[0],
                    int(kp[0][i][1].item()) + top_left[1],
                )
                pt2 = (
                    int(kp[0][j][0].item()) + top_left[0],
                    int(kp[0][j][1].item()) + top_left[1],
                )
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

            if kp[0].numel() == 0:
                print("⚠️ No keypoints found for loitering person.")
                continue
            
# NEW FUNCTION — draw labels for aggression/loitering stacked above box
def draw_styled_label(frame, x, y, text, color, font_scale=0.55, font_thickness=2):
    """
    Draws a styled label box with given text at (x, y).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    box_width, box_height = text_size[0] + 10, text_size[1] + 8

    # Background rectangle
    cv2.rectangle(frame, (x, y - box_height), (x + box_width, y), color, -1)

    # Text overlay
    cv2.putText(
        frame,
        text,
        (x + 5, y - 5),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        lineType=cv2.LINE_AA,
    )
