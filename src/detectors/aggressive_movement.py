import cv2
from ultralytics import YOLO
from src.utils.draw_utils import draw_skeleton_with_lines

# COCO Keypoint Definitions
KEYPOINT_NAMES = {
    0: "Nose",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle",
}

KEYPOINTS_TO_DRAW = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def load_pose_model(pose_model_path):
    """
    Loads a YOLOv8 pose model for skeleton detection.
    """
    return YOLO(pose_model_path)


def making_skeleton(full_frame, pose_model, person_crop, top_left):
    """
    Detects pose keypoints for a given cropped person region and overlays
    skeleton with joints and lines back onto the original frame.

    Args:
        full_frame (ndarray): Original video frame.
        pose_model (YOLO): Loaded YOLOv8-pose model.
        person_crop (ndarray): Cropped person image.
        top_left (tuple): (x1, y1) coordinates of the crop in the full frame.

    Returns:
        Annotated frame with skeleton overlaid.
    """
    if person_crop is None or person_crop.size == 0:
        return full_frame

    results = pose_model.predict(person_crop, verbose=False)[0]

    if results.keypoints is not None and hasattr(results.keypoints, "data"):
        keypoints = results.keypoints.data
        draw_skeleton_with_lines(full_frame, keypoints, top_left, threshold=0.5)

    return full_frame
