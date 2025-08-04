import cv2
from ultralytics import YOLO
from src.utils.draw_utils import draw_skeleton_with_lines
import numpy as np

# COCO Keypoint Definitions
POSE_CONNECTIONS = [
    (5, 7),
    (7, 9),  # Left arm
    (6, 8),
    (8, 10),  # Right arm
    (5, 6),  # Shoulders
    (11, 13),
    (13, 15),  # Left leg
    (12, 14),
    (14, 16),  # Right leg
    (11, 12),  # Hips
    (5, 11),
    (6, 12),  # Torso
]

KEYPOINTS_TO_DRAW = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def load_pose_model(pose_model_path):
    """
    Loads a YOLOv8 pose model for skeleton detection.
    """
    return YOLO(pose_model_path)


SKELETON_COLOR = (255, 255, 255)
CIRCLE_COLOR = (255, 0, 255)


def check_keypoint_confidence(*keypoints, threshold=0.5):
    return all(len(kp) >= 3 and kp[2] >= threshold for kp in keypoints)


def making_skeleton(
    full_frame,
    pose_model: YOLO,
    crop: np.ndarray,
    top_left: tuple,
    visualize_crop=False,
    return_keypoints=False,
):
    """
    Draws skeleton keypoints on full_frame using pose estimation from the cropped person image.

    Args:
        full_frame (np.ndarray): Original frame.
        pose_model (YOLO): Ultralytics pose model.
        crop (np.ndarray): Cropped person image.
        top_left (tuple): (x1, y1) offset of crop relative to full_frame.
        visualize_crop (bool): Optional debug view.
        return_keypoints (bool): If True, returns normalized keypoints.

    Returns:
        full_frame (np.ndarray) or (np.ndarray, keypoints_tensor) if return_keypoints is True
    """
    x_offset, y_offset = top_left

    if crop is None or crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
        print("⚠️ Invalid or too small crop, skipping skeleton drawing.")
        return (full_frame, None) if return_keypoints else full_frame

    if visualize_crop:
        cv2.imshow("Pose Crop", crop)
        cv2.waitKey(1)

    try:
        results = pose_model(crop, verbose=False)[0]
        keypoints = results.keypoints

        if (
            keypoints is None
            or not hasattr(keypoints, "xyn")
            or len(keypoints.xyn) == 0
        ):
            print("⚠️ No keypoints detected.")
            return (full_frame, None) if return_keypoints else full_frame

        # Use normalized keypoints with confidence: shape (1, 17, 3)
        keypoints_tensor = keypoints.xyn

        # Convert to pixel coordinates for drawing
        keypoints_np = keypoints.xy[0].cpu().numpy()  # shape (17, 2)
    except Exception as e:
        print(f"❌ Pose estimation error: {e}")
        return (full_frame, None) if return_keypoints else full_frame

    # Draw keypoints
    for i, (x, y) in enumerate(keypoints_np):
        x_global, y_global = int(x + x_offset), int(y + y_offset)
        cv2.circle(full_frame, (x_global, y_global), 4, CIRCLE_COLOR, -1)

    for pt1, pt2 in POSE_CONNECTIONS:
        if pt1 < len(keypoints_np) and pt2 < len(keypoints_np):
            x1, y1 = keypoints_np[pt1]
            x2, y2 = keypoints_np[pt2]
            x1, y1 = int(x1 + x_offset), int(y1 + y_offset)
            x2, y2 = int(x2 + x_offset), int(y2 + y_offset)
            cv2.line(full_frame, (x1, y1), (x2, y2), SKELETON_COLOR, 2)

    return (full_frame, keypoints_tensor) if return_keypoints else full_frame


def compute_vector_angle(v1, v2):
    """Compute angle (degrees) between two vectors."""
    unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))


def compute_elbow_angle(elbow, shoulder, wrist):
    """
    Compute the angle at the elbow joint (in degrees)
    between the upper arm (shoulder to elbow) and forearm (wrist to elbow).

    Args:
        elbow (array-like): [x, y] of elbow joint
        shoulder (array-like): [x, y] of shoulder joint
        wrist (array-like): [x, y] of wrist joint

    Returns:
        float: Elbow angle in degrees
    """
    vec1 = np.array(shoulder[:2]) - np.array(elbow[:2])
    vec2 = np.array(wrist[:2]) - np.array(elbow[:2])

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cosine_angle = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)


def compute_arm_metrics(kp_prev, kp_curr, shoulder_idx, elbow_idx, wrist_idx):
    try:
        s1, e1, w1 = kp_prev[shoulder_idx], kp_prev[elbow_idx], kp_prev[wrist_idx]
        s2, e2, w2 = kp_curr[shoulder_idx], kp_curr[elbow_idx], kp_curr[wrist_idx]
    except IndexError as e:
        print(f"❌ IndexError while accessing keypoints: {e}")
        return 0, 0

    # Check minimum length (x, y at least)
    if any(len(p) < 2 for p in [s1, e1, w1, s2, e2, w2]):
        print("⚠️ Incomplete keypoint data for arm metrics.")
        return 0, 0

    # Check confidence if available (length == 3)
    if all(len(p) >= 3 for p in [s1, e1, w1, s2, e2, w2]):
        if not check_keypoint_confidence(s1, e1, w1, s2, e2, w2):
            print("⚠️ Keypoints have low confidence.")
            return 0, 0

    # Extract just the x, y coordinates for vector math
    e1_xy, s1_xy, w1_xy = np.array(e1[:2]), np.array(s1[:2]), np.array(w1[:2])
    e2_xy = np.array(e2[:2])

    angle = compute_elbow_angle(e1_xy, s1_xy, w1_xy)
    speed = np.linalg.norm(e2_xy - e1_xy)

    return angle, speed


def detect_aggressive_arm_motion(
    kp_history, pid, threshold_angle_change=45, threshold_speed=10
):
    """
    Detects aggressive movement based on right or left arm velocity + angular change.

    Args:
        kp_history (dict): Dictionary of {pid: [keypoints_t-1, keypoints_t]}
        pid (int): Person ID
        threshold_angle_change (float): min angular delta for aggression
        threshold_speed (float): min wrist speed (px/frame)

    Returns:
        bool: True if aggression detected
    """
    if pid not in kp_history or len(kp_history[pid]) < 2:
        return False

    kp_prev = kp_history[pid][-2][0].cpu().numpy()
    kp_curr = kp_history[pid][-1][0].cpu().numpy()

    def check_arm_aggression(s_prev, e_prev, w_prev, s_curr, e_curr, w_curr):
        if not check_keypoint_confidence(
            s_prev, e_prev, w_prev, s_curr, e_curr, w_curr
        ):
            return False

        vec_upper_prev = e_prev[:2] - s_prev[:2]
        vec_fore_prev = w_prev[:2] - e_prev[:2]
        angle_prev = compute_vector_angle(vec_upper_prev, vec_fore_prev)

        vec_upper_curr = e_curr[:2] - s_curr[:2]
        vec_fore_curr = w_curr[:2] - e_curr[:2]
        angle_curr = compute_vector_angle(vec_upper_curr, vec_fore_curr)

        angle_change = abs(angle_curr - angle_prev)
        wrist_speed = np.linalg.norm(w_curr[:2] - w_prev[:2])

        return angle_change > threshold_angle_change and wrist_speed > threshold_speed

    # Right arm
    if check_arm_aggression(
        kp_prev[6], kp_prev[8], kp_prev[10], kp_curr[6], kp_curr[8], kp_curr[10]
    ):
        return True

    # Left arm
    if check_arm_aggression(
        kp_prev[5], kp_prev[7], kp_prev[9], kp_curr[5], kp_curr[7], kp_curr[9]
    ):
        return True

    return False


def detect_aggression_low_body_motion(
    kp_history, pid, angle_thresh=40, speed_thresh=8, hip_motion_thresh=3
):
    """
    Detect aggression when lower body (hips) is mostly stationary but upper body is active.

    Args:
        kp_history (dict): Person ID mapped to list of keypoints tensors.
        pid (int): Person ID.
        angle_thresh (float): Min angle change for aggression.
        speed_thresh (float): Min wrist speed (px/frame).
        hip_motion_thresh (float): Max hip movement allowed (i.e. stationary).

    Returns:
        bool: True if upper-body aggression while lower body is still.
    """
    if pid not in kp_history or len(kp_history[pid]) < 2:
        return False

    kp_prev = kp_history[pid][-2][0].cpu().numpy()
    kp_curr = kp_history[pid][-1][0].cpu().numpy()

    # Check hip motion
    lhip_prev, rhip_prev = kp_prev[11], kp_prev[12]
    lhip_curr, rhip_curr = kp_curr[11], kp_curr[12]

    # Confidence check
    if not check_keypoint_confidence(lhip_prev, rhip_prev, lhip_curr, rhip_curr):
        return False

    # Hip displacement
    hip_prev = (lhip_prev[:2] + rhip_prev[:2]) / 2
    hip_curr = (lhip_curr[:2] + rhip_curr[:2]) / 2
    hip_movement = np.linalg.norm(hip_curr - hip_prev)

    if hip_movement > hip_motion_thresh:
        return False  # Too much movement → not low body motion

    # Now apply upper-body logic (reuse check_arm_aggression)
    return detect_aggressive_arm_motion(
        kp_history,
        pid,
        threshold_angle_change=angle_thresh,
        threshold_speed=speed_thresh,
    )


def detect_side_or_back_aggression(kp_history, pid, speed_thresh=10):
    """
    Detect aggressive behavior from side/back view based on asymmetric visibility and speed.

    Args:
        kp_history (dict): Person ID mapped to keypoints.
        pid (int): Person ID.
        speed_thresh (float): Wrist movement threshold.

    Returns:
        bool: True if side-view aggression detected.
    """
    if pid not in kp_history or len(kp_history[pid]) < 2:
        return False

    kp_prev = kp_history[pid][-2][0].cpu().numpy()
    kp_curr = kp_history[pid][-1][0].cpu().numpy()

    # Extract left and right wrists
    lw_prev, rw_prev = kp_prev[9], kp_prev[10]
    lw_curr, rw_curr = kp_curr[9], kp_curr[10]

    # Check if confidence scores exist (length == 3)
    if not (len(lw_prev) >= 3 and len(rw_prev) >= 3):
        print("⚠️ Missing confidence in wrist keypoints.")
        return False
    
    if len(lw_prev) < 3 or len(rw_prev) < 3:
        return False

    lw_conf, rw_conf = lw_prev[2], rw_prev[2]

    # Side view assumption: one hand is visible, one is not
    side_view = (lw_conf > 0.5 and rw_conf < 0.3) or (rw_conf > 0.5 and lw_conf < 0.3)
    if not side_view:
        return False

    # Speed of visible wrist
    if lw_conf > 0.5 and rw_conf < 0.3:
        wrist_speed = np.linalg.norm(lw_curr[:2] - lw_prev[:2])
    elif rw_conf > 0.5 and lw_conf < 0.3:
        wrist_speed = np.linalg.norm(rw_curr[:2] - rw_prev[:2])
    else:
        return False

    return wrist_speed > speed_thresh


def detect_both_hands_aggression(kp_history, pid, angle_thresh=35, speed_thresh=8):
    """
    Detects aggression using both arms (right and left).

    Args:
        kp_history (dict): Dictionary of {pid: [keypoints_t-1, keypoints_t]}
        pid (int): Person ID
        angle_thresh (float): Minimum angle change to count as aggression.
        speed_thresh (float): Minimum wrist speed to count as aggression.

    Returns:
        bool: True if both arms show aggressive motion.
    """
    if pid not in kp_history or len(kp_history[pid]) < 2:
        return False

    kp_prev = kp_history[pid][-2][0].cpu().numpy()
    kp_curr = kp_history[pid][-1][0].cpu().numpy()

    # Right arm: 6 → 8 → 10
    right_angle, right_speed = compute_arm_metrics(kp_prev, kp_curr, 6, 8, 10)

    # Left arm: 5 → 7 → 9
    left_angle, left_speed = compute_arm_metrics(kp_prev, kp_curr, 5, 7, 9)

    right_aggressive = right_angle > angle_thresh and right_speed > speed_thresh
    left_aggressive = left_angle > angle_thresh and left_speed > speed_thresh

    return right_aggressive and left_aggressive


def too_far_for_skeleton(bbox, min_area_threshold=3000):
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return area < min_area_threshold


def track_object_disappearance(object_tracker, current_frame_count, disappearance_threshold=10):
    disappeared = []
    to_delete = []
    for obj_id, info in object_tracker.items():
        if current_frame_count - info["last_seen"] > disappearance_threshold:
            disappeared.append((obj_id, info))
            to_delete.append(obj_id)  # Optional: clean up tracker

    for obj_id in to_delete:
        del object_tracker[obj_id]
    return disappeared



def is_too_close(person_a, person_b, threshold=40):
    xa1, ya1, xa2, ya2 = person_a["bbox"]
    xb1, yb1, xb2, yb2 = person_b["bbox"]

    center_a = ((xa1 + xa2) // 2, (ya1 + ya2) // 2)
    center_b = ((xb1 + xb2) // 2, (yb1 + yb2) // 2)

    dist = np.linalg.norm(np.array(center_a) - np.array(center_b))
    return dist < threshold