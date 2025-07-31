import numpy as np

def is_stationary(kp_history, pid, movement_thresh=3):
    if pid not in kp_history or len(kp_history[pid]) < 2:
        return False

    kp_prev = kp_history[pid][-2][0].cpu().numpy()
    kp_curr = kp_history[pid][-1][0].cpu().numpy()

    try:
        lhip_prev, rhip_prev = kp_prev[11], kp_prev[12]
        lhip_curr, rhip_curr = kp_curr[11], kp_curr[12]
    except IndexError:
        return False

    if any(len(p) < 2 for p in [lhip_prev, rhip_prev, lhip_curr, rhip_curr]):
        return False

    hip_prev = (np.array(lhip_prev[:2]) + np.array(rhip_prev[:2])) / 2
    hip_curr = (np.array(lhip_curr[:2]) + np.array(rhip_curr[:2])) / 2
    movement = np.linalg.norm(hip_curr - hip_prev)

    return movement < movement_thresh

def is_defensive_pose(keypoints):
    try:
        lshoulder = keypoints[5]
        rshoulder = keypoints[6]
        lhand = keypoints[9]
        rhand = keypoints[10]

        # Check keypoints have confidence value
        if len(lhand) < 3 or len(rhand) < 3 or len(lshoulder) < 3 or len(rshoulder) < 3:
            return False

        return (
            (lhand[1] < lshoulder[1] if lhand[2] > 0.5 else False) or
            (rhand[1] < rshoulder[1] if rhand[2] > 0.5 else False)
        )
    except Exception:
        return False
    
    
def detect_victim(pid, kp_history, person_tracker):
    if pid not in kp_history or len(kp_history[pid]) < 2:
        return False

    stationary = is_stationary(kp_history, pid)
    hands_up = is_defensive_pose(kp_history[pid][-1])
    crouching = is_ducking(kp_history[pid][-1])

    if stationary and (hands_up or crouching):
        person_tracker[pid]["state"] = "blue"
        person_tracker[pid]["victim_candidate"] = True
        return True

    return False


def is_ducking(kp_curr):
    keypoints = kp_curr[0].cpu().numpy()
    try:
        nose = keypoints[0]
        lshoulder, rshoulder = keypoints[5], keypoints[6]
    except IndexError:
        return False

    if any(len(k) < 3 or k[2] < 0.5 for k in [nose, lshoulder, rshoulder]):
        return False

    shoulder_avg_y = (lshoulder[1] + rshoulder[1]) / 2
    return nose[1] > shoulder_avg_y + 20  # head lower than shoulders → crouching

