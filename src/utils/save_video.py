# File: src/utils/save_video.py

import cv2

def init_video_writer(input_cap, output_path: str, codec: str = 'mp4v'):
    """
    Initializes and returns a cv2.VideoWriter object.

    Args:
        input_cap: OpenCV VideoCapture object (already opened).
        output_path (str): Where to save the annotated video.
        codec (str): FourCC codec (default: 'mp4v').

    Returns:
        cv2.VideoWriter: Video writer object ready to write frames.
    """
    width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return out