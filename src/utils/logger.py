# src/utils/logger.py

import json
import os

LOG_FILE = "outputs/logs/event_log.json"

def init_logger():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump([], f)  # start with an empty list

def log_event(frame_count, timestamp, event_type, pid, extra_info=None):
    log_entry = {
        "frame": frame_count,
        "timestamp": timestamp,
        "event": event_type,
        "pid": pid,
    }
    if extra_info:
        log_entry.update(extra_info)

    with open(LOG_FILE, "r+") as f:
        data = json.load(f)
        data.append(log_entry)
        f.seek(0)
        json.dump(data, f, indent=2)
