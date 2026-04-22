import json
import os
from datetime import datetime

LOG_FILE = "logs/history.json"


def ensure_log_file():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)


def log_result(filename, model_name, detections):
    """
    detections: list of dicts
    example:
    [
        {
            "label": "car",
            "verdict": "Possible Damage",
            "confidence": 81.0
        }
    ]
    """

    ensure_log_file()

    log_entry = {
        "filename": filename,
        "model": model_name,
        "detections": detections,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(LOG_FILE, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data.insert(0, log_entry)  # newest first
        f.seek(0)
        json.dump(data, f, indent=2)