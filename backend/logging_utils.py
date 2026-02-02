import json
import os
import threading
from datetime import datetime, timezone


class JsonlLogger:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, payload: dict):
        if "ts" not in payload:
            payload["ts"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
