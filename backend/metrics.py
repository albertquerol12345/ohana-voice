import threading


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "windows_total": 0,
            "windows_skipped": 0,
            "windows_transcribed": 0,
            "dropped_windows": 0,
            "events_accepted": 0,
        }

    def inc(self, key: str, value: int = 1):
        with self._lock:
            self._data[key] = self._data.get(key, 0) + value

    def set(self, key: str, value):
        with self._lock:
            self._data[key] = value

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)
