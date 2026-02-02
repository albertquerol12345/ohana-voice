import os
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soundfile as sf


def normalize_audio(samples: np.ndarray, target_rms: float, max_gain: float) -> np.ndarray:
    if samples.size == 0:
        return samples
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 1e-6:
        return samples
    gain = min(target_rms / rms, max_gain)
    normalized = samples * gain
    peak = float(np.max(np.abs(normalized)))
    if peak > 0.95:
        normalized = normalized * (0.95 / peak)
    return normalized.astype(np.float32)


@dataclass
class WindowItem:
    window_id: int
    audio: np.ndarray
    rms: float
    peak: float
    noise_floor: float
    gate_active: bool
    ts_mono: float


class RingBuffer:
    def __init__(self, size: int):
        self.size = size
        self.data = np.zeros(size, dtype=np.float32)
        self.write_idx = 0
        self.filled = 0

    def append(self, samples: np.ndarray):
        n = len(samples)
        if n == 0:
            return
        if n >= self.size:
            self.data[:] = samples[-self.size :]
            self.write_idx = 0
            self.filled = self.size
            return
        end = self.write_idx + n
        if end <= self.size:
            self.data[self.write_idx : end] = samples
        else:
            first = self.size - self.write_idx
            self.data[self.write_idx :] = samples[:first]
            self.data[: end % self.size] = samples[first:]
        self.write_idx = end % self.size
        self.filled = min(self.size, self.filled + n)

    def get_last(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        if n > self.filled:
            n = self.filled
        start = (self.write_idx - n) % self.size
        if start + n <= self.size:
            return self.data[start : start + n].copy()
        first = self.size - start
        return np.concatenate((self.data[start:], self.data[: n - first])).copy()


class AudioStream:
    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        queue_max: int,
        device: str | None = None,
        record_queue_max: int = 50,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self.queue = queue.Queue(maxsize=queue_max)
        self.record_queue_max = record_queue_max
        self.dropped_frames = 0
        self.record_dropped_chunks = 0
        self._stream = None
        self._recorder = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass
        samples = indata[:, 0].copy()
        try:
            self.queue.put_nowait(samples)
        except queue.Full:
            self.dropped_frames += 1
        if self._recorder is not None:
            if not self._recorder.enqueue(samples):
                self.record_dropped_chunks += 1

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
            device=self.device,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.stop_recording()

    def start_recording(self, path: str) -> bool:
        if self._recorder is not None:
            return False
        self.record_dropped_chunks = 0
        self._recorder = AudioRecorder(path, self.sample_rate, self.record_queue_max)
        self._recorder.start()
        return True

    def stop_recording(self):
        if self._recorder is None:
            return
        self._recorder.stop()
        self._recorder.join(timeout=5.0)
        self._recorder = None


class AudioRecorder(threading.Thread):
    def __init__(self, path: str, sample_rate: int, queue_max: int):
        super().__init__(daemon=True)
        self.path = path
        self.sample_rate = sample_rate
        self.queue = queue.Queue(maxsize=queue_max)
        self._stop_event = threading.Event()
        self.dropped_chunks = 0

    def enqueue(self, samples: np.ndarray) -> bool:
        try:
            self.queue.put_nowait(samples.copy())
            return True
        except queue.Full:
            self.dropped_chunks += 1
            return False

    def stop(self):
        self._stop_event.set()

    def run(self):
        dir_path = os.path.dirname(self.path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with sf.SoundFile(
            self.path,
            mode="w",
            samplerate=self.sample_rate,
            channels=1,
            subtype="PCM_16",
        ) as file:
            while not self._stop_event.is_set() or not self.queue.empty():
                try:
                    block = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                file.write(block)


class WindowGenerator(threading.Thread):
    def __init__(
        self,
        audio_stream: AudioStream,
        window_sec: float,
        hop_sec: float,
        ring_sec: float,
        use_gate: bool,
        gate_factor: float,
        gate_min_rms: float,
        gate_hangover_sec: float,
        gate_noise_alpha: float,
        gate_noise_floor_init: float,
        normalize_audio: bool,
        target_rms: float,
        max_gain: float,
        window_queue: queue.Queue,
        drop_oldest: bool,
        metrics,
        logger,
        state_cb,
        cooldown_cb,
        state_interval_sec: float,
        state_send_levels: bool,
    ):
        super().__init__(daemon=True)
        self.audio_stream = audio_stream
        self.sample_rate = audio_stream.sample_rate
        self.window_samples = int(window_sec * self.sample_rate)
        self.hop_samples = int(hop_sec * self.sample_rate)
        self.ring = RingBuffer(int(ring_sec * self.sample_rate))
        self.use_gate = use_gate
        self.gate_factor = gate_factor
        self.gate_min_rms = gate_min_rms
        self.gate_hangover_sec = gate_hangover_sec
        self.gate_noise_alpha = gate_noise_alpha
        self.noise_floor = gate_noise_floor_init
        self.enable_normalize = normalize_audio
        self.target_rms = target_rms
        self.max_gain = max_gain
        self.window_queue = window_queue
        self.drop_oldest = drop_oldest
        self.metrics = metrics
        self.logger = logger
        self.state_cb = state_cb
        self.cooldown_cb = cooldown_cb
        self.state_interval_sec = state_interval_sec
        self.state_send_levels = state_send_levels
        self._stop_event = threading.Event()
        self._last_state_ts = 0.0
        self._samples_since_hop = 0
        self._window_id = 0
        self._gate_until = 0.0

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                samples = self.audio_stream.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.ring.append(samples)
            self._samples_since_hop += len(samples)
            while self._samples_since_hop >= self.hop_samples:
                self._samples_since_hop -= self.hop_samples
                self._emit_window()

    def _emit_window(self):
        if self.ring.filled < self.window_samples:
            return
        now = time.monotonic()
        window = self.ring.get_last(self.window_samples)
        rms = float(np.sqrt(np.mean(window ** 2)))
        peak = float(np.max(np.abs(window))) if window.size else 0.0
        threshold = max(self.noise_floor * self.gate_factor, self.gate_min_rms)

        gate_active = True
        if self.use_gate:
            if rms >= threshold:
                self._gate_until = now + self.gate_hangover_sec
                gate_active = True
            elif now < self._gate_until:
                gate_active = True
            else:
                gate_active = False

            if not gate_active:
                self.noise_floor = (1.0 - self.gate_noise_alpha) * self.noise_floor + (
                    self.gate_noise_alpha * rms
                )
                self.noise_floor = max(self.noise_floor, self.gate_min_rms)

        state = "detecting" if gate_active else "listening"
        if self.cooldown_cb and self.cooldown_cb(now):
            state = "cooldown"
        if self.state_cb and now - self._last_state_ts >= self.state_interval_sec:
            payload = {"type": "state", "state": state}
            if self.state_send_levels:
                payload["level"] = {
                    "rms": rms,
                    "peak": peak,
                    "noise_floor": self.noise_floor,
                    "threshold": threshold,
                }
            self.state_cb(payload)
            self._last_state_ts = now

        self.metrics.inc("windows_total", 1)
        if not gate_active:
            self.metrics.inc("windows_skipped", 1)
            self.logger.write(
                {
                    "type": "window",
                    "window_id": self._window_id,
                    "gate": False,
                    "rms": rms,
                    "peak": peak,
                    "noise_floor": self.noise_floor,
                    "threshold": threshold,
                    "reason": "gate_off",
                }
            )
            self._window_id += 1
            return

        window_for_model = window
        if self.enable_normalize:
            window_for_model = normalize_audio(window, self.target_rms, self.max_gain)

        item = WindowItem(
            window_id=self._window_id,
            audio=window_for_model,
            rms=rms,
            peak=peak,
            noise_floor=self.noise_floor,
            gate_active=gate_active,
            ts_mono=now,
        )
        self._window_id += 1

        if self.window_queue.full():
            self.metrics.inc("dropped_windows", 1)
            if self.drop_oldest:
                try:
                    self.window_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.window_queue.put_nowait(item)
                except queue.Full:
                    pass
            return

        self.window_queue.put_nowait(item)
        self.metrics.inc("windows_transcribed", 1)
