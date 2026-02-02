import asyncio
import audioop
import json
import os
import queue
import threading
from collections import deque
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import websockets
import webrtcvad
import sounddevice as sd

from kws_utils import SAMPLE_RATE, log_mel, make_melspec, pad_or_trim
from kws_train import KWSNet

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
MODEL_PATH = Path(os.environ.get("KWS_MODEL_PATH", Path(__file__).with_name("kws_model.pt")))
STATIC_PORT = int(os.environ.get("STATIC_PORT", "8010"))
WS_PORT = int(os.environ.get("WS_PORT", "2700"))
LOG_DIR = ROOT_DIR / "logs"
LOG_PATH = Path(os.environ.get("DETECTION_LOG_PATH", LOG_DIR / "detections.jsonl"))

USE_VAD = os.environ.get("KWS_USE_VAD", "1") != "0"
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "1"))
VAD_FRAME_MS = int(os.environ.get("VAD_FRAME_MS", "20"))
VAD_START_FRAMES = int(os.environ.get("VAD_START_FRAMES", "1"))
VAD_END_FRAMES = int(os.environ.get("VAD_END_FRAMES", "8"))
VAD_PRE_ROLL_FRAMES = int(os.environ.get("VAD_PRE_ROLL_FRAMES", "5"))
VAD_FORCE_RMS = int(os.environ.get("VAD_FORCE_RMS", "40"))
VAD_MAX_FRAMES = int(os.environ.get("VAD_MAX_FRAMES", "120"))

AUTO_GAIN = os.environ.get("KWS_AUTO_GAIN", "1") != "0"
INPUT_GAIN = float(os.environ.get("KWS_INPUT_GAIN", "1.2"))
TARGET_RMS = int(os.environ.get("KWS_TARGET_RMS", "2400"))
MIN_RMS = int(os.environ.get("KWS_MIN_RMS", "40"))
MAX_GAIN = float(os.environ.get("KWS_MAX_GAIN", "12.0"))

MIN_SECONDS = float(os.environ.get("KWS_MIN_SECONDS", "0.25"))
ACCEPT_PROB = float(os.environ.get("KWS_ACCEPT_PROB", "0.6"))
MIN_MARGIN = float(os.environ.get("KWS_MIN_MARGIN", "0.08"))


def apply_gain(frame: bytes) -> bytes:
    if not AUTO_GAIN and INPUT_GAIN == 1.0:
        return frame
    out = frame
    if INPUT_GAIN != 1.0:
        out = audioop.mul(out, 2, INPUT_GAIN)
    if AUTO_GAIN:
        rms = audioop.rms(out, 2)
        effective_rms = max(rms, MIN_RMS)
        boost = min(MAX_GAIN, TARGET_RMS / max(effective_rms, 1))
        if boost > 1.0:
            out = audioop.mul(out, 2, boost)
    return out


class DetectionLogger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict):
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class KWSModel:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Missing model: {path}")
        payload = torch.load(path, map_location="cpu")
        self.labels = payload["labels"]
        self.window_seconds = float(payload.get("window_seconds", 1.0))
        self.model = KWSNet(len(self.labels))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.mels = make_melspec()

    def predict(self, samples: torch.Tensor):
        window_samples = int(self.window_seconds * SAMPLE_RATE)
        wave = pad_or_trim(samples, window_samples)
        mel = self.mels(wave)
        mel = log_mel(mel)
        mel = mel.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(mel)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        top_prob, top_idx = torch.max(probs, dim=0)
        second_prob = float(torch.topk(probs, 2).values[1]) if probs.numel() > 1 else 0.0
        return {
            "label": self.labels[int(top_idx)],
            "prob": float(top_prob),
            "second_prob": second_prob,
        }


class RecognizerWorker:
    def __init__(self, model: KWSModel, send_cb, logger: DetectionLogger):
        self.model = model
        self.send_cb = send_cb
        self.logger = logger
        self._lock = threading.Lock()
        self._active = False
        self._stop_event = threading.Event()
        self._audio_queue = queue.Queue()
        self._thread = None

    def start(self):
        with self._lock:
            if self._active:
                return False
            self._active = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def stop(self):
        with self._lock:
            if not self._active:
                return False
            self._stop_event.set()
            self._active = False
            return True

    def _run(self):
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS) if USE_VAD else None
        frame_samples = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000.0))
        frame_bytes = frame_samples * 2
        pre_buffer = deque(maxlen=VAD_PRE_ROLL_FRAMES)
        speech_active = False
        speech_frames = 0
        silence_frames = 0
        utterance_frames = 0
        recorded = []

        def finalize(reason: str):
            nonlocal recorded, speech_active, speech_frames, silence_frames, utterance_frames
            if not recorded:
                return
            audio_bytes = b"".join(recorded)
            recorded = []
            speech_active = False
            speech_frames = 0
            silence_frames = 0
            utterance_frames = 0

            samples = torch.frombuffer(audio_bytes, dtype=torch.int16).float() / 32768.0
            duration = samples.numel() / SAMPLE_RATE
            if duration < MIN_SECONDS:
                return
            result = self.model.predict(samples)
            accept = result["prob"] >= ACCEPT_PROB and (
                result["prob"] - result["second_prob"]
            ) >= MIN_MARGIN
            self.logger.write(
                {
                    "type": "final",
                    "engine": "kws",
                    "reason": reason,
                    "matched": result["label"],
                    "prob": result["prob"],
                    "second_prob": result["second_prob"],
                    "accept": accept,
                    "duration": duration,
                }
            )
            if accept:
                self.send_cb(
                    {
                        "type": "result",
                        "key": result["label"],
                        "heard": result["label"],
                        "confidence": result["prob"],
                        "similarity": None,
                    }
                )
                self._stop_event.set()

        def callback(indata, frames, time_info, status):
            if status:
                self.send_cb({"type": "status", "status": "audio_warning", "detail": str(status)})
            if self._stop_event.is_set():
                return
            self._audio_queue.put(bytes(indata))

        try:
            blocksize = frame_samples
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=blocksize,
                dtype="int16",
                channels=1,
                callback=callback,
            ):
                self.send_cb({"type": "status", "status": "listening"})
                while not self._stop_event.is_set():
                    try:
                        data = self._audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    for offset in range(0, len(data), frame_bytes):
                        frame = data[offset : offset + frame_bytes]
                        if len(frame) < frame_bytes:
                            continue
                        raw_rms = audioop.rms(frame, 2)
                        boosted = apply_gain(frame)
                        is_speech = True
                        if USE_VAD:
                            is_speech = vad.is_speech(frame, SAMPLE_RATE)
                            if not is_speech and raw_rms >= VAD_FORCE_RMS:
                                is_speech = True

                        if not speech_active:
                            pre_buffer.append(boosted)
                            if is_speech:
                                speech_frames += 1
                            else:
                                speech_frames = 0
                            if speech_frames >= VAD_START_FRAMES:
                                speech_active = True
                                silence_frames = 0
                                utterance_frames = 0
                                recorded = list(pre_buffer)
                                pre_buffer.clear()
                        else:
                            recorded.append(boosted)
                            utterance_frames += 1
                            if is_speech:
                                silence_frames = 0
                            else:
                                silence_frames += 1
                                if silence_frames >= VAD_END_FRAMES:
                                    finalize("vad_end")
                                    break
                            if utterance_frames >= VAD_MAX_FRAMES:
                                finalize("vad_max")
                                break
        except Exception as exc:
            self.send_cb({"type": "error", "message": str(exc)})
        finally:
            if recorded:
                finalize("stop_flush")
            self.send_cb({"type": "status", "status": "idle"})
            with self._lock:
                self._active = False


class StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)


def start_static_server(port: int):
    httpd = ThreadingHTTPServer(("", port), StaticHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


async def main():
    model = KWSModel(MODEL_PATH)
    logger = DetectionLogger(LOG_PATH)
    loop = asyncio.get_running_loop()
    clients = set()

    async def broadcast(payload):
        if not clients:
            return
        data = json.dumps(payload, ensure_ascii=False)
        stale = []
        for ws in clients:
            try:
                await ws.send(data)
            except Exception:
                stale.append(ws)
        for ws in stale:
            clients.discard(ws)

    def send_cb(payload):
        asyncio.run_coroutine_threadsafe(broadcast(payload), loop)

    recognizer = RecognizerWorker(model, send_cb, logger)

    async def handler(ws):
        clients.add(ws)
        await ws.send(json.dumps({"type": "status", "status": "connected"}))
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                action = data.get("action")
                if action == "start":
                    started = recognizer.start()
                    if started:
                        await ws.send(json.dumps({"type": "status", "status": "starting"}))
                elif action == "stop":
                    stopped = recognizer.stop()
                    if stopped:
                        await ws.send(json.dumps({"type": "status", "status": "stopping"}))
        finally:
            clients.discard(ws)

    start_static_server(STATIC_PORT)
    print(f"Static server: http://localhost:{STATIC_PORT}")
    print(f"WebSocket server: ws://localhost:{WS_PORT}")
    print(f"Model: {MODEL_PATH}")

    async with websockets.serve(handler, "", WS_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
