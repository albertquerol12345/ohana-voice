import asyncio
import audioop
import json
import os
import queue
import re
import threading
import unicodedata
from collections import deque
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import sounddevice as sd
import websockets
import webrtcvad
from vosk import KaldiRecognizer, Model

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
KEYWORDS_PATH = os.path.join(os.path.dirname(__file__), "keywords.json")
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "vosk-model-es-0.42")
FALLBACK_MODEL_PATH = os.path.join(ROOT_DIR, "vosk-model-small-es-0.42")
MODEL_PATH = os.environ.get("VOSK_MODEL_PATH")
if not MODEL_PATH:
    MODEL_PATH = DEFAULT_MODEL_PATH if os.path.isdir(DEFAULT_MODEL_PATH) else FALLBACK_MODEL_PATH
STATIC_PORT = int(os.environ.get("STATIC_PORT", "8010"))
WS_PORT = int(os.environ.get("WS_PORT", "2700"))
SAMPLE_RATE = 16000

USE_VAD = os.environ.get("VOSK_USE_VAD", "0") != "0"
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "1"))
VAD_FRAME_MS = int(os.environ.get("VAD_FRAME_MS", "20"))
VAD_START_FRAMES = int(os.environ.get("VAD_START_FRAMES", "1"))
VAD_END_FRAMES = int(os.environ.get("VAD_END_FRAMES", "8"))
VAD_PRE_ROLL_FRAMES = int(os.environ.get("VAD_PRE_ROLL_FRAMES", "5"))
VAD_FORCE_RMS = int(os.environ.get("VAD_FORCE_RMS", "40"))
VAD_MAX_FRAMES = int(os.environ.get("VAD_MAX_FRAMES", "120"))

AUTO_GAIN = os.environ.get("VOSK_AUTO_GAIN", "1") != "0"
INPUT_GAIN = float(os.environ.get("VOSK_INPUT_GAIN", "1.2"))
TARGET_RMS = int(os.environ.get("VOSK_TARGET_RMS", "2400"))
MIN_RMS = int(os.environ.get("VOSK_MIN_RMS", "40"))
MAX_GAIN = float(os.environ.get("VOSK_MAX_GAIN", "12.0"))

LOG_DIR = os.path.join(ROOT_DIR, "logs")
LOG_PATH = os.environ.get("DETECTION_LOG_PATH", os.path.join(LOG_DIR, "detections.jsonl"))
WORDS_PATH = os.path.join(MODEL_PATH, "graph", "words.txt")

FINAL_SIMILARITY = float(os.environ.get("VOSK_FINAL_SIMILARITY", "0.78"))
PARTIAL_SIMILARITY = float(os.environ.get("VOSK_PARTIAL_SIMILARITY", "0.84"))
PARTIAL_MIN_CHARS = int(os.environ.get("VOSK_PARTIAL_MIN_CHARS", "4"))
PARTIAL_STABILITY = int(os.environ.get("VOSK_PARTIAL_STABILITY", "2"))

FALLBACK_ALIASES = {
    "ojana": ["ohana", "johana"],
    "big ojana": ["big ohana", "big johana"],
    "suli": ["sully", "sulli"],
    "uasoski": ["wazowski"],
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, payload: dict):
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_word_list():
    if not os.path.isfile(WORDS_PATH):
        return None
    vocab = set()
    with open(WORDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            vocab.add(parts[0])
    return vocab


def phrase_in_vocab(phrase: str, vocab: set | None) -> bool:
    if vocab is None:
        return True
    for token in phrase.lower().split():
        if token not in vocab:
            return False
    return True


def load_keywords(vocab: set | None):
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    alias_map = {}
    alias_entries = []
    grammar_set = set()
    collisions = []
    missing_vocab = []

    for key, aliases in data.items():
        phonetic = aliases.get("phonetic") if isinstance(aliases, dict) else aliases
        for alias in phonetic or []:
            alias = alias.strip().lower()
            if not alias:
                continue
            variants = {alias}
            normalized_alias = normalize_text(alias)
            if normalized_alias and normalized_alias != alias:
                variants.add(normalized_alias)
            for fallback in FALLBACK_ALIASES.get(alias, []):
                variants.add(fallback)
            if not alias.startswith("la "):
                variants.add(f"la {alias}")
                for fallback in FALLBACK_ALIASES.get(alias, []):
                    variants.add(f"la {fallback}")
            for variant in variants:
                if not phrase_in_vocab(variant, vocab):
                    missing_vocab.append({"alias": alias, "variant": variant})
                    continue
                normalized = normalize_text(variant)
                if normalized in alias_map:
                    if alias_map[normalized] != key:
                        collisions.append((normalized, alias_map[normalized], key))
                    continue
                alias_map[normalized] = key
                alias_entries.append((normalized, key))
                grammar_set.add(variant)

    grammar = json.dumps(sorted(grammar_set), ensure_ascii=False)
    return alias_map, alias_entries, grammar, collisions, missing_vocab


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def similarity_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    dist = levenshtein_distance(a, b)
    return 1.0 - (dist / max(len(a), len(b)))


class RecognizerWorker:
    def __init__(
        self,
        model: Model,
        grammar: str,
        alias_map: dict,
        alias_entries: list,
        send_cb,
        logger: DetectionLogger,
    ):
        self.model = model
        self.grammar = grammar
        self.alias_map = alias_map
        self.alias_entries = alias_entries
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
        def make_recognizer():
            rec = KaldiRecognizer(self.model, SAMPLE_RATE, self.grammar)
            rec.SetWords(True)
            return rec

        recognizer = make_recognizer()
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS) if USE_VAD else None
        frame_samples = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000.0))
        frame_bytes = frame_samples * 2
        pre_buffer = deque(maxlen=VAD_PRE_ROLL_FRAMES)
        speech_active = False
        speech_frames = 0
        silence_frames = 0
        utterance_frames = 0

        last_partial_key = None
        partial_hits = 0

        def best_match(normalized: str, threshold: float, min_chars: int):
            if not normalized or len(normalized) < min_chars:
                return None, 0.0, None
            if normalized in self.alias_map:
                return self.alias_map[normalized], 1.0, normalized
            best_key = None
            best_score = 0.0
            best_alias = None
            for alias_norm, key in self.alias_entries:
                score = similarity_score(normalized, alias_norm)
                if score > best_score:
                    best_score = score
                    best_key = key
                    best_alias = alias_norm
            if best_score >= threshold:
                return best_key, best_score, best_alias
            return None, best_score, best_alias

        def handle_partial(text: str):
            nonlocal last_partial_key, partial_hits
            normalized = normalize_text(text)
            key, score, alias = best_match(normalized, PARTIAL_SIMILARITY, PARTIAL_MIN_CHARS)
            if not key:
                if normalized != last_partial_key:
                    partial_hits = 0
                return
            if key == last_partial_key:
                partial_hits += 1
            else:
                last_partial_key = key
                partial_hits = 1
            self.logger.write(
                {
                    "type": "partial",
                    "engine": "kaldi",
                    "heard": text,
                    "normalized": normalized,
                    "matched": key,
                    "score": score,
                    "alias": alias,
                    "hits": partial_hits,
                }
            )
            if partial_hits >= PARTIAL_STABILITY:
                self.send_cb(
                    {
                        "type": "result",
                        "key": key,
                        "heard": text,
                        "confidence": score,
                        "similarity": score,
                    }
                )
                self._stop_event.set()

        def handle_result(result: dict, reason: str):
            text = (result.get("text") or "").strip()
            normalized = normalize_text(text)
            matched, score, alias = best_match(normalized, FINAL_SIMILARITY, 1)
            accept = matched is not None
            self.logger.write(
                {
                    "type": "final",
                    "engine": "kaldi",
                    "reason": reason,
                    "heard": text,
                    "normalized": normalized,
                    "matched": matched,
                    "score": score,
                    "alias": alias,
                    "accept": accept,
                }
            )
            if accept:
                self.send_cb(
                    {
                        "type": "result",
                        "key": matched,
                        "heard": text,
                        "confidence": score,
                        "similarity": score,
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
                        boosted = apply_gain(frame)

                        if not USE_VAD:
                            if recognizer.AcceptWaveform(boosted):
                                result = json.loads(recognizer.Result())
                                handle_result(result, "accept_waveform")
                                if self._stop_event.is_set():
                                    break
                            else:
                                partial = json.loads(recognizer.PartialResult()).get("partial", "")
                                if partial:
                                    handle_partial(partial)
                            continue

                        raw_rms = audioop.rms(boosted, 2)
                        is_speech = vad.is_speech(boosted, SAMPLE_RATE) if vad else True
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
                                recognizer = make_recognizer()
                                for buffered in pre_buffer:
                                    recognizer.AcceptWaveform(buffered)
                                pre_buffer.clear()
                        else:
                            if recognizer.AcceptWaveform(boosted):
                                result = json.loads(recognizer.Result())
                                handle_result(result, "accept_waveform")
                                if self._stop_event.is_set():
                                    break
                            else:
                                partial = json.loads(recognizer.PartialResult()).get("partial", "")
                                if partial:
                                    handle_partial(partial)
                            utterance_frames += 1
                            if is_speech:
                                silence_frames = 0
                            else:
                                silence_frames += 1
                                if silence_frames >= VAD_END_FRAMES:
                                    result = json.loads(recognizer.FinalResult())
                                    handle_result(result, "vad_end")
                                    speech_active = False
                                    speech_frames = 0
                                    silence_frames = 0
                                    utterance_frames = 0
                                    pre_buffer.clear()
                                    recognizer = make_recognizer()
                                    break
                            if utterance_frames >= VAD_MAX_FRAMES:
                                result = json.loads(recognizer.FinalResult())
                                handle_result(result, "vad_max")
                                speech_active = False
                                speech_frames = 0
                                silence_frames = 0
                                utterance_frames = 0
                                pre_buffer.clear()
                                recognizer = make_recognizer()
                                break
        except Exception as exc:
            self.send_cb({"type": "error", "message": str(exc)})
        finally:
            if USE_VAD and speech_active:
                result = json.loads(recognizer.FinalResult())
                handle_result(result, "stop_flush")
            self.send_cb({"type": "status", "status": "idle"})
            with self._lock:
                self._active = False


class StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)


def start_static_server(port: int):
    httpd = ThreadingHTTPServer(("", port), StaticHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


async def main():
    if not os.path.isdir(MODEL_PATH):
        print("Vosk model not found.")
        print("Set VOSK_MODEL_PATH or download the model.")
        return

    vocab = load_word_list()
    alias_map, alias_entries, grammar, collisions, missing_vocab = load_keywords(vocab)
    model = Model(MODEL_PATH)
    logger = DetectionLogger(LOG_PATH)
    for alias, kept, ignored in collisions:
        logger.write({"type": "collision", "alias": alias, "kept": kept, "ignored": ignored})
    for item in missing_vocab:
        logger.write({"type": "vocab_skip", "alias": item["alias"], "variant": item["variant"]})

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

    recognizer = RecognizerWorker(model, grammar, alias_map, alias_entries, send_cb, logger)

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
