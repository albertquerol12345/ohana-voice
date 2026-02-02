import asyncio
import audioop
import difflib
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
MODEL_PATH = os.environ.get(
    "VOSK_MODEL_PATH",
    os.path.join(ROOT_DIR, "vosk-model-small-es-0.42"),
)
STATIC_PORT = int(os.environ.get("STATIC_PORT", "8000"))
WS_PORT = int(os.environ.get("WS_PORT", "2700"))
SAMPLE_RATE = 16000
MIN_CONFIDENCE = 0.6
MIN_SIMILARITY = 0.6
MIN_MARGIN = 0.0
MIN_PHONETIC_SIMILARITY = 0.6
MAX_FUZZY_TOKENS = 4
USE_GRAMMAR = os.environ.get("VOSK_USE_GRAMMAR", "1") != "0"
INCLUDE_ASR_ALIASES = os.environ.get("VOSK_INCLUDE_ASR_ALIASES", "0") != "0"
USE_VAD = os.environ.get("VOSK_USE_VAD", "1") != "0"
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "0"))
VAD_FRAME_MS = int(os.environ.get("VAD_FRAME_MS", "20"))
VAD_START_FRAMES = int(os.environ.get("VAD_START_FRAMES", "1"))
VAD_END_FRAMES = int(os.environ.get("VAD_END_FRAMES", "8"))
VAD_PRE_ROLL_FRAMES = int(os.environ.get("VAD_PRE_ROLL_FRAMES", "8"))
AUTO_GAIN = os.environ.get("VOSK_AUTO_GAIN", "1") != "0"
INPUT_GAIN = float(os.environ.get("VOSK_INPUT_GAIN", "1.2"))
TARGET_RMS = int(os.environ.get("VOSK_TARGET_RMS", "2600"))
MIN_RMS = int(os.environ.get("VOSK_MIN_RMS", "40"))
MAX_GAIN = float(os.environ.get("VOSK_MAX_GAIN", "16.0"))
VAD_FORCE_RMS = int(os.environ.get("VAD_FORCE_RMS", "40"))
VAD_MAX_FRAMES = int(os.environ.get("VAD_MAX_FRAMES", "120"))
PARTIAL_COMMIT = os.environ.get("VOSK_PARTIAL_COMMIT", "1") != "0"
PARTIAL_STABLE_COUNT = int(os.environ.get("VOSK_PARTIAL_STABLE", "2"))
PARTIAL_MIN_SIMILARITY = float(os.environ.get("VOSK_PARTIAL_MIN_SIMILARITY", "0.8"))
PARTIAL_INTERVAL_FRAMES = int(os.environ.get("VOSK_PARTIAL_INTERVAL_FRAMES", "2"))
LOG_DIR = os.path.join(ROOT_DIR, "logs")
LOG_PATH = os.environ.get("DETECTION_LOG_PATH", os.path.join(LOG_DIR, "detections.jsonl"))
STOPWORDS = {
    "a",
    "al",
    "de",
    "del",
    "el",
    "ella",
    "ellas",
    "ellos",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "mi",
    "mis",
    "por",
    "favor",
    "porfavor",
    "pon",
    "ponme",
    "dame",
    "quiero",
    "una",
    "un",
    "unas",
    "unos",
    "hamburguesa",
    "burger",
    "burguer",
}
TOKEN_MAP = {
    "by": "bi",
}

BIG_TOKENS = {"big", "bi"}
OHANA_TOKENS = {"ohana", "ojana", "oana", "johanna", "hoanna"}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_match(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    tokens = []
    for token in normalized.split():
        token = TOKEN_MAP.get(token, token)
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return " ".join(tokens)


def phonetic_key(text: str) -> str:
    if not text:
        return ""
    text = normalize_text(text)
    text = text.replace("j", "h")
    text = text.replace("v", "b")
    text = text.replace("z", "s")
    text = text.replace("w", "u")
    text = text.replace("ll", "y")
    text = text.replace("qu", "k")
    text = text.replace("q", "k")
    text = text.replace("c", "k")
    text = text.replace("h", "")
    text = text.replace(" ", "")
    text = re.sub(r"(.)\1+", r"\1", text)
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


def load_keywords(use_grammar: bool, include_asr: bool):
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    alias_map = {}
    alias_entries = []
    grammar_set = set()
    collisions = []

    for key, aliases in data.items():
        if isinstance(aliases, dict):
            phonetic = aliases.get("phonetic") or []
            asr = aliases.get("asr") or []
            alias_list = [(alias, "phonetic") for alias in phonetic]
            if include_asr:
                alias_list.extend((alias, "asr") for alias in asr)
        else:
            alias_list = [(alias, "phonetic") for alias in aliases]

        for alias, source in alias_list:
            alias = alias.strip().lower()
            if not alias:
                continue
            variants = {alias}
            if not alias.startswith("la "):
                variants.add(f"la {alias}")
            for variant in variants:
                normalized = normalize_text(variant)
                if normalized in alias_map:
                    if alias_map[normalized]["key"] != key:
                        collisions.append((normalized, alias_map[normalized]["key"], key))
                    continue
                entry = {
                    "alias": normalized,
                    "key": key,
                    "source": source,
                    "phonetic": phonetic_key(normalized),
                }
                alias_map[normalized] = entry
                alias_entries.append(entry)
                grammar_set.add(variant)

    grammar = None
    if use_grammar:
        grammar_set.add("[unk]")
        grammar = json.dumps(sorted(grammar_set), ensure_ascii=False)
    return alias_map, alias_entries, grammar, collisions


def result_confidence(result: dict):
    words = result.get("result") or []
    confs = [w.get("conf") for w in words if "conf" in w]
    if not confs:
        return None
    return sum(confs) / len(confs)


def match_keyword(text: str, alias_map: dict, alias_entries: list, use_fuzzy: bool):
    normalized = normalize_text(text)
    cleaned = clean_text_for_match(text)
    if not normalized:
        return None, None, None, None, cleaned, normalized

    candidates = []

    def add_candidate(value: str):
        if value and value not in candidates:
            candidates.append(value)

    if cleaned:
        add_candidate(cleaned)
    if normalized and normalized not in candidates:
        add_candidate(normalized)
    if cleaned:
        tokens = cleaned.split()
        if len(tokens) > 1:
            max_len = min(len(tokens), MAX_FUZZY_TOKENS)
            for size in range(1, max_len + 1):
                for index in range(0, len(tokens) - size + 1):
                    add_candidate(" ".join(tokens[index : index + size]))

    # Rule: if both "big/bi" and "ohana variants" appear, force Big Ohana.
    if cleaned:
        tokens = set(cleaned.split())
        has_big = bool(tokens & BIG_TOKENS)
        has_ohana = bool(tokens & OHANA_TOKENS)
        if not has_big:
            for prefix in ("big", "bi"):
                if cleaned.startswith(prefix) and any(o in cleaned for o in OHANA_TOKENS):
                    has_big = True
                    has_ohana = True
                    break
        if has_big and has_ohana:
            entry = {"alias": "big_ohana_rule", "key": "Big Ohana", "source": "rule"}
            return entry, 1.0, cleaned, "rule", cleaned, normalized

    for candidate in candidates:
        entry = alias_map.get(candidate)
        if entry:
            return entry, 1.0, candidate, "exact", cleaned, normalized
    if not use_fuzzy:
        return None, None, None, None, cleaned, normalized

    best_key = None
    best_score = 0.0
    best_alias = None
    best_source = None
    best_candidate = None
    second_best = 0.0

    for candidate in candidates:
        if len(candidate.split()) > MAX_FUZZY_TOKENS:
            continue
        for entry in alias_entries:
            score = difflib.SequenceMatcher(None, candidate, entry["alias"]).ratio()
            if score > best_score:
                second_best = best_score
                best_score = score
                best_key = entry["key"]
                best_alias = entry["alias"]
                best_source = entry["source"]
                best_candidate = candidate
            elif score > second_best:
                second_best = score

    if best_score >= MIN_SIMILARITY and (best_score - second_best) >= MIN_MARGIN:
        return (
            {"alias": best_alias, "key": best_key, "source": best_source},
            best_score,
            best_candidate,
            "fuzzy",
            cleaned,
            normalized,
        )
    best_phonetic = None
    best_phonetic_score = 0.0
    best_phonetic_candidate = None
    best_phonetic_alias = None
    best_phonetic_source = None
    for candidate in candidates:
        if len(candidate.split()) > MAX_FUZZY_TOKENS:
            continue
        candidate_phonetic = phonetic_key(candidate)
        if not candidate_phonetic:
            continue
        for entry in alias_entries:
            score = difflib.SequenceMatcher(
                None, candidate_phonetic, entry["phonetic"]
            ).ratio()
            if score > best_phonetic_score:
                best_phonetic_score = score
                best_phonetic = entry["key"]
                best_phonetic_alias = entry["alias"]
                best_phonetic_source = entry["source"]
                best_phonetic_candidate = candidate

    if best_phonetic and best_phonetic_score >= MIN_PHONETIC_SIMILARITY:
        return (
            {"alias": best_phonetic_alias, "key": best_phonetic, "source": best_phonetic_source},
            best_phonetic_score,
            best_phonetic_candidate,
            "phonetic",
            cleaned,
            normalized,
        )
    return None, None, None, None, cleaned, normalized


class RecognizerWorker:
    def __init__(
        self,
        model: Model,
        grammar: str | None,
        alias_map: dict,
        alias_entries: list,
        send_cb,
        use_grammar: bool,
        logger: DetectionLogger,
    ):
        self.model = model
        self.grammar = grammar
        self.alias_map = alias_map
        self.alias_entries = alias_entries
        self.send_cb = send_cb
        self.use_grammar = use_grammar
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
            if self.use_grammar and self.grammar:
                rec = KaldiRecognizer(self.model, SAMPLE_RATE, self.grammar)
            else:
                rec = KaldiRecognizer(self.model, SAMPLE_RATE)
            rec.SetWords(True)
            return rec

        recognizer = make_recognizer()
        pending_result = None
        sent_result = False
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS) if USE_VAD else None
        frame_samples = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000.0))
        frame_bytes = frame_samples * 2
        pre_buffer = deque(maxlen=VAD_PRE_ROLL_FRAMES)
        speech_active = False
        speech_frames = 0
        silence_frames = 0
        utterance_frames = 0
        partial_key = None
        partial_hits = 0

        def reset_partial():
            nonlocal partial_key, partial_hits
            partial_key = None
            partial_hits = 0

        def send_match(
            entry: dict | None,
            similarity,
            used_text: str | None,
            method: str | None,
            cleaned: str,
            normalized: str,
            text: str,
            conf,
        ) -> bool:
            nonlocal pending_result, sent_result
            if not entry or sent_result:
                return False
            pending_result = {
                "type": "result",
                "key": entry["key"],
                "heard": text,
                "confidence": conf,
                "similarity": similarity,
                "alias": entry["alias"],
            }
            self.send_cb(pending_result)
            sent_result = True
            self._stop_event.set()
            reset_partial()
            return True

        def handle_result(result: dict, reason: str):
            nonlocal pending_result
            text = (result.get("text") or "").strip()
            if not text:
                return
            entry, similarity, used_text, method, cleaned, normalized = match_keyword(
                text,
                self.alias_map,
                self.alias_entries,
                use_fuzzy=True,
            )
            conf = result_confidence(result)
            key = entry["key"] if entry else None
            alias = entry["alias"] if entry else None
            source = entry["source"] if entry else None
            accept = False
            if entry:
                if method == "exact":
                    accept = True
                else:
                    accept = True
                if accept:
                    send_match(entry, similarity, used_text, method, cleaned, normalized, text, conf)
            self.logger.write(
                {
                    "type": "final",
                    "reason": reason,
                    "heard": text,
                    "normalized": normalized,
                    "cleaned": cleaned,
                    "used": used_text,
                    "matched": key,
                    "alias": alias,
                    "source": source,
                    "method": method,
                    "confidence": conf,
                    "similarity": similarity,
                    "grammar": self.use_grammar,
                    "include_asr_aliases": INCLUDE_ASR_ALIASES,
                    "vad": USE_VAD,
                }
            )

        def handle_partial(text: str):
            nonlocal partial_key, partial_hits
            entry, similarity, used_text, method, cleaned, normalized = match_keyword(
                text,
                self.alias_map,
                self.alias_entries,
                use_fuzzy=True,
            )
            if not entry:
                reset_partial()
                return False
            if method != "exact":
                if similarity is None or similarity < PARTIAL_MIN_SIMILARITY:
                    reset_partial()
                    return False
                compact = (used_text or "").replace(" ", "")
                if len(compact) < 4:
                    reset_partial()
                    return False
            key = entry["key"]
            if key == partial_key:
                partial_hits += 1
            else:
                partial_key = key
                partial_hits = 1
            if partial_hits >= PARTIAL_STABLE_COUNT:
                self.logger.write(
                    {
                        "type": "partial_commit",
                        "reason": "partial_commit",
                        "heard": text,
                        "normalized": normalized,
                        "cleaned": cleaned,
                        "used": used_text,
                        "matched": entry["key"],
                        "alias": entry["alias"],
                        "source": entry["source"],
                        "method": method,
                        "confidence": None,
                        "similarity": similarity,
                        "grammar": self.use_grammar,
                        "include_asr_aliases": INCLUDE_ASR_ALIASES,
                        "vad": USE_VAD,
                    }
                )
                return send_match(entry, similarity, used_text, method, cleaned, normalized, text, None)
            return False

        def callback(indata, frames, time_info, status):
            if status:
                self.send_cb({"type": "status", "status": "audio_warning", "detail": str(status)})
            if self._stop_event.is_set():
                return
            self._audio_queue.put(bytes(indata))

        try:
            blocksize = frame_samples if USE_VAD else 8000
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

                    if USE_VAD:
                        for offset in range(0, len(data), frame_bytes):
                            frame = data[offset : offset + frame_bytes]
                            if len(frame) < frame_bytes:
                                continue
                            raw_rms = audioop.rms(frame, 2)
                            boosted = apply_gain(frame)
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
                                    reset_partial()
                                    recognizer = make_recognizer()
                                    for buffered in pre_buffer:
                                        recognizer.AcceptWaveform(buffered)
                                    pre_buffer.clear()
                            else:
                                recognizer.AcceptWaveform(boosted)
                                utterance_frames += 1
                                if PARTIAL_COMMIT and utterance_frames % PARTIAL_INTERVAL_FRAMES == 0:
                                    partial_text = json.loads(recognizer.PartialResult()).get(
                                        "partial", ""
                                    )
                                    if partial_text and handle_partial(partial_text):
                                        break
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
                                        reset_partial()
                                        pre_buffer.clear()
                                        recognizer = make_recognizer()
                                if speech_active and utterance_frames >= VAD_MAX_FRAMES:
                                    result = json.loads(recognizer.FinalResult())
                                    handle_result(result, "vad_max")
                                    speech_active = False
                                    speech_frames = 0
                                    silence_frames = 0
                                    utterance_frames = 0
                                    reset_partial()
                                    pre_buffer.clear()
                                    recognizer = make_recognizer()
                    else:
                        if recognizer.AcceptWaveform(data):
                            result = json.loads(recognizer.Result())
                            handle_result(result, "asr_final")
                        else:
                            partial = json.loads(recognizer.PartialResult()).get("partial", "")
                            if partial:
                                if PARTIAL_COMMIT and handle_partial(partial):
                                    continue
                                self.send_cb({"type": "partial", "partial": partial})
        except Exception as exc:
            self.send_cb({"type": "error", "message": str(exc)})
        finally:
            if USE_VAD and speech_active:
                result = json.loads(recognizer.FinalResult())
                handle_result(result, "stop_flush")
            if pending_result and not sent_result:
                self.send_cb(pending_result)
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

    alias_map, alias_entries, grammar, collisions = load_keywords(
        USE_GRAMMAR, INCLUDE_ASR_ALIASES
    )
    model = Model(MODEL_PATH)
    logger = DetectionLogger(LOG_PATH)
    for alias, kept, ignored in collisions:
        logger.write({"type": "collision", "alias": alias, "kept": kept, "ignored": ignored})

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

    recognizer = RecognizerWorker(
        model, grammar, alias_map, alias_entries, send_cb, USE_GRAMMAR, logger
    )

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
    print(f"Grammar mode: {'on' if USE_GRAMMAR else 'off'}")
    print(f"ASR aliases: {'on' if INCLUDE_ASR_ALIASES else 'off'}")
    print(f"Detection log: {LOG_PATH}")

    async with websockets.serve(handler, "", WS_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
