import argparse
import json
import os
import queue
import wave
from pathlib import Path

import sounddevice as sd
import webrtcvad

from dtw_utils import SAMPLE_RATE

ROOT_DIR = Path(__file__).resolve().parents[1]
KEYWORDS_PATH = Path(__file__).with_name("keywords.json")


def load_keywords():
    with KEYWORDS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    keys = []
    prompts = {}
    for key, aliases in data.items():
        keys.append(key)
        prompt = key
        if isinstance(aliases, dict):
            phonetic = aliases.get("phonetic") or []
            if phonetic:
                prompt = phonetic[0]
        elif aliases:
            prompt = aliases[0]
        prompts[key] = prompt
    return keys, prompts


def write_wav(path: Path, pcm_bytes: bytes):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def record_once(
    vad: webrtcvad.Vad,
    frame_ms: int,
    start_frames: int,
    end_frames: int,
    pre_roll: int,
    max_seconds: float,
):
    frame_samples = int(SAMPLE_RATE * (frame_ms / 1000.0))
    frame_bytes = frame_samples * 2
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            return
        q.put(bytes(indata))

    max_frames = int(max_seconds * 1000 / frame_ms)
    pre_buffer = []
    speech_active = False
    speech_frames = 0
    silence_frames = 0
    recorded = []

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=frame_samples,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while True:
            data = q.get()
            for offset in range(0, len(data), frame_bytes):
                frame = data[offset : offset + frame_bytes]
                if len(frame) < frame_bytes:
                    continue
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                if not speech_active:
                    pre_buffer.append(frame)
                    if len(pre_buffer) > pre_roll:
                        pre_buffer.pop(0)
                    if is_speech:
                        speech_frames += 1
                    else:
                        speech_frames = 0
                    if speech_frames >= start_frames:
                        speech_active = True
                        silence_frames = 0
                        recorded.extend(pre_buffer)
                        pre_buffer = []
                else:
                    recorded.append(frame)
                    if is_speech:
                        silence_frames = 0
                    else:
                        silence_frames += 1
                        if silence_frames >= end_frames:
                            return b"".join(recorded)
                    if len(recorded) >= max_frames:
                        return b"".join(recorded)


def parse_keys(value: str | None, all_keys: list[str]) -> list[str]:
    if not value:
        return all_keys
    wanted = [item.strip() for item in value.split(",") if item.strip()]
    selected = []
    for key in all_keys:
        if key in wanted:
            selected.append(key)
    missing = [item for item in wanted if item not in all_keys]
    if missing:
        raise SystemExit(f"Unknown keys: {', '.join(missing)}")
    return selected


def next_index_for_key(key_dir: Path, key: str) -> int:
    prefix = f"{key}_"
    max_idx = 0
    for wav_path in key_dir.glob("*.wav"):
        stem = wav_path.stem
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix) :]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx + 1


def main():
    parser = argparse.ArgumentParser(description="Record keyword samples for DTW.")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "dtw_samples"))
    parser.add_argument("--keys", default=None, help="Comma-separated list of keys")
    parser.add_argument("--plan", default=None, help="JSON plan with per-key counts")
    parser.add_argument("--max-seconds", type=float, default=3.0)
    parser.add_argument("--vad", type=int, default=1)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--start-frames", type=int, default=1)
    parser.add_argument("--end-frames", type=int, default=8)
    parser.add_argument("--pre-roll", type=int, default=5)
    args = parser.parse_args()

    keys, prompts = load_keywords()
    plan = None
    if args.plan:
        plan_path = Path(args.plan)
        if not plan_path.exists():
            raise SystemExit(f"Plan not found: {plan_path}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        selected_keys = [k for k in keys if k in plan]
    else:
        selected_keys = parse_keys(args.keys, keys)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vad = webrtcvad.Vad(args.vad)

    for key in selected_keys:
        prompt = prompts[key]
        key_dir = out_dir / key
        key_dir.mkdir(parents=True, exist_ok=True)
        repeats = plan.get(key, args.repeats) if plan else args.repeats
        print(f"\n== {key} ==")
        print(f"Di: {prompt}")
        idx = 0
        start_idx = next_index_for_key(key_dir, key)
        while idx < repeats:
            input(f"[{idx + 1}/{repeats}] Pulsa Enter y habla...")
            audio = record_once(
                vad,
                args.frame_ms,
                args.start_frames,
                args.end_frames,
                args.pre_roll,
                args.max_seconds,
            )
            if not audio or len(audio) < SAMPLE_RATE * 2 * 0.2:
                print("Muy corto, repite.")
                continue
            filename = f"{key}_{start_idx + idx:03d}.wav"
            path = key_dir / filename
            write_wav(path, audio)
            print(f"Guardado: {path}")
            idx += 1


if __name__ == "__main__":
    main()
