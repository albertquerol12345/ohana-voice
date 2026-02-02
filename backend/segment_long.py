import argparse
import json
import shutil
import wave
from collections import deque
from pathlib import Path

import webrtcvad

SAMPLE_RATE = 16000


def read_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.getnframes()
        data = wf.readframes(frames)
    if channels != 1:
        raise ValueError(f"{path} not mono (channels={channels})")
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path} sample rate {rate} != {SAMPLE_RATE}")
    return data


def write_wav(path: Path, data: bytes):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data)


def segment_audio(
    data: bytes,
    vad: webrtcvad.Vad,
    frame_ms: int,
    start_frames: int,
    end_frames: int,
    pre_roll: int,
    max_frames: int,
    force_frames: int | None = None,
):
    frame_samples = int(SAMPLE_RATE * (frame_ms / 1000.0))
    frame_bytes = frame_samples * 2
    pre_buffer = deque(maxlen=pre_roll)
    segments = []
    current = []
    speech_active = False
    speech_frames = 0
    silence_frames = 0
    utterance_frames = 0

    for offset in range(0, len(data), frame_bytes):
        frame = data[offset : offset + frame_bytes]
        if len(frame) < frame_bytes:
            continue
        is_speech = vad.is_speech(frame, SAMPLE_RATE)
        if not speech_active:
            pre_buffer.append(frame)
            if is_speech:
                speech_frames += 1
            else:
                speech_frames = 0
            if speech_frames >= start_frames:
                speech_active = True
                silence_frames = 0
                utterance_frames = 0
                current = list(pre_buffer)
                pre_buffer.clear()
        else:
            current.append(frame)
            utterance_frames += 1
            if is_speech:
                silence_frames = 0
            else:
                silence_frames += 1
                if silence_frames >= end_frames:
                    segments.append(b"".join(current))
                    current = []
                    speech_active = False
                    speech_frames = 0
                    silence_frames = 0
                    utterance_frames = 0
                    pre_buffer.clear()
                    continue
            if force_frames and utterance_frames >= force_frames:
                segments.append(b"".join(current))
                current = []
                speech_active = False
                speech_frames = 0
                silence_frames = 0
                utterance_frames = 0
                pre_buffer.clear()
            elif utterance_frames >= max_frames:
                segments.append(b"".join(current))
                current = []
                speech_active = False
                speech_frames = 0
                silence_frames = 0
                utterance_frames = 0
                pre_buffer.clear()

    if current:
        segments.append(b"".join(current))
    return segments


def seconds_from_bytes(data: bytes):
    return len(data) / (SAMPLE_RATE * 2)


def main():
    parser = argparse.ArgumentParser(description="Segment long recordings into samples.")
    parser.add_argument("--in-dir", default="/home/albert/ohana_voice_mvp/dtw_long")
    parser.add_argument("--out-dir", default="/home/albert/ohana_voice_mvp/dtw_samples_long")
    parser.add_argument("--clean", type=int, default=1)
    parser.add_argument("--vad", type=int, default=1)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--start-frames", type=int, default=1)
    parser.add_argument("--end-frames", type=int, default=8)
    parser.add_argument("--pre-roll", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=2.8)
    parser.add_argument("--force-seconds", type=float, default=0.0)
    parser.add_argument("--min-seconds", type=float, default=0.25)
    parser.add_argument("--expected", type=int, default=25)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.exists():
        raise SystemExit(f"Missing input dir: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    vad = webrtcvad.Vad(args.vad)
    max_frames = int((args.max_seconds * 1000) / args.frame_ms)
    force_frames = (
        int((args.force_seconds * 1000) / args.frame_ms) if args.force_seconds > 0 else None
    )
    summary = {}

    for wav_path in sorted(in_dir.glob("*.wav")):
        key = wav_path.stem
        data = read_wav(wav_path)
        segments = segment_audio(
            data,
            vad,
            args.frame_ms,
            args.start_frames,
            args.end_frames,
            args.pre_roll,
            max_frames,
            force_frames,
        )
        key_dir = out_dir / key
        if args.clean and key_dir.exists():
            shutil.rmtree(key_dir)
        key_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        skipped = 0
        too_long = 0
        for idx, segment in enumerate(segments, start=1):
            duration = seconds_from_bytes(segment)
            if duration < args.min_seconds:
                skipped += 1
                continue
            if duration > args.max_seconds:
                too_long += 1
            out_path = key_dir / f"{key}_{idx:03d}.wav"
            write_wav(out_path, segment)
            kept += 1

        summary[key] = {
            "segments_total": len(segments),
            "kept": kept,
            "skipped_short": skipped,
            "too_long": too_long,
            "expected": args.expected,
        }
        print(f"{key}: kept={kept} segments={len(segments)}")

    summary_path = out_dir / "segment_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
