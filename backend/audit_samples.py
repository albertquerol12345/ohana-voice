import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def load_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.getnframes()
        data = wf.readframes(frames)
    if channels != 1:
        raise ValueError(f"{path} not mono (channels={channels})")
    samples = np.frombuffer(data, dtype=np.int16)
    return rate, samples


def frame_rms(samples: np.ndarray, frame_size: int):
    if samples.size < frame_size:
        return np.array([], dtype=np.float32)
    trim = samples[: samples.size - (samples.size % frame_size)]
    frames = trim.reshape(-1, frame_size).astype(np.float32)
    return np.sqrt(np.mean(frames * frames, axis=1))


def analyze_file(path: Path):
    rate, samples = load_wav(path)
    duration = samples.size / float(rate)
    peak = int(np.max(np.abs(samples))) if samples.size else 0
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0
    clip_ratio = float(np.mean(np.abs(samples) >= 30000)) if samples.size else 0.0
    silence_ratio = float(np.mean(np.abs(samples) <= 300)) if samples.size else 1.0

    frame_size = int(rate * 0.02)
    frms = frame_rms(samples, frame_size)
    voice_ratio = float(np.mean(frms > 600)) if frms.size else 0.0

    return {
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "clip_ratio": clip_ratio,
        "silence_ratio": silence_ratio,
        "voice_ratio": voice_ratio,
        "rate": rate,
    }


def robust_bounds(values: list[float], mad_scale: float = 2.5):
    if not values:
        return (0.0, 0.0)
    med = float(np.median(values))
    mad = float(np.median(np.abs(np.array(values) - med)))
    if mad == 0:
        return (med * 0.5, med * 1.8)
    lower = med - mad_scale * mad
    upper = med + mad_scale * mad
    return (max(lower, 0.0), max(upper, 0.0))


def main():
    parser = argparse.ArgumentParser(description="Audit DTW samples.")
    parser.add_argument("--samples", default="/home/albert/ohana_voice_mvp/dtw_samples")
    parser.add_argument("--out", default="/home/albert/ohana_voice_mvp/dtw_audit.json")
    args = parser.parse_args()

    sample_dir = Path(args.samples)
    if not sample_dir.exists():
        raise SystemExit(f"Missing samples dir: {sample_dir}")

    report = {}
    overall_issues = []

    for key_dir in sorted(sample_dir.iterdir()):
        if not key_dir.is_dir():
            continue
        key = key_dir.name
        files = sorted(key_dir.glob("*.wav"))
        metrics = {}
        durations = []
        rmss = []
        silences = []
        voices = []
        for wav_path in files:
            try:
                info = analyze_file(wav_path)
            except Exception as exc:
                overall_issues.append({"file": str(wav_path), "issue": str(exc)})
                continue
            metrics[str(wav_path)] = info
            durations.append(info["duration"])
            rmss.append(info["rms"])
            silences.append(info["silence_ratio"])
            voices.append(info["voice_ratio"])

        d_low, d_high = robust_bounds(durations, mad_scale=2.5)
        r_low, r_high = robust_bounds(rmss, mad_scale=2.8)
        s_low, s_high = robust_bounds(silences, mad_scale=3.0)
        v_low, v_high = robust_bounds(voices, mad_scale=3.0)
        flagged = []
        for wav_path, info in metrics.items():
            reasons = []
            if info["rate"] != SAMPLE_RATE:
                reasons.append("bad_sample_rate")
            if info["duration"] < 0.25 or info["duration"] > 3.0:
                reasons.append("duration_out_of_range")
            if info["duration"] < d_low or info["duration"] > d_high:
                reasons.append("duration_outlier")
            if info["rms"] < r_low or info["rms"] > r_high * 1.6:
                reasons.append("rms_outlier")
            if info["clip_ratio"] > 0.002:
                reasons.append("clipping")
            if info["silence_ratio"] > max(0.85, min(0.98, s_high)):
                reasons.append("too_silent")
            if info["voice_ratio"] < min(0.15, max(0.02, v_low)):
                reasons.append("low_voice_ratio")
            if reasons:
                flagged.append({"file": wav_path, "reasons": reasons, **info})

        report[key] = {
            "count": len(files),
            "duration_median": float(np.median(durations)) if durations else 0.0,
            "rms_median": float(np.median(rmss)) if rmss else 0.0,
            "silence_median": float(np.median(silences)) if silences else 0.0,
            "voice_median": float(np.median(voices)) if voices else 0.0,
            "flagged": flagged,
            "metrics": metrics,
        }

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"report": report, "issues": overall_issues}, f, ensure_ascii=False, indent=2)
    print(f"Audit saved to {out_path}")

    total_flagged = sum(len(item["flagged"]) for item in report.values())
    print(f"Flagged samples: {total_flagged}")

    retake_path = out_path.with_name("dtw_retake.txt")
    plan_path = out_path.with_name("dtw_retake_plan.json")
    with retake_path.open("w", encoding="utf-8") as f:
        for key, info in report.items():
            for item in info["flagged"]:
                f.write(f"{item['file']}\t{','.join(item['reasons'])}\n")
    print(f"Retake list: {retake_path}")

    plan = {key: len(info["flagged"]) for key, info in report.items() if info["flagged"]}
    with plan_path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(f"Retake plan: {plan_path}")


if __name__ == "__main__":
    main()
