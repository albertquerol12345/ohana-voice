import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from backend.config import Config
from backend.decision import DecisionEngine
from backend.matcher import Matcher
from backend.whisper_engine import WhisperManager


def load_audio(path: str, target_sr: int) -> np.ndarray:
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if sr != target_sr:
        data = resample_poly(data, target_sr, sr)
    return data.astype(np.float32)


def iter_windows(audio: np.ndarray, window_samples: int, hop_samples: int):
    if audio.size < window_samples:
        pad = window_samples - audio.size
        audio = np.pad(audio, (pad, 0), mode="constant")
    for start in range(0, max(1, audio.size - window_samples + 1), hop_samples):
        yield audio[start : start + window_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Root folder with subfolders per burger")
    parser.add_argument("--output", default="logs/eval_summary.json")
    parser.add_argument("--mode", choices=["window", "clip"], default="window")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_env()
    if args.device:
        cfg.device = args.device
    if args.compute_type:
        cfg.compute_type = args.compute_type
    if args.no_fallback:
        cfg.enable_fallback = False

    keywords_path = cfg.keywords_path or os.path.join(os.getcwd(), "backend", "keywords.json")
    matcher = Matcher(
        keywords_path,
        observed_aliases_path=cfg.observed_aliases_path
        or os.path.join(os.getcwd(), "backend", "keywords_observed.json"),
        include_asr=cfg.include_asr_aliases,
        include_observed=cfg.include_observed_aliases,
    )
    hotwords = cfg.hotwords
    if cfg.hotwords_from_keywords and not hotwords:
        try:
            with open(keywords_path, "r", encoding="utf-8") as f:
                keys = list(json.load(f).keys())
            hotwords = ", ".join(keys)
        except Exception:
            hotwords = ""
    whisper = WhisperManager(
        primary_name=cfg.model_primary,
        fallback_name=cfg.model_fallback if cfg.enable_fallback else None,
        device=cfg.device,
        compute_type=cfg.compute_type,
        beam_size=cfg.beam_size,
        temperature=cfg.temperature,
        hotwords=hotwords,
    )

    window_samples = int(cfg.window_sec * cfg.sample_rate)
    hop_samples = int(cfg.hop_sec * cfg.sample_rate)

    totals = Counter()
    per_class = defaultdict(Counter)
    confusion = Counter()

    for root, _, files in os.walk(args.dataset):
        for name in files:
            if not name.lower().endswith(".wav"):
                continue
            label = os.path.basename(root)
            path = os.path.join(root, name)
            audio = load_audio(path, cfg.sample_rate)
            decision = DecisionEngine(
                thresh_best=cfg.thresh_best,
                thresh_margin=cfg.thresh_margin,
                margin_relax_score=cfg.margin_relax_score,
                margin_relax=cfg.margin_relax,
                margin_strict_score=cfg.margin_strict_score,
                margin_strict=cfg.margin_strict,
                hysteresis_k=cfg.hysteresis_k,
                hysteresis_window_sec=cfg.hysteresis_window_sec,
                cooldown_sec=cfg.cooldown_sec,
                max_events_per_min=cfg.max_events_per_min,
                mute_sec=cfg.mute_sec,
                same_key_block_sec=cfg.same_key_block_sec,
                related_key_groups=cfg.related_key_groups,
            )

            predicted = None
            best_score_seen = 0.0
            best_key_seen = None
            second_seen = 0.0

            if args.mode == "clip":
                primary = whisper.transcribe_primary(audio, cfg.language_primary)
                candidates = matcher.match(
                    primary.text,
                    max_tokens=cfg.max_tokens,
                    ngram_max_tokens=cfg.ngram_max_tokens,
                    topn=5,
                )
                if candidates:
                    best_score_seen = candidates[0].score
                    best_key_seen = candidates[0].key
                    second_seen = candidates[1].score if len(candidates) > 1 else 0.0
                predicted = best_key_seen
            else:
                now = 0.0
                for window in iter_windows(audio, window_samples, hop_samples):
                    primary = whisper.transcribe_primary(window, cfg.language_primary)
                    candidates = matcher.match(
                        primary.text,
                        max_tokens=cfg.max_tokens,
                        ngram_max_tokens=cfg.ngram_max_tokens,
                        topn=5,
                    )
                    if candidates:
                        best_score_seen = max(best_score_seen, candidates[0].score)
                        if candidates[0].score > best_score_seen:
                            best_key_seen = candidates[0].key
                        second_seen = candidates[1].score if len(candidates) > 1 else 0.0
                    best = candidates[0] if candidates else None
                    second = candidates[1] if len(candidates) > 1 else None
                    margin = (best.score - second.score) if best and second else (best.score if best else 0)
                    result = decision.decide(best.key if best else None, best.score if best else 0.0, margin, now)
                    if result.accepted and best:
                        predicted = best.key
                        break
                    now += cfg.hop_sec

            totals["total"] += 1
            per_class[label]["total"] += 1
            if predicted is None:
                totals["no_detect"] += 1
                per_class[label]["no_detect"] += 1
                confusion[(label, "none")] += 1
            elif predicted == label:
                totals["correct"] += 1
                per_class[label]["correct"] += 1
                confusion[(label, predicted)] += 1
            else:
                totals["wrong"] += 1
                per_class[label]["wrong"] += 1
                confusion[(label, predicted)] += 1

    summary = {
        "totals": totals,
        "per_class": per_class,
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
        "config": {
            "model_primary": cfg.model_primary,
            "model_fallback": cfg.model_fallback if cfg.enable_fallback else None,
            "window_sec": cfg.window_sec,
            "hop_sec": cfg.hop_sec,
            "thresh_best": cfg.thresh_best,
            "thresh_margin": cfg.thresh_margin,
            "hysteresis_k": cfg.hysteresis_k,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(summary["totals"])
    print(f"Summary: {args.output}")


if __name__ == "__main__":
    main()
