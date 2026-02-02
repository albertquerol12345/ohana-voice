import argparse
import json
import os
import re

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from backend.audio_stream import normalize_audio
from backend.config import Config
from backend.decision import DecisionEngine
from backend.matcher import Matcher, normalize_text
from backend.whisper_engine import WhisperManager


def load_audio(path: str, target_sr: int) -> np.ndarray:
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if sr != target_sr:
        data = resample_poly(data, target_sr, sr)
    return data.astype(np.float32)


def should_fallback(cfg: Config, best_score: float, margin: float, backlog: int) -> bool:
    if not cfg.enable_fallback:
        return False
    if backlog > cfg.max_backlog_for_fallback:
        return False
    band = cfg.fallback_score_band
    if (cfg.thresh_best - band) <= best_score <= (cfg.thresh_best + band):
        return True
    if cfg.fallback_margin_low <= margin <= cfg.fallback_margin_high:
        return True
    return False


def extract_best(candidates):
    best = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None
    best_score = best.score if best else 0.0
    second_score = second.score if second else 0.0
    margin = best_score - second_score
    return best, second, best_score, margin


def short_alias_override(cfg: Config, decision: DecisionEngine, best, best_score: float, margin: float):
    if not best:
        return None
    alias_len = len(normalize_text(best.alias))
    if alias_len > cfg.short_alias_max_len:
        return None
    if best_score < max(cfg.thresh_best, cfg.short_alias_score):
        return None
    min_margin = decision._margin_for_score(best_score)
    if margin < min_margin:
        return None
    return 1


def high_confidence_override(cfg: Config, best_score: float, margin: float):
    if best_score < cfg.high_confidence_score:
        return None
    if margin < cfg.high_confidence_margin:
        return None
    return 1


def mid_confidence_override(cfg: Config, best, best_score: float, margin: float):
    if not best:
        return None
    alias_len = len(normalize_text(best.alias))
    if alias_len < cfg.mid_confidence_min_len:
        return None
    if best_score < cfg.mid_confidence_score:
        return None
    if margin < cfg.mid_confidence_margin:
        return None
    return 1


def key_hysteresis_override(cfg: Config, best):
    if not best:
        return None
    keys = {k.strip() for k in cfg.hysteresis_override_keys.split(",") if k.strip()}
    if best.key in keys:
        return 1
    return None


def pick_segment_candidates(cfg: Config, matcher: Matcher, text: str):
    normalized = normalize_text(text)
    segments = [normalized]
    if cfg.split_on_and:
        segments = [
            seg.strip()
            for seg in re.split(
                r"[.,;]|\\b(?:y|e|una|un|otra|otro|ponme|pon|ponga|poneme|ponmela|ponle|dame)\\b",
                normalized,
            )
            if seg.strip()
        ]
    chosen = None
    chosen_idx = 0
    for idx, segment in enumerate(segments):
        candidates = matcher.match(
            segment,
            max_tokens=cfg.max_tokens,
            ngram_max_tokens=cfg.ngram_max_tokens,
            topn=max(cfg.debug_topn, 2),
        )
        if not candidates:
            continue
        if chosen is None:
            chosen = (segment, candidates)
            chosen_idx = idx
            continue
        if candidates[0].score > chosen[1][0].score + 1e-6:
            chosen = (segment, candidates)
            chosen_idx = idx
        elif abs(candidates[0].score - chosen[1][0].score) <= 1e-6 and idx < chosen_idx:
            chosen = (segment, candidates)
            chosen_idx = idx
    if chosen is None:
        candidates = matcher.match(
            normalized,
            max_tokens=cfg.max_tokens,
            ngram_max_tokens=cfg.ngram_max_tokens,
            topn=max(cfg.debug_topn, 2),
        )
        return normalized, candidates, segments
    return chosen[0], chosen[1], segments


def collect_segment_candidates(cfg: Config, matcher: Matcher, segments: list[str]):
    entries = []
    for segment in segments:
        candidates = matcher.match(
            segment,
            max_tokens=cfg.max_tokens,
            ngram_max_tokens=cfg.ngram_max_tokens,
            topn=max(cfg.debug_topn, 2),
        )
        best, second, best_score, margin = extract_best(candidates)
        entries.append(
            {
                "segment": segment,
                "candidates": candidates,
                "best": best,
                "second": second,
                "best_score": best_score,
                "margin": margin,
            }
        )
    return entries


def passes_thresholds(cfg: Config, decision: DecisionEngine, best, best_score, margin) -> bool:
    if not best:
        return False
    if best_score < cfg.thresh_best:
        return False
    min_margin = decision._margin_for_score(best_score)
    if margin < min_margin:
        return False
    return True


def apply_margin_override(cfg: Config, decision: DecisionEngine, best, best_score, margin) -> float:
    if not best:
        return margin
    keys = {k.strip() for k in cfg.margin_override_keys.split(",") if k.strip()}
    if best.key in keys and best_score >= cfg.thresh_best:
        min_margin = decision._margin_for_score(best_score)
        return max(margin, min_margin)
    return margin


def main():
    parser = argparse.ArgumentParser(description="Replay a session wav through the detector pipeline")
    parser.add_argument("audio", help="Path to session wav")
    parser.add_argument("--output", default=None, help="JSONL output path")
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

    audio = load_audio(args.audio, cfg.sample_rate)
    window_samples = int(cfg.window_sec * cfg.sample_rate)
    hop_samples = int(cfg.hop_sec * cfg.sample_rate)

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

    base = os.path.splitext(os.path.basename(args.audio))[0]
    output = args.output or os.path.join(
        os.path.dirname(args.audio), f"{base}_replay.jsonl"
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)

    noise_floor = cfg.gate_noise_floor_init
    gate_until = 0.0
    events = []

    with open(output, "w", encoding="utf-8") as f:
        for idx, start in enumerate(range(0, len(audio) - window_samples + 1, hop_samples)):
            window = audio[start : start + window_samples]
            now = start / cfg.sample_rate
            rms = float(np.sqrt(np.mean(window ** 2)))
            peak = float(np.max(np.abs(window))) if window.size else 0.0
            threshold = max(noise_floor * cfg.gate_factor, cfg.gate_min_rms)

            gate_active = True
            if cfg.use_gate:
                if rms >= threshold:
                    gate_until = now + cfg.gate_hangover_sec
                    gate_active = True
                elif now < gate_until:
                    gate_active = True
                else:
                    gate_active = False

                if not gate_active:
                    noise_floor = (1.0 - cfg.gate_noise_alpha) * noise_floor + (
                        cfg.gate_noise_alpha * rms
                    )
                    noise_floor = max(noise_floor, cfg.gate_min_rms)

            payload = {
                "type": "window",
                "window_id": idx,
                "t": round(now, 3),
                "rms": rms,
                "peak": peak,
                "noise_floor": noise_floor,
                "threshold": threshold,
                "gate": gate_active,
            }

            if not gate_active:
                payload["decision"] = "gate_off"
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                continue

            window_for_model = (
                normalize_audio(window, cfg.target_rms, cfg.max_gain)
                if cfg.normalize_audio
                else window
            )
            primary = whisper.transcribe_primary(window_for_model, cfg.language_primary)
            segment_text, candidates, segments = pick_segment_candidates(cfg, matcher, primary.text)
            best, second, best_score, margin = extract_best(candidates)
            margin = apply_margin_override(cfg, decision, best, best_score, margin)
            used_fallback = False
            fallback_text = None
            if should_fallback(cfg, best_score, margin, backlog=0):
                fallback_lang = cfg.language_fallback or cfg.language_primary
                fallback = whisper.transcribe_fallback(window_for_model, fallback_lang)
                segment_text, fallback_candidates, segments = pick_segment_candidates(
                    cfg, matcher, fallback.text
                )
                fb_best, fb_second, fb_score, fb_margin = extract_best(fallback_candidates)
                prefer_keys = {k.strip() for k in cfg.fallback_prefer_keys.split(",") if k.strip()}
                prefer_fallback = False
                if fb_best and fb_best.key in prefer_keys:
                    if fb_score >= best_score - cfg.fallback_prefer_delta:
                        prefer_fallback = True
                if prefer_fallback or fb_score > best_score or fb_margin > margin:
                    used_fallback = True
                    best, second, best_score, margin = fb_best, fb_second, fb_score, fb_margin
                    candidates = fallback_candidates
                    fallback_text = fallback.text
                    margin = apply_margin_override(cfg, decision, best, best_score, margin)

            list_entries = []
            list_mode = cfg.list_mode and len(segments) >= cfg.list_min_segments
            if list_mode:
                list_entries = collect_segment_candidates(cfg, matcher, segments)

            events = []
            if list_mode:
                for entry in list_entries:
                    best_item = entry["best"]
                    best_score_item = entry["best_score"]
                    margin_item = apply_margin_override(
                        cfg, decision, best_item, best_score_item, entry["margin"]
                    )
                    if not passes_thresholds(cfg, decision, best_item, best_score_item, margin_item):
                        continue
                    hysteresis_override = key_hysteresis_override(cfg, best_item)
                    if hysteresis_override is None:
                        hysteresis_override = short_alias_override(
                            cfg, decision, best_item, best_score_item, margin_item
                        )
                    if hysteresis_override is None:
                        hysteresis_override = high_confidence_override(cfg, best_score_item, margin_item)
                    if hysteresis_override is None:
                        hysteresis_override = mid_confidence_override(
                            cfg, best_item, best_score_item, margin_item
                        )
                    result = decision.decide(
                        best_item.key if best_item else None,
                        best_score_item,
                        margin_item,
                        now,
                        hysteresis_override=hysteresis_override or 1,
                        bypass_cooldown=True,
                        bypass_repeat=True,
                        skip_cooldown_set=True,
                    )
                    if result.accepted and best_item:
                        events.append(
                            {
                                "key": best_item.key,
                                "confidence": best_score_item,
                                "margin": margin_item,
                            }
                        )
                if events:
                    decision.bump_cooldown(now, cfg.list_cooldown_sec)
            if not events:
                hysteresis_override = key_hysteresis_override(cfg, best)
                if hysteresis_override is None:
                    hysteresis_override = short_alias_override(cfg, decision, best, best_score, margin)
                if hysteresis_override is None:
                    hysteresis_override = high_confidence_override(cfg, best_score, margin)
                if hysteresis_override is None:
                    hysteresis_override = mid_confidence_override(cfg, best, best_score, margin)
                result = decision.decide(
                    best.key if best else None,
                    best_score,
                    margin,
                    now,
                    hysteresis_override=hysteresis_override,
                )

            payload.update(
                {
                    "raw_text": primary.text,
                    "segment_text": segment_text,
                    "norm_text": normalize_text(primary.text),
                    "best_key": best.key if best else None,
                    "best_score": best_score,
                    "best_alias": best.alias if best else None,
                    "best_ngram": best.ngram if best else None,
                    "second_key": second.key if second else None,
                    "second_score": second.score if second else 0.0,
                    "margin": margin,
                    "decision": "list" if events else result.reason,
                    "used_fallback": used_fallback,
                }
            )
            if fallback_text is not None:
                payload["fallback_text"] = fallback_text
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if events:
                for ev in events:
                    event = {
                        "type": "event",
                        "t": round(now, 3),
                        "key": ev["key"],
                        "confidence": ev["confidence"],
                        "margin": ev["margin"],
                    }
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            elif result.accepted and best:
                event = {
                    "type": "event",
                    "t": round(now, 3),
                    "key": best.key,
                    "confidence": best_score,
                    "margin": margin,
                }
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(f"Replay log: {output}")
    print(f"Events: {len(events)}")
    for event in events:
        print(f"{event['t']:.2f}s {event['key']} ({event['confidence']:.2f})")


if __name__ == "__main__":
    main()
