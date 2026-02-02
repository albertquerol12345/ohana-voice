import queue
import re
import threading
import time

from backend.matcher import normalize_text


class Processor(threading.Thread):
    def __init__(
        self,
        window_queue: queue.Queue,
        whisper,
        matcher,
        decision,
        metrics,
        logger,
        broadcaster,
        config,
    ):
        super().__init__(daemon=True)
        self.window_queue = window_queue
        self.whisper = whisper
        self.matcher = matcher
        self.decision = decision
        self.metrics = metrics
        self.logger = logger
        self.broadcaster = broadcaster
        self.config = config
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _should_fallback(self, best_score: float, margin: float, backlog: int) -> bool:
        if not self.config.enable_fallback:
            return False
        if backlog > self.config.max_backlog_for_fallback:
            return False
        score_band = self.config.fallback_score_band
        if (self.config.thresh_best - score_band) <= best_score <= (
            self.config.thresh_best + score_band
        ):
            return True
        if self.config.fallback_margin_low <= margin <= self.config.fallback_margin_high:
            return True
        return False

    def _extract_best(self, candidates):
        best = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        best_score = best.score if best else 0.0
        second_score = second.score if second else 0.0
        margin = best_score - second_score
        return best, second, best_score, margin

    def _short_alias_override(self, best, best_score, margin) -> int | None:
        if not best:
            return None
        alias_len = len(normalize_text(best.alias))
        if alias_len > self.config.short_alias_max_len:
            return None
        if best_score < max(self.config.thresh_best, self.config.short_alias_score):
            return None
        min_margin = self.decision._margin_for_score(best_score)
        if margin < min_margin:
            return None
        return 1

    def _high_confidence_override(self, best_score, margin) -> int | None:
        if best_score < self.config.high_confidence_score:
            return None
        if margin < self.config.high_confidence_margin:
            return None
        return 1

    def _key_hysteresis_override(self, best) -> int | None:
        if not best:
            return None
        keys = {k.strip() for k in self.config.hysteresis_override_keys.split(",") if k.strip()}
        if best.key in keys:
            return 1
        return None

    def _mid_confidence_override(self, best, best_score, margin) -> int | None:
        if not best:
            return None
        alias_len = len(normalize_text(best.alias))
        if alias_len < self.config.mid_confidence_min_len:
            return None
        if best_score < self.config.mid_confidence_score:
            return None
        if margin < self.config.mid_confidence_margin:
            return None
        return 1

    def _pick_segment_candidates(self, text: str):
        normalized = normalize_text(text)
        segments = [normalized]
        if self.config.split_on_and:
            segments = [
                seg.strip()
                for seg in re.split(r"[.,;]|\\b(?:y|e|una|un|otra|otro|ponme|pon|ponga|poneme|ponmela|ponle|dame)\\b", normalized)
                if seg.strip()
            ]
        chosen = None
        chosen_idx = 0
        for idx, segment in enumerate(segments):
            candidates = self.matcher.match(
                segment,
                max_tokens=self.config.max_tokens,
                ngram_max_tokens=self.config.ngram_max_tokens,
                topn=max(self.config.debug_topn, 2),
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
            candidates = self.matcher.match(
                normalized,
                max_tokens=self.config.max_tokens,
                ngram_max_tokens=self.config.ngram_max_tokens,
                topn=max(self.config.debug_topn, 2),
            )
            return normalized, candidates, segments
        return chosen[0], chosen[1], segments

    def _collect_segment_candidates(self, segments: list[str]):
        entries = []
        for segment in segments:
            candidates = self.matcher.match(
                segment,
                max_tokens=self.config.max_tokens,
                ngram_max_tokens=self.config.ngram_max_tokens,
                topn=max(self.config.debug_topn, 2),
            )
            best, second, best_score, margin = self._extract_best(candidates)
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

    def _passes_thresholds(self, best, best_score, margin) -> bool:
        if not best:
            return False
        if best_score < self.config.thresh_best:
            return False
        min_margin = self.decision._margin_for_score(best_score)
        if margin < min_margin:
            return False
        return True

    def _apply_margin_override(self, best, best_score, margin) -> float:
        if not best:
            return margin
        keys = {k.strip() for k in self.config.margin_override_keys.split(",") if k.strip()}
        if best.key in keys and best_score >= self.config.thresh_best:
            min_margin = self.decision._margin_for_score(best_score)
            return max(margin, min_margin)
        return margin

    def run(self):
        while not self._stop_event.is_set():
            try:
                item = self.window_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            backlog = self.window_queue.qsize()
            primary = self.whisper.transcribe_primary(item.audio, self.config.language_primary)
            segment_text, candidates, segments = self._pick_segment_candidates(primary.text)
            best, second, best_score, margin = self._extract_best(candidates)
            margin = self._apply_margin_override(best, best_score, margin)
            used_fallback = False
            fallback = None
            if self._should_fallback(best_score, margin, backlog):
                fallback_lang = self.config.language_fallback or self.config.language_primary
                fallback = self.whisper.transcribe_fallback(item.audio, fallback_lang)
                segment_text, fallback_candidates, segments = self._pick_segment_candidates(
                    fallback.text
                )
                fb_best, fb_second, fb_best_score, fb_margin = self._extract_best(
                    fallback_candidates
                )
                prefer_keys = {
                    k.strip()
                    for k in self.config.fallback_prefer_keys.split(",")
                    if k.strip()
                }
                prefer_fallback = False
                if fb_best and fb_best.key in prefer_keys:
                    if fb_best_score >= best_score - self.config.fallback_prefer_delta:
                        prefer_fallback = True
                if prefer_fallback or fb_best_score > best_score or fb_margin > margin:
                    used_fallback = True
                    best, second, best_score, margin = (
                        fb_best,
                        fb_second,
                        fb_best_score,
                        fb_margin,
                    )
                    candidates = fallback_candidates
                    margin = self._apply_margin_override(best, best_score, margin)

            now = time.monotonic()
            list_entries = []
            list_mode = self.config.list_mode and len(segments) >= self.config.list_min_segments
            if list_mode:
                list_entries = self._collect_segment_candidates(segments)

            events = []
            if list_mode:
                for entry in list_entries:
                    best_item = entry["best"]
                    best_score_item = entry["best_score"]
                    margin_item = self._apply_margin_override(
                        best_item, best_score_item, entry["margin"]
                    )
                    if not self._passes_thresholds(best_item, best_score_item, margin_item):
                        continue
                    hysteresis_override = self._key_hysteresis_override(best_item)
                    if hysteresis_override is None:
                        hysteresis_override = self._short_alias_override(
                        best_item, best_score_item, margin_item
                        )
                    if hysteresis_override is None:
                        hysteresis_override = self._high_confidence_override(
                            best_score_item, margin_item
                        )
                    if hysteresis_override is None:
                        hysteresis_override = self._mid_confidence_override(
                            best_item, best_score_item, margin_item
                        )
                    decision = self.decision.decide(
                        best_item.key if best_item else None,
                        best_score_item,
                        margin_item,
                        now,
                        hysteresis_override=hysteresis_override or 1,
                        bypass_cooldown=True,
                        bypass_repeat=True,
                        skip_cooldown_set=True,
                    )
                    if decision.accepted and best_item:
                        events.append(
                            {
                                "key": best_item.key,
                                "confidence": best_score_item,
                                "margin": margin_item,
                                "alias": best_item.alias,
                                "ngram": best_item.ngram,
                                "segment": entry["segment"],
                            }
                        )
                if events:
                    self.decision.bump_cooldown(now, self.config.list_cooldown_sec)
            if not events:
                hysteresis_override = self._key_hysteresis_override(best)
                if hysteresis_override is None:
                    hysteresis_override = self._short_alias_override(best, best_score, margin)
                if hysteresis_override is None:
                    hysteresis_override = self._high_confidence_override(best_score, margin)
                if hysteresis_override is None:
                    hysteresis_override = self._mid_confidence_override(best, best_score, margin)
                decision = self.decision.decide(
                    best.key if best else None,
                    best_score,
                    margin,
                    now,
                    hysteresis_override=hysteresis_override,
                )

            payload = {
                "type": "window",
                "window_id": item.window_id,
                "rms": item.rms,
                "peak": item.peak,
                "noise_floor": item.noise_floor,
                "gate": item.gate_active,
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
                "decision": "list" if events else decision.reason,
                "used_fallback": used_fallback,
                "backlog": backlog,
            }
            if fallback is not None:
                payload["fallback_text"] = fallback.text
            if self.config.debug:
                payload["topn"] = [
                    {"key": cand.key, "score": cand.score, "alias": cand.alias, "ngram": cand.ngram}
                    for cand in candidates[: self.config.debug_topn]
                ]
                payload["segments"] = segments
                if events:
                    payload["list_events"] = events
            self.logger.write(payload)

            if events:
                for entry in events:
                    self.metrics.inc("events_accepted", 1)
                    self.logger.write(
                        {
                            "type": "event",
                            "window_id": item.window_id,
                            "key": entry["key"],
                            "confidence": entry["confidence"],
                            "margin": entry["margin"],
                            "hysteresis": 1,
                            "cooldown_until": self.decision.cooldown_until(),
                            "used_fallback": used_fallback,
                            "list": True,
                        }
                    )
                    event = {
                        "type": "detect",
                        "key": entry["key"],
                        "confidence": entry["confidence"],
                        "text": primary.text if not used_fallback else fallback.text,
                        "norm": normalize_text(primary.text if not used_fallback else fallback.text),
                    }
                    if self.config.debug:
                        event["debug"] = {
                            "best_score": entry["confidence"],
                            "margin": entry["margin"],
                            "topn": payload.get("topn", []),
                            "segment": entry["segment"],
                        }
                    self.broadcaster.publish(event)
                self.broadcaster.publish(
                    {
                        "type": "state",
                        "state": "cooldown",
                        "cooldown_left": self.decision.cooldown_until(),
                    }
                )
            elif decision.accepted and best:
                self.metrics.inc("events_accepted", 1)
                self.logger.write(
                    {
                        "type": "event",
                        "window_id": item.window_id,
                        "key": best.key,
                        "confidence": best_score,
                        "margin": margin,
                        "hysteresis": decision.hysteresis_count,
                        "cooldown_until": decision.cooldown_until,
                        "used_fallback": used_fallback,
                    }
                )
                event = {
                    "type": "detect",
                    "key": best.key,
                    "confidence": best_score,
                    "text": primary.text if not used_fallback else fallback.text,
                    "norm": normalize_text(primary.text if not used_fallback else fallback.text),
                }
                if self.config.debug:
                    event["debug"] = {
                        "best_score": best_score,
                        "margin": margin,
                        "topn": payload.get("topn", []),
                    }
                self.broadcaster.publish(event)
                self.broadcaster.publish(
                    {
                        "type": "state",
                        "state": "cooldown",
                        "cooldown_left": self.decision.cooldown_until(),
                    }
                )
