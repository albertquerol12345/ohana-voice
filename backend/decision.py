import time
from collections import deque
from dataclasses import dataclass
import threading


@dataclass
class DecisionResult:
    accepted: bool
    reason: str
    cooldown_until: float | None = None
    hysteresis_count: int = 0
    muted_until: float | None = None


class DecisionEngine:
    def __init__(
        self,
        thresh_best: float,
        thresh_margin: float,
        margin_relax_score: float,
        margin_relax: float,
        margin_strict_score: float,
        margin_strict: float,
        hysteresis_k: int,
        hysteresis_window_sec: float,
        cooldown_sec: float,
        max_events_per_min: int,
        mute_sec: float,
        same_key_block_sec: float = 1.2,
        related_key_groups: str = "",
    ):
        self.thresh_best = thresh_best
        self.thresh_margin = thresh_margin
        self.margin_relax_score = margin_relax_score
        self.margin_relax = margin_relax
        self.margin_strict_score = margin_strict_score
        self.margin_strict = margin_strict
        self.hysteresis_k = hysteresis_k
        self.hysteresis_window_sec = hysteresis_window_sec
        self.cooldown_sec = cooldown_sec
        self.max_events_per_min = max_events_per_min
        self.mute_sec = mute_sec
        self.same_key_block_sec = same_key_block_sec
        self._related_groups = []
        for group in related_key_groups.split(";"):
            items = [item.strip() for item in group.split("|") if item.strip()]
            if len(items) >= 2:
                self._related_groups.append(set(items))

        self._lock = threading.Lock()
        self._cooldown_until = 0.0
        self._last_key = None
        self._last_accepted_ts = 0.0
        self._candidate_key = None
        self._candidate_count = 0
        self._candidate_start = 0.0
        self._accepted_timestamps = deque()
        self._mute_until = 0.0

    def _margin_for_score(self, score: float) -> float:
        if score >= self.margin_relax_score:
            return self.margin_relax
        if score <= self.margin_strict_score:
            return self.margin_strict
        return self.thresh_margin

    def in_cooldown(self, now: float) -> bool:
        with self._lock:
            return now < self._cooldown_until or now < self._mute_until

    def cooldown_until(self) -> float:
        with self._lock:
            return max(self._cooldown_until, self._mute_until)

    def decide(
        self,
        best_key: str | None,
        best_score: float,
        margin: float,
        now: float,
        hysteresis_override: int | None = None,
        bypass_cooldown: bool = False,
        bypass_repeat: bool = False,
        skip_cooldown_set: bool = False,
    ) -> DecisionResult:
        hysteresis_k = self.hysteresis_k if hysteresis_override is None else hysteresis_override
        with self._lock:
            if now < self._mute_until:
                return DecisionResult(False, "mute", muted_until=self._mute_until)
            if not bypass_cooldown and now < self._cooldown_until:
                return DecisionResult(False, "cooldown", cooldown_until=self._cooldown_until)

            if not best_key:
                self._candidate_key = None
                self._candidate_count = 0
                return DecisionResult(False, "no_candidate")

            if best_score < self.thresh_best:
                self._candidate_key = None
                self._candidate_count = 0
                return DecisionResult(False, "low_score")

            min_margin = self._margin_for_score(best_score)
            if margin < min_margin:
                self._candidate_key = None
                self._candidate_count = 0
                return DecisionResult(False, "low_margin")

            if self._candidate_key != best_key or (now - self._candidate_start) > self.hysteresis_window_sec:
                self._candidate_key = best_key
                self._candidate_count = 1
                self._candidate_start = now
            else:
                self._candidate_count += 1

            if self._candidate_count < hysteresis_k:
                return DecisionResult(False, "hysteresis", hysteresis_count=self._candidate_count)

            if not bypass_repeat and (
                self._last_key == best_key
                or self._is_related(self._last_key, best_key)
            ) and (now - self._last_accepted_ts) < self.same_key_block_sec:
                return DecisionResult(False, "repeat")

            self._accepted_timestamps.append(now)
            while self._accepted_timestamps and now - self._accepted_timestamps[0] > 60.0:
                self._accepted_timestamps.popleft()
            if len(self._accepted_timestamps) > self.max_events_per_min:
                self._mute_until = now + self.mute_sec
                return DecisionResult(False, "rate_limit", muted_until=self._mute_until)

            if not skip_cooldown_set:
                self._cooldown_until = now + self.cooldown_sec
            self._last_key = best_key
            self._last_accepted_ts = now
            self._candidate_key = None
            self._candidate_count = 0
            return DecisionResult(
                True, "accepted", cooldown_until=self._cooldown_until, hysteresis_count=hysteresis_k
            )

    def bump_cooldown(self, now: float, sec: float):
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, now + sec)

    def _is_related(self, key_a: str | None, key_b: str | None) -> bool:
        if not key_a or not key_b:
            return False
        for group in self._related_groups:
            if key_a in group and key_b in group:
                return True
        return False
