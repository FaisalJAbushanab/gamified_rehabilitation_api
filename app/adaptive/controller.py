"""
Algorithm S3-4: adaptive response time limit (GAG only).

Server-authoritative and pure: the caller passes the most recent completed
trials *newest first*; this module takes the first ``window_n`` of them.
Definitions (see IMPLEMENTATION_PLAN.md section 12):
  correct        trial outcome was correct
  response_time  speech-onset latency of the final presentation; a presentation
                 that expired with no speech counts as its full window
  timed_out      any presentation in the trial expired with no speech
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from ..config import ADAPTIVE_VERSION, AdaptiveConfig


@dataclass(frozen=True)
class TrialStat:
    correct: bool
    response_time_ms: int
    timed_out: bool


@dataclass
class AdaptiveResult:
    old_ms: int
    new_ms: int
    rule: str  # decrease, hard_increase, soft_increase, no_change, insufficient_history
    n: int
    accuracy: float | None
    mean_response_ms: float | None
    timeout_rate: float | None
    version: str = ADAPTIVE_VERSION

    def as_dict(self) -> dict:
        return asdict(self)


def initial_limit_ms(severity: str | None, cfg: AdaptiveConfig) -> int:
    return cfg.start_ms_by_severity.get((severity or "").lower(), cfg.default_start_ms)


def clamp(value: int, cfg: AdaptiveConfig) -> int:
    return max(cfg.min_ms, min(cfg.max_ms, value))


def next_limit(current_ms: int, recent_newest_first: Sequence[TrialStat], cfg: AdaptiveConfig) -> AdaptiveResult:
    recent = list(recent_newest_first[: cfg.window_n])
    n = len(recent)
    if n < cfg.min_history:
        return AdaptiveResult(current_ms, current_ms, "insufficient_history", n, None, None, None)

    acc = sum(t.correct for t in recent) / n
    mean_rt = sum(t.response_time_ms for t in recent) / n
    timeout_rate = sum(t.timed_out for t in recent) / n

    if acc > cfg.acc_high and mean_rt < cfg.fast_resp * current_ms:
        rule, delta = "decrease", -cfg.decrease_ms
    elif timeout_rate > cfg.timeout_high:
        rule, delta = "hard_increase", cfg.hard_increase_ms
    elif acc < cfg.acc_low:
        rule, delta = "soft_increase", cfg.soft_increase_ms
    else:
        rule, delta = "no_change", 0

    return AdaptiveResult(
        old_ms=current_ms,
        new_ms=clamp(current_ms + delta, cfg),
        rule=rule,
        n=n,
        accuracy=round(acc, 4),
        mean_response_ms=round(mean_rt, 1),
        timeout_rate=round(timeout_rate, 4),
    )
