"""
Algorithm S3-3: GANVA two-stage classification (ACCEPT / REJECT / FLAG).

Besides the live decision, every call returns a *shadow decision*: what Stage 1
and Stage 2 would have decided if the confidence check were ignored. For FLAG
attempts this is what makes the flag-scoring question in S3 answerable and the
60/70/80 % confidence sensitivity analysis reproducible from stored records.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Iterable, Literal

from ..config import GANVA_VERSION, GanvaConfig
from .normalize import normalize_arabic_text
from .similarity import best_span_similarity

Decision = Literal["ACCEPT", "REJECT", "FLAG"]


@dataclass
class GanvaResult:
    decision: Decision
    reason: str  # stage1_accept, stage1_reject, stage2_accept, stage2_reject, low_confidence, no_confidence, asr_error, no_speech
    shadow_decision: Literal["ACCEPT", "REJECT"] | None
    stage_reached: int | None  # 1 or 2; None when the transcript was empty or ASR failed
    normalized_transcript: str
    stage1_similarity: float | None
    matched_span: str | None
    stage2_similarity: float | None
    matched_associate: str | None
    asr_confidence: float | None
    ganva_version: str
    t_ganva_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def _stages(transcript: str, target: str, associates: Iterable[str], cfg: GanvaConfig) -> dict:
    s1 = best_span_similarity(transcript, target, cfg.multiword_rule)
    out = {
        "stage1_similarity": round(s1.similarity, 4),
        "matched_span": s1.span,
        "stage2_similarity": None,
        "matched_associate": None,
    }
    if s1.similarity >= cfg.stage1_threshold:
        return {**out, "shadow": "ACCEPT", "reason": "stage1_accept", "stage": 1}
    if s1.similarity < cfg.stage2_low:
        return {**out, "shadow": "REJECT", "reason": "stage1_reject", "stage": 1}
    best_sim, best_assoc = 0.0, None
    for assoc in associates:
        m = best_span_similarity(transcript, assoc, cfg.multiword_rule)
        if m.similarity > best_sim:
            best_sim, best_assoc = m.similarity, assoc
        if m.similarity >= cfg.stage1_threshold:
            return {
                **out,
                "stage2_similarity": round(m.similarity, 4),
                "matched_associate": assoc,
                "shadow": "ACCEPT",
                "reason": "stage2_accept",
                "stage": 2,
            }
    return {
        **out,
        "stage2_similarity": round(best_sim, 4) if best_assoc else None,
        "matched_associate": None,
        "shadow": "REJECT",
        "reason": "stage2_reject",
        "stage": 2,
    }


def ganva_classify(
    asr_transcript: str | None,
    asr_confidence: float | None,
    target: str,
    semantic_associates: Iterable[str],
    cfg: GanvaConfig,
    asr_ok: bool = True,
) -> GanvaResult:
    t0 = time.perf_counter()
    norm = normalize_arabic_text(asr_transcript)

    def done(decision: Decision, reason: str, stages: dict | None) -> GanvaResult:
        stages = stages or {}
        return GanvaResult(
            decision=decision,
            reason=reason,
            shadow_decision=stages.get("shadow"),
            stage_reached=stages.get("stage"),
            normalized_transcript=norm,
            stage1_similarity=stages.get("stage1_similarity"),
            matched_span=stages.get("matched_span"),
            stage2_similarity=stages.get("stage2_similarity"),
            matched_associate=stages.get("matched_associate"),
            asr_confidence=asr_confidence,
            ganva_version=GANVA_VERSION,
            t_ganva_ms=round((time.perf_counter() - t0) * 1000, 3),
        )

    if not asr_ok:
        return done("FLAG", "asr_error", None)
    if not norm:
        return done("REJECT", "no_speech", None)

    stages = _stages(asr_transcript or "", target, list(semantic_associates), cfg)

    if asr_confidence is None and cfg.flag_when_confidence_missing:
        return done("FLAG", "no_confidence", stages)
    if asr_confidence is not None and asr_confidence < cfg.asr_confidence_flag:
        return done("FLAG", "low_confidence", stages)
    return done(stages["shadow"], stages["reason"], stages)
