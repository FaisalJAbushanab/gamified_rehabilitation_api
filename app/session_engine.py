"""
Server-side therapy session state machine (GAG).

A *trial* is one trained item from its first presentation to its final outcome.
A *presentation* is one cue level (0 = cue-free, 1..5 = semantic, phonemic,
visual, written word, full auditory model) with its own response window; each
presentation produces one ``Attempt`` (a response or a timeout).

Flow per presentation
    response -> ASR -> GANVA -> ACCEPT  -> trial correct (points by cue level)
                              REJECT  -> next cue level (after L5: one repetition, then trial incorrect)
                              FLAG    -> wait for clinician (override) or the default decision
    timeout  -> treated as REJECT
After each completed trial: adaptive limit (S3-4), gamification, then either the
next trial or, once active time reaches the session duration, session end.
"""
from __future__ import annotations

import random
import secrets
import time
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import storage
from .adaptive import TrialStat, initial_limit_ms, next_limit
from .asr import transcribe
from .content import is_placeholder_file, is_placeholder_text, same_file_content
from .config import ADAPTIVE_VERSION, APP_VERSION, GANVA_VERSION, STUDY, config_hash, get_settings
from .db import utcnow
from .ganva import ganva_classify
from .models import (
    CUE_TYPES,
    AdaptiveEvent,
    Attempt,
    FlagReview,
    GamificationEvent,
    Participant,
    Stimulus,
    TherapySession,
    Trial,
    User,
)

CUE_LEVEL_REQUIREMENTS = {
    1: ("text", "audio"),  # semantic: Gulf Arabic script + recording
    2: ("audio",),  # phonemic: initial sound recording
    3: ("image",),  # visual: salient-feature picture
    4: (),  # written word: rendered from the target
    5: ("audio",),  # full auditory model
}


def conflict(msg: str) -> HTTPException:
    return HTTPException(409, msg)


# --- content ------------------------------------------------------------------

def trained_stimuli(db: Session) -> list[Stimulus]:
    return list(
        db.scalars(
            select(Stimulus)
            .where(Stimulus.word_list == STUDY.session.trained_list, Stimulus.active.is_(True), Stimulus.probe_only.is_(False))
            .order_by(Stimulus.id)
        )
    )


def cue_part_status(stim: Stimulus, cue, need: str) -> str:
    """'ok', 'missing' or 'placeholder' for one required part of a cue."""
    if cue is None:
        return "missing"
    if need == "text":
        if not cue.text:
            return "missing"
        return "placeholder" if is_placeholder_text(cue.text) else "ok"
    path = getattr(cue, f"{need}_path")
    if not path:
        return "missing"
    if is_placeholder_file(path) or (need == "image" and cue.level == 3 and same_file_content(path, stim.image_path)):
        return "placeholder"  # demo file, or an L3 cue that just repeats the item picture
    return "ok"


def content_problems(db: Session) -> list[dict]:
    problems = []
    for s in trained_stimuli(db):
        missing = []
        if not s.image_path:
            missing.append("picture")
        elif is_placeholder_file(s.image_path):
            missing.append("picture (demo placeholder)")
        cues = {c.level: c for c in s.cues}
        for level, needs in CUE_LEVEL_REQUIREMENTS.items():
            for need in needs:
                status = cue_part_status(s, cues.get(level), need)
                if status != "ok":
                    label = f"L{level} {CUE_TYPES[level]} {need}"
                    missing.append(label if status == "missing" else f"{label} (demo placeholder)")
        if missing:
            problems.append({"stimulus_id": s.id, "code": s.code, "english": s.english, "target": s.target, "missing": missing})
    return problems


# --- helpers ------------------------------------------------------------------

def _active_seconds(session: TherapySession, now: datetime | None = None) -> float:
    now = now or utcnow()
    extra = (now - session.active_since).total_seconds() if session.status == "active" and session.active_since else 0.0
    return session.active_seconds + max(0.0, extra)


def _active_trial(session: TherapySession) -> Trial | None:
    for t in reversed(session.trials):
        if t.status == "active":
            return t
    return None


def _pass_order(session: TherapySession, stim_ids: list[int], pass_no: int) -> list[int]:
    ids = sorted(stim_ids)
    random.Random(session.order_seed * 1000 + pass_no).shuffle(ids)
    return ids


def _create_next_trial(db: Session, session: TherapySession) -> Trial:
    stim_ids = [s.id for s in trained_stimuli(db)]
    if not stim_ids:
        raise conflict("There are no trained items (List A) to practise.")
    k = len(session.trials)
    pass_no, idx = k // len(stim_ids) + 1, k % len(stim_ids)
    trial = Trial(
        session_id=session.id,
        stimulus_id=_pass_order(session, stim_ids, pass_no)[idx],
        pass_no=pass_no,
        order_index=k,
    )
    session.current_pass = pass_no
    db.add(trial)
    db.flush()
    db.refresh(session)
    return trial


def _media(path: str) -> str | None:
    return storage.media_url(path)


# --- read models --------------------------------------------------------------

def presentation(session: TherapySession) -> dict | None:
    trial = _active_trial(session)
    if trial is None:
        return None
    stim = trial.stimulus
    level = trial.current_cue_level
    cue = None
    if level >= 1:
        asset = next((c for c in stim.cues if c.level == level), None)
        cue = {
            "level": level,
            "type": CUE_TYPES[level],
            "text": stim.target if level == 4 else (asset.text if asset else ""),
            "audio_url": _media(asset.audio_path) if asset else None,
            "image_url": _media(asset.image_path) if asset else None,
        }
    return {
        "trial_id": trial.id,
        "stimulus_id": stim.id,
        "code": stim.code,
        "word_class": stim.word_class,
        "english": stim.english,
        "target": stim.target,  # clinician panel only; the patient view shows it at L4/L5
        "image_url": _media(stim.image_path),
        "cue_level": level,
        "is_repetition": trial.current_is_repetition,
        "cue": cue,
        "window_ms": session.participant.adaptive_limit_ms,
        "pass_no": trial.pass_no,
        "order_index": trial.order_index,
        "awaiting_attempt_id": trial.pending_attempt_id,
    }


def session_summary(session: TherapySession) -> dict:
    completed = [t for t in session.trials if t.status == "completed"]
    return {
        "id": session.id,
        "participant_id": session.participant_id,
        "study_code": session.participant.study_code,
        "session_no": session.session_no,
        "status": session.status,
        "started_at": session.started_at.isoformat(),
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "end_reason": session.end_reason,
        "active_seconds": round(_active_seconds(session), 1),
        "duration_seconds": STUDY.session.duration_min * 60,
        "trials_completed": len(completed),
        "trials_correct": sum(t.outcome == "correct" for t in completed),
        "pass_no": session.current_pass,
        "trained_count": None,
        "start_limit_ms": session.start_limit_ms,
        "end_limit_ms": session.end_limit_ms,
        "config_hash": session.config_hash,
        "app_version": session.app_version,
        "ganva_version": session.ganva_version,
    }


def session_state(db: Session, session: TherapySession) -> dict:
    db.flush()  # refresh() reloads from the database; unflushed changes would be lost
    db.refresh(session)
    p = session.participant
    summary = session_summary(session)
    summary["trained_count"] = len(trained_stimuli(db))
    return {
        "session": summary,
        "presentation": presentation(session) if session.status in ("active", "paused") else None,
        "limit_ms": p.adaptive_limit_ms,
        "participant": {"id": p.id, "study_code": p.study_code, "total_points": p.total_points, "level": p.level},
        "flag_wait_s": STUDY.session.flag_clinician_wait_s,
        "vad": {"silence_ms": STUDY.session.vad_silence_ms, "min_speech_ms": STUDY.session.vad_min_speech_ms},
    }


def attempt_view(a: Attempt) -> dict:
    return {
        "id": a.id,
        "kind": a.kind,
        "cue_level": a.cue_level,
        "is_repetition": a.is_repetition,
        "transcript": a.transcript,
        "asr_confidence": a.asr_confidence,
        "asr_error": a.asr_error,
        "stage1_similarity": a.stage1_similarity,
        "stage_reached": a.stage_reached,
        "matched_associate": a.matched_associate,
        "ganva_decision": a.ganva_decision,
        "ganva_reason": a.ganva_reason,
        "shadow_decision": a.shadow_decision,
        "effective_decision": a.effective_decision,
        "resolution": a.resolution,
        "t_asr_ms": a.t_asr_ms,
        "t_ganva_ms": a.t_ganva_ms,
        "t_server_total_ms": a.t_server_total_ms,
        "has_audio": bool(a.audio_ref),
    }


# --- lifecycle ----------------------------------------------------------------

def start_session(db: Session, participant: Participant, operator: User) -> TherapySession:
    if participant.arm != "GAG":
        raise conflict("Only participants in the Gamified App Group have app sessions.")
    db.flush()  # autoflush is off: pending status changes must be visible to the queries below
    existing = db.scalars(
        select(TherapySession).where(
            TherapySession.participant_id == participant.id, TherapySession.status.in_(("active", "paused"))
        )
    ).first()
    if existing:
        return existing
    done = db.scalar(
        select(func.count()).select_from(TherapySession).where(
            TherapySession.participant_id == participant.id, TherapySession.status.in_(("completed", "abandoned"))
        )
    )
    if done >= STUDY.session.total_sessions:
        raise conflict(f"{participant.study_code} has already had all {STUDY.session.total_sessions} sessions.")
    if get_settings().require_complete_content:
        problems = content_problems(db)
        if problems:
            raise conflict(
                f"{len(problems)} trained items are missing cue materials or still use demo placeholders. "
                "Open Content to see what is missing."
            )
    if participant.adaptive_limit_ms is None:
        start = initial_limit_ms(participant.severity, STUDY.adaptive)
        participant.adaptive_limit_ms = start
        db.add(AdaptiveEvent(participant_id=participant.id, old_ms=start, new_ms=start, rule="initial", n=0, version=ADAPTIVE_VERSION))
    now = utcnow()
    session = TherapySession(
        participant_id=participant.id,
        session_no=done + 1,
        status="active",
        started_at=now,
        active_since=now,
        operator_id=operator.id,
        app_version=APP_VERSION,
        ganva_version=GANVA_VERSION,
        adaptive_version=ADAPTIVE_VERSION,
        config_hash=config_hash(),
        config_snapshot=STUDY.model_dump(mode="json"),
        order_seed=secrets.randbelow(2**31 - 1),
        start_limit_ms=participant.adaptive_limit_ms,
    )
    db.add(session)
    db.flush()
    db.refresh(session)
    _create_next_trial(db, session)
    return session


def pause_session(session: TherapySession) -> None:
    if session.status != "active":
        raise conflict("Only a running session can be paused.")
    session.active_seconds = _active_seconds(session)
    session.active_since = None
    session.status = "paused"


def resume_session(session: TherapySession) -> None:
    if session.status != "paused":
        raise conflict("Only a paused session can be resumed.")
    session.status = "active"
    session.active_since = utcnow()


def end_session(db: Session, session: TherapySession, reason: str, status: str = "completed") -> list[str]:
    if session.status not in ("active", "paused"):
        raise conflict("This session has already ended.")
    session.active_seconds = _active_seconds(session)
    session.active_since = None
    session.status = status
    session.ended_at = utcnow()
    session.end_reason = reason
    for t in session.trials:
        if t.status == "active":
            t.status = "abandoned"
            t.ended_at = session.ended_at
    if STUDY.adaptive.update_every == "session":
        _update_adaptive(db, session, None)
    session.end_limit_ms = session.participant.adaptive_limit_ms
    return _session_end_badges(db, session) if status == "completed" else []


# --- attempts -----------------------------------------------------------------

def _check_presentation(session: TherapySession, trial: Trial, cue_level: int) -> None:
    if session.status != "active":
        raise conflict("The session is not running. Resume it first.")
    if trial.session_id != session.id or trial.status != "active":
        raise conflict("This item is no longer active.")
    if trial.pending_attempt_id is not None:
        raise conflict("Waiting for the clinician to decide the flagged response.")
    if cue_level != trial.current_cue_level:
        raise conflict("The screen is out of date. Reload the session.")


def _duplicate(db: Session, client_uuid: str) -> Attempt | None:
    return db.scalars(select(Attempt).where(Attempt.client_uuid == client_uuid)).first()


def submit_response(
    db: Session,
    session: TherapySession,
    trial: Trial,
    *,
    client_uuid: str,
    cue_level: int,
    speech_onset_ms: int | None,
    presented_at: str,
    audio: bytes,
    mime: str,
    filename: str | None,
    dev_transcript: str | None = None,
    dev_confidence: float | None = None,
) -> dict:
    dup = _duplicate(db, client_uuid)
    if dup:
        return {"attempt": attempt_view(dup), "action": "duplicate", "state": session_state(db, session)}
    _check_presentation(session, trial, cue_level)
    t0 = time.perf_counter()
    settings = get_settings()
    participant = session.participant
    stim = trial.stimulus

    wav, duration_ms, convert_error = b"", None, ""
    try:
        wav, duration_ms = storage.to_wav16k(audio, storage.suffix_for_mime(mime, filename))
    except Exception as e:  # noqa: BLE001 - any conversion failure becomes a FLAG, never a silent INCORRECT
        convert_error = str(e)

    if convert_error:
        from .asr import ASRResult

        asr = ASRResult(False, settings.asr_provider, locale=settings.asr_locale, error=convert_error)
    else:
        asr = transcribe(wav, settings, dev_transcript, dev_confidence)

    g = ganva_classify(
        asr.transcript,
        asr.confidence,
        stim.target,
        [a.text for a in stim.associates],
        STUDY.ganva,
        asr_ok=asr.ok,
    )
    audio_ref = storage.save_encrypted_audio(wav) if (wav and participant.consent_audio_retention) else None

    attempt = Attempt(
        trial_id=trial.id,
        client_uuid=client_uuid,
        kind="response",
        cue_level=cue_level,
        is_repetition=trial.current_is_repetition,
        window_ms=participant.adaptive_limit_ms,
        presented_at_client=presented_at,
        speech_onset_ms=speech_onset_ms,
        audio_ref=audio_ref,
        audio_mime=mime or "",
        audio_duration_ms=duration_ms,
        asr_provider=asr.provider,
        asr_model=asr.model,
        asr_locale=asr.locale,
        transcript=asr.transcript,
        asr_confidence=asr.confidence,
        n_best=asr.n_best,
        asr_error=asr.error,
        normalized_transcript=g.normalized_transcript,
        stage1_similarity=g.stage1_similarity,
        matched_span=g.matched_span,
        stage_reached=g.stage_reached,
        stage2_similarity=g.stage2_similarity,
        matched_associate=g.matched_associate,
        ganva_decision=g.decision,
        ganva_reason=g.reason,
        shadow_decision=g.shadow_decision,
        t_asr_ms=asr.latency_ms if asr.ok else None,
        t_ganva_ms=g.t_ganva_ms,
        ganva_version=g.ganva_version,
        config_hash=session.config_hash,
    )
    db.add(attempt)
    db.flush()

    if g.decision == "FLAG":
        attempt.resolution = "pending"
        trial.pending_attempt_id = attempt.id
        outcome = {"action": "await_clinician"}
    else:
        attempt.effective_decision = g.decision
        attempt.resolution = "auto"
        outcome = _advance(db, session, trial, attempt)
    attempt.t_server_total_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"attempt": attempt_view(attempt), **outcome, "state": session_state(db, session)}


def submit_timeout(db: Session, session: TherapySession, trial: Trial, *, client_uuid: str, cue_level: int, presented_at: str) -> dict:
    dup = _duplicate(db, client_uuid)
    if dup:
        return {"attempt": attempt_view(dup), "action": "duplicate", "state": session_state(db, session)}
    _check_presentation(session, trial, cue_level)
    window = session.participant.adaptive_limit_ms
    attempt = Attempt(
        trial_id=trial.id,
        client_uuid=client_uuid,
        kind="timeout",
        cue_level=cue_level,
        is_repetition=trial.current_is_repetition,
        window_ms=window,
        presented_at_client=presented_at,
        effective_decision="REJECT",
        resolution="timeout",
        config_hash=session.config_hash,
        ganva_version=GANVA_VERSION,
    )
    db.add(attempt)
    db.flush()
    trial.timed_out_any = True
    outcome = _advance(db, session, trial, attempt)
    return {"attempt": attempt_view(attempt), **outcome, "state": session_state(db, session)}


def resolve_flag(db: Session, session: TherapySession, attempt: Attempt, decision: str, user: User | None, how: str) -> dict:
    trial = attempt.trial
    if trial.pending_attempt_id != attempt.id or attempt.resolution != "pending":
        raise conflict("This response has already been decided.")
    if session.status != "active":
        raise conflict("The session is not running. Resume it first.")
    attempt.effective_decision = decision
    attempt.resolution = how
    if user is not None:
        attempt.override_by = user.id
        attempt.override_at = utcnow()
        db.add(FlagReview(attempt_id=attempt.id, reviewer_id=user.id, human_decision=decision, note="in-session decision"))
    trial.pending_attempt_id = None
    outcome = _advance(db, session, trial, attempt)
    return {"attempt": attempt_view(attempt), **outcome, "state": session_state(db, session)}


# --- progression --------------------------------------------------------------

def _advance(db: Session, session: TherapySession, trial: Trial, attempt: Attempt) -> dict:
    cfg = STUDY.session
    response_ms = attempt.speech_onset_ms if (attempt.kind == "response" and attempt.speech_onset_ms is not None) else attempt.window_ms
    if attempt.effective_decision == "ACCEPT":
        if attempt.is_repetition:
            return _complete_trial(db, session, trial, "incorrect", attempt, response_ms, STUDY.gamification.repetition_points, "trial_incorrect")
        points = STUDY.gamification.points_by_cue_level.get(attempt.cue_level, 0)
        return _complete_trial(db, session, trial, "correct", attempt, response_ms, points, "trial_correct")
    if not attempt.is_repetition and trial.current_cue_level < cfg.max_cue_level:
        trial.current_cue_level += 1
        return {"action": "next_cue"}
    if not attempt.is_repetition and cfg.repetition_after_l5_failure:
        trial.current_is_repetition = True
        return {"action": "repetition"}
    return _complete_trial(db, session, trial, "incorrect", attempt, response_ms, 0, "trial_incorrect")


def _complete_trial(db, session, trial, outcome, attempt, response_ms, points, action) -> dict:
    trial.status = "completed"
    trial.outcome = outcome
    trial.final_cue_level = attempt.cue_level
    trial.ended_at = utcnow()
    trial.response_time_ms = response_ms
    trial.points = points
    participant = session.participant
    badges: list[str] = []
    if points:
        participant.total_points += points
        db.add(GamificationEvent(participant_id=participant.id, session_id=session.id, trial_id=trial.id, kind="points", value=points))
    db.flush()
    db.refresh(session)
    badges += _trial_badges(db, session, trial)
    adaptive = None
    if STUDY.adaptive.update_every == "trial":
        adaptive = _update_adaptive(db, session, trial)

    session_complete = False
    if _active_seconds(session) >= STUDY.session.duration_min * 60:
        badges += end_session(db, session, "duration_reached")
        session_complete = True
    else:
        _create_next_trial(db, session)
    return {
        "action": action,
        "points": points,
        "badges": badges,
        "adaptive": adaptive,
        "session_complete": session_complete,
    }


def _recent_trial_stats(db: Session, participant_id: int) -> list[TrialStat]:
    cfg = STUDY.adaptive
    q = (
        select(Trial)
        .join(TherapySession, Trial.session_id == TherapySession.id)
        .where(TherapySession.participant_id == participant_id, Trial.status == "completed")
        .order_by(Trial.ended_at.desc(), Trial.id.desc())
    )
    if cfg.window_scope == "trials":
        trials = list(db.scalars(q.limit(cfg.window_n)))
    else:
        sessions = list(
            db.scalars(
                select(TherapySession.id)
                .where(TherapySession.participant_id == participant_id, TherapySession.status.in_(("completed", "abandoned")))
                .order_by(TherapySession.ended_at.desc(), TherapySession.id.desc())
                .limit(cfg.window_n)
            )
        )
        trials = list(db.scalars(q.where(Trial.session_id.in_(sessions)))) if sessions else []
    return [TrialStat(t.outcome == "correct", t.response_time_ms or 0, t.timed_out_any) for t in trials]


def _update_adaptive(db: Session, session: TherapySession, trial: Trial | None) -> dict:
    participant = session.participant
    stats = _recent_trial_stats(db, participant.id)
    cfg = STUDY.adaptive
    if cfg.window_scope == "sessions":
        # pooled trials of the window's sessions all count; bypass the trial-count window
        cfg = cfg.model_copy(update={"window_n": len(stats) or 1, "min_history": 1})
    res = next_limit(participant.adaptive_limit_ms, stats, cfg)
    participant.adaptive_limit_ms = res.new_ms
    db.add(
        AdaptiveEvent(
            participant_id=participant.id,
            session_id=session.id,
            trial_id=trial.id if trial else None,
            old_ms=res.old_ms,
            new_ms=res.new_ms,
            rule=res.rule,
            n=res.n,
            accuracy=res.accuracy,
            mean_response_ms=res.mean_response_ms,
            timeout_rate=res.timeout_rate,
            version=res.version,
        )
    )
    return res.as_dict()


# --- gamification -------------------------------------------------------------

def _has_badge(db: Session, participant_id: int, code: str) -> bool:
    return db.scalar(
        select(func.count()).select_from(GamificationEvent).where(
            GamificationEvent.participant_id == participant_id, GamificationEvent.kind == "badge", GamificationEvent.code == code
        )
    ) > 0


def _award(db: Session, session: TherapySession, code: str, trial: Trial | None = None) -> list[str]:
    if _has_badge(db, session.participant_id, code):
        return []
    db.add(GamificationEvent(participant_id=session.participant_id, session_id=session.id, trial_id=trial.id if trial else None, kind="badge", code=code))
    return [code]


def _trial_badges(db: Session, session: TherapySession, trial: Trial) -> list[str]:
    out: list[str] = []
    completed = [t for t in session.trials if t.status == "completed"]
    streak = 0
    for t in reversed(completed):
        if t.outcome != "correct":
            break
        streak += 1
    for n in STUDY.gamification.streak_badges:
        if streak == n:
            out += _award(db, session, f"streak_{n}", trial)
    same_pass = [t for t in completed if t.pass_no == trial.pass_no]
    if len(same_pass) == len(trained_stimuli(db)) and all(t.outcome == "correct" and t.final_cue_level == 0 for t in same_pass):
        out += _award(db, session, "perfect_pass", trial)
    p = session.participant
    new_level = sum(1 for th in STUDY.gamification.level_thresholds if p.total_points >= th)
    if new_level > p.level:
        p.level = new_level
        db.add(GamificationEvent(participant_id=p.id, session_id=session.id, trial_id=trial.id, kind="level", value=new_level))
        out.append(f"level_{new_level}")
    return out


def _session_end_badges(db: Session, session: TherapySession) -> list[str]:
    out: list[str] = []
    if session.session_no == 1:
        out += _award(db, session, "first_session")
    for n in (5, 10, 15):
        if session.session_no == n:
            out += _award(db, session, f"sessions_{n}")
    return out
