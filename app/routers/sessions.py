import statistics
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import session_engine as engine
from ..config import STUDY
from ..db import get_db
from ..models import AdaptiveEvent, Attempt, Participant, TherapySession, Trial, User
from ..security import STAFF, audit, current_user, require_roles
from .participants import get_participant_for

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _session_for(db: Session, session_id: int, user: User) -> TherapySession:
    s = db.get(TherapySession, session_id)
    if s is None:
        raise HTTPException(404, "Session not found.")
    get_participant_for(db, s.participant_id, user)  # access check
    return s


def _trial_for(db: Session, session: TherapySession, trial_id: int) -> Trial:
    t = db.get(Trial, trial_id)
    if t is None or t.session_id != session.id:
        raise HTTPException(404, "Item not found in this session.")
    return t


def _attempt_for(db: Session, session: TherapySession, attempt_id: int) -> Attempt:
    a = db.get(Attempt, attempt_id)
    if a is None or a.trial.session_id != session.id:
        raise HTTPException(404, "Response not found in this session.")
    return a


class StartIn(BaseModel):
    participant_id: int


@router.post("")
def start(body: StartIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    p = get_participant_for(db, body.participant_id, user)
    s = engine.start_session(db, p, user)
    audit(db, user, "start_or_resume", "session", s.id, participant=p.study_code)
    db.commit()
    return engine.session_state(db, s)


@router.get("")
def list_sessions(participant_id: int | None = None, status: str | None = None, db: Session = Depends(get_db), _: User = Depends(require_roles(*STAFF))):
    q = select(TherapySession).order_by(TherapySession.started_at.desc())
    if participant_id:
        q = q.where(TherapySession.participant_id == participant_id)
    if status:
        q = q.where(TherapySession.status.in_(status.split(",")))
    return [engine.session_summary(s) for s in db.scalars(q.limit(500))]


@router.get("/{session_id}/state")
def state(session_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    return engine.session_state(db, _session_for(db, session_id, user))


@router.post("/{session_id}/pause")
def pause(session_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    s = _session_for(db, session_id, user)
    engine.pause_session(s)
    audit(db, user, "pause", "session", s.id)
    db.commit()
    return engine.session_state(db, s)


@router.post("/{session_id}/resume")
def resume(session_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    s = _session_for(db, session_id, user)
    engine.resume_session(s)
    audit(db, user, "resume", "session", s.id)
    db.commit()
    return engine.session_state(db, s)


class EndIn(BaseModel):
    reason: str = "ended_by_clinician"
    status: Literal["completed", "abandoned"] = "completed"


@router.post("/{session_id}/end")
def end(session_id: int, body: EndIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    s = _session_for(db, session_id, user)
    badges = engine.end_session(db, s, body.reason, body.status)
    audit(db, user, "end", "session", s.id, reason=body.reason, status=body.status)
    db.commit()
    return {**engine.session_state(db, s), "badges": badges}


@router.post("/{session_id}/trials/{trial_id}/response")
async def response(
    session_id: int,
    trial_id: int,
    client_uuid: str = Form(...),
    cue_level: int = Form(...),
    presented_at: str = Form(""),
    speech_onset_ms: int | None = Form(None),
    audio: UploadFile = File(...),
    dev_transcript: str | None = Form(None),
    dev_confidence: float | None = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    s = _session_for(db, session_id, user)
    t = _trial_for(db, s, trial_id)
    data = await audio.read()
    if not data:
        raise HTTPException(422, "The recording is empty.")
    out = engine.submit_response(
        db, s, t,
        client_uuid=client_uuid, cue_level=cue_level, speech_onset_ms=speech_onset_ms, presented_at=presented_at,
        audio=data, mime=audio.content_type or "", filename=audio.filename,
        dev_transcript=dev_transcript, dev_confidence=dev_confidence,
    )
    db.commit()
    return out


class TimeoutIn(BaseModel):
    client_uuid: str
    cue_level: int
    presented_at: str = ""


@router.post("/{session_id}/trials/{trial_id}/timeout")
def timeout(session_id: int, trial_id: int, body: TimeoutIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    s = _session_for(db, session_id, user)
    t = _trial_for(db, s, trial_id)
    out = engine.submit_timeout(db, s, t, client_uuid=body.client_uuid, cue_level=body.cue_level, presented_at=body.presented_at)
    db.commit()
    return out


class DecideIn(BaseModel):
    decision: Literal["ACCEPT", "REJECT"]


@router.post("/{session_id}/attempts/{attempt_id}/decide")
def decide(session_id: int, attempt_id: int, body: DecideIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    """D11: the supervising SLP decides a flagged response on the tablet."""
    s = _session_for(db, session_id, user)
    out = engine.resolve_flag(db, s, _attempt_for(db, s, attempt_id), body.decision, user, "clinician")
    audit(db, user, "decide_flag", "attempt", attempt_id, decision=body.decision)
    db.commit()
    return out


@router.post("/{session_id}/attempts/{attempt_id}/default")
def default_decision(session_id: int, attempt_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """D11 fallback: nobody decided within the wait time; the configured default applies and the flag stays in the review queue."""
    s = _session_for(db, session_id, user)
    out = engine.resolve_flag(db, s, _attempt_for(db, s, attempt_id), STUDY.session.flag_default_decision, None, "default")
    db.commit()
    return out


class FeedbackIn(BaseModel):
    client_total_ms: float | None = None
    shown_at: str = ""


@router.post("/{session_id}/attempts/{attempt_id}/feedback")
def feedback_shown(session_id: int, attempt_id: int, body: FeedbackIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Latency logging: the client reports when feedback appeared (audio end -> feedback shown)."""
    s = _session_for(db, session_id, user)
    a = _attempt_for(db, s, attempt_id)
    a.t_client_total_ms = body.client_total_ms
    a.feedback_shown_at_client = body.shown_at
    db.commit()
    return {"ok": True}


@router.get("/{session_id}/attempts")
def session_attempts(session_id: int, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    s = _session_for(db, session_id, user)
    out = []
    for t in s.trials:
        for a in t.attempts:
            out.append({**engine.attempt_view(a), "trial_id": t.id, "order_index": t.order_index, "code": t.stimulus.code, "target": t.stimulus.target})
    return out


def _quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) < 4:
        return {"median": round(statistics.median(values), 1), "q1": None, "q3": None, "n": len(values)}
    q = statistics.quantiles(values, n=4)
    return {"median": round(q[1], 1), "q1": round(q[0], 1), "q3": round(q[2], 1), "n": len(values)}


@router.get("/{session_id}/analytics")
def analytics(session_id: int, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    s = _session_for(db, session_id, user)
    trials = [t for t in s.trials if t.status == "completed"]
    attempts = [a for t in s.trials for a in t.attempts]
    responses = [a for a in attempts if a.kind == "response"]
    by_level: dict[int, int] = {}
    for t in trials:
        if t.outcome == "correct":
            by_level[t.final_cue_level] = by_level.get(t.final_cue_level, 0) + 1
    events = db.scalars(select(AdaptiveEvent).where(AdaptiveEvent.session_id == s.id).order_by(AdaptiveEvent.id))
    summary = engine.session_summary(s)
    return {
        "session": summary,
        "trials": len(trials),
        "correct": sum(t.outcome == "correct" for t in trials),
        "correct_cue_free": by_level.get(0, 0),
        "correct_by_cue_level": by_level,
        "incorrect": sum(t.outcome == "incorrect" for t in trials),
        "timeouts": sum(a.kind == "timeout" for a in attempts),
        "responses": len(responses),
        "flags": sum(a.ganva_decision == "FLAG" for a in responses),
        "flag_rate": round(sum(a.ganva_decision == "FLAG" for a in responses) / len(responses), 4) if responses else None,
        "decisions": {k: sum(a.ganva_decision == k for a in responses) for k in ("ACCEPT", "REJECT", "FLAG")},
        "stage2_accepts": sum(a.ganva_reason == "stage2_accept" for a in responses),
        "response_time_ms": _quantiles([t.response_time_ms for t in trials if t.response_time_ms is not None]),
        "latency_server_ms": _quantiles([a.t_server_total_ms for a in responses if a.t_server_total_ms is not None]),
        "latency_asr_ms": _quantiles([a.t_asr_ms for a in responses if a.t_asr_ms is not None]),
        "latency_client_ms": _quantiles([a.t_client_total_ms for a in responses if a.t_client_total_ms is not None]),
        "window_trajectory": [{"trial_id": e.trial_id, "old_ms": e.old_ms, "new_ms": e.new_ms, "rule": e.rule} for e in events],
        "points": sum(t.points for t in trials),
        "active_minutes": round(summary["active_seconds"] / 60, 1),
        "passes": s.current_pass,
    }
