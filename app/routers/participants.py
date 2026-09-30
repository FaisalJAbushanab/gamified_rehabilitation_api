from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import AdaptiveEvent, GamificationEvent, Participant, TherapySession, User
from ..security import STAFF, audit, current_user, hash_password, require_roles
from ..session_engine import session_summary

router = APIRouter(prefix="/api/participants", tags=["participants"])

Severity = Literal["mild", "moderate", "severe", "profound"]


class ParticipantIn(BaseModel):
    study_code: str = Field(pattern=r"^[A-Z]{2}\d{2,3}$")
    arm: Literal["GAG", "TSG"]
    severity: Severity
    dysarthria: bool = False
    consent_audio_retention: bool = False
    notes: str = ""
    patient_password: str | None = Field(default=None, min_length=8)


class ParticipantPatch(BaseModel):
    severity: Severity | None = None
    dysarthria: bool | None = None
    consent_audio_retention: bool | None = None
    notes: str | None = None


def view(db: Session, p: Participant) -> dict:
    sessions = list(db.scalars(select(TherapySession).where(TherapySession.participant_id == p.id)))
    return {
        "id": p.id,
        "study_code": p.study_code,
        "arm": p.arm,
        "severity": p.severity,
        "dysarthria": p.dysarthria,
        "consent_audio_retention": p.consent_audio_retention,
        "notes": p.notes,
        "adaptive_limit_ms": p.adaptive_limit_ms,
        "total_points": p.total_points,
        "level": p.level,
        "enrolled_at": p.enrolled_at.isoformat(),
        "sessions_done": sum(s.status in ("completed", "abandoned") for s in sessions),
        "active_session_id": next((s.id for s in sessions if s.status in ("active", "paused")), None),
        "has_login": p.user_id is not None,
    }


def get_participant_for(db: Session, participant_id: int, user: User) -> Participant:
    p = db.get(Participant, participant_id)
    if p is None:
        raise HTTPException(404, "Participant not found.")
    if user.role == "patient" and p.user_id != user.id:
        raise HTTPException(403, "Your account does not have access to this.")
    if user.role not in STAFF and user.role != "patient":
        raise HTTPException(403, "Your account does not have access to this.")
    return p


@router.get("")
def list_participants(db: Session = Depends(get_db), _: User = Depends(require_roles(*STAFF))):
    return [view(db, p) for p in db.scalars(select(Participant).order_by(Participant.study_code))]


@router.post("", status_code=201)
def create_participant(body: ParticipantIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    if db.scalars(select(Participant).where(Participant.study_code == body.study_code)).first():
        raise HTTPException(409, f"{body.study_code} already exists.")
    p = Participant(**body.model_dump(exclude={"patient_password"}))
    if body.patient_password:
        if db.scalars(select(User).where(User.username == body.study_code)).first():
            raise HTTPException(409, f"A login named {body.study_code} already exists.")
        u = User(username=body.study_code, password_hash=hash_password(body.patient_password), role="patient")
        db.add(u)
        db.flush()
        p.user_id = u.id
    db.add(p)
    db.flush()
    audit(db, user, "create", "participant", p.id, study_code=p.study_code, arm=p.arm)
    db.commit()
    return view(db, p)


@router.get("/{participant_id}")
def get_participant(participant_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    return view(db, get_participant_for(db, participant_id, user))


@router.patch("/{participant_id}")
def patch_participant(participant_id: int, body: ParticipantPatch, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    p = get_participant_for(db, participant_id, user)
    changes = body.model_dump(exclude_none=True)
    for k, v in changes.items():
        setattr(p, k, v)
    audit(db, user, "update", "participant", p.id, **changes)
    db.commit()
    return view(db, p)


@router.get("/{participant_id}/sessions")
def participant_sessions(participant_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    p = get_participant_for(db, participant_id, user)
    rows = db.scalars(select(TherapySession).where(TherapySession.participant_id == p.id).order_by(TherapySession.session_no))
    return [session_summary(s) for s in rows]


@router.get("/{participant_id}/adaptive")
def participant_adaptive(participant_id: int, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    p = get_participant_for(db, participant_id, user)
    rows = db.scalars(select(AdaptiveEvent).where(AdaptiveEvent.participant_id == p.id).order_by(AdaptiveEvent.id))
    return [
        {
            "id": e.id,
            "session_id": e.session_id,
            "trial_id": e.trial_id,
            "computed_at": e.computed_at.isoformat(),
            "old_ms": e.old_ms,
            "new_ms": e.new_ms,
            "rule": e.rule,
            "n": e.n,
            "accuracy": e.accuracy,
            "mean_response_ms": e.mean_response_ms,
            "timeout_rate": e.timeout_rate,
        }
        for e in rows
    ]


@router.get("/{participant_id}/gamification")
def participant_gamification(participant_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    p = get_participant_for(db, participant_id, user)
    events = list(db.scalars(select(GamificationEvent).where(GamificationEvent.participant_id == p.id, GamificationEvent.kind != "points").order_by(GamificationEvent.id)))
    return {
        "total_points": p.total_points,
        "level": p.level,
        "badges": [{"code": e.code, "at": e.created_at.isoformat()} for e in events if e.kind == "badge"],
        "levels": [{"level": e.value, "at": e.created_at.isoformat()} for e in events if e.kind == "level"],
    }
