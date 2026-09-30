"""Flag review queue, attempt audio playback, and blinded validation rating."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import storage, validation
from ..db import get_db
from ..models import Attempt, FlagReview, User, ValidationItem, ValidationRating, ValidationSample
from ..security import STAFF, audit, require_roles
from ..session_engine import attempt_view

router = APIRouter(prefix="/api", tags=["review"])


# --- audio --------------------------------------------------------------------

@router.get("/attempts/{attempt_id}/audio")
def attempt_audio(attempt_id: int, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF, "rater"))):
    a = db.get(Attempt, attempt_id)
    if a is None or not a.audio_ref:
        raise HTTPException(404, "No recording is stored for this response (no consent for retention, or none was made).")
    audit(db, user, "play_audio", "attempt", a.id)
    db.commit()
    return Response(storage.load_encrypted_audio(a.audio_ref), media_type="audio/wav", headers={"Cache-Control": "no-store"})


# --- flags --------------------------------------------------------------------

def flag_view(a: Attempt, review: FlagReview | None) -> dict:
    t = a.trial
    return {
        **attempt_view(a),
        "trial_id": t.id,
        "session_id": t.session_id,
        "session_no": t.session.session_no,
        "study_code": t.session.participant.study_code,
        "code": t.stimulus.code,
        "target": t.stimulus.target,
        "english": t.stimulus.english,
        "image_url": storage.media_url(t.stimulus.image_path),
        "created_at": a.created_at.isoformat(),
        "review": {"decision": review.human_decision, "note": review.note, "at": review.reviewed_at.isoformat()} if review else None,
    }


@router.get("/flags")
def flags(status: Literal["pending", "reviewed", "all"] = "pending", db: Session = Depends(get_db), _: User = Depends(require_roles(*STAFF))):
    reviews = {r.attempt_id: r for r in db.scalars(select(FlagReview))}
    rows = db.scalars(select(Attempt).where(Attempt.ganva_decision == "FLAG").order_by(Attempt.id.desc()).limit(1000))
    out = []
    for a in rows:
        r = reviews.get(a.id)
        if status == "pending" and r is not None:
            continue
        if status == "reviewed" and r is None:
            continue
        out.append(flag_view(a, r))
    return out


class ReviewIn(BaseModel):
    decision: Literal["ACCEPT", "REJECT"]
    note: str = ""


@router.post("/flags/{attempt_id}/review")
def review_flag(attempt_id: int, body: ReviewIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    a = db.get(Attempt, attempt_id)
    if a is None or a.ganva_decision != "FLAG":
        raise HTTPException(404, "Flagged response not found.")
    r = db.scalars(select(FlagReview).where(FlagReview.attempt_id == a.id)).first()
    if r is None:
        r = FlagReview(attempt_id=a.id, reviewer_id=user.id, human_decision=body.decision, note=body.note)
        db.add(r)
    else:
        r.human_decision, r.note, r.reviewer_id = body.decision, body.note, user.id
    audit(db, user, "review_flag", "attempt", a.id, decision=body.decision)
    db.commit()
    return flag_view(a, r)


# --- validation ---------------------------------------------------------------

class SampleIn(BaseModel):
    seed: int
    items_per_session: int = Field(default=15, ge=1, le=60)
    second_rater_fraction: float = Field(default=0.10, ge=0, le=1)
    description: str = ""


def sample_view(db: Session, s: ValidationSample) -> dict:
    items = list(db.scalars(select(ValidationItem).where(ValidationItem.sample_id == s.id)))
    ids = [i.id for i in items]
    ratings = list(db.scalars(select(ValidationRating).where(ValidationRating.item_id.in_(ids)))) if ids else []
    return {
        "id": s.id,
        "created_at": s.created_at.isoformat(),
        "seed": s.seed,
        "items_per_session": s.items_per_session,
        "second_rater_fraction": s.second_rater_fraction,
        "description": s.description,
        "items": len(items),
        "second_rater_items": sum(i.needs_second for i in items),
        "rated_by_1": sum(r.rater_no == 1 for r in ratings),
        "rated_by_2": sum(r.rater_no == 2 for r in ratings),
        "consensus": sum(r.rater_no == 0 for r in ratings),
    }


@router.get("/validation/samples")
def list_samples(db: Session = Depends(get_db), _: User = Depends(require_roles(*STAFF, "rater"))):
    return [sample_view(db, s) for s in db.scalars(select(ValidationSample).order_by(ValidationSample.id.desc()))]


@router.post("/validation/samples", status_code=201)
def create_sample(body: SampleIn, db: Session = Depends(get_db), user: User = Depends(require_roles("admin"))):
    s = validation.create_sample(db, user.id, body.seed, body.items_per_session, body.second_rater_fraction, body.description)
    audit(db, user, "create_sample", "validation", s.id, seed=body.seed)
    db.commit()
    return sample_view(db, s)


@router.get("/validation/samples/{sample_id}/next")
def next_item(sample_id: int, rater_no: int = Query(ge=0, le=2), db: Session = Depends(get_db), _: User = Depends(require_roles("rater", "admin"))):
    """Blinded: returns only the picture, the target and the recording. No transcript, no GANVA output, no participant details."""
    it = validation.next_item_for_rater(db, sample_id, rater_no)
    if it is None:
        return {"item": None}
    a = db.get(Attempt, it.attempt_id)
    stim = a.trial.stimulus
    remaining = len(list(db.scalars(select(ValidationItem.id).where(ValidationItem.sample_id == sample_id))))
    return {
        "item": {
            "item_id": it.id,
            "attempt_id": a.id,
            "target": stim.target,
            "english": stim.english,
            "image_url": storage.media_url(stim.image_path),
            "has_audio": bool(a.audio_ref),
        },
        "sample_items": remaining,
    }


class RatingIn(BaseModel):
    item_id: int
    rater_no: Literal[0, 1, 2]
    score: Literal["correct", "incorrect"]


@router.post("/validation/ratings", status_code=201)
def rate(body: RatingIn, db: Session = Depends(get_db), user: User = Depends(require_roles("rater", "admin"))):
    it = db.get(ValidationItem, body.item_id)
    if it is None:
        raise HTTPException(404, "Item not found.")
    if body.rater_no == 2 and not it.needs_second:
        raise HTTPException(409, "This item is not in the second-rater subsample.")
    other = db.scalars(select(ValidationRating).where(ValidationRating.item_id == it.id, ValidationRating.rater_no != body.rater_no)).all()
    if body.rater_no in (1, 2) and any(r.rater_id == user.id and r.rater_no in (1, 2) for r in other):
        raise HTTPException(409, "The second rating must come from a different rater.")
    existing = db.scalars(select(ValidationRating).where(ValidationRating.item_id == it.id, ValidationRating.rater_no == body.rater_no)).first()
    if existing:
        raise HTTPException(409, "This item already has a rating from that rater slot.")
    db.add(ValidationRating(item_id=it.id, rater_id=user.id, rater_no=body.rater_no, score=body.score))
    db.commit()
    return {"ok": True}


@router.get("/validation/samples/{sample_id}/metrics")
def metrics(sample_id: int, db: Session = Depends(get_db), _: User = Depends(require_roles(*STAFF))):
    if db.get(ValidationSample, sample_id) is None:
        raise HTTPException(404, "Sample not found.")
    return validation.compute_metrics(db, sample_id)
