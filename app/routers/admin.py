"""Configuration view and pseudonymised CSV exports."""
import csv
import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from ..config import ADAPTIVE_VERSION, APP_VERSION, GANVA_VERSION, STUDY, config_hash, get_settings
from ..db import get_db
from ..models import (
    AdaptiveEvent,
    Attempt,
    FlagReview,
    GamificationEvent,
    Participant,
    Stimulus,
    TherapySession,
    Trial,
    User,
    ValidationItem,
    ValidationRating,
)
from ..security import STAFF, audit, current_user, require_roles

router = APIRouter(prefix="/api", tags=["admin"])


@router.get("/config")
def get_config(_: User = Depends(current_user)):
    s = get_settings()
    return {
        "app_version": APP_VERSION,
        "ganva_version": GANVA_VERSION,
        "adaptive_version": ADAPTIVE_VERSION,
        "config_hash": config_hash(),
        "study": STUDY.model_dump(mode="json"),
        "asr": {"provider": s.asr_provider, "fallback": s.asr_fallback, "locale": s.asr_locale},
        "env": s.env,
        "manual_asr": s.asr_provider == "manual" and s.env != "prod",
        "require_complete_content": s.require_complete_content,
    }


# Exports never include usernames, password hashes, notes or audio references.
EXPORTS = {
    "participants": (Participant, {"user_id", "notes"}),
    "stimuli": (Stimulus, set()),
    "sessions": (TherapySession, {"config_snapshot", "operator_id"}),
    "trials": (Trial, set()),
    "attempts": (Attempt, {"audio_ref", "override_by"}),
    "adaptive_events": (AdaptiveEvent, set()),
    "flag_reviews": (FlagReview, {"reviewer_id"}),
    "gamification_events": (GamificationEvent, set()),
    "validation_items": (ValidationItem, set()),
    "validation_ratings": (ValidationRating, set()),
}


@router.get("/export/{name}.csv")
def export(name: str, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    if name not in EXPORTS:
        raise HTTPException(404, f"Unknown export. Available: {', '.join(EXPORTS)}.")
    model, excluded = EXPORTS[name]
    cols = [c.key for c in inspect(model).columns if c.key not in excluded]
    codes = {p.id: p.study_code for p in db.scalars(select(Participant))}
    buf = io.StringIO()
    w = csv.writer(buf)
    header = cols + (["study_code"] if "participant_id" in cols else [])
    w.writerow(header)
    for row in db.scalars(select(model).order_by(model.id)):
        vals = [getattr(row, c) for c in cols]
        if "participant_id" in cols:
            vals.append(codes.get(row.participant_id, ""))
        w.writerow(["" if v is None else v for v in vals])
    audit(db, user, "export", name)
    db.commit()
    return StreamingResponse(
        iter(["﻿" + buf.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'},
    )
