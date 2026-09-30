import csv
import io
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import storage
from ..db import get_db
from ..models import CUE_TYPES, Associate, CueAsset, Stimulus, User
from ..security import STAFF, audit, require_roles
from ..content import is_placeholder_file
from ..session_engine import content_problems, cue_part_status

router = APIRouter(prefix="/api/stimuli", tags=["stimuli"])

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_TYPES = {".m4a", ".mp3", ".wav", ".webm", ".mp4", ".aac", ".ogg"}


def cue_view(c: CueAsset, s: Stimulus) -> dict:
    return {
        "level": c.level,
        "type": c.cue_type,
        "text": c.text,
        "audio_url": storage.media_url(c.audio_path),
        "image_url": storage.media_url(c.image_path),
        # demo placeholders: the content check treats these as missing
        "placeholder_text": cue_part_status(s, c, "text") == "placeholder",
        "placeholder_audio": cue_part_status(s, c, "audio") == "placeholder",
        "placeholder_image": cue_part_status(s, c, "image") == "placeholder",
    }


def view(s: Stimulus, full: bool = False) -> dict:
    d = {
        "id": s.id,
        "code": s.code,
        "list": s.word_list,
        "word_class": s.word_class,
        "number": s.number,
        "english": s.english,
        "target": s.target,
        "grammatical": s.grammatical,
        "phonemic_cue_label": s.phonemic_cue_label,
        "semantic_cue_en": s.semantic_cue_en,
        "image_url": storage.media_url(s.image_path),
        "image_placeholder": is_placeholder_file(s.image_path),
        "probe_only": s.probe_only,
        "active": s.active,
        "associate_count": len(s.associates),
        "cues": [cue_view(c, s) for c in s.cues],
    }
    if full:
        d["psycholinguistic"] = s.psycholinguistic
        d["associates"] = [
            {"id": a.id, "text": a.text, "kind": a.kind, "source": a.source, "raters_agreeing": a.raters_agreeing, "raters_total": a.raters_total}
            for a in s.associates
        ]
    return d


def _get(db: Session, stimulus_id: int) -> Stimulus:
    s = db.get(Stimulus, stimulus_id)
    if s is None:
        raise HTTPException(404, "Item not found.")
    return s


def _suffix(upload: UploadFile, allowed: set[str], what: str) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(422, f"The {what} must be one of: {', '.join(sorted(allowed))}.")
    return suffix


@router.get("")
def list_stimuli(list: str | None = None, db: Session = Depends(get_db), _: User = Depends(require_roles(*STAFF))):
    q = select(Stimulus).order_by(Stimulus.word_list, Stimulus.word_class, Stimulus.number)
    if list:
        q = q.where(Stimulus.word_list == list)
    return [view(s) for s in db.scalars(q)]


@router.get("/content-check")
def content_check(db: Session = Depends(get_db), _: User = Depends(require_roles(*STAFF))):
    problems = content_problems(db)
    return {"complete": not problems, "problems": problems}


@router.get("/{stimulus_id}")
def get_stimulus(stimulus_id: int, db: Session = Depends(get_db), _: User = Depends(require_roles(*STAFF))):
    return view(_get(db, stimulus_id), full=True)


class StimulusPatch(BaseModel):
    target: str | None = Field(default=None, min_length=1, max_length=64)
    active: bool | None = None


@router.patch("/{stimulus_id}")
def patch_stimulus(stimulus_id: int, body: StimulusPatch, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    s = _get(db, stimulus_id)
    changes = body.model_dump(exclude_none=True)
    for k, v in changes.items():
        setattr(s, k, v)
    audit(db, user, "update", "stimulus", s.id, **changes)
    db.commit()
    return view(s, full=True)


@router.put("/{stimulus_id}/picture")
async def put_picture(stimulus_id: int, image: UploadFile = File(...), db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    s = _get(db, stimulus_id)
    s.image_path = storage.save_media(await image.read(), "pictures", _suffix(image, IMAGE_TYPES, "picture"))
    audit(db, user, "upload_picture", "stimulus", s.id)
    db.commit()
    return view(s, full=True)


@router.put("/{stimulus_id}/cues/{level}")
async def put_cue(
    stimulus_id: int,
    level: int,
    text: str | None = Form(None),
    audio: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    clear_audio: bool = Form(False),
    clear_image: bool = Form(False),
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(*STAFF)),
):
    if level not in CUE_TYPES:
        raise HTTPException(422, "Cue level must be 1 to 5.")
    s = _get(db, stimulus_id)
    cue = next((c for c in s.cues if c.level == level), None)
    if cue is None:
        cue = CueAsset(stimulus_id=s.id, level=level, cue_type=CUE_TYPES[level])
        db.add(cue)
    if text is not None:
        cue.text = text.strip()
    if clear_audio:
        cue.audio_path = ""
    if clear_image:
        cue.image_path = ""
    if audio is not None and audio.filename:
        cue.audio_path = storage.save_media(await audio.read(), "cues", _suffix(audio, AUDIO_TYPES, "recording"))
    if image is not None and image.filename:
        cue.image_path = storage.save_media(await image.read(), "cues", _suffix(image, IMAGE_TYPES, "picture"))
    audit(db, user, "update_cue", "stimulus", s.id, level=level)
    db.commit()
    db.refresh(s)
    return view(s, full=True)


class AssociateIn(BaseModel):
    text: str = Field(min_length=1, max_length=128)
    kind: str = "variant"
    source: str = ""
    raters_agreeing: int | None = None
    raters_total: int | None = None


@router.post("/{stimulus_id}/associates", status_code=201)
def add_associate(stimulus_id: int, body: AssociateIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    s = _get(db, stimulus_id)
    if any(a.text == body.text.strip() for a in s.associates):
        raise HTTPException(409, "That associate is already listed for this item.")
    db.add(Associate(stimulus_id=s.id, **{**body.model_dump(), "text": body.text.strip()}))
    audit(db, user, "add_associate", "stimulus", s.id, text=body.text)
    db.commit()
    db.refresh(s)
    return view(s, full=True)


@router.delete("/{stimulus_id}/associates/{associate_id}")
def delete_associate(stimulus_id: int, associate_id: int, db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    a = db.get(Associate, associate_id)
    if a is None or a.stimulus_id != stimulus_id:
        raise HTTPException(404, "Associate not found.")
    audit(db, user, "delete_associate", "stimulus", stimulus_id, text=a.text)
    db.delete(a)
    db.commit()
    return {"ok": True}


@router.post("/associates/import")
async def import_associates(file: UploadFile = File(...), db: Session = Depends(get_db), user: User = Depends(require_roles(*STAFF))):
    """CSV columns: code (e.g. A-N01) or english, associate, kind, source, raters_agreeing, raters_total."""
    text = (await file.read()).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    by_code = {s.code: s for s in db.scalars(select(Stimulus))}
    by_en = {s.english.lower(): s for s in by_code.values()}
    added, skipped, errors = 0, 0, []
    for i, row in enumerate(reader, start=2):
        s = by_code.get((row.get("code") or "").strip()) or by_en.get((row.get("english") or "").strip().lower())
        assoc = (row.get("associate") or "").strip()
        if s is None or not assoc:
            errors.append(f"line {i}: unknown item or empty associate")
            continue
        if any(a.text == assoc for a in s.associates):
            skipped += 1
            continue
        toint = lambda v: int(v) if (v or "").strip().isdigit() else None  # noqa: E731
        s.associates.append(
            Associate(text=assoc, kind=(row.get("kind") or "variant").strip(), source=(row.get("source") or "").strip(),
                      raters_agreeing=toint(row.get("raters_agreeing")), raters_total=toint(row.get("raters_total")))
        )
        added += 1
    audit(db, user, "import_associates", "stimulus", "", added=added, skipped=skipped, errors=len(errors))
    db.commit()
    return {"added": added, "skipped_duplicates": skipped, "errors": errors[:50]}
