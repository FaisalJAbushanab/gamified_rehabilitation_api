"""
Management commands.

    python -m app.cli create-user --username admin --role admin
    python -m app.cli import-stimuli --csv seed/stimuli_s2.csv --pictures "../Gulfian Arabic/Pictures"
    python -m app.cli import-cues --dir /path/to/cue_files
    python -m app.cli demo-content          # dev only: placeholder cues so sessions can run
    python -m app.cli content-check

Cue file naming for import-cues (one folder, any mix):
    A-N01_L1.txt   Gulf Arabic semantic cue script (UTF-8)
    A-N01_L1.m4a   semantic cue recording
    A-N01_L2.m4a   phonemic cue recording (initial sound)
    A-N01_L3.jpg   visual cue picture (salient feature)
    A-N01_L5.m4a   full auditory model recording
Audio: .m4a .mp3 .wav .webm .aac .ogg   Images: .jpg .jpeg .png .webp
"""
from __future__ import annotations

import argparse
import csv
import getpass
import re
import subprocess
import sys
from pathlib import Path

from sqlalchemy import select

from . import storage
from .config import get_settings
from .db import SessionLocal
from .models import CUE_TYPES, CueAsset, Stimulus, User
from .routers.stimuli import AUDIO_TYPES, IMAGE_TYPES
from .security import hash_password
from .session_engine import content_problems

PSYCHO_FIELDS = [
    "syllables", "phonemes", "frequency_pm", "frequency_source", "rt_ms", "name_agreement",
    "visual_complexity", "image_agreement", "imageability", "familiarity", "aoa",
]


def _cue(db, stim: Stimulus, level: int) -> CueAsset:
    c = next((c for c in stim.cues if c.level == level), None)
    if c is None:
        c = CueAsset(stimulus_id=stim.id, level=level, cue_type=CUE_TYPES[level])
        stim.cues.append(c)
    return c


def create_user(args) -> None:
    password = args.password or getpass.getpass("Password (min 8 characters): ")
    if len(password) < 8:
        sys.exit("Password must be at least 8 characters.")
    with SessionLocal() as db:
        if db.scalars(select(User).where(User.username == args.username)).first():
            sys.exit(f"User {args.username} already exists.")
        db.add(User(username=args.username, password_hash=hash_password(password), role=args.role))
        db.commit()
    print(f"created {args.role} {args.username}")


def _find_picture(pictures: Path, english: str) -> Path | None:
    key = english.lower().replace(" ", "")
    for sub in ("Gulf Nouns", "Gulf Verbs"):
        folder = pictures / sub
        if folder.is_dir():
            for f in folder.iterdir():
                if f.stem.lower().replace(" ", "") == key and f.suffix.lower() in IMAGE_TYPES:
                    return f
    return None


def import_stimuli(args) -> None:
    pictures = Path(args.pictures) if args.pictures else None
    missing_pictures = []
    with SessionLocal() as db:
        existing = {s.code: s for s in db.scalars(select(Stimulus))}
        n = 0
        for r in csv.DictReader(open(args.csv, encoding="utf-8")):
            s = existing.get(r["code"]) or Stimulus(code=r["code"])
            s.word_list, s.word_class, s.number = r["list"], r["word_class"], int(r["number"])
            s.english, s.target, s.grammatical = r["english"], r["target"], r["grammatical"]
            s.phonemic_cue_label, s.semantic_cue_en = r["phonemic_cue_label"], r["semantic_cue_en"]
            s.psycholinguistic = {k: r.get(k) for k in PSYCHO_FIELDS}
            s.probe_only = r["list"] == "B"
            if s.id is None:
                db.add(s)
                db.flush()
            if pictures:
                pic = _find_picture(pictures, r["english"])
                if pic:
                    s.image_path = storage.save_media(pic.read_bytes(), "pictures", pic.suffix)
                else:
                    missing_pictures.append(f'{r["code"]} {r["english"]}')
            if s.word_list == "A":
                # L2 phonemic: the letter in brackets, e.g. "m (م)" -> "م"
                m = re.search(r"\(([^)]+)\)", s.phonemic_cue_label or "")
                c = _cue(db, s, 2)
                if not c.text and m:
                    c.text = m.group(1)
            n += 1
        db.commit()
    print(f"imported {n} items")
    if missing_pictures:
        print("no picture found for: " + ", ".join(missing_pictures))


def import_cues(args) -> None:
    folder = Path(args.dir)
    pattern = re.compile(r"^([AB]-[NV]\d{2})_L([1-5])$", re.I)
    done, skipped = 0, []
    with SessionLocal() as db:
        by_code = {s.code.upper(): s for s in db.scalars(select(Stimulus))}
        for f in sorted(folder.iterdir()):
            m = pattern.match(f.stem)
            suffix = f.suffix.lower()
            if not m or m.group(1).upper() not in by_code:
                skipped.append(f.name)
                continue
            stim, level = by_code[m.group(1).upper()], int(m.group(2))
            c = _cue(db, stim, level)
            if suffix == ".txt":
                c.text = f.read_text(encoding="utf-8-sig").strip()
            elif suffix in AUDIO_TYPES:
                c.audio_path = storage.save_media(f.read_bytes(), "cues", suffix)
            elif suffix in IMAGE_TYPES:
                c.image_path = storage.save_media(f.read_bytes(), "cues", suffix)
            else:
                skipped.append(f.name)
                continue
            done += 1
        db.commit()
    print(f"imported {done} cue files")
    if skipped:
        print("skipped (name or type not recognised): " + ", ".join(skipped))


def _tone_wav(seconds: float, freq: int) -> bytes:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "t.wav"
        subprocess.run(
            [storage.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             f"sine=frequency={freq}:duration={seconds}", "-ar", "16000", "-ac", "1", str(out)],
            check=True,
        )
        return out.read_bytes()


def demo_content(args) -> None:
    if get_settings().env == "prod":
        sys.exit("demo-content is disabled in production.")
    tones = {1: _tone_wav(0.6, 440), 2: _tone_wav(0.3, 660), 5: _tone_wav(0.5, 550)}
    filled = 0
    with SessionLocal() as db:
        for s in db.scalars(select(Stimulus).where(Stimulus.word_list == "A")):
            if not s.image_path:
                s.image_path = storage.save_media(_placeholder_png(), "pictures", ".png")
                filled += 1
            c1 = _cue(db, s, 1)
            if not c1.text:
                c1.text = f"[DEMO – needs Gulf Arabic script] {s.semantic_cue_en}"
                filled += 1
            for level in (1, 2, 5):
                c = _cue(db, s, level)
                if not c.audio_path:
                    c.audio_path = storage.save_media(tones[level], "cues", ".wav")
                    filled += 1
            c3 = _cue(db, s, 3)
            if not c3.image_path and s.image_path:
                c3.image_path = s.image_path
                filled += 1
        db.commit()
    print(f"filled {filled} placeholder assets (DEMO ONLY; replace before clinical use)")


def _placeholder_png() -> bytes:
    import base64

    # 1x1 grey PNG
    return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def check(args) -> None:
    with SessionLocal() as db:
        problems = content_problems(db)
    if not problems:
        print("all trained items have complete cue materials")
        return
    for p in problems:
        print(f'{p["code"]} {p["english"]} ({p["target"]}): ' + "; ".join(p["missing"]))
    print(f"\n{len(problems)} of 30 trained items are incomplete")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m app.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("create-user")
    p.add_argument("--username", required=True)
    p.add_argument("--role", required=True, choices=["admin", "clinician", "rater", "patient"])
    p.add_argument("--password")
    p.set_defaults(fn=create_user)
    p = sub.add_parser("import-stimuli")
    p.add_argument("--csv", default="seed/stimuli_s2.csv")
    p.add_argument("--pictures")
    p.set_defaults(fn=import_stimuli)
    p = sub.add_parser("import-cues")
    p.add_argument("--dir", required=True)
    p.set_defaults(fn=import_cues)
    sub.add_parser("demo-content").set_defaults(fn=demo_content)
    sub.add_parser("content-check").set_defaults(fn=check)
    from .demo_validation import run as demo_validation

    p = sub.add_parser("demo-validation", help="dev only: synthetic rated validation sample")
    p.add_argument("--reset", action="store_true", help="remove existing demo data first")
    p.set_defaults(fn=demo_validation)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
