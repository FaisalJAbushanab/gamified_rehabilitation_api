"""
DEV ONLY: build a synthetic, fully rated validation sample so the Validation page
and the rater workflow can be explored before real data exists.

Everything goes through the real code paths: session engine, GANVA, adaptive
window, validation sampling and metrics. Only the patient, the speech recogniser
and the human raters are simulated:

- patient: says the target, an approved associate, a near-miss (one letter off),
  a wrong word, or nothing, with probabilities by severity;
- ASR: sometimes garbles a correct answer; confidence is lower for severe and
  dysarthric speech, so some responses are flagged;
- raters: score the true answer, with a small disagreement rate; consensus
  resolves disagreements.

Participants are DM01-DM06, notes say "DEMO - synthetic data". Refused in prod.
"""
from __future__ import annotations

import random
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from sqlalchemy import select

from . import storage
from .config import get_settings
from .db import SessionLocal
from .models import Attempt, Participant, User, ValidationItem, ValidationRating
from .session_engine import end_session, resolve_flag, start_session, submit_response, submit_timeout, trained_stimuli
from .validation import create_sample

PROFILES = [
    # code, severity, dysarthria, P(correct target), P(associate), P(near-miss), P(silence); rest = wrong word
    ("DM01", "mild", False, 0.72, 0.08, 0.06, 0.02),
    ("DM02", "mild", False, 0.68, 0.08, 0.08, 0.03),
    ("DM03", "moderate", False, 0.50, 0.08, 0.12, 0.08),
    ("DM04", "moderate", True, 0.45, 0.06, 0.14, 0.10),
    ("DM05", "severe", False, 0.30, 0.05, 0.15, 0.20),
    ("DM06", "profound", True, 0.18, 0.04, 0.14, 0.30),
]
SESSIONS_PER_PARTICIPANT = 3
ITEMS_PER_SESSION = 12


def _tone() -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "t.wav"
        subprocess.run([storage.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "sine=frequency=240:duration=0.5", "-ar", "16000", "-ac", "1", str(out)], check=True)
        return out.read_bytes()


def _near_miss(word: str, rng: random.Random) -> str:
    letters = list(word)
    i = rng.randrange(len(letters))
    if rng.random() < 0.5 and len(letters) > 2:
        del letters[i]
    else:
        letters[i] = rng.choice("بتدرسعفقلمنهو")
    return "".join(letters)


def _reset(db) -> None:
    """Remove all demo participants, their sessions, attempts, audio and demo samples."""
    from sqlalchemy import delete

    from .models import AdaptiveEvent, FlagReview, GamificationEvent, TherapySession, Trial, ValidationSample

    pids = [p.id for p in db.scalars(select(Participant).where(Participant.study_code.like("DM%"), Participant.notes.like("DEMO%")))]
    sids = list(db.scalars(select(TherapySession.id).where(TherapySession.participant_id.in_(pids))))
    tids = list(db.scalars(select(Trial.id).where(Trial.session_id.in_(sids))))
    attempts = list(db.scalars(select(Attempt).where(Attempt.trial_id.in_(tids))))
    aids = [a.id for a in attempts]
    for a in attempts:
        if a.audio_ref:
            (storage.audio_root() / a.audio_ref).unlink(missing_ok=True)
    samples = [x.id for x in db.scalars(select(ValidationSample).where(ValidationSample.description.like("DEMO%")))]
    items = list(db.scalars(select(ValidationItem.id).where(ValidationItem.sample_id.in_(samples))))
    db.execute(delete(ValidationRating).where(ValidationRating.item_id.in_(items)))
    db.execute(delete(ValidationItem).where(ValidationItem.sample_id.in_(samples)))
    db.execute(delete(ValidationSample).where(ValidationSample.id.in_(samples)))
    db.execute(delete(FlagReview).where(FlagReview.attempt_id.in_(aids)))
    db.execute(delete(AdaptiveEvent).where(AdaptiveEvent.participant_id.in_(pids)))
    db.execute(delete(GamificationEvent).where(GamificationEvent.participant_id.in_(pids)))
    db.execute(delete(Attempt).where(Attempt.id.in_(aids)))
    db.execute(delete(Trial).where(Trial.id.in_(tids)))
    db.execute(delete(TherapySession).where(TherapySession.id.in_(sids)))
    db.execute(delete(Participant).where(Participant.id.in_(pids)))
    db.commit()
    print(f"removed {len(pids)} demo participants, {len(sids)} sessions, {len(aids)} responses, {len(samples)} demo samples")


def run(args=None) -> None:
    s = get_settings()
    if s.env == "prod":
        sys.exit("demo-validation is disabled in production.")
    if s.asr_provider != "manual":
        sys.exit("demo-validation needs ASR_PROVIDER=manual (it simulates the recogniser).")
    rng = random.Random(2026)
    tone = _tone()
    truth: dict[str, bool] = {}  # attempt client_uuid -> did the patient really name it correctly

    with SessionLocal() as db:
        if db.scalars(select(Participant).where(Participant.study_code == "DM01")).first():
            if not getattr(args, "reset", False):
                sys.exit("Demo data already exists (DM01-DM06). Run with --reset to rebuild it.")
            _reset(db)
        operator = db.scalars(select(User).where(User.role.in_(("clinician", "admin")))).first()
        admin = db.scalars(select(User).where(User.role == "admin")).first()
        raters = list(db.scalars(select(User).where(User.role == "rater").order_by(User.id)))
        if not (operator and admin and len(raters) >= 2):
            sys.exit("Needs one admin, one clinician/admin and two rater accounts.")
        targets = [st.target for st in trained_stimuli(db)]

        for code, severity, dys, p_ok, p_assoc, p_near, p_silent in PROFILES:
            p = Participant(study_code=code, arm="GAG", severity=severity, dysarthria=dys,
                            consent_audio_retention=True, notes="DEMO - synthetic data, not a real participant")
            db.add(p)
            db.flush()
            for _ in range(SESSIONS_PER_PARTICIPANT):
                sess = start_session(db, p, operator)
                done = 0
                while done < ITEMS_PER_SESSION:
                    trial = next(t for t in reversed(sess.trials) if t.status == "active")
                    stim = trial.stimulus
                    # a little better with each cue level
                    boost = 0.06 * trial.current_cue_level
                    r = rng.random()
                    assocs = [a.text for a in stim.associates]
                    if r < p_silent:
                        out = submit_timeout(db, sess, trial, client_uuid=str(uuid.uuid4()), cue_level=trial.current_cue_level, presented_at="")
                    else:
                        r = rng.random()
                        if r < p_ok + boost:
                            said, correct = stim.target, True
                        elif r < p_ok + boost + p_assoc and assocs:
                            said, correct = rng.choice(assocs), True
                        elif r < p_ok + boost + p_assoc + p_near:
                            # humans credit about half of phonological near-misses
                            said, correct = _near_miss(stim.target, rng), rng.random() < 0.5
                        else:
                            said, correct = rng.choice([t for t in targets if t != stim.target]), False
                        # simulated ASR: garbles some answers, lower confidence for severe/dysarthric speech
                        heard = _near_miss(said, rng) if rng.random() < (0.25 if dys else 0.10) else said
                        base = {"mild": 0.9, "moderate": 0.84, "severe": 0.78, "profound": 0.72}[severity] - (0.12 if dys else 0)
                        conf = max(0.2, min(0.99, rng.gauss(base, 0.1)))
                        cu = str(uuid.uuid4())
                        truth[cu] = correct
                        out = submit_response(db, sess, trial, client_uuid=cu, cue_level=trial.current_cue_level,
                                              speech_onset_ms=int(rng.uniform(1500, 9000)), presented_at="",
                                              audio=tone, mime="audio/wav", filename="r.wav",
                                              dev_transcript=heard, dev_confidence=round(conf, 2))
                        if out["action"] == "await_clinician":
                            a = db.get(Attempt, out["attempt"]["id"])
                            out = resolve_flag(db, sess, a, "ACCEPT" if correct else "REJECT", operator, "clinician")
                    db.flush()
                    if out["action"] in ("trial_correct", "trial_incorrect"):
                        done += 1
                end_session(db, sess, "demo", "completed")
            db.commit()
            print(f"{code}: {SESSIONS_PER_PARTICIPANT} sessions")

        # draw only from the demo sessions: temporarily hide other ended sessions from the sampler
        from .models import TherapySession

        others = list(db.scalars(select(TherapySession).join(Participant).where(~Participant.study_code.like("DM%"), TherapySession.status.in_(("completed", "abandoned")))))
        saved = {o.id: o.status for o in others}
        for o in others:
            o.status = "hidden_for_demo"
        db.flush()
        sample = create_sample(db, admin.id, seed=48213, items_per_session=6, second_fraction=0.2,
                               description="DEMO - synthetic data (DM01-DM06), for exploring the page only")
        for o in others:
            o.status = saved[o.id]
        db.flush()

        items = list(db.scalars(select(ValidationItem).where(ValidationItem.sample_id == sample.id)))
        disagreements = 0
        for it in items:
            a = db.get(Attempt, it.attempt_id)
            real = truth.get(a.client_uuid, False)
            s1 = real if rng.random() > 0.04 else not real
            db.add(ValidationRating(item_id=it.id, rater_id=raters[0].id, rater_no=1, score="correct" if s1 else "incorrect"))
            if it.needs_second:
                s2 = real if rng.random() > 0.06 else not real
                db.add(ValidationRating(item_id=it.id, rater_id=raters[1].id, rater_no=2, score="correct" if s2 else "incorrect"))
                if s1 != s2:
                    disagreements += 1
                    db.add(ValidationRating(item_id=it.id, rater_id=admin.id, rater_no=0, score="correct" if real else "incorrect"))
        db.commit()
        print(f"sample {sample.id}: {len(items)} responses rated, {sum(i.needs_second for i in items)} double-rated, {disagreements} resolved by consensus")
