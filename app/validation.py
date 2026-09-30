"""
GANVA validation against blinded human scoring (Supplementary S3).

Sampling: for every attended session, a random sample of ``items_per_session``
distinct trained items, balanced across nouns and verbs; every response attempt
in the sampled trials is rated. A stratified ``second_rater_fraction`` of items
(largest-remainder allocation across participant x session) is double-rated.

Gold standard: the consensus score (rater_no 0) when present, else rater 1.
Positive class = "correct".

Flagged attempts are reported under three explicit policies:
  exclude    FLAG attempts removed
  shadow     GANVA's Stage 1/2 decision ignoring the confidence check
  effective  the decision the session actually used (clinician or default)
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Attempt, Participant, Stimulus, TherapySession, Trial, ValidationItem, ValidationRating, ValidationSample

POLICIES = ("exclude", "shadow", "effective")


# --- sampling -----------------------------------------------------------------

def _balanced_pick(trials: list[Trial], k: int, rng: random.Random) -> list[Trial]:
    by_class: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_class[t.stimulus.word_class].append(t)
    for lst in by_class.values():
        rng.shuffle(lst)
    classes = sorted(by_class)
    rng.shuffle(classes)
    picked: list[Trial] = []
    while len(picked) < k and any(by_class.values()):
        for c in classes:
            if by_class[c] and len(picked) < k:
                picked.append(by_class[c].pop())
    return picked


def create_sample(db: Session, user_id: int, seed: int, items_per_session: int = 15, second_fraction: float = 0.10, description: str = "") -> ValidationSample:
    rng = random.Random(seed)
    sample = ValidationSample(created_by=user_id, seed=seed, items_per_session=items_per_session, second_rater_fraction=second_fraction, description=description)
    db.add(sample)
    db.flush()
    sessions = list(
        db.scalars(select(TherapySession).where(TherapySession.status.in_(("completed", "abandoned"))).order_by(TherapySession.id))
    )
    groups: dict[tuple[int, int], list[ValidationItem]] = defaultdict(list)
    for s in sessions:
        first_per_item: dict[int, Trial] = {}
        for t in s.trials:
            if t.status == "completed" and t.stimulus_id not in first_per_item:
                first_per_item[t.stimulus_id] = t
        for t in _balanced_pick(list(first_per_item.values()), items_per_session, rng):
            for a in t.attempts:
                if a.kind == "response":
                    item = ValidationItem(sample_id=sample.id, attempt_id=a.id, trial_id=t.id)
                    db.add(item)
                    groups[(s.participant_id, s.id)].append(item)
    db.flush()
    total = sum(len(v) for v in groups.values())
    target = round(second_fraction * total)
    quotas = {g: second_fraction * len(v) for g, v in groups.items()}
    alloc = {g: math.floor(q) for g, q in quotas.items()}
    remainder = sorted(groups, key=lambda g: (quotas[g] - alloc[g], rng.random()), reverse=True)
    for g in remainder[: max(0, target - sum(alloc.values()))]:
        alloc[g] += 1
    for g, items in groups.items():
        for item in rng.sample(items, min(alloc[g], len(items))):
            item.needs_second = True
    return sample


def next_item_for_rater(db: Session, sample_id: int, rater_no: int) -> ValidationItem | None:
    rated = select(ValidationRating.item_id).where(ValidationRating.rater_no == rater_no)
    q = select(ValidationItem).where(ValidationItem.sample_id == sample_id, ValidationItem.id.not_in(rated))
    if rater_no == 2:
        q = q.where(ValidationItem.needs_second.is_(True))
    if rater_no == 0:  # consensus only where raters 1 and 2 disagree
        items = list(db.scalars(q.where(ValidationItem.needs_second.is_(True))))
        for it in items:
            scores = {r.rater_no: r.score for r in db.scalars(select(ValidationRating).where(ValidationRating.item_id == it.id))}
            if 1 in scores and 2 in scores and scores[1] != scores[2]:
                return it
        return None
    return db.scalars(q.order_by(ValidationItem.id)).first()


# --- metrics ------------------------------------------------------------------

def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs of (ganva_positive, human_positive)."""
    tp = sum(g and h for g, h in pairs)
    fp = sum(g and not h for g, h in pairs)
    fn = sum((not g) and h for g, h in pairs)
    tn = sum((not g) and (not h) for g, h in pairs)
    n = len(pairs)
    assert tp + fp + fn + tn == n, "confusion counts must sum to the number of trials"
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else None
    spec = tn / (tn + fp) if tn + fp else None
    acc = (tp + tn) / n if n else None
    rd = lambda x: round(x, 3) if x is not None else None  # noqa: E731
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": rd(p), "recall": rd(r), "f1": rd(f1), "specificity": rd(spec), "accuracy": rd(acc)}


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def kappa_with_ci(a: list[bool], b: list[bool], seed: int = 2026, reps: int = 2000) -> dict:
    k = cohen_kappa(a, b)
    n = len(a)
    agreement = round(sum(x == y for x, y in zip(a, b)) / n, 4) if n else None
    if k is None or n < 2:
        # kappa is undefined when both raters used one category only
        return {"n": n, "agreement": agreement, "kappa": k, "ci95": None}
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        kb = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        if kb is not None:
            boots.append(kb)
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    return {"n": n, "agreement": agreement, "kappa": round(k, 3), "ci95": [round(lo, 3), round(hi, 3)]}


def _ganva_positive(a: Attempt, policy: str, threshold: float | None = None) -> bool | None:
    flagged = a.ganva_decision == "FLAG"
    if threshold is not None:  # sensitivity analysis: re-derive the flag from stored confidence
        flagged = a.asr_error != "" or a.asr_confidence is None or a.asr_confidence < threshold
    if not flagged:
        return (a.shadow_decision or a.ganva_decision) == "ACCEPT"
    if policy == "exclude":
        return None
    if policy == "shadow":
        return a.shadow_decision == "ACCEPT" if a.shadow_decision else False
    return a.effective_decision == "ACCEPT"


def compute_metrics(db: Session, sample_id: int) -> dict:
    items = list(db.scalars(select(ValidationItem).where(ValidationItem.sample_id == sample_id)))
    ratings: dict[int, dict[int, str]] = defaultdict(dict)
    for r in db.scalars(select(ValidationRating).join(ValidationItem).where(ValidationItem.sample_id == sample_id)):
        ratings[r.item_id][r.rater_no] = r.score
    rows = []
    for it in items:
        sc = ratings.get(it.id, {})
        gold = sc.get(0) or sc.get(1)
        if gold is None:
            continue
        a = db.get(Attempt, it.attempt_id)
        t = a.trial
        p: Participant = t.session.participant
        stim: Stimulus = t.stimulus
        rows.append({"a": a, "human": gold == "correct", "participant": p, "stim": stim, "first": a.cue_level == 0 and not a.is_repetition})

    def table(rs, policy, threshold=None):
        pairs = []
        for r in rs:
            g = _ganva_positive(r["a"], policy, threshold)
            if g is not None:
                pairs.append((g, r["human"]))
        return confusion(pairs)

    def flag_rate(rs, threshold=None):
        if not rs:
            return None
        if threshold is None:
            f = sum(r["a"].ganva_decision == "FLAG" for r in rs)
        else:
            f = sum(r["a"].asr_error != "" or r["a"].asr_confidence is None or r["a"].asr_confidence < threshold for r in rs)
        return round(f / len(rs), 4)

    def block(rs):
        return {"flag_rate": flag_rate(rs), **{pol: table(rs, pol) for pol in POLICIES}}

    subgroups = {
        "all": rows,
        "non_dysarthric": [r for r in rows if not r["participant"].dysarthria],
        "dysarthric": [r for r in rows if r["participant"].dysarthria],
        "nouns": [r for r in rows if r["stim"].word_class == "noun"],
        "verbs": [r for r in rows if r["stim"].word_class == "verb"],
    }
    for sev in ("mild", "moderate", "severe", "profound"):
        subgroups[f"severity_{sev}"] = [r for r in rows if r["participant"].severity == sev]

    per_participant = {}
    for code in sorted({r["participant"].study_code for r in rows}):
        rs = [r for r in rows if r["participant"].study_code == code]
        per_participant[code] = {"severity": rs[0]["participant"].severity, "dysarthria": rs[0]["participant"].dysarthria, **block(rs)}

    borderline = [r for r in rows if r["a"].stage_reached == 2]
    stage2 = {
        "borderline_attempts": len(borderline),
        "resolved_by_associate": sum(r["a"].ganva_reason == "stage2_accept" or (r["a"].shadow_decision == "ACCEPT" and r["a"].stage_reached == 2) for r in borderline),
        "metrics_shadow": table(borderline, "shadow"),
    }
    sensitivity = {
        f"{int(th * 100)}%": {"flag_rate": flag_rate(rows, th), "exclude": table(rows, "exclude", th)} for th in (0.6, 0.7, 0.8)
    }

    double = [(ratings[it.id][1], ratings[it.id][2]) for it in items if it.needs_second and 1 in ratings.get(it.id, {}) and 2 in ratings.get(it.id, {})]
    reliability = kappa_with_ci([x == "correct" for x, _ in double], [y == "correct" for _, y in double])
    reliability["double_rated_items_planned"] = sum(it.needs_second for it in items)

    return {
        "sample_id": sample_id,
        "items": len(items),
        "rated": len(rows),
        "unit": "attempt",
        "subgroups": {k: block(v) for k, v in subgroups.items()},
        "first_presentation_only": block([r for r in rows if r["first"]]),
        "per_participant": per_participant,
        "stage2_isolation": stage2,
        "confidence_sensitivity": sensitivity,
        "inter_rater": reliability,
    }
