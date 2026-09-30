import random

import pytest

from app.config import GanvaConfig
from app.ganva import (
    best_span_similarity,
    calculate_similarity,
    ganva_classify,
    levenshtein_distance,
    normalize_arabic_text,
)

CFG = GanvaConfig()


# --- S3-1 normalisation -------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("مَدْرَسَةٌ", "مدرسه"),  # harakat + tanween removed, ta marbuta -> ha
        ("أسد", "اسد"),  # alef hamza above
        ("إبرة", "ابره"),  # alef hamza below
        ("آلة", "اله"),  # alef madda
        ("ٱلبيت", "البيت"),  # alef wasla U+0671
        ("مستشفى", "مستشفي"),  # alef maqsura
        ("طائرة", "طايره"),  # yeh-hamza -> yeh
        ("مؤمن", "مومن"),  # waw-hamza via NFD
        ("شدّة", "شده"),  # shadda
        ("مـهـرج", "مهرج"),  # tatweel
        ("  مهرج   كبير ", "مهرج كبير"),  # trim + collapse
        ("مهرج.", "مهرج"),  # punctuation
        ("مهرج؟", "مهرج"),
        ("‏مهرج‎", "مهرج"),  # direction marks
        ("هذا، مهرج", "هذا مهرج"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize_arabic_text(raw) == expected


def test_normalize_idempotent():
    for w in ["مَدْرَسَةٌ", "أسد", "طائرة", "ٱلبيت"]:
        once = normalize_arabic_text(w)
        assert normalize_arabic_text(once) == once


# --- S3-2 similarity ----------------------------------------------------------

def _lev_reference(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def test_levenshtein_matches_reference():
    rng = random.Random(7)
    alphabet = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
    for _ in range(500):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 9)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 9)))
        assert levenshtein_distance(a, b) == _lev_reference(a, b)


def test_similarity_formula():
    assert calculate_similarity("مهرج", "مهرج") == 1.0
    assert calculate_similarity("مدرسة", "مدرسه") == 1.0  # equal after normalisation
    assert calculate_similarity("", "") == 1.0
    assert calculate_similarity("مهرج", "") == 0.0
    # one substitution in 4 letters -> 0.75
    assert calculate_similarity("مهرج", "مهرح") == pytest.approx(0.75)
    # one deletion from 5 letters -> 0.8
    assert calculate_similarity("ستاره", "ستار") == pytest.approx(0.8)


def test_best_span_handles_carrier_phrase():
    m = best_span_similarity("هذا مهرج", "مهرج")
    assert m.similarity == 1.0 and m.span == "مهرج"
    full = best_span_similarity("هذا مهرج", "مهرج", multiword_rule="full_only")
    assert full.similarity < 0.7


# --- S3-3 classification ------------------------------------------------------

def test_stage1_accept():
    r = ganva_classify("مهرج", 0.92, "مهرج", [], CFG)
    assert (r.decision, r.reason, r.stage_reached) == ("ACCEPT", "stage1_accept", 1)


def test_stage1_reject_below_050():
    r = ganva_classify("كرسي", 0.95, "مهرج", ["بهلوان"], CFG)
    assert (r.decision, r.reason) == ("REJECT", "stage1_reject")
    assert r.stage1_similarity < 0.5


def test_stage2_accept_and_reject():
    # target ستارة vs plural ستاير: similarity 0.6 -> Stage 2 (morphological variant)
    target, said = "ستارة", "ستاير"
    sim = best_span_similarity(said, target).similarity
    assert 0.5 <= sim < 0.7
    ok = ganva_classify(said, 0.9, target, ["ستاير"], CFG)
    assert (ok.decision, ok.reason, ok.stage_reached) == ("ACCEPT", "stage2_accept", 2)
    assert ok.matched_associate == "ستاير"
    no = ganva_classify(said, 0.9, target, ["شباك"], CFG)
    assert (no.decision, no.reason) == ("REJECT", "stage2_reject")


def test_flag_low_confidence_keeps_shadow_decision():
    r = ganva_classify("مهرج", 0.55, "مهرج", [], CFG)
    assert r.decision == "FLAG" and r.reason == "low_confidence"
    assert r.shadow_decision == "ACCEPT"
    assert r.stage1_similarity == 1.0


def test_confidence_threshold_boundary():
    assert ganva_classify("مهرج", 0.70, "مهرج", [], CFG).decision == "ACCEPT"
    assert ganva_classify("مهرج", 0.6999, "مهرج", [], CFG).decision == "FLAG"


def test_flag_missing_confidence_and_asr_error():
    assert ganva_classify("مهرج", None, "مهرج", [], CFG).reason == "no_confidence"
    err = ganva_classify(None, None, "مهرج", [], CFG, asr_ok=False)
    assert (err.decision, err.reason, err.shadow_decision) == ("FLAG", "asr_error", None)


def test_empty_transcript_rejected():
    r = ganva_classify("  ", 0.9, "مهرج", [], CFG)
    assert (r.decision, r.reason) == ("REJECT", "no_speech")


def test_result_is_serialisable_and_versioned():
    d = ganva_classify("مهرج", 0.9, "مهرج", [], CFG).as_dict()
    assert d["ganva_version"] and d["t_ganva_ms"] >= 0
