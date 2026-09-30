import uuid

import pytest

from app.config import STUDY

from .conftest import login


def make_participant(client, h, code="PE01", arm="GAG", severity="moderate", consent=True, dysarthria=False):
    r = client.post("/api/participants", headers=h, json={
        "study_code": code, "arm": arm, "severity": severity, "dysarthria": dysarthria,
        "consent_audio_retention": consent, "patient_password": "patientpw1",
    })
    assert r.status_code == 201, r.text
    return r.json()


def respond(client, h, state, transcript, confidence=0.9, onset=3000, wav=b""):
    p = state["presentation"]
    r = client.post(
        f"/api/sessions/{state['session']['id']}/trials/{p['trial_id']}/response",
        headers=h,
        data={"client_uuid": str(uuid.uuid4()), "cue_level": p["cue_level"], "speech_onset_ms": onset,
              "presented_at": "2026-10-01T10:00:00Z", "dev_transcript": transcript,
              **({"dev_confidence": confidence} if confidence is not None else {})},
        files={"audio": ("a.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    return r.json()


def timeout(client, h, state):
    p = state["presentation"]
    r = client.post(f"/api/sessions/{state['session']['id']}/trials/{p['trial_id']}/timeout", headers=h,
                    json={"client_uuid": str(uuid.uuid4()), "cue_level": p["cue_level"]})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture()
def running(client, tone_wav):
    h = login(client, "slp")
    client.demo_content()
    p = make_participant(client, h)
    r = client.post("/api/sessions", headers=h, json={"participant_id": p["id"]})
    assert r.status_code == 200, r.text
    return client, h, r.json(), p, tone_wav


def test_session_blocked_until_content_complete(client):
    h = login(client, "slp")
    p = make_participant(client, h)
    r = client.post("/api/sessions", headers=h, json={"participant_id": p["id"]})
    assert r.status_code == 409 and "missing cue materials" in r.json()["detail"]
    chk = client.get("/api/stimuli/content-check", headers=h).json()
    assert not chk["complete"] and len(chk["problems"]) == 30


def test_tsg_participants_have_no_app_sessions(client):
    h = login(client, "slp")
    client.demo_content()
    p = make_participant(client, h, code="PC01", arm="TSG")
    assert client.post("/api/sessions", headers=h, json={"participant_id": p["id"]}).status_code == 409


def test_start_state(running):
    client, h, st, p, _ = running
    pres = st["presentation"]
    assert pres["cue_level"] == 0 and pres["cue"] is None
    assert st["limit_ms"] == 45_000 == pres["window_ms"]  # moderate start
    assert st["session"]["session_no"] == 1
    assert st["session"]["trained_count"] == 30


def test_accept_scores_points_and_moves_on(running):
    client, h, st, p, wav = running
    target = st["presentation"]["target"]
    out = respond(client, h, st, f"هذا {target}", wav=wav)
    assert out["attempt"]["ganva_decision"] == "ACCEPT"
    assert out["action"] == "trial_correct" and out["points"] == 100
    assert out["state"]["presentation"]["trial_id"] != st["presentation"]["trial_id"]
    assert out["state"]["participant"]["total_points"] == 100
    assert out["attempt"]["has_audio"]  # consent -> encrypted audio stored
    assert out["attempt"]["t_server_total_ms"] is not None


def test_full_cue_ladder_then_repetition_then_incorrect(running):
    client, h, st, p, wav = running
    levels = []
    for _ in range(6):
        out = respond(client, h, st, "كلام غلط تماما", wav=wav)
        assert out["attempt"]["ganva_decision"] == "REJECT"
        levels.append(out["action"])
        st = out["state"]
        if out["action"] == "next_cue":
            assert st["presentation"]["cue"]["level"] == st["presentation"]["cue_level"]
    assert levels == ["next_cue"] * 5 + ["repetition"]
    pres = st["presentation"]
    assert pres["cue_level"] == 5 and pres["is_repetition"]
    assert pres["cue"]["type"] == "model"
    out = respond(client, h, st, pres["target"], wav=wav)  # correct repetition still counts as incorrect naming
    assert out["action"] == "trial_incorrect" and out["points"] == 0


def test_written_cue_shows_target(running):
    client, h, st, p, wav = running
    for _ in range(4):
        st = respond(client, h, st, "غلط", wav=wav)["state"]
    cue = st["presentation"]["cue"]
    assert cue["level"] == 4 and cue["type"] == "written" and cue["text"] == st["presentation"]["target"]


def test_flag_then_clinician_decides(running):
    client, h, st, p, wav = running
    target = st["presentation"]["target"]
    out = respond(client, h, st, target, confidence=0.5, wav=wav)
    assert out["action"] == "await_clinician"
    assert out["attempt"]["ganva_decision"] == "FLAG" and out["attempt"]["shadow_decision"] == "ACCEPT"
    aid = out["attempt"]["id"]
    assert out["state"]["presentation"]["awaiting_attempt_id"] == aid
    # no new response accepted while waiting
    r = client.post(f"/api/sessions/{st['session']['id']}/trials/{st['presentation']['trial_id']}/timeout", headers=h,
                    json={"client_uuid": str(uuid.uuid4()), "cue_level": 0})
    assert r.status_code == 409
    d = client.post(f"/api/sessions/{st['session']['id']}/attempts/{aid}/decide", headers=h, json={"decision": "ACCEPT"})
    assert d.status_code == 200 and d.json()["action"] == "trial_correct"
    # decided in-session -> reviewed, not pending
    assert client.get("/api/flags", headers=h).json() == []
    assert len(client.get("/api/flags?status=reviewed", headers=h).json()) == 1


def test_flag_default_keeps_it_in_review_queue(running):
    client, h, st, p, wav = running
    out = respond(client, h, st, "شي", confidence=None, wav=wav)  # no confidence -> FLAG
    assert out["attempt"]["ganva_reason"] == "no_confidence"
    ph = login(client, "PE01", "patientpw1")
    d = client.post(f"/api/sessions/{st['session']['id']}/attempts/{out['attempt']['id']}/default", headers=ph)
    assert d.status_code == 200 and d.json()["action"] == "next_cue"
    pending = client.get("/api/flags", headers=h).json()
    assert len(pending) == 1 and pending[0]["resolution"] == "default"
    rv = client.post(f"/api/flags/{pending[0]['id']}/review", headers=h, json={"decision": "REJECT", "note": "unclear"})
    assert rv.status_code == 200 and client.get("/api/flags", headers=h).json() == []


def test_patient_cannot_decide_flags(running):
    client, h, st, p, wav = running
    out = respond(client, h, st, "شي", confidence=0.3, wav=wav)
    ph = login(client, "PE01", "patientpw1")
    r = client.post(f"/api/sessions/{st['session']['id']}/attempts/{out['attempt']['id']}/decide", headers=ph, json={"decision": "ACCEPT"})
    assert r.status_code == 403


def test_asr_failure_is_flag_not_incorrect(running):
    client, h, st, p, _ = running
    out = respond(client, h, st, "مهرج", wav=b"not audio at all")
    assert out["attempt"]["ganva_decision"] == "FLAG" and out["attempt"]["ganva_reason"] == "asr_error"


def test_timeout_advances_and_counts(running):
    client, h, st, p, wav = running
    out = timeout(client, h, st)
    assert out["action"] == "next_cue" and out["attempt"]["kind"] == "timeout"


def test_adaptive_decreases_after_three_fast_correct_trials(running):
    client, h, st, p, wav = running
    for i in range(3):
        out = respond(client, h, st, st["presentation"]["target"], onset=2000, wav=wav)
        st = out["state"]
        if i < 2:
            assert out["adaptive"]["rule"] == "insufficient_history"
    assert out["adaptive"]["rule"] == "decrease" and out["adaptive"]["new_ms"] == 40_000
    assert st["limit_ms"] == 40_000 == st["presentation"]["window_ms"]
    events = client.get(f"/api/participants/{p['id']}/adaptive", headers=h).json()
    assert [e["rule"] for e in events] == ["initial", "insufficient_history", "insufficient_history", "decrease"]


def test_adaptive_hard_increase_on_timeouts(running):
    client, h, st, p, wav = running
    for _ in range(3):
        for _ in range(6):  # levels 0-5 all time out
            out = timeout(client, h, st)
            st = out["state"]
        out = timeout(client, h, st)  # repetition times out
        assert out["action"] == "trial_incorrect"
        st = out["state"]
    assert out["adaptive"]["rule"] == "hard_increase" and st["limit_ms"] == 55_000


def test_session_ends_at_duration(running, monkeypatch):
    client, h, st, p, wav = running
    monkeypatch.setattr(STUDY.session, "duration_min", 0)
    out = respond(client, h, st, st["presentation"]["target"], wav=wav)
    assert out["session_complete"] and out["state"]["session"]["status"] == "completed"
    assert out["state"]["presentation"] is None
    assert "first_session" in out["badges"]
    nxt = client.post("/api/sessions", headers=h, json={"participant_id": p["id"]}).json()
    assert nxt["session"]["session_no"] == 2


def test_items_are_all_30_trained_in_random_order(running):
    client, h, st, p, wav = running
    seen = []
    for _ in range(30):
        seen.append(st["presentation"]["code"])
        st = respond(client, h, st, st["presentation"]["target"], wav=wav)["state"]
    assert sorted(seen) == sorted({c for c in seen}) and len(set(seen)) == 30
    assert all(c.startswith("A-") for c in seen)
    assert st["presentation"]["pass_no"] == 2


def test_pause_resume_and_end(running):
    client, h, st, p, _ = running
    sid = st["session"]["id"]
    assert client.post(f"/api/sessions/{sid}/pause", headers=h).json()["session"]["status"] == "paused"
    assert client.post(f"/api/sessions/{sid}/resume", headers=h).json()["session"]["status"] == "active"
    r = client.post(f"/api/sessions/{sid}/end", headers=h, json={"reason": "fatigue", "status": "abandoned"})
    assert r.json()["session"]["status"] == "abandoned"


def test_idempotent_upload(running):
    client, h, st, p, wav = running
    p0 = st["presentation"]
    data = {"client_uuid": "fixed-uuid-1", "cue_level": 0, "dev_transcript": p0["target"], "dev_confidence": 0.9}
    url = f"/api/sessions/{st['session']['id']}/trials/{p0['trial_id']}/response"
    first = client.post(url, headers=h, data=data, files={"audio": ("a.wav", wav, "audio/wav")}).json()
    again = client.post(url, headers=h, data=data, files={"audio": ("a.wav", wav, "audio/wav")}).json()
    assert again["action"] == "duplicate" and again["attempt"]["id"] == first["attempt"]["id"]


def test_audio_is_encrypted_at_rest(running, tmp_path):
    client, h, st, p, wav = running
    out = respond(client, h, st, st["presentation"]["target"], wav=wav)
    files = list((tmp_path / "audio").iterdir())
    assert len(files) == 1 and files[0].read_bytes()[:4] == b"GAV1" and b"RIFF" not in files[0].read_bytes()
    audio = client.get(f"/api/attempts/{out['attempt']['id']}/audio", headers=h)
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"


def test_no_consent_no_audio(client, tone_wav):
    h = login(client, "slp")
    client.demo_content()
    p = make_participant(client, h, code="PE02", consent=False)
    st = client.post("/api/sessions", headers=h, json={"participant_id": p["id"]}).json()
    out = respond(client, h, st, st["presentation"]["target"], wav=tone_wav)
    assert not out["attempt"]["has_audio"]


def test_role_access(client):
    hr = login(client, "r1")
    assert client.get("/api/participants", headers=hr).status_code == 403
    assert client.get("/api/participants").status_code == 401
    h = login(client, "slp")
    make_participant(client, h, code="PE01")
    make_participant(client, h, code="PE02")
    ph = login(client, "PE01", "patientpw1")
    other = client.get("/api/participants", headers=h).json()[1]["id"]
    assert client.get(f"/api/participants/{other}", headers=ph).status_code == 403


def test_export_is_pseudonymised(running):
    client, h, st, p, wav = running
    respond(client, h, st, st["presentation"]["target"], wav=wav)
    csv_text = client.get("/api/export/attempts.csv", headers=h).text
    assert "audio_ref" not in csv_text.splitlines()[0] and "transcript" in csv_text
    parts = client.get("/api/export/participants.csv", headers=h).text
    assert "PE01" in parts and "user_id" not in parts.splitlines()[0]


def test_validation_end_to_end(running):
    client, h, st, p, wav = running
    # a small session: 6 trials with a mix of outcomes
    script = [("ok", 0.9), ("bad", 0.9), ("ok", 0.5), ("ok", 0.9), ("bad", 0.9), ("ok", 0.9)]
    for kind, conf in script:
        target = st["presentation"]["target"]
        out = respond(client, h, st, target if kind == "ok" else "كلمة ثانية بعيدة", confidence=conf, wav=wav)
        if out["action"] == "await_clinician":
            out = client.post(f"/api/sessions/{st['session']['id']}/attempts/{out['attempt']['id']}/decide", headers=h, json={"decision": "ACCEPT"}).json()
        while out["action"] in ("next_cue", "repetition"):
            out = timeout(client, h, out["state"])
        st = out["state"]
    client.post(f"/api/sessions/{st['session']['id']}/end", headers=h, json={})

    ha = login(client, "admin")
    s = client.post("/api/validation/samples", headers=ha, json={"seed": 11, "items_per_session": 15, "second_rater_fraction": 0.5}).json()
    assert s["items"] == 6  # only response attempts in sampled trials (timeouts are not rated)
    h1, h2 = login(client, "r1"), login(client, "r2")
    # blinded item: no transcript, no GANVA output
    nxt = client.get(f"/api/validation/samples/{s['id']}/next?rater_no=1", headers=h1).json()["item"]
    assert set(nxt) == {"item_id", "attempt_id", "target", "english", "image_url", "has_audio"}
    while nxt:
        client.post("/api/validation/ratings", headers=h1, json={"item_id": nxt["item_id"], "rater_no": 1, "score": "correct"})
        nxt = client.get(f"/api/validation/samples/{s['id']}/next?rater_no=1", headers=h1).json()["item"]
    nxt = client.get(f"/api/validation/samples/{s['id']}/next?rater_no=2", headers=h1).json()["item"]
    r = client.post("/api/validation/ratings", headers=h1, json={"item_id": nxt["item_id"], "rater_no": 2, "score": "correct"})
    assert r.status_code == 409  # same person cannot be rater 2
    count2 = 0
    while nxt:
        client.post("/api/validation/ratings", headers=h2, json={"item_id": nxt["item_id"], "rater_no": 2, "score": "correct"})
        count2 += 1
        nxt = client.get(f"/api/validation/samples/{s['id']}/next?rater_no=2", headers=h2).json()["item"]
    assert count2 == 3

    m = client.get(f"/api/validation/samples/{s['id']}/metrics", headers=h).json()
    all_ = m["subgroups"]["all"]
    for pol in ("exclude", "shadow", "effective"):
        t = all_[pol]
        assert t["tp"] + t["fp"] + t["fn"] + t["tn"] == t["n"]
    assert all_["exclude"]["n"] == 5 and all_["shadow"]["n"] == 6
    assert all_["flag_rate"] == pytest.approx(1 / 6, abs=1e-3)
    assert set(m["confidence_sensitivity"]) == {"60%", "70%", "80%"}
    assert m["inter_rater"]["n"] == 3 and m["inter_rater"]["agreement"] == 1.0
    assert "PE01" in m["per_participant"]
