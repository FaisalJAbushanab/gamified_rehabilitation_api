import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tone_wav() -> bytes:
    from app.storage import ffmpeg_exe

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "t.wav"
        subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=0.4",
                        "-ar", "16000", "-ac", "1", str(out)], check=True)
        return out.read_bytes()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ASR_PROVIDER", "manual")
    monkeypatch.setenv("REQUIRE_COMPLETE_CONTENT", "true")

    from app import config, db, security

    config.get_settings.cache_clear()
    security.jwt_secret.cache_clear()
    security.audio_key.cache_clear()
    db.reset_engine()

    from app import models  # noqa: F401
    from app.cli import demo_content, import_stimuli

    db.Base.metadata.create_all(db.get_engine())
    seed = Path(__file__).resolve().parent.parent / "seed" / "stimuli_s2.csv"
    import_stimuli(argparse.Namespace(csv=str(seed), pictures=None))

    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.models import User
    from app.security import hash_password

    with db.SessionLocal() as s:
        for name, role in [("admin", "admin"), ("slp", "clinician"), ("r1", "rater"), ("r2", "rater")]:
            s.add(User(username=name, password_hash=hash_password("password123"), role=role))
        s.commit()

    c = TestClient(create_app())
    c.demo_content = lambda: demo_content(argparse.Namespace())
    yield c
    db.reset_engine()
    config.get_settings.cache_clear()


def login(client, username, password="password123") -> dict:
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
