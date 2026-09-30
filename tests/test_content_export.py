"""export-content -> import-cues round trip, and demo placeholders never exported."""
import argparse

from sqlalchemy import select

from app import db, storage
from app.cli import export_content, import_cues
from app.content import tone_wav
from app.models import Associate, Stimulus


def _real_content():
    """Give item A-N01 real (non-demo) cues on top of the demo placeholders."""
    with db.SessionLocal() as s:
        st = s.scalars(select(Stimulus).where(Stimulus.code == "A-N01")).one()
        cues = {c.level: c for c in st.cues}
        cues[1].text = "واحد يضحّك الناس"
        cues[1].audio_path = storage.save_media(tone_wav(0.7, 300), "cues", ".wav")  # not a demo tone
        cues[3].image_path = storage.save_media(b"\x89PNG real visual cue", "cues", ".png")
        st.associates.append(Associate(text="المهرج", kind="morphological", source="test"))
        s.commit()


def test_export_skips_placeholders_and_round_trips(client, tmp_path):
    client.demo_content()
    _real_content()
    out = tmp_path / "export"
    export_content(argparse.Namespace(dir=str(out), force=False))
    names = {f.name for f in out.iterdir()}
    assert {"A-N01_L1.txt", "A-N01_L1.wav", "A-N01_L3.png", "associates.csv", "manifest.csv"} <= names
    assert "A-N02_L1.wav" not in names and "A-N02_L3.png" not in names  # demo tone / picture copy
    assert not any(n.startswith("A-N02_L5") for n in names)

    # wipe A-N01's content, then import the export back
    with db.SessionLocal() as s:
        st = s.scalars(select(Stimulus).where(Stimulus.code == "A-N01")).one()
        for c in st.cues:
            if c.level in (1, 3):
                c.text, c.audio_path, c.image_path = "", "", ""
        st.associates.clear()
        s.commit()
    import_cues(argparse.Namespace(dir=str(out)))
    with db.SessionLocal() as s:
        st = s.scalars(select(Stimulus).where(Stimulus.code == "A-N01")).one()
        cues = {c.level: c for c in st.cues}
        assert cues[1].text == "واحد يضحّك الناس"
        assert (storage.media_root() / cues[1].audio_path).read_bytes() == tone_wav(0.7, 300)
        assert (storage.media_root() / cues[3].image_path).read_bytes() == b"\x89PNG real visual cue"
        assert [a.text for a in st.associates] == ["المهرج"]
