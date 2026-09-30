"""
Demo placeholder assets and their detection.

`demo-content` fills missing cues with generated tones, a 1x1 picture and "[DEMO"
texts so development sessions can run. Those must never reach a participant, so
the content check recognises them by fingerprint (the generated files are
byte-identical every time), and an L3 visual cue that is merely a copy of the
item picture is also treated as a placeholder.
"""
from __future__ import annotations

import base64
import hashlib
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from . import storage

DEMO_TEXT_PREFIX = "[DEMO"
DEMO_TONES = {1: (0.6, 440), 2: (0.3, 660), 5: (0.5, 550)}  # cue level -> (seconds, Hz)
_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def tone_wav(seconds: float, freq: int) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "t.wav"
        subprocess.run(
            [storage.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             f"sine=frequency={freq}:duration={seconds}", "-ar", "16000", "-ac", "1", str(out)],
            check=True,
        )
        return out.read_bytes()


def placeholder_png() -> bytes:
    return _PLACEHOLDER_PNG


@lru_cache
def demo_fingerprints() -> frozenset[str]:
    blobs = [tone_wav(*spec) for spec in DEMO_TONES.values()] + [_PLACEHOLDER_PNG]
    return frozenset(hashlib.sha256(b).hexdigest() for b in blobs)


@lru_cache(maxsize=4096)
def _file_digest(rel_path: str) -> str | None:
    # media files are write-once (random names), so caching by path is safe
    path = storage.media_root() / rel_path
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def is_placeholder_file(rel_path: str) -> bool:
    return bool(rel_path) and _file_digest(rel_path) in demo_fingerprints()


def same_file_content(a: str, b: str) -> bool:
    """True when two media paths hold byte-identical files (e.g. an L3 cue copied from the item picture)."""
    if not a or not b:
        return False
    if a == b:
        return True
    da, db = _file_digest(a), _file_digest(b)
    return da is not None and da == db


def is_placeholder_text(text: str) -> bool:
    return (text or "").strip().startswith(DEMO_TEXT_PREFIX)
