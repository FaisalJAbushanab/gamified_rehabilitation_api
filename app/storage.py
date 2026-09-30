"""
File storage.

- Stimulus media (pictures, cue audio): not personal data, served as static
  files from ``data_dir/media``.
- Participant response audio: personal health data, encrypted at rest with
  AES-256-GCM in ``data_dir/audio``; only decrypted for authorised staff.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import get_settings
from .security import audio_key

_MAGIC = b"GAV1"


def media_root() -> Path:
    p = get_settings().data_dir / "media"
    p.mkdir(parents=True, exist_ok=True)
    return p


def audio_root() -> Path:
    p = get_settings().data_dir / "audio"
    p.mkdir(parents=True, exist_ok=True)
    return p


def media_url(rel_path: str) -> str | None:
    return f"/media/{rel_path}" if rel_path else None


def save_media(data: bytes, subdir: str, suffix: str) -> str:
    """Store a stimulus media file; returns the path relative to media_root."""
    rel = Path(subdir) / f"{uuid.uuid4().hex}{suffix.lower()}"
    dest = media_root() / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return rel.as_posix()


def save_encrypted_audio(data: bytes) -> str:
    nonce = os.urandom(12)
    blob = _MAGIC + nonce + AESGCM(audio_key()).encrypt(nonce, data, _MAGIC)
    ref = f"{uuid.uuid4().hex}.enc"
    path = audio_root() / ref
    path.write_bytes(blob)
    path.chmod(0o600)
    return ref


def load_encrypted_audio(ref: str) -> bytes:
    if "/" in ref or "\\" in ref or ".." in ref:
        raise ValueError("invalid audio reference")
    blob = (audio_root() / ref).read_bytes()
    if blob[:4] != _MAGIC:
        raise ValueError("not an encrypted audio file")
    return AESGCM(audio_key()).decrypt(blob[4:16], blob[16:], _MAGIC)


def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def to_wav16k(data: bytes, suffix: str = ".webm") -> tuple[bytes, int]:
    """Convert any browser recording (webm/opus, mp4/aac from iPad Safari, wav) to 16 kHz mono PCM WAV.

    Returns (wav_bytes, duration_ms).
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"in{suffix}"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(data)
        proc = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0 or not dst.exists():
            raise RuntimeError(f"audio conversion failed: {proc.stderr.strip()[:300]}")
        wav = dst.read_bytes()
    pcm_bytes = max(0, len(wav) - 44)
    return wav, int(pcm_bytes / 2 / 16000 * 1000)


def suffix_for_mime(mime: str, filename: str | None = None) -> str:
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[1].lower()
    m = (mime or "").lower()
    if "mp4" in m or "aac" in m or "m4a" in m:
        return ".mp4"
    if "wav" in m:
        return ".wav"
    if "ogg" in m:
        return ".ogg"
    return ".webm"
