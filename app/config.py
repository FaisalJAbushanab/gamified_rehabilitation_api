"""
Study configuration and deployment settings.

Study parameters (GANVA thresholds, adaptive controller, session rules, points)
are code constants on purpose: they are part of the protocol, change only
through version control, and are fingerprinted by ``config_hash()`` which is
stamped on every session and attempt record.

Deployment settings (database URL, secrets, ASR credentials) come from the
environment / ``.env`` via ``Settings``.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_VERSION = "2.0.0"
GANVA_VERSION = "2.0.0"
ADAPTIVE_VERSION = "2.0.0"

BASE_DIR = Path(__file__).resolve().parent.parent


class GanvaConfig(BaseModel):
    """Algorithm S3-3 constants."""

    stage1_threshold: float = 0.70  # ACCEPT at or above
    stage2_low: float = 0.50  # REJECT below; [stage2_low, stage1_threshold) goes to Stage 2
    asr_confidence_flag: float = 0.70  # FLAG below
    flag_when_confidence_missing: bool = True
    # D5: compare the target against the full transcript and every contiguous
    # n-gram of up to (target tokens + 1) tokens, and keep the best match.
    multiword_rule: Literal["full_only", "best_ngram"] = "best_ngram"


class AdaptiveConfig(BaseModel):
    """Algorithm S3-4 constants (D2-D4)."""

    window_scope: Literal["trials", "sessions"] = "trials"
    window_n: int = 3
    min_history: int = 3  # no adjustment until the window is full
    update_every: Literal["trial", "session"] = "trial"
    min_ms: int = 20_000
    max_ms: int = 60_000
    decrease_ms: int = 5_000
    soft_increase_ms: int = 5_000
    hard_increase_ms: int = 10_000
    acc_high: float = 0.70
    acc_low: float = 0.50
    fast_resp: float = 0.60
    timeout_high: float = 0.30
    start_ms_by_severity: dict[str, int] = Field(
        default_factory=lambda: {
            "mild": 30_000,
            "moderate": 45_000,
            "severe": 60_000,
            "profound": 60_000,
        }
    )
    default_start_ms: int = 45_000


class SessionConfig(BaseModel):
    """Session engine rules (D8-D11)."""

    duration_min: int = 60  # D9: passes repeat until this much active time, then the current trial finishes
    total_sessions: int = 15
    trained_list: str = "A"
    max_cue_level: int = 5  # presentations run 0 (cue-free) .. 5 (full auditory model)
    repetition_after_l5_failure: bool = True  # D10
    flag_clinician_wait_s: int = 20  # D11: then the default decision applies
    flag_default_decision: Literal["ACCEPT", "REJECT"] = "REJECT"
    vad_silence_ms: int = 1_500  # D8: recording ends after this much silence following speech
    vad_min_speech_ms: int = 150


class GamificationConfig(BaseModel):
    """D12."""

    points_by_cue_level: dict[int, int] = Field(
        default_factory=lambda: {0: 100, 1: 75, 2: 50, 3: 35, 4: 25, 5: 10}
    )
    repetition_points: int = 0
    level_thresholds: list[int] = Field(
        default_factory=lambda: [0, 1_000, 3_000, 6_000, 10_000, 15_000, 21_000, 28_000, 36_000, 45_000]
    )
    streak_badges: list[int] = Field(default_factory=lambda: [3, 5, 10])


class StudyConfig(BaseModel):
    ganva: GanvaConfig = Field(default_factory=GanvaConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    gamification: GamificationConfig = Field(default_factory=GamificationConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["dev", "prod", "test"] = "dev"
    data_dir: Path = BASE_DIR / "data"
    database_url: str = ""  # default: sqlite file in data_dir
    cors_origins: list[str] = ["http://127.0.0.1:5173", "http://localhost:5173"]

    jwt_secret: str = ""  # required in prod
    access_token_minutes: int = 60 * 8
    refresh_token_days: int = 14

    audio_key_b64: str = ""  # 32-byte AES-256 key, base64; required in prod

    asr_provider: Literal["google", "azure", "manual"] = "manual"
    asr_fallback: Literal["google", "azure", "none"] = "none"
    asr_locale: str = "ar-SA"  # Saudi Arabic (Najdi study group); ar-AE etc. for Gulf
    asr_timeout_s: float = 5.0
    google_api_key: str = ""
    azure_speech_key: str = ""
    azure_speech_region: str = ""

    # Dev only: lets sessions run before all cue assets exist.
    require_complete_content: bool = True

    @property
    def sqlalchemy_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'naming_app.db'}"


STUDY = StudyConfig()


def config_hash(study: StudyConfig = STUDY) -> str:
    payload = {
        "ganva_version": GANVA_VERSION,
        "adaptive_version": ADAPTIVE_VERSION,
        "study": study.model_dump(mode="json"),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@lru_cache
def get_settings() -> Settings:
    return Settings()
