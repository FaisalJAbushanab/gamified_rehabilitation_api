"""ORM models. See IMPLEMENTATION_PLAN.md section 3."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, utcnow

ROLES = ("admin", "clinician", "rater", "patient")


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Participant(Base):
    __tablename__ = "participants"
    id: Mapped[int] = mapped_column(primary_key=True)
    study_code: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    arm: Mapped[str] = mapped_column(String(8))  # GAG | TSG
    severity: Mapped[str] = mapped_column(String(16))  # mild | moderate | severe | profound
    dysarthria: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_audio_retention: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    adaptive_limit_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_points: Mapped[int] = mapped_column(Integer, default=0)
    level: Mapped[int] = mapped_column(Integer, default=1)
    enrolled_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    user: Mapped[User | None] = relationship()


class Stimulus(Base):
    __tablename__ = "stimuli"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(16), unique=True)  # e.g. A-N01
    word_list: Mapped[str] = mapped_column(String(1))  # A trained | B untrained probe
    word_class: Mapped[str] = mapped_column(String(8))  # noun | verb
    number: Mapped[int] = mapped_column(Integer)
    english: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(64))
    grammatical: Mapped[str] = mapped_column(String(16), default="")  # gender or transitivity
    psycholinguistic: Mapped[dict] = mapped_column(JSON, default=dict)
    phonemic_cue_label: Mapped[str] = mapped_column(String(32), default="")
    semantic_cue_en: Mapped[str] = mapped_column(Text, default="")
    image_path: Mapped[str] = mapped_column(String(255), default="")
    probe_only: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    cues: Mapped[list["CueAsset"]] = relationship(back_populates="stimulus", cascade="all, delete-orphan", order_by="CueAsset.level")
    associates: Mapped[list["Associate"]] = relationship(back_populates="stimulus", cascade="all, delete-orphan")


CUE_TYPES = {1: "semantic", 2: "phonemic", 3: "visual", 4: "written", 5: "model"}


class CueAsset(Base):
    __tablename__ = "cue_assets"
    __table_args__ = (UniqueConstraint("stimulus_id", "level"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    stimulus_id: Mapped[int] = mapped_column(ForeignKey("stimuli.id", ondelete="CASCADE"))
    level: Mapped[int] = mapped_column(Integer)  # 1..5
    cue_type: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text, default="")
    audio_path: Mapped[str] = mapped_column(String(255), default="")
    image_path: Mapped[str] = mapped_column(String(255), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    stimulus: Mapped[Stimulus] = relationship(back_populates="cues")


class Associate(Base):
    __tablename__ = "associates"
    __table_args__ = (UniqueConstraint("stimulus_id", "text"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    stimulus_id: Mapped[int] = mapped_column(ForeignKey("stimuli.id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(24), default="variant")  # variant | morphological | synonym | semantic
    source: Mapped[str] = mapped_column(String(128), default="")
    raters_agreeing: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raters_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    stimulus: Mapped[Stimulus] = relationship(back_populates="associates")


class TherapySession(Base):
    __tablename__ = "therapy_sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"), index=True)
    session_no: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(12), default="active")  # active | paused | completed | abandoned
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    active_since: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    active_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    end_reason: Mapped[str] = mapped_column(String(64), default="")
    operator_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    app_version: Mapped[str] = mapped_column(String(16))
    ganva_version: Mapped[str] = mapped_column(String(16))
    adaptive_version: Mapped[str] = mapped_column(String(16))
    config_hash: Mapped[str] = mapped_column(String(32))
    config_snapshot: Mapped[dict] = mapped_column(JSON)
    order_seed: Mapped[int] = mapped_column(Integer)
    current_pass: Mapped[int] = mapped_column(Integer, default=1)
    start_limit_ms: Mapped[int] = mapped_column(Integer)
    end_limit_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    participant: Mapped[Participant] = relationship()
    trials: Mapped[list["Trial"]] = relationship(back_populates="session", order_by="Trial.order_index")


class Trial(Base):
    __tablename__ = "trials"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("therapy_sessions.id", ondelete="CASCADE"), index=True)
    stimulus_id: Mapped[int] = mapped_column(ForeignKey("stimuli.id"))
    pass_no: Mapped[int] = mapped_column(Integer)
    order_index: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(12), default="active")  # active | completed
    current_cue_level: Mapped[int] = mapped_column(Integer, default=0)
    current_is_repetition: Mapped[bool] = mapped_column(Boolean, default=False)
    pending_attempt_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # FLAG awaiting clinician
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(12), nullable=True)  # correct | incorrect
    final_cue_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timed_out_any: Mapped[bool] = mapped_column(Boolean, default=False)
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    points: Mapped[int] = mapped_column(Integer, default=0)

    session: Mapped[TherapySession] = relationship(back_populates="trials")
    stimulus: Mapped[Stimulus] = relationship()
    attempts: Mapped[list["Attempt"]] = relationship(back_populates="trial", order_by="Attempt.id")


class Attempt(Base):
    """One presentation (cue level) and its response or timeout."""

    __tablename__ = "attempts"
    id: Mapped[int] = mapped_column(primary_key=True)
    trial_id: Mapped[int] = mapped_column(ForeignKey("trials.id", ondelete="CASCADE"), index=True)
    client_uuid: Mapped[str] = mapped_column(String(64), unique=True)
    kind: Mapped[str] = mapped_column(String(12))  # response | timeout
    cue_level: Mapped[int] = mapped_column(Integer)
    is_repetition: Mapped[bool] = mapped_column(Boolean, default=False)
    window_ms: Mapped[int] = mapped_column(Integer)
    presented_at_client: Mapped[str] = mapped_column(String(40), default="")
    speech_onset_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    audio_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    audio_mime: Mapped[str] = mapped_column(String(64), default="")
    audio_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # ASR
    asr_provider: Mapped[str] = mapped_column(String(16), default="")
    asr_model: Mapped[str] = mapped_column(String(64), default="")
    asr_locale: Mapped[str] = mapped_column(String(16), default="")
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    asr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_best: Mapped[list] = mapped_column(JSON, default=list)
    asr_error: Mapped[str] = mapped_column(Text, default="")
    # GANVA
    normalized_transcript: Mapped[str] = mapped_column(Text, default="")
    stage1_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    matched_span: Mapped[str | None] = mapped_column(Text, nullable=True)
    stage_reached: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stage2_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    matched_associate: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ganva_decision: Mapped[str | None] = mapped_column(String(8), nullable=True)  # ACCEPT | REJECT | FLAG
    ganva_reason: Mapped[str] = mapped_column(String(24), default="")
    shadow_decision: Mapped[str | None] = mapped_column(String(8), nullable=True)
    # resolution used by the session engine
    effective_decision: Mapped[str | None] = mapped_column(String(8), nullable=True)  # ACCEPT | REJECT
    resolution: Mapped[str] = mapped_column(String(12), default="")  # auto | clinician | default | timeout | pending
    override_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    override_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # latency
    t_asr_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    t_ganva_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    t_server_total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    t_client_total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    feedback_shown_at_client: Mapped[str] = mapped_column(String(40), default="")
    # provenance
    ganva_version: Mapped[str] = mapped_column(String(16), default="")
    config_hash: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    trial: Mapped[Trial] = relationship(back_populates="attempts")


class AdaptiveEvent(Base):
    __tablename__ = "adaptive_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"), index=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("therapy_sessions.id"), nullable=True)
    trial_id: Mapped[int | None] = mapped_column(ForeignKey("trials.id"), nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    old_ms: Mapped[int] = mapped_column(Integer)
    new_ms: Mapped[int] = mapped_column(Integer)
    rule: Mapped[str] = mapped_column(String(24))
    n: Mapped[int] = mapped_column(Integer)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_response_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    timeout_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    version: Mapped[str] = mapped_column(String(16))


class FlagReview(Base):
    __tablename__ = "flag_reviews"
    id: Mapped[int] = mapped_column(primary_key=True)
    attempt_id: Mapped[int] = mapped_column(ForeignKey("attempts.id", ondelete="CASCADE"), unique=True)
    reviewer_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    human_decision: Mapped[str] = mapped_column(String(8))  # ACCEPT | REJECT
    note: Mapped[str] = mapped_column(Text, default="")
    reviewed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ValidationSample(Base):
    __tablename__ = "validation_samples"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    seed: Mapped[int] = mapped_column(Integer)
    items_per_session: Mapped[int] = mapped_column(Integer)
    second_rater_fraction: Mapped[float] = mapped_column(Float)
    description: Mapped[str] = mapped_column(Text, default="")


class ValidationItem(Base):
    __tablename__ = "validation_items"
    __table_args__ = (UniqueConstraint("sample_id", "attempt_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey("validation_samples.id", ondelete="CASCADE"), index=True)
    attempt_id: Mapped[int] = mapped_column(ForeignKey("attempts.id"))
    trial_id: Mapped[int] = mapped_column(ForeignKey("trials.id"))
    needs_second: Mapped[bool] = mapped_column(Boolean, default=False)


class ValidationRating(Base):
    __tablename__ = "validation_ratings"
    __table_args__ = (UniqueConstraint("item_id", "rater_no"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("validation_items.id", ondelete="CASCADE"), index=True)
    rater_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    rater_no: Mapped[int] = mapped_column(Integer)  # 1 first rater, 2 second rater, 0 consensus
    score: Mapped[str] = mapped_column(String(10))  # correct | incorrect
    rated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class GamificationEvent(Base):
    __tablename__ = "gamification_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"), index=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("therapy_sessions.id"), nullable=True)
    trial_id: Mapped[int | None] = mapped_column(ForeignKey("trials.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(12))  # points | badge | level
    value: Mapped[int] = mapped_column(Integer, default=0)
    code: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(48))
    entity: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[str] = mapped_column(String(32), default="")
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
