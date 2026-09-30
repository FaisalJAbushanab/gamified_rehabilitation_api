"""Authentication (JWT), password hashing (argon2), role checks, secrets and audit."""
from __future__ import annotations

import base64
import secrets
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .db import get_db, utcnow
from .models import AuditLog, User

_hasher = PasswordHasher()
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, Exception):
        return False


def _dev_secret_file(settings: Settings, name: str, nbytes: int) -> str:
    """Dev/test only: persist a random secret under data_dir so restarts keep working."""
    path: Path = settings.data_dir / name
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(base64.b64encode(secrets.token_bytes(nbytes)).decode())
        path.chmod(0o600)
    return path.read_text().strip()


@lru_cache
def jwt_secret() -> str:
    s = get_settings()
    if s.jwt_secret:
        return s.jwt_secret
    if s.env == "prod":
        raise RuntimeError("JWT_SECRET must be set in production")
    return _dev_secret_file(s, "dev_jwt.secret", 48)


@lru_cache
def audio_key() -> bytes:
    s = get_settings()
    raw = s.audio_key_b64
    if not raw:
        if s.env == "prod":
            raise RuntimeError("AUDIO_KEY_B64 (32-byte AES-256 key, base64) must be set in production")
        raw = _dev_secret_file(s, "dev_audio.key", 32)
    key = base64.b64decode(raw)
    if len(key) != 32:
        raise RuntimeError("AUDIO_KEY_B64 must decode to exactly 32 bytes (AES-256)")
    return key


def make_token(user: User, kind: str = "access") -> str:
    s = get_settings()
    ttl = timedelta(minutes=s.access_token_minutes) if kind == "access" else timedelta(days=s.refresh_token_days)
    payload = {"sub": str(user.id), "role": user.role, "kind": kind, "exp": utcnow() + ttl}
    return jwt.encode(payload, jwt_secret(), algorithm="HS256")


def decode_token(token: str, kind: str = "access") -> dict:
    try:
        payload = jwt.decode(token, jwt_secret(), algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Your sign-in has expired. Sign in again.")
    if payload.get("kind") != kind:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong token type.")
    return payload


def current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to continue.")
    payload = decode_token(creds.credentials)
    user = db.get(User, int(payload["sub"]))
    if user is None or not user.active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "This account is not active.")
    return user


def require_roles(*roles: str):
    def dep(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Your account does not have access to this.")
        return user

    return dep


STAFF = ("admin", "clinician")


def audit(db: Session, actor: User | None, action: str, entity: str, entity_id: object = "", **detail) -> None:
    db.add(AuditLog(actor_id=actor.id if actor else None, action=action, entity=entity, entity_id=str(entity_id), detail=detail))
